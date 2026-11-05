"""Count every native OLMES request on the CPU without loading or scoring a model."""

import argparse
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from gpu_config import active_gpu
import shlex
import time
import traceback

import torch
import transformers
from lm_eval.api.model import LM
from oe_eval import launch, run_eval
from oe_eval.models.eleuther_huggingface import HFLM_Verbose

from evaluation.tokenizer_adapter import TokenizerAdapter

SUITES = ("olmo3:base_easy:qa_bpb", "olmo3:base_easy:math_bpb", "olmo3:base_easy:code_bpb")
OLMES_REVISION = "5a51f502d463b8cdc4a2dcad7d7096c41ff1197e"
MAX_LENGTH = 20480
BATCH_SIZE = 8


class TokensObserved(Exception):
    """Stop immediately after native request encoding, before inference or metrics."""


class TokenObserver(HFLM_Verbose):
    def __init__(self, tokenizer, max_length=MAX_LENGTH, batch_size=BATCH_SIZE):
        LM.__init__(self)
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.tokenizer_size = len(tokenizer)
        self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
        self.add_bos_token = False
        self.custom_prefix_token_id = None
        self.logits_cache = True
        self._max_length = max_length
        self.batch_size_per_gpu = batch_size
        self._device = torch.device("cpu")
        self._rank = 0
        self._world_size = 1
        self.reset_counts()

    def reset_counts(self):
        self.groups = {}
        self.windows = 0
        self.context_tokens = 0
        self.continuation_tokens = 0
        self.scored_tokens = 0
        self.effective_input_tokens = 0
        self.left_truncated_windows = 0
        self.invalid_windows = 0
        self.max_raw_length = 0
        self.max_continuation_length = 0

    def _loglikelihood_tokens_verbose(self, requests, **kwargs):
        for _, context, continuation in requests:
            raw_length = len(context) + len(continuation)
            effective = min(self.max_length, raw_length - 1)
            self.windows += 1
            self.context_tokens += len(context)
            self.continuation_tokens += len(continuation)
            self.scored_tokens += sum(token >= 0 for token in continuation)
            self.effective_input_tokens += max(0, effective)
            self.left_truncated_windows += raw_length - 1 > self.max_length
            self.invalid_windows += not context or not continuation or len(continuation) > self.max_length
            self.max_raw_length = max(self.max_raw_length, raw_length)
            self.max_continuation_length = max(self.max_continuation_length, len(continuation))
            # The native causal logits cache groups by the complete, untruncated input.
            key = hashlib.sha256(array("q", context + continuation[:-1]).tobytes()).digest()
            self.groups.setdefault(key, (raw_length, max(0, effective)))
        raise TokensObserved()

    def statistics(self):
        ordered = sorted(self.groups.values(), key=lambda entry: -entry[0])
        shapes = Counter()
        padded_tokens = 0
        attention_elements = 0
        for start in range(0, len(ordered), self.batch_size):
            chunk = ordered[start:start + self.batch_size]
            length = max(entry[1] for entry in chunk)
            shape = (len(chunk), length)
            shapes[shape] += 1
            padded_tokens += shape[0] * shape[1]
            attention_elements += shape[0] * shape[1] ** 2
        return {
            "encoded_windows": self.windows,
            "context_tokens": self.context_tokens,
            "continuation_tokens": self.continuation_tokens,
            "scored_tokens": self.scored_tokens,
            "effective_input_tokens_before_cache": self.effective_input_tokens,
            "unique_model_inputs": len(ordered),
            "input_tokens_after_native_logits_cache": sum(entry[1] for entry in ordered),
            "padded_forward_tokens": padded_tokens,
            "padded_attention_elements": attention_elements,
            "forward_batches": sum(shapes.values()),
            "max_raw_length": self.max_raw_length,
            "max_continuation_length": self.max_continuation_length,
            "left_truncated_windows": self.left_truncated_windows,
            "invalid_windows": self.invalid_windows,
            "forward_shapes": [{"batch": batch, "length": length, "count": count}
                               for (batch, length), count in sorted(shapes.items())],
        }


def resolve_native_configs():
    arguments = launch._parser.parse_args([
        "--model", "autoresearch", "--task", *SUITES, "--dry-run",
        "--batch-size", str(BATCH_SIZE),
        "--model-args", json.dumps({"max_length": MAX_LENGTH, "add_bos_token": False}),
    ])
    command = launch.launch_eval(vars(arguments))["command"]
    parts = shlex.split(command)
    assert parts[:3] == ["python", "-m", "oe_eval.run_eval"], parts[:3]
    return run_eval.process_eval_args(vars(run_eval._parser.parse_args(parts[3:])))


def observe_instances(observer, instances, chunk_size=256):
    grouped = defaultdict(list)
    for instance in instances:
        grouped[instance.request_type].append(instance.request)
    for kind, requests in grouped.items():
        if kind not in ("loglikelihood", "loglikelihood_rolling"):
            raise ValueError(f"A BPB suite requested unexpected inference: {kind}")
        method = getattr(observer, kind + "_verbose")
        for start in range(0, len(requests), chunk_size):
            try:
                method(requests[start:start + chunk_size], disable_tqdm=True)
            except TokensObserved:
                continue
            raise AssertionError("The observer must stop before scoring any requests.")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(active_gpu().results_directory() / "olmes-token-counts"))
    parser.add_argument("--tokenizer", default=str(active_gpu().cache_directory("tokenizer") / "tokenizer.pkl"))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config_path = output / "native-configs.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = resolve_native_configs()
        atomic_json(config_path, config)
    tasks = config["tasks_config"]
    tokenizer_path = Path(args.tokenizer)
    tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    existing_report = output / "report.json"
    if existing_report.exists():
        recorded = json.loads(existing_report.read_text())
        expected = {"tokenizer_sha256": tokenizer_sha256, "olmes_revision": OLMES_REVISION,
                    "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
                    "native_configs_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise ValueError("Use a separate count directory for different evaluation inputs.")
    tokenizer = TokenizerAdapter(tokenizer_path, MAX_LENGTH)
    observer = TokenObserver(tokenizer)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "olmes_revision": OLMES_REVISION,
        "suites": list(SUITES), "expected_leaf_tasks": len(tasks),
        "native_configs_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "tokenizer_sha256": tokenizer_sha256,
        "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
        "add_bos_token": False, "logits_cache": True,
        "quality_evaluations": 0, "model_weights_loaded": False,
        "all_native_tasks_examples_prompts_and_scoring_unchanged": True,
        "tasks": [], "errors": [], "complete": False,
    }
    for index, task_config in enumerate(tasks):
        task_path = output / f"task-{index:03d}.json"
        if task_path.exists():
            task_report = json.loads(task_path.read_text())
            assert task_report["native_config"] == task_config
            if "request_document_ids" not in task_report:
                task_report["request_document_ids"] = task_report["examples"]
                if task_report["resolved_task_config"]["metric_kwargs"].get("uncond_docid_offset"):
                    assert task_report["examples"] % 2 == 0
                    task_report["examples"] //= 2
                atomic_json(task_path, task_report)
            report["tasks"].append(task_report)
            continue
        observer.reset_counts()
        task_started = time.monotonic()
        alias = task_config.get("metadata", {}).get("alias", task_config["task_name"])
        print(json.dumps({"starting_task": alias, "index": index, "of": len(tasks)}), flush=True)
        try:
            task = run_eval.load_task(task_config)
            task.download()
            task.set_model(observer, config["model_config"])
            task.build_all_requests()
            instances = task._instances
            assert instances is not None and len(instances) > 0
            prepared = time.monotonic()
            observe_instances(observer, instances)
            stats = observer.statistics()
            offset = task.task_config["metric_kwargs"].get("uncond_docid_offset")
            documents = {instance.doc_id for instance in instances}
            examples = {index - offset if offset and index >= offset else index for index in documents}
            task_report = {
                "index": index, "alias": alias, "native_config": task_config,
                "resolved_task_config": task.task_config, "task_hash": task._task_hash,
                "examples": len(examples), "request_document_ids": len(documents),
                "requests": len(instances),
                "request_types": dict(Counter(instance.request_type for instance in instances)),
                "dataset_fingerprints": {name: getattr(split, "_fingerprint", None)
                                         for name, split in getattr(task, "dataset", {}).items()},
                "download_and_build_seconds": prepared - task_started,
                "tokenize_seconds": time.monotonic() - prepared,
                **stats,
            }
            atomic_json(task_path, task_report)
            report["tasks"].append(task_report)
            print(json.dumps({key: task_report[key] for key in
                              ("index", "alias", "examples", "requests", "padded_forward_tokens",
                               "max_raw_length", "invalid_windows", "tokenize_seconds")}), flush=True)
            del instances, task
        except Exception as error:
            item = {"index": index, "alias": alias, "type": type(error).__name__, "message": str(error)}
            report["errors"].append(item)
            print(json.dumps(item), flush=True)
            traceback.print_exc()
        atomic_json(output / "report.json", report)
    report["complete"] = len(report["tasks"]) == len(tasks) and not report["errors"]
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds_this_process"] = time.monotonic() - started
    for key in ("examples", "requests", "encoded_windows", "context_tokens", "continuation_tokens",
                "scored_tokens", "padded_forward_tokens", "padded_attention_elements",
                "forward_batches", "invalid_windows"):
        report["total_" + key] = sum(task[key] for task in report["tasks"])
    atomic_json(output / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("tasks", "errors")}), flush=True)
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
