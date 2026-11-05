"""Count every correct-answer BPB measurement on the CPU, without running a model."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from gpu_config import active_gpu

from evaluation.count_tokens import TokenObserver, atomic_json, observe_instances
from evaluation.gold_bpb import gold_request_indices
from evaluation.protocol import read_requests, sha256_file
from evaluation.tokenizer_adapter import TokenizerAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", default=str(active_gpu().results_directory() / "olmes-native-requests"))
    parser.add_argument("--tokenizer", default=str(active_gpu().cache_directory("tokenizer") / "tokenizer.pkl"))
    parser.add_argument("--output", default=str(active_gpu().results_directory() / "olmes-gold-token-counts.json"))
    args = parser.parse_args()
    directory = Path(args.requests)
    manifest = json.loads((directory/"manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Use the complete native request bundle.")
    observer = TokenObserver(TokenizerAdapter(args.tokenizer, manifest["max_length"]),
                             manifest["max_length"], manifest["batch_size"])
    report = dict(complete=False, started_at=datetime.now(timezone.utc).isoformat(),
                  computation="correct_answer_only", olmes_revision=manifest["olmes_revision"],
                  suites=manifest["suites"], tokenizer_sha256=sha256_file(args.tokenizer),
                  source_manifest_sha256=sha256_file(directory/"manifest.json"), tasks=[],
                  model_weights_loaded=False, quality_evaluations=0)
    output = Path(args.output)
    for entry in manifest["tasks"]:
        _, instances = read_requests(directory, entry)
        selected = gold_request_indices(instances, entry["resolved_task_config"])
        if len(selected) != entry["examples"]:
            raise ValueError("Keep every original example exactly once.")
        observer.reset_counts()
        observer._max_length = entry["resolved_task_config"].get("model_max_length") or manifest["max_length"]
        observe_instances(observer, [instances[index] for index in selected])
        task = dict(index=entry["index"], alias=entry["alias"], examples=entry["examples"],
                    source_requests=entry["requests"], computed_requests=len(selected),
                    selected_source_indices_sha256=hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
                    **observer.statistics())
        if task["invalid_windows"]:
            raise ValueError("A correct-answer measurement has an invalid native context window.")
        report["tasks"].append(task)
        atomic_json(output, report)
        print(json.dumps({name:task[name] for name in
                          ("index", "alias", "examples", "source_requests", "computed_requests", "padded_forward_tokens")}), flush=True)
    if len(report["tasks"]) != len(manifest["tasks"]):
        raise ValueError("Count every native leaf task.")
    report.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat())
    for name in ("examples", "source_requests", "computed_requests", "context_tokens", "continuation_tokens",
                 "padded_forward_tokens", "forward_batches", "invalid_windows"):
        report["total_"+name] = sum(task[name] for task in report["tasks"])
    atomic_json(output, report)
    print(json.dumps({key:value for key,value in report.items() if key != "tasks"}), flush=True)


if __name__ == "__main__":
    main()
