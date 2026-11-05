"""Serialize complete native OLMES requests before renting the GPU."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from gpu_config import active_gpu
import shutil
import time

from oe_eval import run_eval
from oe_eval.utils import save_jsonl

from evaluation.count_tokens import BATCH_SIZE, MAX_LENGTH, OLMES_REVISION, SUITES, TokenObserver, atomic_json
from evaluation.tokenizer_adapter import TokenizerAdapter


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", default=str(active_gpu().results_directory() / "olmes-token-counts"))
    parser.add_argument("--output", default=str(active_gpu().results_directory() / "olmes-native-requests"))
    parser.add_argument("--tokenizer", default=str(active_gpu().cache_directory("tokenizer") / "tokenizer.pkl"))
    args = parser.parse_args()
    counts_dir, output = Path(args.counts), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    counts = json.loads((counts_dir / "report.json").read_text())
    config_file = counts_dir / "native-configs.json"
    config = json.loads(config_file.read_text())
    assert counts["complete"] and counts["total_invalid_windows"] == 0 and not counts["errors"]
    assert counts["olmes_revision"] == OLMES_REVISION and tuple(counts["suites"]) == SUITES
    assert file_sha256(config_file) == counts["native_configs_sha256"]
    assert file_sha256(args.tokenizer) == counts["tokenizer_sha256"]
    assert len(counts["tasks"]) == len(config["tasks_config"]) == counts["expected_leaf_tasks"]
    model = TokenObserver(TokenizerAdapter(args.tokenizer, MAX_LENGTH))
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "olmes_revision": OLMES_REVISION, "suites": list(SUITES),
        "native_configs_sha256": counts["native_configs_sha256"],
        "tokenizer_sha256": counts["tokenizer_sha256"],
        "count_report_sha256": file_sha256(counts_dir / "report.json"),
        "native_config": config, "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
        "tasks": [], "complete": False, "quality_evaluations": 0,
    }
    for index, (task_config, counted) in enumerate(zip(config["tasks_config"], counts["tasks"], strict=True)):
        metadata_file = output / f"task-{index:03d}.json"
        if metadata_file.exists():
            cached = json.loads(metadata_file.read_text())
            assert cached["native_config"] == task_config
            assert file_sha256(output / cached["requests_file"]) == cached["requests_sha256"]
            report["tasks"].append(cached)
            continue
        started = time.monotonic()
        task = run_eval.load_task(task_config)
        print(json.dumps({"preparing_task": counted["alias"], "index": index}), flush=True)
        task.download()
        task.set_model(model, config["model_config"])
        task.build_all_requests()
        instances = task._instances
        assert instances is not None and len(instances) == counted["requests"]
        assert task._task_hash == counted["task_hash"]
        fingerprints = {name: getattr(split, "_fingerprint", None) for name, split in task.dataset.items()}
        assert fingerprints == counted["dataset_fingerprints"]
        native_file = output / f"task-{index:03d}-requests.jsonl"
        # Use the native serializer without its optional log-only prompt truncation.
        save_jsonl(native_file, [instance.to_dict() for instance in instances])
        native_sha = file_sha256(native_file)
        compressed_file = native_file.with_suffix(native_file.suffix + ".gz")
        with native_file.open("rb") as source, compressed_file.open("wb") as destination:
            with gzip.GzipFile(filename="", mode="wb", fileobj=destination, compresslevel=1, mtime=0) as compressor:
                shutil.copyfileobj(source, compressor)
        entry = {
            "index": index, "alias": counted["alias"], "native_config": task_config,
            "resolved_task_config": task.task_config, "task_hash": task._task_hash,
            "examples": counted["examples"], "requests": counted["requests"],
            "dataset_fingerprints": fingerprints,
            "requests_file": compressed_file.name,
            "requests_sha256": file_sha256(compressed_file),
            "uncompressed_requests_sha256": native_sha,
            "uncompressed_bytes": native_file.stat().st_size,
            "compressed_bytes": compressed_file.stat().st_size,
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(metadata_file, entry)
        native_file.unlink()
        report["tasks"].append(entry)
        atomic_json(output / "manifest.json", report)
        print(json.dumps({key: entry[key] for key in
                          ("index", "alias", "examples", "requests", "compressed_bytes", "elapsed_seconds")}), flush=True)
        del instances, task
    report["complete"] = True
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["total_requests"] = sum(task["requests"] for task in report["tasks"])
    report["total_examples"] = sum(task["examples"] for task in report["tasks"])
    report["total_compressed_bytes"] = sum(task["compressed_bytes"] for task in report["tasks"])
    assert report["total_requests"] == counts["total_requests"]
    assert report["total_examples"] == counts["total_examples"]
    atomic_json(output / "manifest.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("native_config", "tasks")}), flush=True)


if __name__ == "__main__":
    main()
