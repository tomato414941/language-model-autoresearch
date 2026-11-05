from gpu_config import active_gpu
"""Verify the token geometry of the exact serialized native request bundle."""

import argparse
import json
from pathlib import Path

from evaluation.count_tokens import BATCH_SIZE, MAX_LENGTH, OLMES_REVISION, SUITES, TokenObserver, atomic_json, observe_instances
from evaluation.protocol import read_requests, sha256_file
from evaluation.tokenizer_adapter import TokenizerAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", default="results/olmes-native-requests")
    parser.add_argument("--counts", default="results/olmes-token-counts/report.json")
    parser.add_argument("--output", default="results/olmes-frozen-request-token-counts.json")
    parser.add_argument("--tokenizer", default=str(active_gpu().cache_directory("tokenizer") / "tokenizer.pkl"))
    args = parser.parse_args()
    directory = Path(args.requests)
    manifest = json.loads((directory / "manifest.json").read_text())
    previous = json.loads(Path(args.counts).read_text())
    assert manifest["complete"] and previous["complete"]
    assert sha256_file(args.tokenizer) == manifest["tokenizer_sha256"] == previous["tokenizer_sha256"]
    observer = TokenObserver(TokenizerAdapter(args.tokenizer, MAX_LENGTH), MAX_LENGTH, BATCH_SIZE)
    report = {key: previous[key] for key in
              ("olmes_revision", "suites", "native_configs_sha256", "tokenizer_sha256", "max_length", "batch_size", "expected_leaf_tasks")}
    report.update(tasks=[], errors=[], complete=False, quality_evaluations=0, model_weights_loaded=False)
    assert report["olmes_revision"] == OLMES_REVISION and tuple(report["suites"]) == SUITES
    differences = []
    for entry, counted in zip(manifest["tasks"], previous["tasks"], strict=True):
        _, instances = read_requests(directory, entry)
        observer.reset_counts()
        observer._max_length = entry["resolved_task_config"].get("model_max_length") or MAX_LENGTH
        observe_instances(observer, instances)
        stats = observer.statistics()
        for key in ("context_tokens", "continuation_tokens", "padded_forward_tokens", "forward_batches", "invalid_windows"):
            if stats[key] != counted[key]:
                differences.append(dict(index=entry["index"], alias=entry["alias"], field=key,
                                        initial=counted[key], serialized=stats[key]))
        report["tasks"].append(dict(index=entry["index"], alias=entry["alias"], examples=entry["examples"],
                                    requests=entry["requests"], **stats))
        print(json.dumps({key: report["tasks"][-1][key] for key in
                          ("index", "alias", "padded_forward_tokens", "invalid_windows")}), flush=True)
    report["complete"] = len(report["tasks"]) == report["expected_leaf_tasks"]
    report["matches_initial_native_counts"] = not differences
    report["initial_count_differences"] = differences
    for key in ("examples", "requests", "context_tokens", "continuation_tokens", "padded_forward_tokens", "invalid_windows"):
        report["total_" + key] = sum(task[key] for task in report["tasks"])
    atomic_json(Path(args.output), report)
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}), flush=True)


if __name__ == "__main__":
    main()
