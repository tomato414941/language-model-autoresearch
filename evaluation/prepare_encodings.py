"""Prepare a native gold-answer tokenization cache on CPU without loading model weights."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from evaluation.count_tokens import OLMES_REVISION, SUITES, TokenObserver, atomic_json, observe_instances
from evaluation.encoding_cache import EncodingWriter, input_identity, use_encodings
from evaluation.gold_bpb import gold_request_indices
from evaluation.protocol import read_requests
from evaluation.tokenizer_adapter import TokenizerAdapter


def prepare_cache(requests, tokenizer, output):
    requests, output = Path(requests), Path(output)
    manifest_path = requests / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["complete"] or manifest["olmes_revision"] != OLMES_REVISION or tuple(manifest["suites"]) != SUITES:
        raise ValueError("Prepare the complete pinned native requests before caching their encodings.")
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("Keep every existing native encoding preparation artifact.")
    observer = TokenObserver(TokenizerAdapter(tokenizer, manifest["max_length"]),
                             manifest["max_length"], manifest["batch_size"])
    writer = EncodingWriter(output, input_identity(observer, tokenizer, manifest_path))
    started = time.monotonic()
    report = dict(complete=False, started_at=datetime.now(timezone.utc).isoformat(), identity=writer.identity,
                  quality_evaluations=0, model_weights_loaded=False, tasks=[], examples=0)
    atomic_json(report_path, report)
    try:
        with use_encodings(observer, writer):
            for entry in manifest["tasks"]:
                _, instances = read_requests(requests, entry)
                indices = gold_request_indices(instances, entry["resolved_task_config"])
                if len(indices) != entry["examples"]:
                    raise ValueError("Cache every original example's correct-answer measurement.")
                observer.reset_counts()
                observer._max_length = entry["resolved_task_config"].get("model_max_length") or manifest["max_length"]
                observe_instances(observer, [instances[index] for index in indices])
                report["tasks"].append(dict(alias=entry["alias"], examples=len(indices), statistics=observer.statistics()))
                report["examples"] += len(indices)
                atomic_json(report_path, report)
                print(json.dumps(dict(alias=entry["alias"], examples=len(indices), encoded=True)), flush=True)
        if report["examples"] != manifest["total_examples"]:
            raise ValueError("The cache must cover all original examples.")
        writer.finish()
    finally:
        writer.close()
    report.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - started, native_tasks=len(report["tasks"]),
                  cache_bytes=output.stat().st_size, recorded_calls=writer.calls,
                  unique_encodings=writer.metadata["unique_encodings"], stored_tokens=writer.stored_tokens)
    atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = prepare_cache(args.requests, args.tokenizer, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}), flush=True)


if __name__ == "__main__":
    main()
