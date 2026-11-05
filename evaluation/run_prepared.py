"""Evaluate one experiment checkpoint once on all three native OLMES suites."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from evaluation.model import CheckpointModel
from evaluation.arc_macro import APPROVED_CONFIG_FIX
from evaluation.protocol import evaluate_prepared, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("The complete quality evaluation runs on the rented GPU.")
    model = CheckpointModel(args.checkpoint, args.tokenizer)
    torch.cuda.reset_peak_memory_stats()
    suites, manifest = evaluate_prepared(model, args.requests, args.output)
    report = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "olmes_revision": manifest["olmes_revision"],
        "approved_config_fix": APPROVED_CONFIG_FIX,
        "suites": suites, "complete": True, "leaf_tasks": len(manifest["tasks"]),
        "examples": manifest["total_examples"], "requests": manifest["total_requests"],
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "model_source_sha256": model.source_sha256,
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "native_request_manifest_sha256": sha256_file(Path(args.requests) / "manifest.json"),
        "metrics_all_sha256": sha256_file(Path(args.output) / "metrics-all.jsonl"),
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
    }
    with (Path(args.output) / "summary.json").open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
