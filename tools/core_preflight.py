"""Check every official CORE prompt on CPU without running or scoring a model."""

import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import prepare


class PromptPrepared(Exception):
    pass


def main():
    torch.set_num_threads(2)
    tasks = prepare.verify_core_assets()
    evaluator = prepare._load_nanochat_core()
    evaluate_example = evaluator.__globals__["evaluate_task"].__globals__["evaluate_example"]
    tokenizer = prepare.Tokenizer.from_directory()
    records = []

    def inspect_task(model, tokenizer, data, device, metadata):
        label = tasks[len(records)]["label"]
        record = dict(task=label, examples=len(data), max_sequence_len=0, max_batch_tokens=0)
        started = time.monotonic()

        class ShapeRecorder:
            def __call__(self, input_ids):
                record["max_sequence_len"] = max(record["max_sequence_len"], input_ids.shape[1])
                record["max_batch_tokens"] = max(record["max_batch_tokens"], input_ids.numel())
                raise PromptPrepared

        recorder = ShapeRecorder()
        for index in range(len(data)):
            try:
                evaluate_example(index, recorder, tokenizer, data, "cpu", metadata)
            except PromptPrepared:
                pass
            else:
                raise RuntimeError("Prompt preparation did not reach the model interface.")
            if (index + 1) % 1000 == 0:
                print(f"Prompt preflight: {label} {index + 1}/{len(data)}", flush=True)
        record["seconds"] = time.monotonic() - started
        records.append(record)
        print(json.dumps(record), flush=True)
        # No model prediction exists in this preflight. Discard the synthetic aggregate.
        return 0.0

    evaluator.__globals__["evaluate_task"] = inspect_task
    evaluator.__globals__["print0"] = lambda *args, **kwargs: None
    evaluator(object(), tokenizer, "cpu", max_per_task=-1)
    report = dict(
        nanochat_revision=prepare.CORE_NANOCHAT_REVISION,
        bundle_sha256=prepare.CORE_BUNDLE_SHA256,
        tokenizer_sha256=prepare._file_sha256(Path(prepare.TOKENIZER_DIR) / "tokenizer.pkl"),
        tasks=records,
        total_examples=sum(record["examples"] for record in records),
        max_sequence_len=max(record["max_sequence_len"] for record in records),
        max_batch_tokens=max(record["max_batch_tokens"] for record in records),
    )
    destination = ROOT / "results" / "core-preflight.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print("Complete CORE prompt preflight finished; no model quality was measured.", flush=True)


if __name__ == "__main__":
    main()
