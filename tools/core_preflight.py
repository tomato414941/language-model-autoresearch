"""Check every official CORE prompt on CPU without running or scoring a model."""

import argparse
from collections import Counter
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/core-preflight.json')
    parser.add_argument('--inspect-spans', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    tasks = prepare.verify_core_assets()
    evaluator = prepare._load_nanochat_core()
    core_globals = evaluator.__globals__["evaluate_task"].__globals__
    evaluate_example = core_globals["evaluate_example"]
    tokenizer = prepare.Tokenizer.from_directory()
    records = []
    active_record = [None]
    active_example = [None]

    if args.inspect_spans:
        def observe_batch(original):
            def observed(*args, **kwargs):
                result = original(*args, **kwargs)
                tokens, starts, ends = result
                record = active_record[0]
                record['span_sequences'] += len(tokens)
                empty = [i for i, (start, end) in enumerate(zip(starts, ends)) if start == end]
                record['empty_span_sequences'] += len(empty)
                record['nonpositive_start_sequences'] += sum(start <= 0 for start in starts)
                if empty:
                    record['examples_with_empty_span'] += 1
                    if len(record['empty_span_examples']) < 10:
                        record['empty_span_examples'].append(dict(
                            index=active_example[0], options=empty,
                            starts=list(starts), ends=list(ends),
                            sequence_lengths=[len(sequence) for sequence in tokens],
                        ))
                return result
            return observed

        for name in ('batch_sequences_mc', 'batch_sequences_schema', 'batch_sequences_lm'):
            core_globals[name] = observe_batch(core_globals[name])

    def inspect_task(model, tokenizer, data, device, metadata):
        label = tasks[len(records)]["label"]
        record = dict(task=label, examples=len(data), max_sequence_len=0, max_batch_tokens=0,
                      input_sequences=0, input_tokens=0)
        if args.inspect_spans:
            record.update(span_sequences=0, empty_span_sequences=0, examples_with_empty_span=0,
                          nonpositive_start_sequences=0, empty_span_examples=[], task_type=metadata['task_type'])
            active_record[0] = record
        shapes = Counter()
        started = time.monotonic()

        class ShapeRecorder:
            def __call__(self, input_ids):
                batch, length = input_ids.shape
                shapes[batch, length] += 1
                record['input_sequences'] += batch
                record['input_tokens'] += input_ids.numel()
                record["max_sequence_len"] = max(record["max_sequence_len"], input_ids.shape[1])
                record["max_batch_tokens"] = max(record["max_batch_tokens"], input_ids.numel())
                raise PromptPrepared

        recorder = ShapeRecorder()
        for index in range(len(data)):
            active_example[0] = index
            try:
                evaluate_example(index, recorder, tokenizer, data, "cpu", metadata)
            except PromptPrepared:
                pass
            else:
                raise RuntimeError("Prompt preparation did not reach the model interface.")
            if (index + 1) % 1000 == 0:
                print(f"Prompt preflight: {label} {index + 1}/{len(data)}", flush=True)
        record["seconds"] = time.monotonic() - started
        record['shapes'] = [dict(batch=batch, sequence_len=length, calls=calls)
                            for (batch, length), calls in sorted(shapes.items())]
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != 'shapes'}), flush=True)
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
        total_input_sequences=sum(record['input_sequences'] for record in records),
        total_input_tokens=sum(record['input_tokens'] for record in records),
        max_sequence_len=max(record["max_sequence_len"] for record in records),
        max_batch_tokens=max(record["max_batch_tokens"] for record in records),
    )
    if args.inspect_spans:
        report['span_inspection'] = dict(
            core_source_sha256=prepare._file_sha256(Path(prepare._core_source_dir()) / 'nanochat/core_eval.py'),
            span_sequences=sum(record['span_sequences'] for record in records),
            empty_span_sequences=sum(record['empty_span_sequences'] for record in records),
            examples_with_empty_span=sum(record['examples_with_empty_span'] for record in records),
            nonpositive_start_sequences=sum(record['nonpositive_start_sequences'] for record in records),
            purpose='Observational metadata only: original batch results are returned unchanged; no predictions, scores or adoption gates.',
        )
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print("Complete CORE prompt preflight finished; no model quality was measured.", flush=True)


if __name__ == "__main__":
    main()
