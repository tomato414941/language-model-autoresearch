# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Work on `main`**: do not create branches.
2. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — the fixed frame (allowed data, time budget, evaluation, acceptance threshold) and a default tokenizer and dataloader. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
3. **Rent the GPU**: experiments run on one rented H100 SXM, never on the machine you run on (see the rules at the end). Never write hosts, keys or IP addresses into the repository.
4. **Prepare the data and evaluations**: on the processing server (see the rules at the end), run `uv run prepare.py --num-shards 10` to train the tokenizer and prepare the complete pinned CORE evaluation. Bring the tokenizer and the entire CORE cache when renting, then run `uv run prepare.py` on the rented machine to download the remaining training shards. Finish the CORE integration and its CPU checks before measuring the baseline.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 30 minutes** (wall clock training time, excluding startup/compilation): one run in each daily rental of at most 1 hour. You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, the tokenizer, the sequence length, how the data is read, etc. The tokenizer, `MAX_SEQ_LEN` and dataloader in `prepare.py` are defaults you may replace; CORE must run unchanged on whatever tokenizer you use.
- Prepare anything beforehand that needs only the CPU, such as a tokenizer, tables or statistics built from the allowed training data. Run it on the processing server; its code may live in files of its own. It does not count toward the 30-minute budget.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed allowed data, time budget, evaluation and acceptance threshold.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness, task set, examples, prompts, few-shot settings, scoring, aggregation, or acceptance threshold. CORE uses nanochat's unchanged implementation and official 22-task bundle, with every example (`max_per_task=-1`). `evaluate_bpb` stays unchanged as a reference metric.
- Train on data that is not allowed. The allowed data is what `prepare.py` downloads, now FineWeb-Edu only. Only the human adds data, by adding it to `prepare.py`. Never train on the validation shard, the tokenizer included, and do not filter the allowed data by its content.
- Continue an earlier run's training. Every run trains from scratch and ends within its rental.
- Use other models in training: no distillation from a teacher, and no other model's weights or outputs.

**The goal is simple: improve CORE.** A higher CORE score is better. Keep a candidate when its unrounded CORE score is higher than the current best (`prepare.core_improves`). `val_bpb` is recorded for reference only and never determines adoption. Every run still trains for 30 minutes and must finish, including both evaluations, within the 1-hour rental.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful CORE gains, but it should not blow up dramatically.

**Simplicity criterion**: Among candidates that improve CORE, prefer simpler implementations. Neither simplicity nor a better `val_bpb` makes a candidate that does not improve CORE a keep.

**The first run**: Establish the baseline only after the complete CORE evaluation is ready. Keep the initial training architecture, optimizer, hyperparameters and seed unchanged; the evaluation calls are added beforehand. Measure the full CORE suite and reference `val_bpb` once under this frame.

**No result-driven reruns**: A completed evaluation is final for that run. Do not change seeds or repeat training or evaluation until a better number appears. Preserve every attempt and its complete per-task output. Missing, partial, non-finite or failed CORE results cannot justify adoption.

**Bundles**: With one run a day, one change per run is far too slow. Combine several promising changes into one experiment, a bundle, and list every change in the description. If a bundle is discarded, its parts are not rejected: split or recombine them in later experiments.

**Discarded is not forever**: An idea can return in a genuinely different bundle or configuration, or after the baseline changes. Describe the new hypothesis before running it. Never repeat an unchanged experiment to replace an unfavorable completed result.

**Follow the latest work**: Between rentals, search the web for what is new: speedrun (modded-nanogpt) and nanochat records, recent arXiv papers on small-model pretraining, and other people's autoresearch results. Put the source of each idea in its description.

## Output format

Once the script finishes it prints a summary like this (illustrative values, not a measured baseline):

```
---
core_metric:      0.12345678901234566
val_bpb:          0.997900
training_seconds: 1800.1
total_seconds:    2400.0
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 30 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^core_metric:\|^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	core_metric	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. CORE achieved, stored at full round-trip precision (17 significant digits); use `NA` if incomplete
3. reference `val_bpb` achieved (e.g. 1.234567); use `NA` if unavailable
4. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short description listing every bundled change and its source

Keep each run's full log and `results/core.json` (nanochat's native result dictionary with raw and centered scores for all 22 tasks) and `results/core.csv` (nanochat's native CSV format). Use the unrounded JSON/TSV CORE value for decisions, not the rounded CSV display. Apply `prepare.core_improves(candidate, incumbent)` to the recorded scores.

Example:

```
commit	core_metric	val_bpb	memory_gb	status	description
a1b2c3d	0.12	0.997900	44.0	keep	baseline under the fixed CORE frame
b2c3d4e	0.125	0.993200	44.2	keep	example bundle improving CORE; source URL
c3d4e5f	0.123	0.985000	44.0	discard	example bundle lowering CORE despite a better val_bpb; source URL
d4e5f6g	NA	NA	0.0	crash	double model width (OOM); source URL
```

## The experiment loop

The experiment runs on `main`.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment on the rented GPU: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep -e '^core_metric:' -e '^val_bpb:' -e '^peak_vram_mb:' run.log`; preserve the complete CORE JSON and CSV.
6. Require a successful exit and complete finite results for all 22 CORE tasks. If either is missing, record the attempt as failed and inspect `tail -n 50 run.log`. A technical fix can be prepared for a later run under the daily limit; do not rerun a completed evaluation to obtain a better score.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. Keep the commit only if `prepare.core_improves(candidate_core, best_core)` is true: a higher CORE score than the current best. The baseline is the initial reference, not a candidate requiring an earlier score.
9. Otherwise, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment takes 30 minutes of training plus startup and eval, and must end within the 1-hour rental. If it cannot, kill it and treat it as a failure (discard and revert).

**Crashes**: Record every failed attempt. Fix technical errors in advance of a later daily run, or discard a fundamentally broken idea. Never substitute a partial CORE average or relax the suite or threshold to rescue a failed run.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period. In this repository the loop alternates the daily rental and the work between rentals (see the rules at the end).

## Rules of this repository

- **Rentals.** Rent one H100 SXM at most once a day and for at most 1 hour, counting only this loop's own rentals (other projects' rentals on the same account do not count): the setup, the day's 30-minute training and its eval must fit in it. Delete it right after. When you rent, set a timer that deletes it at 1 hour, in case you stop responding.
- **Between rentals.** Use this time for everything that helps the work ahead, not only the next rental. If you think something will be needed later, do it, even if it takes more than a day. For example: analyze the results, search for the latest work, prepare the next experiment as a ready-to-run commit so that rented time goes to training only, and make renting faster and cheaper.
- **CPU work.** Rented GPU time is only for the training. Everything that needs no GPU runs beforehand on the processing server in your environment notes: preparing the tokenizer, installing and checking code (for example, that the prepared experiment builds and takes a training step with a small model on the CPU), and analysis. Never run such work on the machine you run on. Keep your files in their own directory there and leave other projects' files alone.
- **Source and secrets.** Edit source and configuration only in the local Git repository, then send them to the processing server; its copy is not a Git checkout. Keep keys in the ignored `.env`, and have tools read environment variables. Add `.env` to `.gitignore` before creating it. Never commit secrets or machine-specific connection details.
- **Evaluation.** CORE is DCLM's full 22-task suite as implemented by [nanochat](https://github.com/karpathy/nanochat/tree/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd). `prepare.py` pins the unchanged upstream task evaluator and aggregation function to this revision, and pins the official `eval_bundle.zip` by SHA-256. Keep the original task data, task order, all examples, prompts, few-shot sampling, delimiters, scoring and centered aggregation. Do not reduce tasks or examples, change formats, move the acceptance threshold after seeing results, or rerun until numbers improve. Autoresearch's held-out FineWeb-Edu `evaluate_bpb` remains unchanged and is recorded only as a reference. Evaluation changes are limited to adding public evaluations, and only the user decides to add one; agents must not revise, replace or remove an existing evaluation or silently update its pinned version.
- The frame is the user's to change: the data, `prepare.py`, `program.md`, the dependencies, the GPU, the time budget and the daily hour. Propose additions to the public evaluations with evidence and wait for the user's decision. After a user-approved addition or another user-approved frame change, measure the current best once under the full new frame before comparing candidates. Implement this CORE frame before its baseline; do not compare earlier BPB-only runs against CORE candidates.
