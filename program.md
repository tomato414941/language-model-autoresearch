# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — the fixed frame (allowed data, time budget, evaluation) and a default tokenizer and dataloader. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
2. **Rent the GPU**: experiments run on one rented H100 SXM, never on the machine you run on (see the rules at the end).
3. **Prepare the data**: on the processing server (see the rules at the end), run `uv run prepare.py --num-shards 10` to train the tokenizer. Bring the tokenizer when renting, then run `uv run prepare.py` on the rented machine to download the remaining training shards.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 30 minutes** (wall clock training time, excluding startup/compilation): one run in each daily rental of at most 1 hour. You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, the tokenizer, the sequence length, how the data is read, etc. The tokenizer, `MAX_SEQ_LEN` and dataloader in `prepare.py` are defaults you may replace. Save a replacement tokenizer where `prepare.py` reads it, in the same format, so that `evaluate_bpb` runs unchanged on it.
- Prepare anything beforehand that needs only the CPU, such as a tokenizer, tables or statistics built from the allowed training data. Run it on the processing server; its code may live in files of its own. It does not count toward the 30-minute budget.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed allowed data, time budget and evaluation.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation. The metric is AI2's three OLMES suites `olmo3:base_easy:qa_bpb`, `olmo3:base_easy:math_bpb` and `olmo3:base_easy:code_bpb` (see the rules at the end).
- Train on data that is not allowed. The allowed data is what `prepare.py` downloads: FineWeb-Edu, and FineMath's `finemath-3plus` and `finemath-4plus` subsets ([HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) at revision `e92b25a616738fe95dc186b64dfb19f9c8525594`, ODC-By 1.0), and Common Pile's openly licensed educational code ([common-pile/stackv2_edu_filtered](https://huggingface.co/datasets/common-pile/stackv2_edu_filtered) at revision `c354dbe88469a1153e97c6a63ac50591849654de`; each file keeps its own open license). Only the human adds data. Never train on the validation shard, the tokenizer included, and do not filter the allowed data by its content.
- Continue an earlier run's training. Every run trains from scratch and ends within its rental.
- Use other models in training: no distillation from a teacher, and no other model's weights or outputs.

**The goal is simple: lower the bits per byte on AI2's three OLMES suites for QA, math and code.** Compare a candidate with the current best only on the evaluations both were measured with. When both have the three suites, keep the candidate when its rates of improvement, averaged over the suites, are positive (`prepare.suites_improve`): each suite counts equally, as the share by which its BPB falls. Otherwise keep it when its `val_bpb` is lower (`prepare.bpb_improves`). Since the time budget is fixed, you don't need to worry about training time — it's always 30 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes, with its evaluation, within the 1-hour rental.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.1% improvement that adds 20 lines of hacky code? Probably not worth it. A 0.1% improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

**Measure once**: Each experiment is measured once, and the noise in that single number is accepted. Never repeat a run, add seeds or average runs to make a number more certain, and never rerun until a better number appears. Later bundles and retried ideas correct wrong calls. Preserve every attempt and its full log. Missing, partial or non-finite suite results cannot justify adoption.

**Bundles**: With one run a day, one change per run is far too slow. Combine several promising changes into one experiment, a bundle, and list every change in the description. If a bundle is discarded, its parts are not rejected: split or recombine them in later experiments.

**Discarded is not forever**: An idea can return in a genuinely different bundle or configuration, or after the baseline changes. Describe the new hypothesis before running it.

**Follow the latest work**: Between rentals, search the web for what is new: speedrun (modded-nanogpt) and nanochat records, recent arXiv papers on small-model pretraining, and other people's autoresearch results. Put the source of each idea in its description.

## Output format

Once the script finishes it prints a summary like this (illustrative values, not a measured run):

```
---
qa_bpb:           0.912345
math_bpb:         1.234567
code_bpb:         1.456789
val_bpb:          0.997900
training_seconds: 1800.1
total_seconds:    1925.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   2998.6
num_steps:        5718
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 30 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^qa_bpb:\|^math_bpb:\|^code_bpb:\|^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 8 columns:

```
commit	qa_bpb	math_bpb	code_bpb	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2.–4. qa_bpb, math_bpb and code_bpb achieved (e.g. 0.912345) — use 0.000000 for crashes
5. reference val_bpb achieved (e.g. 1.234567) — use 0.000000 if unavailable
6. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
7. status: `keep`, `discard`, or `crash`
8. short description listing every bundled change and its source

Example:

```
commit	qa_bpb	math_bpb	code_bpb	val_bpb	memory_gb	status	description
a1b2c3d	0.912345	1.234567	1.456789	0.997900	44.0	keep	baseline
b2c3d4e	0.908100	1.230000	1.450000	0.993200	44.2	keep	example bundle: deeper model, shorter warmdown; source URLs
c3d4e5f	0.915000	1.236000	1.455000	0.990000	44.0	discard	example bundle: GeLU activation, larger batch; source URLs
d4e5f6g	0.000000	0.000000	0.000000	0.000000	0.0	crash	double model width (OOM); source URL
```

## The experiment loop

The experiment runs on `main`.

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment on the rented GPU: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^qa_bpb:\|^math_bpb:\|^code_bpb:\|^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Record the attempt as failed and inspect `tail -n 50 run.log`. A technical fix can be prepared for a later run under the daily limit.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If the candidate improves on the evaluations both share (see the goal above), you "advance" the branch, keeping the git commit
9. Otherwise, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment takes 30 minutes of training plus startup and eval, and must end within the 1-hour rental. If it cannot, kill it and treat it as a failure (discard and revert).

**Crashes**: Record every failed attempt. Fix technical errors in advance of a later daily run, or discard a fundamentally broken idea.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period. In this repository the loop alternates the daily rental and the work between rentals (see the rules at the end).

## Rules of this repository

- **Rentals.** Rent one H100 SXM at most once a day and for at most 1 hour, counting only this loop's own rentals (other projects' rentals on the same account do not count): the setup, the day's 30-minute training and its eval must fit in it. Delete it right after. When you rent, set a timer that deletes it at 1 hour, in case you stop responding.
- **Between rentals.** Use this time for everything that helps the work ahead, not only the next rental. If you think something will be needed later, do it, even if it takes more than a day. For example: analyze the results, search for the latest work, prepare the next experiment as a ready-to-run commit so that rented time goes to training only, and make renting faster and cheaper.
- **CPU work.** Rented GPU time is only for the training. Everything that needs no GPU runs beforehand on the processing server in your environment notes: preparing the tokenizer, installing and checking code (for example, that the prepared experiment builds and takes a training step with a small model on the CPU), and analysis. Never run such work on the machine you run on. Keep your files in their own directory there and leave other projects' files alone.
- **Source and secrets.** Edit source and configuration only in the local Git repository, then send them to the processing server; its copy is not a Git checkout. Keep keys in the ignored `.env`, and have tools read environment variables. Add `.env` to `.gitignore` before creating it.
- **Evaluation.** The metric is AI2's three OLMES suites `olmo3:base_easy:qa_bpb`, `olmo3:base_easy:math_bpb` and `olmo3:base_easy:code_bpb`: each is the macro average over its tasks of the bits per byte of the correct answer. Run them with AI2's `olmes` at a pinned revision, with every task, example, prompt and scoring unchanged; only connect them to the model and tokenizer; the one approved addition is `primary_metric: macro` for the ARC-Easy/ARC-Challenge group, which upstream lacks. The user approved adding `olmes` as a dependency and implementing the suites in `prepare.py`; this and the data downloads below are the changes to `prepare.py` allowed now. `evaluate_bpb` on the held-out FineWeb-Edu validation shard is measured unchanged in every run as the common yardstick.
- **Data.** The user added FineMath and Common Pile's educational code. Implement their downloads in `prepare.py` at the pinned revisions, each into its own directory, so that the default dataloader still reads FineWeb-Edu only and the measured baseline stays unchanged.
- The frame is the user's to change: the data, `prepare.py`, `program.md`, the dependencies, the GPU, the time budget and the daily hour. Never re-measure an earlier model when the measurement changes. A new evaluation is measured from the next run on and starts deciding once the current best has it. If a change leaves no shared evaluation comparable (for example a new time budget or GPU), the next run becomes the new baseline without a comparison.
