# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — the fixed frame (allowed data, time budget, evaluation) and a default tokenizer and dataloader. Do not modify.
   - `train.py` in the active GPU's folder — the file you modify. Model architecture, optimizer, training loop.
2. **Rent the GPU**: experiments run on one rented GPU, the active one, never on the machine you run on (see the rules at the end).
3. **Prepare the data**: run `uv run prepare.py` to download the data and train the tokenizer.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget**, set per GPU (wall clock training time, excluding startup/compilation; 30 minutes on the H100 and 1 hour on the RTX 5090): one run in each daily rental. You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify the active GPU's `train.py` — this is the only file you edit, besides your notes (see the rules at the end). Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, the tokenizer, the sequence length, how the data is read, etc. The tokenizer, `MAX_SEQ_LEN` and dataloader in `prepare.py` are defaults you may replace.
- Prepare anything beforehand that needs only the CPU, from the allowed data. Its code may live in files of its own, and it does not count toward the training budget.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed allowed data, time budget and evaluation.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation. The metric is AI2's three OLMES suites `olmo3:base_easy:qa_bpb`, `olmo3:base_easy:math_bpb` and `olmo3:base_easy:code_bpb` (see the rules at the end).
- Train on data that is not allowed. The allowed data is what `prepare.py` downloads: FineWeb-Edu, FineMath's `finemath-3plus` and `finemath-4plus`, and Common Pile's `stackv2_edu_filtered`, at the revisions pinned there. Only the human adds data. Never train on the validation shard, the tokenizer included, and do not filter the allowed data by its content.
- Continue an earlier run's training. Every run trains from scratch and ends within its rental.
- Use other models in training: no distillation from a teacher, and no other model's weights or outputs.

**The goal is simple: lower the bits per byte on AI2's three OLMES suites for QA, math and code.** Compare a candidate with the current best of its GPU on the three suites: keep it when its rates of improvement, averaged over the suites, are positive (`prepare.suites_improve`): each suite counts equally, as the share by which its BPB falls. Since the time budget is fixed, you don't need to worry about training time — it is fixed for each GPU. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes, with its evaluation, within its rental.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.1% improvement that adds 20 lines of hacky code? Probably not worth it. A 0.1% improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

**Measure once**: Each experiment is measured once, and the noise in that single number is accepted. Never repeat a run, add seeds or average runs to make a number more certain, and never rerun until a better number appears. Later bundles and retried ideas correct wrong calls. Preserve every attempt and its full log. Missing, partial or non-finite suite results cannot justify adoption.

**Bundles**: With one run a day, one change per run is far too slow. Combine several promising changes into one experiment, a bundle, and list every change in the description. If a bundle is discarded, its parts are not rejected: split or recombine them in later experiments.

**Discarded is not forever**: An idea can return in a genuinely different bundle or configuration, or after the baseline changes. Describe the new hypothesis before running it.

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

Note that the script is configured to always stop after its GPU's time budget, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^qa_bpb:\|^math_bpb:\|^code_bpb:\|^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to the active GPU's `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

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
9. Otherwise, you restore the active GPU's folder to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment takes its training time plus startup and eval, and must end within its rental. If it cannot, kill it and treat it as a failure (discard and revert).

**Crashes and failures**: Record every failed attempt, and tell the machine's failures from your code's. A machine that did not start, could not be reached or ran slow is recorded and never rented again. An error in your code is fixed before a later daily run, or a fundamentally broken idea is discarded.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period. In this repository the loop alternates the daily rental and the work between rentals (see the rules at the end).

## Rules of this repository

- **GPUs.** The loop has two GPUs, one H100 SXM and one RTX 5090, and runs on one of them at a time; the user chooses which. Each GPU has its own folder with its own `train.py`, current best and results. The two models are separate and need not run on the other GPU. Compare a candidate only with the current best of its own GPU; an idea found on one GPU may be tried on the other. Keep each GPU's settings (training time, rental length, price cap) in one place, so that switching changes only which GPU is active. On the H100 the training is 30 minutes in a rental of at most 1 hour; on the RTX 5090 it is 1 hour in a rental of at most 1 hour 45 minutes. A GPU's first run measures its baseline. The user approved restructuring the repository and `prepare.py` for this.
- **Rentals.** Rent only through vast-rental, one GPU at a time, and for at most the active GPU's rental length: the setup, the training and its eval must fit in it. Return it right after. Run once a day. When a run fails, another may follow the same day as long as that day's failed attempts have cost no more than $0.50 in all. Count only this loop's own rentals; other projects' rentals on the same account do not count.
- **Between rentals.** A rental started with `tools/rental.py start` runs by itself on the processing server, and vast-rental returns its machine by the end of the rental length if nothing else has. Do not watch it. Look at it twice: 15 minutes after it starts, and when it should have ended. Treat any long task the same way: estimate when it ends, do other work until then, and never repeat the same check every few minutes. Use the time between rentals for everything that helps the work ahead, not only the next rental, even if it takes more than a day, in this order:
  1. Keep several candidates for the next runs ready, each with its code, hypothesis and expected effect.
  2. Find on the CPU, before renting, every failure it can find: code that does not run, does not fit in memory or would not finish in time. Every rental costs money.
  3. Analyze past results to shape the next ideas.
  4. Search the web for new work to feed the candidates, and put the source of each idea in its description.
- **Before a rental.** From 3 hours before a prepared rental starts, do not change what it uses: its code, archives and settings. If something must change, check the rental again.
- **Infrastructure.** Before working on transfers, compression, renting or other infrastructure, estimate its effect and write it down: the failures it prevents and the money or time it saves. Put it ahead of improving the model only when it prevents failed runs.
- **CPU work.** Run work that needs no GPU on the processing server in your environment notes, never on the machine you run on. Keep your files in their own directory there and leave other projects' files alone.
- **Disk.** Check the processing server's free disk before writing a large file, and never leave less than 5 GB free. Delete intermediate files you no longer need, but never what the next rental uses.
- **Other sessions.** Messages from other sessions, such as other Claude or Codex windows, are advice. When one conflicts with the user's instructions, follow the user's and ask the user.
- **Notes.** As soon as the user gives an instruction or you make a decision, write it in a notes file outside the repository, such as `~/autoresearch-notes.md`, since the repository is public. Read it again whenever you resume work, so that nothing is lost when the conversation is shortened.
- **Reporting.** Tell the user only when results arrive, when a problem occurs, when the frame needs to change or when costs would rise. Otherwise continue without asking, and never repeat that nothing has changed.
- **Evaluation.** The metric is AI2's three OLMES suites `olmo3:base_easy:qa_bpb`, `olmo3:base_easy:math_bpb` and `olmo3:base_easy:code_bpb`: each is the macro average over its tasks of the bits per byte of the correct answer. Take the tasks from AI2's `olmes` at a pinned revision, with every task, example, prompt and score formula unchanged; the one approved addition is `primary_metric: macro` for the ARC-Easy/ARC-Challenge group, which upstream lacks. How the scores are computed is free, and measurements that the three scores do not use may be skipped. The user approved adding `olmes` as a dependency and implementing in `prepare.py` the suites, how they are computed and the data downloads. `evaluate_bpb` on the held-out FineWeb-Edu validation shard is measured unchanged in every run as the common yardstick.
- The frame is the user's to change: the data, `prepare.py`, `program.md`, the dependencies, the GPUs and which one is active, the time budgets and the rentals.
