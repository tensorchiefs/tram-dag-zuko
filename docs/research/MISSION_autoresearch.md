# MISSION: autonomous training-speed research for tramdag

<!--
Launch notes (for the human):
  ssh -t <server> tmux new -As GPU_Rack_tramdag
  git clone https://github.com/tensorchiefs/tramdag && cd tramdag
  claude --remote-control GPU_Rack_tramdag        # /effort medium recommended
  first message: "Read docs/research/MISSION_autoresearch.md and begin."
Prerequisites on the server: claude + gh authenticated, uv or pip available.
-->

You are running a self-directed research loop. Goal: find changes that make
`CausalFlowDAG` training **faster to a fixed quality target** (secondarily: reach
better optima). You have the full repository and this machine (check its GPU). No
human is in the loop — work autonomously, with rigor.

Read first: CLAUDE.md, docs/training-speed.md, experiments/benchmarks/bench_training.py,
src/tramdag/flow.py. The June 2026 benchmark defines your methodology.

> **Script status.** `bench_training.py` and `perf_machine.py` are migrated to
> the current API and live in `experiments/benchmarks/`; both run. Note that
> `fit()` no longer has the `onecycle` and `cosine` schedules, so those rows of
> the June 2026 grid cannot be re-measured. The transform comparison lives in
> `notebooks/demo_tram_dag_colab.py` §6.

## Hard limits

- **At most 20 experiments total.** Then stop, write the report, open the PR.
- **At most one experiment per 30 minutes**, and fewer is better: one well-designed
  experiment beats three rushed ones. If you cannot state a falsifiable hypothesis
  with a predicted effect size, you are not ready to run.
- Run benchmarks as background commands and **wait idly** for them. Do not fill
  waiting time with speculation, re-reading files you already know, or rewriting
  logs. Tokens are the scarce resource; wall-clock is free.
- Keep context lean: your accumulated knowledge lives in RESEARCH_LOG.md and
  IDEAS.md, not in re-derivation. After a context compaction, re-read those two
  files and LEADERBOARD.md — nothing else — and continue.

## Branch & push rules

- Work on **one** branch named `research/<YYYY-MM-DD>-<host>` (UTC date at
  creation, short hostname — e.g. `research/2026-06-15-gpu-rack`). This identifies
  the run by machine and date and lets several machines run without colliding.
- **One branch per run — never fork your own work.** On every startup (including
  after a restart or context compaction) first look for an existing branch with
  `git ls-remote --heads origin 'research/*-<host>'`; if one exists, check it out
  and continue on it. Only create a new `research/<today>-<host>` branch when none
  exists for this host. Do not create a second dated branch mid-run.
- **Never push to main.**
- Commit after every experiment (confirmed or rejected) and **push after every
  commit** — the human follows your progress on GitHub, not by asking you.
- Never touch the frozen CSVs in `data/`. Library changes stay **opt-in**
  (new flag/argument, defaults untouched) until the final report.

## Step 0 (very first action — prove you can check in)

Before any benchmarking, verify the full check-in path works from this machine —
fast, so a broken push/auth setup is caught in seconds instead of after a 30-minute
baseline run. Create your run branch (see naming/reuse rules above) and push a tiny
heartbeat commit:

```bash
git checkout main && git pull
HOST=$(hostname -s)
# reuse this host's branch if it already exists, else create today's
BRANCH=$(git ls-remote --heads origin "research/*-$HOST" | head -1 \
         | sed 's#.*refs/heads/##')
BRANCH=${BRANCH:-research/$(date -u +%F)-$HOST}
git checkout -B "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
mkdir -p docs/research
printf '# Autoresearch heartbeat\n\nStep 0 push round-trip from %s at %s.\n' \
  "$HOST" "$(date -u +%FT%TZ)" > docs/research/HEARTBEAT.md
git add docs/research/HEARTBEAT.md
git commit -m "chore(autoresearch): step 0 push round-trip"
git push -u origin "$BRANCH"
```

Then **confirm the commit is visible on GitHub** (the branch + `HEARTBEAT.md`).
Report the commit URL. Only once this round-trip succeeds, continue to
Experiment #0. If the push fails it is almost certainly the git credential helper
on this machine — run `gh auth status` / `gh auth setup-git` (or check the remote
URL / SSH key), fix it, and retry before doing anything else. Do **not** start the
baseline until a check-in has demonstrably landed on the remote.

## Experiment #0 (the baseline)

Before changing anything: run the existing benchmark suite unchanged on THIS
machine — `experiments/benchmarks/bench_training.py` (full grid, 3 seeds) and
`experiments/benchmarks/perf_machine.py` (hardware fingerprint). All published numbers are
from an Apple-silicon Mac mini; nothing transfers — your baseline is the only
valid comparison point. Commit the results (bench `results.csv`/`ranking.csv`
into `docs/research/baseline/`, the perf JSON into `docs/perf/`) and push.
This is your first *results* commit (after the Step 0 heartbeat), before any
hypothesis. You may delete `docs/research/HEARTBEAT.md` in this commit.

## The metric (non-negotiable)

- **Time-to-target** as defined in `experiments/benchmarks/bench_training.py`: wall-clock
  seconds until val-NLL is within tolerance of the cached long-run reference
  (W1 stroke-ls: tight 1e-3 / practical 5e-3; W2 vaca-ci: 2e-3 / 1e-2).
- Add **W3**: vaca-ci at n=50,000, batch 4096, on this machine's GPU — the
  throughput regime. Cache its reference the same way.
- A change is an IMPROVEMENT only if ALL hold:
  1. median time-to-practical over ≥3 torch seeds improves ≥10% on ≥1 workload,
  2. no workload regresses >5%,
  3. the **full** test suite passes (`uv run pytest tests/ -q`, incl. `slow`) —
     `test_plateau_freeze_preserves_exact_mle` and
     `test_flow_matches_r_reference` are sacred: the exact-MLE property of
     all-`ls` models must survive every change.
- **Test cadence (cost-staged):** during iteration run only the fast subset
  (`uv run pytest tests/ -q -m "not slow"`, ~30 s) plus the specific test(s) your
  change touches. Run the **full** suite (criterion 3) only when a change is about
  to be recorded as a *confirmed* improvement, and again before the final PR — the
  full suite is ~12 min, so running it on every experiment would dominate the run.
- Wall-clock honesty: nothing else heavy on the machine; rerun the baseline
  whenever you doubt machine state; never compare numbers across machine states.

## Runaway protection (hard timeout)

Any experiment can hang (a stuck line search, an unsupported device op, a
fat-fingered epoch count). Bound every run with an **OS-level** timeout — do not
rely on noticing it yourself:

- **Anchor the cap to the baseline wall-clock of the *same scope* you re-run, not
  to a flat ceiling.** Record how long Experiment #0 takes for each scope you'll
  reuse — the full 3-seed grid (`bench_training.py`) and any smaller subset you
  iterate on. Cap a later run at **~2–3× the baseline wall-clock of that same
  scope** and wrap it in `timeout`. A legitimate re-run (even a slightly slower
  variant) finishes inside that; a genuinely hung fit, which would run unbounded,
  is killed. Example: if the full grid took ~T at baseline, run a confirmation
  with `timeout $((3*T_seconds)) uv run python -u bench_training.py ...`.
- **This is separate from pacing.** "≤1 experiment per 30 min" caps *cadence and
  token burn* (idle-wait between experiments) — it is **not** a limit on how long
  a single run may take. If the full grid is ~57 min on this machine, an
  experiment that re-runs it is fine; it just means experiments are naturally
  ~hourly here. Use a **targeted subset** (one workload / fewer seeds, or
  `--quick`) for fast iteration, and reserve the full 3-seed grid for *confirming*
  a candidate — each with its own scope-anchored timeout.
- **On timeout:** the process is killed by the OS and the command returns
  non-zero. Log that run as `INCONCLUSIVE (timed out)` — **never** read a SIGTERM
  as "no improvement". Downrank the idea and move on; **do not auto-retry** the
  same configuration unchanged — it will just hang again.

## Scientific integrity (no gaming the metric)

The time-to-target metric is a **proxy**; the real goal is genuinely faster/better
TRAM-DAG training **in general**. Optimize for that, not for the harness:

- **Broad applicability, not the particularities of the benchmark.** A change that
  helps only W1/W2/W3, only the recorded seeds, or only this machine is **not** an
  improvement. Before believing a candidate win, sanity-check that it generalizes
  (e.g. it still helps on an extra seed or a held-out config/dataset shape). A win
  that evaporates off the measured setup is overfitting to the harness — reject it.
- **Never touch what measures you.** You may **not** edit the tests, the frozen
  data in `data/`, the benchmark targets/tolerances in
  `experiments/benchmarks/bench_training.py`, or the cached reference values, in order to make
  a result look better. *Changing the measurement is not a result.* If a test
  fails, your change is wrong — not the test. Reference values are computed once
  (Experiment #0) and then fixed. **Do not read the target values** either — the
  test assertions, `data/**/truth.json`, or `data/**/ref_ls/**`: run the tests and
  trust pass/fail rather than curve-fitting to the numbers. (You *do* read the
  input data, `data/**/obs.csv`/`rct.csv`, and the harness source — those you must
  fit / run.) On the research machine a `PreToolUse` hook
  (`.claude/hooks/protect_measurement.py`) enforces both rules; treat a `BLOCKED`
  message as a hard stop, not an obstacle to route around.

## Method: hypothesis loop with a lab notebook

Maintain three append-only files in `docs/research/`:

- `RESEARCH_LOG.md` — per experiment: HYPOTHESIS (one sentence, predicted effect
  size), CHANGE (diff summary), COMMANDS, NUMBERS (all seeds), VERDICT
  (CONFIRMED / REJECTED / INCONCLUSIVE), WHAT THIS TEACHES. Negative results are
  first-class citizens — log them fully.
- `IDEAS.md` — ranked backlog: (expected gain × plausibility) / cost. Re-rank
  after every experiment; results should change your beliefs.
- `LEADERBOARD.md` — best known config per workload, with provenance.

Discipline:

- **Profile before optimizing.** Experiment #1 is a torch.profiler breakdown of
  one epoch per workload — where does the time actually go? Form hypotheses from
  data, not vibes.
- One change per experiment. No compound changes; if unavoidable, ablate after.
- Three seeds minimum before any verdict; report medians, keep the spread.

## Starting backlog (seed it, then own it)

profiler baseline → where is the time? · torch.compile on the per-node log-prob ·
TF32 / bf16 autocast on GPU (watch the MLE guard!) · batch-size × lr scaling on
GPU · **warm-start init**: initialize Bernstein θ from the node's marginal
quantiles instead of zeros — could delete most of early training · per-node Adam
betas/eps · LBFGS polish phase after plateau-freeze · evaluate val less often
(per-epoch val eval is overhead) · vectorize/fuse the Python loop over nodes ·
one-hot caching in `_tensorize` · the RQS tail-slope fix (its
symptom is documented in CLAUDE.md and shown in the demo notebook's §6; an
accuracy lever that may also help optimization) ·
gradient accumulation vs huge batches.

## Cadence and end state

After every confirmed improvement (or every 5 experiments): update LEADERBOARD,
write a 5-line status section into the log. Stop at 20 experiments, when the top
of the backlog has expected gain <5%, or when told to stop. Then write
`docs/research/REPORT.md` (what worked, what didn't and why, recommended changes
with evidence) and open a PR via `gh` containing ONLY the validated changes, PR
description = the report's summary table. A human reviews the PR; the log is
your evidence.
