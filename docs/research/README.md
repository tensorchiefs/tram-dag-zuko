# Autoresearch runs

This folder holds the brief and the outputs for the autonomous training-speed
research loop. Start here.

- **[`MISSION_autoresearch.md`](MISSION_autoresearch.md)** — the agent's brief
  (objective, metric, method, hard limits, integrity rules). The instance reads
  this and works from it.
- During a run the instance produces, in this folder:
  `RESEARCH_LOG.md`, `IDEAS.md`, `LEADERBOARD.md`, `baseline/`, and finally
  `REPORT.md` — then opens a PR with the validated changes.

## Division of labour

**You (the human) do the bootstrap; Claude does everything inside the session.**
The rule: anything that must be true at Claude's startup, or that needs
interactive auth, is yours — all in-session git work (branching, commits, pushes)
is the instance's.

This matters specifically for the protection hook and the environment: Claude Code
loads `settings.local.json` and its hooks **once, at startup**, so the guard must
be activated *before* you launch the instance (an agent that could enable its own
guard could also disable it). Likewise the Python env must exist before the
instance runs its first benchmark.

## Launch checklist (on the research machine)

One-time setup (per machine): `claude`, [`uv`](https://docs.astral.sh/uv/),
and `gh` installed and authenticated (`gh auth status` green), repo cloned.

Then for each run:

```bash
ssh -t <rack> tmux new -As GPU_Rack_tramdag        # named, disconnect-proof session
cd ~/tramdag && git pull                           # latest mission + hook + example
uv sync                                            # create/refresh the env
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#   ^ expect True on a GPU box; if False on a GPU machine, fix torch before launching
cp .claude/settings.local.example.json .claude/settings.local.json
#   ^ activates the measurement-protection hook (rack-only; delete the file to disable).
#     If a settings.local.json already exists, MERGE the "hooks" block instead.
gh auth status                                     # confirm pushes will work
claude --remote-control GPU_Rack_tramdag --dangerously-skip-permissions
```

Then send the instance a single message:

> Read docs/research/MISSION_autoresearch.md and begin.

From there the instance runs **Step 0** (create `research/<date>-<host>`, push a
heartbeat — proves check-in), then **Experiment #0** (baseline benchmark, which
also sets the per-experiment timeout anchor), then the bounded experiment loop.

## Guard rails (already wired)

| concern | mechanism |
|---|---|
| disconnect-proof, remote-controllable | `tmux` + `--remote-control GPU_Rack_tramdag` |
| runs unattended without permission prompts | `--dangerously-skip-permissions` |
| can't game the measurement | `.claude/hooks/protect_measurement.py` blocks edits to `tests/`,`data/`,`bench_training.py` and reads of the target values (deterrent) |
| can't push to main | branch-only workflow in the mission (+ enable GitHub branch protection on `main` for hard enforcement) |
| runaway experiment | OS `timeout` at `min(5× baseline, 30 min)` per experiment |
| overfitting to the harness | mission rule: a win must generalize off the benchmark |
| token budget | ≤20 experiments, ≤1 per 30 min, idle waits; run at `/effort medium` |

## Monitoring a run

```bash
ssh -t <rack> tmux attach -t GPU_Rack_tramdag        # watch live
tmux capture-pane -t GPU_Rack_tramdag -p | tail -40  # peek without attaching
```

or just follow the `research/<date>-<host>` branch on GitHub — the instance pushes
after every experiment, so the commit history *is* the progress log.

## Notes

- **Read-blocking is a deterrent, not airtight** — `pytest` and any Python the
  agent runs read files at the OS level, below the tool hook. The write-block is
  the real lock; the load-bearing anti-overfit guard is the generalize-off-harness
  rule.
- The hook is **rack-scoped**: it's only active where `settings.local.json`
  exists (git-ignored), so it never blocks legitimate test edits on a dev machine.
- One run = one branch (`research/<date>-<host>`); restarts reuse it (see the
  mission's branch rules), so the loop survives a crash or context compaction.
