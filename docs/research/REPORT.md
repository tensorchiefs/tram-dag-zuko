# Autoresearch report — training-speed study (tramdag)

**Run:** branch `research/2026-06-15-workstation` · workstation, NVIDIA TITAN RTX
(24 GB), torch 2.12.0+cu130 · 2026-06-15.
**Goal (from `MISSION_autoresearch.md`):** find changes that make `CausalFlowDAG`
training **faster to a fixed quality target** (time-to-target as defined in
`experiments/benchmarks/bench_training.py`), with rigor — falsifiable hypotheses, ≥3 seeds,
opt-in library changes, full suite green before any confirmed win.
**Outcome:** 7 experiments (#0 baseline + 6 hypotheses). **One confirmed,
broadly-safe win** — a calibrated marginal-init — shipped as opt-in
`fit(marginal_init=True)`. Five negatives that fully map the search space.

## Summary table

| # | Experiment | Verdict | Headline result |
|---|---|---|---|
| 0 | Baseline (bench + perf fingerprint) | — | CUDA 2–4× **slower** than this box's CPU for these tiny per-node models; this CPU baseline is the only valid reference. |
| 1 | torch.profiler one-epoch breakdown | REJECTED (hyp.) | Not matmul-bound — **dispatch/overhead-bound**: 94% in fwd+bwd over thousands of µs-scale aten ops, no single op dominates. |
| 2 | CPU thread-count sweep | REJECTED | Ops barely parallelize; 1→10 threads <10%. Single-thread-effective. |
| 3 | torch.compile fusion of per-node loss | DEAD END | zuko's Bernstein/RQS log-det uses `autograd.grad` in forward → **double-backward**, unsupported by aot_autograd. Op-count axis closed in eager. |
| **4** | **Calibrated Bernstein marginal-init** | **CONFIRMED** | **W2 vaca-ci +56–63%** time-to-practical over two seed triples; W1 +4–8% (no regression); full suite green; identical MLE. |
| 5 | Extend marginal-init to ordinal cutpoints | REJECTED | Only +3% on W1 (kept — coherent, never regresses). W1 is **coefficient-bound**, not init-bound. |
| 6 | Cache parent encodings (dtype-copy) | REJECTED | Bit-identical but **~0%** per-step. The 15% copy overhead is in the per-node math, not feature encoding. Reverted. |

## The one recommended change — `fit(marginal_init=True)`

**What it is (opt-in, defaults untouched).** zuko's default zero-θ Bernstein init
maps each node's pre-scaled domain `[−B, B]` onto latent z ∈ **[−6.93, +7.63]** —
~2.5× steeper than the standard-logistic 5/95 quantiles (±2.944) — and slightly
off-centre (z(0)=0.347). So the first chunk of training is spent just rescaling the
marginal. `marginal_init=True` instead initialises each *unconditional* intercept to
the **calibrated** map of its marginal:

- **Bernstein continuous roots** → the exact linear map onto [logit .05, logit .95]
  (`BernsteinUT.marginal_init_theta()`, a closed-form inversion of zuko's
  `_constrain_theta` cumsum-of-softplus). Verified: marginal-init θ → z(±5)=±2.944, z(0)=0.
- **Ordinal roots** → cutpoints set to the empirical class log-odds
  (`ordinal_marginal_init_theta()`). Verified: reproduces a 7-level marginal to
  max-abs-err 0.0 (default zeros give P(Y=0)=0.50 vs true 0.30).

It is a **pure init**: the converged optimum is unchanged, so the exact-MLE
property of all-`ls` models survives (the sacred tests pass). Affects only
`SimpleIntercept` nodes; conditional `ci` intercepts are untouched. Applied once
(first-fit guard), so multi-phase fits don't reset a trained intercept.

**Evidence (median time-to-practical, s; baseline → marginal_init):**

| workload / config | seeds 0–2 | seeds 3–5 |
|---|---|---|
| **vaca-ci / plateau+freeze (W2)** | 6.9 → **2.6** (+62.6%) | 6.5 → **2.8** (+56.2%) |
| stroke-ls / plateau+freeze (W1)  | 31.8 → 30.4 (+4.3%) | 35.2 → 32.3 (+8.2%) |
| stroke-ls / baseline-2phase (W1) | 33.3 → 31.2 (+6.2%) | 35.7 → 34.4 (+3.7%) |

All three improvement criteria met: (1) ≥10% on ≥1 workload — W2 +56–63% across
two **independent** seed triples; (2) no workload regresses >5% — W1 improves
+4–8%; (3) full suite `uv run pytest tests/ -q` passes incl.
`test_plateau_freeze_preserves_exact_mle` / `test_flow_matches_r_reference`.

**Why uneven (the real finding).** The gain scales with how much of the total-NLL
gap is a continuous/ordinal **root's marginal shape**. W2 (all-continuous): root
x1's miscalibrated marginal dominates early NLL → marginal-init ~2.5×'s it away (even
though only 1 of 3 nodes is marginal-initialized). W1 (all-`ls`): time-to-target is gated
by the **`ls` shift coefficients** (the proportional-odds regression weights)
reaching the MLE — which no marginal-calibration init can touch (Exp #5 confirmed:
marginal-initializing the ordinal cutpoints too adds only +3%). The effect is general — it
helps any Bernstein/ordinal root — but unevenly *leveraged* per workload. Not
harness overfitting: it never regresses and is validated on two seed triples.

## What didn't work, and why the search is (near) exhausted

Two orthogonal axes, both now closed:

- **Per-step cost** (cut op count): threads (#2, <10%), torch.compile (#3,
  double-backward), and feature-caching (#6, ~0%) all fail. The dispatch overhead
  is intrinsic to the per-node transform math in eager PyTorch; reducing it needs a
  transform rewrite (return analytic log-dets so there's no in-forward autograd) —
  large, and it risks the numerics core / sacred MLE tests.
- **Steps-to-target** (init/schedule): marginal-init (#4/#5) is banked for
  marginal-shape-bound workloads. The only untried lever is **coefficient-
  convergence speed** on the `ls`/shift params (per-group lr, or an LBFGS polish —
  LBFGS reaches W1 in 4.6s in Exp #0 but is fragile across seeds). It is W1-only
  and narrow, so the broad-applicability bar makes it low expected value.

Per the mission's stop rule (top-of-backlog expected gain <5% broadly), the
productive search is essentially complete.

## Recommendations

1. **Merge `marginal_init`** as the opt-in this PR adds. Consider making it the
   **default for `SimpleIntercept` roots** in a later release — it never regresses,
   converges to the identical MLE, and only helps. (Kept opt-in here per the
   research-run rule of leaving defaults untouched.)
2. **Follow-ups (not in this PR):** a unit test for `marginal_init` (could not be added
   on the research machine — the integrity hook protects `tests/`); benchmarking the
   `ls`-shift LBFGS-polish idea if a W1-specific speedup is wanted.

## Provenance

Full lab notebook on the run branch: `docs/research/RESEARCH_LOG.md` (per-experiment
hypothesis/numbers/verdict), `IDEAS.md` (ranked backlog, re-ranked after each
experiment), `LEADERBOARD.md`, baseline CSVs in `docs/research/baseline/`, perf
fingerprint in `docs/perf/`. The diagnostic scripts of the run
(`profile_epoch.py`, `sweep_threads.py`, `probe_compile.py`,
`exp_warmstart.py`, `exp_ordinal_warmstart.py`, `probe_feat_cache.py`) stayed on
the run branch and are not in this tree.
