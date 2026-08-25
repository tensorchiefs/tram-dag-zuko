# How fast can a CausalFlowDAG train?

This document benchmarks learning-rate schedules, per-node freezing, batch sizes, devices,
and LBFGS. The benchmark ran in June 2026 on an Apple-silicon Mac mini with torch 2.12
(CPU unless noted). To reproduce it, run
[`experiments/benchmarks/bench_training.py`](../experiments/benchmarks/bench_training.py)
(`cd experiments && uv run python -m benchmarks.bench_training`, or `--quick` for
one seed on cpu); the full grid takes ≈ 35 min. For a quick **cross-machine**
comparison, use the self-contained
[`experiments/benchmarks/perf_machine.py`](../experiments/benchmarks/perf_machine.py).
It runs fixed 200-epoch workloads on all available devices and writes a machine
fingerprint to JSON, and it needs nothing but `pip install tramdag`. The raw CSV
is a local run artifact and is not committed.

## The options, and how to use them

**Schedules** (`fit(..., schedule=...)`) control the learning rate over training:

| value | behavior |
|---|---|
| `None` (default) | constant lr — the classic behavior, exactly as before this PR |
| `"plateau"` | **per-node**: a node whose own validation NLL hasn't improved by `min_delta` (default 1e-4) for `plateau_patience` epochs (default 30) gets its lr × 0.3, floored at 1e-3 × the initial lr. Each node decays independently — valid because the per-node losses have independent gradients. |

`"onecycle"` and `"cosine"` were part of this benchmark and lost to
`"plateau"` on every workload; 0.4.0 removed both. The result tables below
keep their measured rows as the record of that decision.

**Early stopping / freezing** (`fit(..., freeze_patience=N)`): if the validation NLL of a
node does not improve for `N` epochs, the fit *freezes* that node. A frozen node leaves
the loss and the backward pass, which saves real compute. Its weights stay fixed from then
on. When all nodes are frozen, the fit returns early. The fit records the freeze epochs in
`flow.history["frozen"]`. Under `schedule="plateau"`, freezing also waits until the lr of
the node is decayed ≥ 100×. This delay prevents a freeze while a smaller step size can
still make progress. Freezing state is per-`fit()`-call: a second `fit` call trains all
nodes again.

**Switching it all off**: for an exact comparison with classical methods (`statsmodels`,
R `polr`/`tram`), omit both arguments. The benchmark changed no defaults, so

```python
flow.fit(train_df, epochs=4000, learning_rate=1e-2, batch_size=512)  # constant lr
flow.fit(train_df, epochs=2000, learning_rate=1e-3, batch_size=512)  # 2nd phase
```

is still the exact-MLE path that `experiments/misc/validate_ls.py` uses. Independent of all
this, `restore_best=False` remains the default (see CHANGELOG). The guard test
`tests/test_fit_schedules.py::test_plateau_freeze_preserves_exact_mle` also shows that
even *with* plateau+freezing the all-`ls` fit lands on the classical MLE within the usual
tolerances.

**LBFGS** is *not* a `fit()` option — it ships as its own method.
[`fit_classical`](fitting.md) runs float64 full-batch L-BFGS with a strong-Wolfe
line search and supersedes the hand-rolled recipe this report benchmarked: it is
deterministic, lands on the exact MLE, matches `statsmodels`/R `polr`, and
refuses non-`ls` specs (minibatch noise also regularizes the MLPs, so flexible
models belong in `fit`). The float32 single-precision variant measured here was
fast (< 2 s) but not robust across seeds (Findings #2) — `fit_classical`'s
float64 upcast is what fixed that.

```python
report = flow.fit_classical(train_df)  # exact classical MLE, seconds
```

## Method: time-to-target, not loss-go-down

Each config runs once. `fit()` records per-epoch validation NLL *and* wall-clock time.
From these records, we measure the seconds until the fit is within a fixed gap of a cached
long-run reference (3 torch seeds, medians):

| workload | model / data | reference NLL | tight tol | practical tol |
|---|---|---|---|---|
| **stroke-ls** | all-`ls` 5-node DAG, frozen `experiments/misc/data/magic-mrclean/ls` (n=1275, full-data MLE) | 10.3042 (train) | +1e-3 | +5e-3 |
| **vaca-ci** | all-`ci` flow, frozen `experiments/paper/data/vaca` (n=5000, 90/10 split) | 4.9632 (val) | +2e-3 | +1e-2 |

*Tight* ≈ exact-MLE equivalence (statsmodels/R-polr match). *Practical* ≈
coefficient-equivalent: a fit with gap ≈ 3e-3 already matches the R reference
coefficients within the tolerances of
[`experiments/misc/validate_ls.py`](../experiments/misc/validate_ls.py). The same
exact-MLE-under-plateau-and-freezing property is pinned on an inline DGP by
`tests/test_fit_schedules.py::test_plateau_freeze_preserves_exact_mle`.

These numbers were measured before the experiment code moved into
`experiments/benchmarks/`; the workloads are unchanged (same frozen data, same
specs), so the timings still stand. Re-running the benchmark reproduces the
machine-independent part exactly: the stroke-ls reference NLL 10.3042.

## Results

![stroke-ls convergence](img/nll_vs_time_stroke-ls.png)
![vaca-ci convergence](img/nll_vs_time_vaca-ci.png)

The table shows median seconds to target (batch 512, cpu). A "—" entry means that the fit
never reached the target within the budget.

| config | stroke-ls practical | stroke-ls tight | vaca-ci practical | vaca-ci tight | self-stops |
|---|---|---|---|---|---|
| baseline two-phase (old default) | 9.0 | **21.4** | 2.1 | 2.8¹ | no (runs 40 s / 15 s) |
| constant 1e-2 | 9.1 | 21.5³ | 2.2 | 2.8¹ | no |
| onecycle (1500 / 300 ep)² | — | — | 3.5 | 4.5 | no |
| onecycle (3000 ep)² | 16.8 | — (gap 1–2e-3) | | | no |
| cosine² | — | — | 2.2 | 3.5¹ | no |
| **plateau + freeze** | **8.9** | — (gap 2e-3) | **2.0** | 2.9 | **yes — 13 s / 4 s total** |
| LBFGS (full-batch) | **1.6** (2/3 seeds) | — (gap 4–8e-3) | n/a | n/a | yes |

¹ transient: the val-NLL curve dips through the target and then drifts away. Stroke needs
the 1e-3 phase to *stay*. Vaca shows mild overfitting. Final gap for the vaca baseline is
0.037. The old 520-epoch budget **underfits** vaca by ~0.03 nats. Plateau+freeze *stays*
at its target.
² `onecycle` and `cosine` were removed from `fit()` in 0.4 — they lost to
plateau on every workload here — so these three rows cannot be re-measured.
³ constant lr at batch 512 stalls at gap 3–7e-3. Only the lr-decay phase closes the last
decade. This is why the two-phase recipe existed.

## Findings

1. **Per-node plateau decay + freezing is the best default-style trainer.** It has the
   same time-to-accuracy as the hand-tuned two-phase schedule, but it needs **no budget
   tuning**. It decays the lr of each node off its own validation curve. It freezes
   converged nodes, which is a real FLOP saving because the per-node NLLs have
   independent gradients. And it **stops itself**: 13 s total vs 40 s for the baseline on
   stroke-ls, 4 s vs 15 s on vaca-ci, at equal or better final NLL.
2. **LBFGS is spectacular but not robust.** Full-batch LBFGS reaches coefficient-level
   accuracy on the classical all-`ls` model in **< 2 s** (vs 9 s for Adam) on 2/3 seeds.
   The third seed stalls at gap 8e-3. An Adam warm start made it *worse* (different
   basin) on every seed. Use it as a fast first shot with the plateau trainer as
   fallback, not as the default.
3. **OneCycle is a "spend exactly this budget" scheduler.** Accuracy arrives only at the
   end of its anneal. At 1500 epochs it misses everything. At 3000 epochs it lands gap
   1–2e-3. But you must know the right budget in advance, and this requirement is the
   problem that we try to remove.
4. **Full-batch loses on time-to-target** despite ~1.6× higher epoch throughput. It makes
   too few optimizer steps per second of compute at these n. Batch 512 is a good default.
   Very large batches (16k) only improved raw throughput at n=50k.
5. **MPS (Apple GPU) is 3–4× slower than the M-series CPU** at these model sizes
   (verified correct: identical reconstruction). Kernel-launch overhead dominates
   sub-millisecond ops. Stay on CPU locally. CUDA on Colab-class GPUs is a different
   regime.
6. **The old defaults waste or under-spend.** Stroke: 4000 epochs budgeted, converged
   work done after ~1500 (freezing recovers the difference automatically). Vaca: 520
   epochs budgeted, ~0.03 nats short of converged. Fixed budgets are wrong in both
   directions. Adaptive stopping fixes both.

## Recommendation

For everyday fits:

```python
flow.fit(
    train,
    val,
    epochs=4000,
    learning_rate=1e-2,
    batch_size=512,
    schedule="plateau",
    plateau_patience=30,
    freeze_patience=120,
)
```

The generous `epochs` value is only a ceiling. The fit stops itself. For exact classical
comparisons where the last 1e-3 matters, append a short constant-lr polish phase
(`epochs=500, learning_rate=1e-3`) after the plateau fit. Or run the old two-phase recipe.

The benchmark itself changed no defaults — that is its own reviewed decision (see
the `restore_best` episode in CHANGELOG.md). Two of its findings have since been
adopted: `plateau_patience` defaults to the 30 recommended here, and `epochs` has no
default at all, because finding 6 is precisely that a fixed budget cannot be right for
every workload. The experiment scripts still run the two-phase constant-lr recipe,
which each states in its own YAML.
