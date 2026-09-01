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

## The recipes, and where they live now

The recipes this benchmark compares are **callbacks** since 0.4 — training
strategies, not part of the model (see [fitting.md](fitting.md)):

| recipe | behavior | today |
|---|---|---|
| constant | one lr for the whole run | `flow.fit(train, epochs=, learning_rate=)` |
| plateau+freeze | **per-node**: a node whose own validation NLL hasn't improved by `min_delta` (1e-4 default; the stroke run uses 1e-5, see the benchmark) for `patience` epochs gets its lr × 0.3, floored at 1e-3 × the start; once decayed ≥ 100× and flat for `freeze` epochs it leaves training (rate 0). Valid because the per-node losses have independent gradients. | `tramdag.callbacks.PerNodePlateau`, a callback over one parameter group per node (`per_node_adam`), measured in `experiments/benchmarks/bench_training.py` |
| global plateau | torch's `ReduceLROnPlateau` on the summed validation NLL — the paper reference's rule | `experiments/paper/helpers.py::fit_paper` |

`"onecycle"` and `"cosine"` lost to `"plateau"` on every workload and were
removed in 0.4; their measured rows below stay as the record.

**The exact-MLE path**: for an exact comparison with classical methods
(`statsmodels`, R `polr`/`tram`) no recipe is needed —

```python
flow.fit(train_df, epochs=4000, learning_rate=1e-2, batch_size=512)  # constant lr
flow.fit(train_df, epochs=2000, learning_rate=1e-3, batch_size=512)  # 2nd phase
```

is still the exact-MLE path (`experiments/misc/validate_ls.py` runs a three-phase variant
of it, 800/700/500 epochs at 1e-2/1e-3/1e-4, batch 256 — cut from
4000/2000/1000 in 2026-09: the same MLE, named-coef gap to statsmodels 1.6e-5
vs 1.8e-5, ~3.5x less wall clock). `fit` keeps the
final weights. The guard test
`tests/test_fit_hooks.py::test_torch_plateau_scheduler_preserves_exact_mle` shows
that a schedule through the hooks still lands the all-`ls` fit on the classical
MLE within the usual tolerances.

**LBFGS** ships as its own method, [`fit_classical`](fitting.md), which
supersedes the hand-rolled recipe this report benchmarked. The float32 variant
measured here was fast (< 2 s) but seed-fragile (Finding #2); `fit_classical`'s
float64 upcast fixed that.

## Method: time-to-target, not loss-go-down

Each config runs once. The benchmark's callback records per-epoch validation NLL *and* wall-clock time.
From these records, we measure the seconds until the fit is within a fixed gap of a cached
long-run reference (3 torch seeds, medians):

| workload | model / data | reference NLL | tight tol | practical tol |
|---|---|---|---|---|
| **stroke-ls** | all-`ls` 5-node DAG, frozen `experiments/misc/data/magic-mrclean/ls` (n=1275, full-data MLE) | 10.3042 (train) | +1e-3 | +5e-3 |
| **vaca-ci** | all-`ci` flow, frozen `experiments/paper/data/vaca` (n=5000, 90/10 split) | 4.9632 (val) | +2e-3 | +1e-2 |

*Tight* ≈ exact-MLE equivalence (statsmodels/R-polr match). *Practical* ≈
coefficient-equivalent: a fit with gap ≈ 3e-3 already matches the R reference
coefficients within the tolerances of
[`experiments/misc/validate_ls.py`](../experiments/misc/validate_ls.py).
The workloads are unchanged since the measurement (same frozen data, same
specs); re-running reproduces the machine-independent stroke-ls reference NLL
10.3042 exactly.

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

The everyday recipe is the global-plateau callback in
[fitting.md](fitting.md#the-recipes-as-callbacks) with a generous `epochs`
ceiling; the per-node self-stopping variant is
`tramdag.callbacks.PerNodePlateau`. One finding became a package default:
`epochs` has no default, because Finding 6 is precisely that a fixed budget
cannot be right for every workload — every in-repo caller states its recipe in
its own YAML.
