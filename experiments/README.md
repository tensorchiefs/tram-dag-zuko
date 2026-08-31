# Experiments — the TRAM-DAG paper replications

Research code, kept out of the installed `tramdag` package. One directory per
**area**, each owning everything it needs:

| area | what it holds |
|---|---|
| [`paper/`](paper/) | the replications of [arXiv:2503.16206](https://arxiv.org/abs/2503.16206), their SCM generators, frozen datasets and expected results |
| [`benchmarks/`](benchmarks/) | training-speed and cross-machine measurements |
| [`misc/`](misc/) | everything else — currently the classical-MLE validation |

A `paper`/`misc` area contains its own `data/`, `ground_truth/`, `results/`
(gitignored), `tests/` and whatever helpers only it needs. `benchmarks/` is the
exception by nature: it measures training speed *on* the other areas' data, so
it reads `misc/data/` and `paper/data/` and commits no ground truth of its own —
its output is a write-up in `docs/`, not a pinned number. Only two files are
shared:
[`common.py`](common.py) (the output layout the workflow reads) and
[`check.py`](check.py) (the ground-truth comparison).

## Running one

Experiments run as modules, from this directory:

```bash
cd experiments
uv run python -m paper.triangle atan-cs        # fit + figures + metrics
uv run python -m check paper triangle-atan-cs  # vs ground truth
uv run python -m paper.check_data              # frozen data regenerates
uv run pytest experiments                      # the area checks (seconds)
uv run pytest experiments/tests                                  # the shared check.py
```

Every run writes to `<area>/results/<name>/`: `metrics.json` (the numbers CI
checks), `report.md` (the table plus figures, posted as a commit comment by the
experiments workflow), `flow.pt`, and `plots/*.png` for the runs that draw
figures — `validate_ls` is a numbers-only comparison and draws none.

## The scripts

| script | dataset | paper | variants |
|---|---|---|---|
| [`paper/triangle.py`](paper/triangle.py) | continuous triangle | Sec. 6.1, App. C.3 | `linear-ls`, `linear-cs`, `atan-cs`, `sin-cs` |
| [`paper/triangle_mixed.py`](paper/triangle_mixed.py) | triangle with an ordinal x3 | Sec. 6.2, App. C.4 | `linear-ls`, `exp-cs` |
| [`paper/vaca.py`](paper/vaca.py) | VACA/CNF bimodal benchmark | Sec. 5.1–5.2, App. C.1 | `flexible` |
| [`paper/carefl.py`](paper/carefl.py) | CAREFL Laplace SCM | Sec. 5.3, App. C.2 | `flexible` |
| [`misc/validate_ls.py`](misc/validate_ls.py) | frozen synthetic cohort | — (framework anchor) | `adam`, `classical` |

Two benchmarks live beside them, measured rather than checked against ground
truth, so they are run by hand and reported in the docs:

| script | measures | output |
|---|---|---|
| [`benchmarks/bench_training.py`](benchmarks/bench_training.py) | time-to-target for lr schedules, batch size, device, L-BFGS | [`docs/training-speed.md`](../docs/training-speed.md) |
| [`benchmarks/perf_machine.py`](benchmarks/perf_machine.py) | fixed 200-epoch throughput per machine and device | [`docs/perf/`](../docs/perf/) |

`perf_machine.py` deliberately depends on nothing but the installed package —
it is meant to be downloaded and run on a machine without a checkout — so it
carries its own copy of the bimodal DGP. `benchmarks/tests/` pins that copy to
the maintained generator, because a drifted copy would silently make the
collected `final_val_nll` values incomparable.

Runtime: at the paper's batch 32 a triangle variant is 500 epochs × 1250
steps over 40 000 rows, about 3.2 s per epoch single-process at 2–4 torch
threads (27 min; 42–69 min on the 2-core CI runners). The step is
overhead-bound — 32 threads make it *slower* (5.5 s per epoch), and eight
variants side by side on one box take 78 min each. Run them one or two at a
time, or with `OMP_NUM_THREADS=2`.

The configs therefore take one documented deviation for CI runtime: batch 256
at lr 0.004 instead of the paper's batch 32 at lr 0.001 — the same 500 epochs
in 8× fewer steps, every ground-truth metric kept (the grid that chose it is
in `docs/paper-replication.md`). Measured on CI (run 32974872751): triangle
jobs 7–11 min instead of 42–69, the whole workflow about 12 min wall. The 500 epochs stay: the linear-shift
coefficients settle after ~40 epochs, the complex shift does not (at 100
epochs the cs-curve error doubles to triples; at 250 mixed exp-cs is still
+60 %).

`vaca.py` and `carefl.py` set `net_input_scaling: minmax`, because the
reference's comparison scripts train in min-max scaled space
(`comparison/utils.R::scale_df`) and their tanh nets saturate on the raw
parents (40% of the VACA rows have `|x| > 2`); the triangle scripts fit
`df_orig`, so those configs leave it `null`.

Which paper figure each variant reproduces — and what is deliberately not
reproduced — is listed in [`paper/PAPER_COVERAGE.md`](paper/PAPER_COVERAGE.md);
every hyperparameter with its source in the R code, the deviations and the
measured numbers are in [`docs/paper-replication.md`](../docs/paper-replication.md).

All five have the same shape: imports, function definitions, a `run(variant)`
function holding the whole experiment, and a `__main__` block whose argparse
call selects the variant.

## Hyperparameters live in YAML, not in code

Each script reads its sibling `<script>.yaml` and **nothing else**: no defaults
in the code, no CLI flags that change a number. `common.py::load_variant` parses
the file and picks the variant's section with `common.py::_config_section`.
Both live here, so the package depends on no config parser and ships no config
helper.
Values shared by several variants are written once under a YAML anchor and
merged with `<<`, which keeps the merge visible in the file.

To change what a run does, edit the YAML. To add a variant, add a section —
`argparse` picks it up automatically, because its choices come from the file.

## Ground truth

`<area>/ground_truth/<result-dir>.json` holds one entry per checked metric, in
one of two forms:

```json
{"beta12": {"value": 1.9825, "atol": 0.05},
 "cs_curve_max_abs_err": {"max": 0.23}}
```

`{value, atol}` is two-sided, for a quantity that should stay where it is.
`{max}` is an upper bound, for an **error measure** — there a smaller number is
a better fit, not a drift, and must not fail the run. Tolerances are per metric
because torch results differ slightly across operating systems and CPUs.
`check.py` fails on a metric outside its tolerance or above its bound, and on a
ground-truth entry the run no longer produces.

A `{max}` bound is only informative in a band: below **1.5x** its measurement it
fails on another machine for no reason, above **4x** it cannot catch a
regression. A `{value, atol}` center decays the other way — it keeps passing
while describing an older run. `check.py` reports both as notes, not failures,
because a tolerance is a judgement call: a bound outside the band, and a
measurement more than half-way to its `atol`. A bound that is *meant* to be wide
says so in a `"why"` string, which is printed instead (it excuses width only —
the too-tight note always fires):

```json
"max_abs_diff_flow_vs_statsmodels": {
  "max": 0.25,
  "why": "the maximum is over a coefficient with 7 of 1275 observations: 0.028 here, 0.113 on the CI runner"}
```

Regenerating ground truth is a deliberate act: run the experiment, review the
figures, then write the new values with a commit message that says what moved
and why.

## The frozen data is a contract

`<area>/data/` is committed input, not a cache. `paper/check_data.py`
regenerates every paper dataset from the seed in its `truth.json` and compares
to 1e-9 (not bit equality: numpy's transcendental functions move their last bits
between releases); `paper/tests/` runs the same comparison in the ordinary test
run. A new seed or changed equations means a **new folder**, never an edit in
place.

`misc/data/magic-mrclean/ls/` is the exception with no generator here: it came
from the stroke simulator that left the repository with the clinical storyline.
Its schema and size are pinned by `misc/tests/` instead, and the generator can
be recovered from the `pre-experiments-cut` tag.
