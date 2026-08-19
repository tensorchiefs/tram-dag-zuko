# Experiments — the TRAM-DAG paper replications

Research code, kept out of the installed `tramdag` package. One directory per
**area**, each owning everything it needs:

| area | what it holds |
|---|---|
| [`paper/`](paper/) | the replications of [arXiv:2503.16206](https://arxiv.org/abs/2503.16206), their SCM generators, frozen datasets and expected results |
| [`benchmarks/`](benchmarks/) | training-speed and cross-machine measurements |
| [`misc/`](misc/) | everything else — currently the classical-MLE validation |

An area contains its own `data/`, `ground_truth/`, `results/` (gitignored),
`tests/` and whatever helpers only it needs. Only two files are shared:
[`common.py`](common.py) (the output layout the workflow reads) and
[`check.py`](check.py) (the ground-truth comparison).

## Running one

Experiments run as modules, from this directory:

```bash
cd experiments
uv run --group experiments python -m paper.triangle atan-cs      # fit + figures + metrics
uv run --group experiments python -m check paper triangle-atan-cs  # vs ground truth
uv run --group experiments python -m paper.check_data             # frozen data regenerates
uv run pytest experiments                                        # the area checks (seconds)
```

Every run writes to `<area>/results/<name>/`: `metrics.json` (the numbers CI
checks), `report.md` (the table plus figures, posted as a commit comment by the
experiments workflow), `plots/*.png` and `flow.pt`.

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

Which paper figure each variant reproduces — and what is deliberately not
reproduced — is listed in [`paper/PAPER_COVERAGE.md`](paper/PAPER_COVERAGE.md).

All five have the same shape: imports, function definitions, a `run(variant)`
function holding the whole experiment, and a `__main__` block whose argparse
call selects the variant.

## Hyperparameters live in YAML, not in code

Each script reads its sibling `<script>.yaml` and **nothing else**: no defaults
in the code, no CLI flags that change a number. The reader is
`tramdag.load_config` — it lives in the framework because the guarantee is
worth having in one place — and it compares the variant's keys against the set
the script declares, failing on a mismatch. A missing key cannot quietly become
a default, and an unused key cannot look effective.
Values shared by several variants are written once under a YAML anchor and
merged with `<<`, which keeps the merge visible in the file.

To change what a run does, edit the YAML. To add a variant, add a section —
`argparse` picks it up automatically, because its choices come from the file.

## Ground truth

`<area>/ground_truth/<result-dir>.json` holds one entry per checked metric:

```json
{"beta12": {"value": 2.0012, "atol": 0.05}}
```

Tolerances are per metric because torch results differ slightly across
operating systems and CPUs. `check.py` fails on a metric outside its
tolerance, and on a ground-truth entry the run no longer produces.

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
