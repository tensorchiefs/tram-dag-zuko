# Experiments — the TRAM-DAG paper replications

Self-contained replication code for [arXiv:2503.16206](https://arxiv.org/abs/2503.16206),
plus the classical-MLE validation that anchors the framework's correctness.
Everything these need lives in this directory: the generators
([`simulations/`](simulations/)), the frozen datasets ([`data/`](data/)), the
hyperparameters (one YAML per script) and the expected results
([`ground_truth/`](ground_truth/)). The installed `tramdag` package does not
contain any of it.

## Running one

```bash
cd experiments
uv run --group experiments python triangle.py atan-cs      # fit + figures + metrics
uv run --group experiments python check.py triangle-atan-cs  # vs committed ground truth
uv run --group experiments python check_data.py             # frozen data still regenerates
```

Every run writes to `results/<name>/` (gitignored): `metrics.json` (the numbers
CI checks), `report.md` (the table plus figures, posted as a commit comment by
the experiments workflow), `plots/*.png` and `flow.pt`.

## The scripts

| script | dataset | paper | variants |
|---|---|---|---|
| [`triangle.py`](triangle.py) | continuous triangle | Sec. 6.1, App. C.3 | `linear-ls`, `atan-cs`, `sin-cs` |
| [`triangle_mixed.py`](triangle_mixed.py) | triangle with an ordinal x3 | Sec. 6.2, App. C.4 | `linear-ls`, `exp-cs` |
| [`vaca.py`](vaca.py) | VACA/CNF bimodal benchmark | Sec. 5.1–5.2, App. C.1 | `flexible` |
| [`carefl.py`](carefl.py) | CAREFL Laplace SCM | Sec. 5.3, App. C.2 | `flexible` |
| [`validate_ls.py`](validate_ls.py) | synthetic stroke cohort | — (framework anchor) | `adam`, `classical` |

All five have the same shape: imports, function definitions, a `run(variant)`
function holding the whole experiment, and a `__main__` block whose argparse
call selects the variant.

## Hyperparameters live in YAML, not in code

Each script reads `<script>.yaml` and **nothing else**: no defaults in the
code, no CLI flags that change a number. The loader compares the variant's
keys against the set the script reads and fails on a mismatch, so a missing
key cannot quietly become a default and an unused key cannot look effective.
Values shared by several variants are written once under a YAML anchor and
merged with `<<`, which keeps the merge visible in the file.

To change what a run does, edit the YAML. To add a variant, add a section —
`argparse` picks it up automatically, because its choices come from the file.

## Ground truth

`ground_truth/<result-dir>.json` holds one entry per checked metric:

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

`data/` is committed input, not a cache. `check_data.py` regenerates every
dataset from the seed in its `truth.json` and compares to 1e-9 (not bit
equality: numpy's transcendental functions move their last bits between
releases). A new seed or changed equations means a **new folder**, never an
edit in place.

`data/magic-mrclean/ls/` is the exception with no generator here: it came from
the stroke simulator that left the repository with the clinical storyline.
Recover it from the `pre-experiments-cut` tag if it ever needs regenerating.
