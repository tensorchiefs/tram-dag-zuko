# Codebase review — 2026-08-17

Full-circle review for stale code, stale files, invalid examples and wrong
docs. Method: four parallel sweeps (docs/ guides; notebooks + experiments;
src/ dead code; top-level docs and config), each cross-checked against the
current source, plus the full test suite. Every finding lists its
resolution. "Fixed here" means a commit on `feat/refactoring`.

## 1. Broken examples and commands

| Where | Problem | Resolution |
|---|---|---|
| `README.md:91` | `effect_modifier_scan(df, "mRS_3m", on="T")` raises `TypeError` — the parameter is `t` since the STE pass (f566170) | Fixed here |
| `docs/stroke-case-study.md:27` | `uv run python all_ls_long.py` — the file is parked in `experiments/stale/` | Fixed here |
| `docs/stroke-case-study.md:111` | Presents `counterfactual_demo.py` as live; it is parked in `stale/` | Fixed here |
| `docs/training-speed.md:50-72` | Manual L-BFGS recipe calls `build_spec` (not importable from the package) and private methods — and `fit_classical` now does all of it | Fixed here: recipe replaced by a pointer to `fit_classical` |
| `docs/perf/REPORT.md:3,22,29` | Three references to `experiments/perf_machine.py`, now in `stale/` (also in the generator template) | Fixed here (report + template) |
| `docs/fitting.md:221` | Links `experiments/bench_training.py`, now in `stale/` | Fixed here |
| `docs/research/MISSION_autoresearch.md` | References parked scripts (`bench_training.py`, `perf_machine.py`, `transforms_tram_dag.py`); Experiment #0 fails as written | Fixed here: paths updated with a "parked, needs API migration" note |

## 2. Stale facts in docs

| Where | Problem | Resolution |
|---|---|---|
| `README.md:9` | Pin advice says `tramdag==0.2.*`; released is 0.3.0 | Fixed here |
| `README.md:13-14` | Garbled sentence ("The structure is of the triangular Adjacency Matrix is") | Fixed here |
| `README.md:205-213` | Layout omits `env.py`, `scores.py`; "training benchmark" is parked; docs list omits `fitting.md`, `notation.md` | Fixed here |
| `CLAUDE.md:134` | "published as tramdag 0.2.0" — 0.3.0 was released 2026-06-19 | Fixed here |
| `CLAUDE.md:117` | Frozen-CSV contract list omits `data/vc-shift/` | Fixed here |
| `CLAUDE.md` (architecture) | `flow.py` method list omits `fit_classical`; simulations bullet omits `vc_shift.py` | Fixed here |
| `CHANGELOG.md:49` | "### Added (0.3.1)" sits inside the 0.4.0 section; 0.3.1 was never released | Fixed here: folded into 0.4.0 |
| `CHANGELOG.md` (0.4.0) | The scan entry still shows the old `on` parameter (renamed to `t` in f566170). The scores feature never shipped (0.3.1 was not released), so the signature is corrected in place — no breaking-change note is needed | Fixed here |
| `src/tramdag/spec.py:64` | Comment claims `_LEGACY` labels are "still accepted by the checkpoint loader" — the pre-0.3 loader path is gone; only `term()` uses them (same stale claim in `tests/test_spec_terms.py:3`) | Fixed here |
| `docs/fitting.md:83-101` | `fit` option list omits `marginal_init=` and `vc_warm_start=` | Fixed here |
| `docs/training-speed.md:7,33,42` | Points to a non-existent `results/` CSV; "this PR" phrasing for merged history | Fixed here |
| `tests/README.md:8-9` | "fast subset (~30 s)" — measured ~2 min locally; full-suite claim also low; file table lists 5 of 16 test files | Fixed here (timings hedged, table completed) |
| `notebooks/demo_tram_dag_colab.py:95,375` | Prose still teaches the removed 0.3 string-label vocabulary (`"ci"`, `"ls"`) | Fixed here (`.ipynb` regenerated via jupytext) |

## 3. Dead code

| Where | Problem | Resolution |
|---|---|---|
| `src/tramdag/scores.py:44` | `CRIT_1PCT` — zero references | Removed |
| `src/tramdag/simulations/vc_shift.py:41` | `COLUMNS` — never referenced, not even in its own module | Removed |
| `src/tramdag/transforms.py` | `StandardLogistic.cdf` — no caller anywhere | Removed |
| `src/tramdag/scores.py:142-144` | Self-acknowledged unreachable branch ("joint LS cannot occur") | Removed, replaced by a comment stating the invariant |
| `src/tramdag/transforms.py` | `ordinal_marginal_init_theta` is cross-module public but missing from `__all__` | Added |

Kept deliberately (checked, not dead):

- `counterfactual_pair` on `_TriangleBase` and `VCLogisticShift` has no caller
  yet, but it is the documented ground-truth read-out of the testing policy
  (CLAUDE.md: validate causal features against `counterfactual_pair`), and
  `MagicMrClean`'s twin is tested. Symmetric API, kept.
- `_LEGACY` labels in `term()`: covered by tests, kept.
- `sup_bb_pvalue` is public but only used internally — kept as API.
- `intercept()` alias: new public name, kept.

## 4. Coverage gaps

| Where | Problem | Resolution |
|---|---|---|
| `spec.py` 0.3-checkpoint shims | The multi-`I` merge and the node-level-transform carry in `spec_from_dict` have zero test coverage — only real 0.3 checkpoints exercise them | Fixed here: test added that feeds 0.3-format dicts |

## 5. Hygiene

| Where | Problem | Resolution |
|---|---|---|
| `.gitignore` | No rule for `texfrag/` (Emacs LaTeX-preview scratch in docs/ and notebooks/) and `docs/notation.html` (generated from notation.md) | Fixed here (files left on disk) |
| `experiments/stale/nihss6_flow.py:14` | Imports `run_name`, `source_arg` — removed from `common.py`; restoring this script needs more than the syntax port | Fixed here: one line in `experiments/stale/README.md` |

## 6. Verified clean

- No doc, notebook or experiment uses the removed 0.3 API (`terms=`,
  node-level `transform=`) or the deleted `Transformation` class.
- All active `experiments/*.py` compile and run against the current API; no
  dead helpers.
- No tracked build artifacts (`__pycache__`, `.ipynb_checkpoints`,
  `.exec.ipynb`); the tracked Colab `.ipynb` is cell-identical to its
  jupytext `.py`.
- All CI workflow references (files, markers, templates) exist.
- All 10 `data/` variant folders carry `truth.json`.
- `docs/scores.md`, `docs/varying-coefficients.md`, `docs/notation.md`:
  examples run as written.

## 7. Next steps (not done on this branch)

1. ~~**`notebooks/intro_tram_dag.py`** prose labels~~ — done: the six
   informal lowercase `ls`/`cs`/`ci` mentions now use the constructor
   names (`LS`, `CS`, `I`), which the notebook actually teaches.
2. ~~**pyproject classifiers**~~ — done: explicit 3.10–3.13 classifiers
   added (matches `requires-python >= 3.10`).
3. ~~**`simulations.REGISTRY`**~~ — done: CLAUDE.md no longer claims
   experiments look generators up via the registry; it maps name → class
   and experiments import the classes directly.
4. **Pre-0.3 checkpoints** (`parents={...}` layout) no longer load anywhere.
   The CHANGELOG 0.4.0 Removed section already declares the loader gone.
   Open owner question: do any pre-0.3 checkpoints survive outside the
   paper monorepo? If yes, they need a one-off converter.
5. **`docs/research/` mission**: the autoresearch mission needs a real
   migration of the parked benchmark scripts before a next run — the path
   fixes here only make the docs honest.
6. **Full-suite timing** in `tests/README.md` should be re-measured on CI
   once this branch merges (nightly job already reports it).
