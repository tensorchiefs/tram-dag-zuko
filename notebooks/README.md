# Notebooks

All notebooks in this repo are [jupytext](https://github.com/mwouts/jupytext)
**percent-format `.py` files** — plain Python with `# %%` cell markers and
markdown in `# %% [markdown]` cells. The `.py` file is always the **source of
truth**.

| notebook | what it is |
|---|---|
| `intro_tram_dag.py` | didactic walkthrough of the TRAM-DAG model (SI/LS/CS, L1–L3, all claims checked against a hand-built SCM; complex intercepts are the one component it does not exercise — see `additive_vs_joint_ci.py`) |
| `demo_tram_dag_colab.py` | 5-minute showcase on the paper's bimodal VACA benchmark ([open in Colab](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)) |
| `additive_vs_joint_ci.py` | joint vs additive complex intercept, and reading per-parent effects out of the additive one with `intercept_contributions` |
| `varying_coefficients.py` | heterogeneous treatment effects: the `VC` head, `varying_coef`, the modifier scan and propensity centering, all scored against a known `beta(x)` |
| `classical_fit_tram_dag.py` | `fit_classical` on all-`ls` models, opening with plain logistic regression on `MASS::birthwt` (a 2-level ordinal node) checked against R `glm`: determinism, the exact MLE against `statsmodels` / R, and the classical-fit-then-keep-training warm start |

`classical_fit_tram_dag.R` is not a notebook. It is the R half of
`classical_fit_tram_dag.py` — every classical reference that notebook hard-codes,
fitted in one script so the numbers can be re-checked instead of trusted. It
needs `tram`, which CI does not install, so it is run by hand:
`Rscript notebooks/classical_fit_tram_dag.R` from the repo root.

Every notebook here is executed by the docs workflow — on pushes to `main` and
`dev-*` branches — which is what keeps them working against the current API. A
notebook that is not in that workflow's executed-notebook loop does not belong
in this directory. On a feature branch, run one by hand:
`MPLBACKEND=Agg uv run python notebooks/<name>.py`.

Data a notebook reads lives in `notebooks/data/` — see its README for
provenance.

## Rules

- **Do not edit `.ipynb` files directly** — edit the `.py` and regenerate.
- **Do not commit `.ipynb` files.** They are git-ignored (embedded base64
  outputs ruin diffs). The single exception is `demo_tram_dag_colab.ipynb`,
  tracked **output-stripped** only so the Open-in-Colab badge works; after
  changing `demo_tram_dag_colab.py`, regenerate it before committing.

## Working with the notebooks

**VS Code / Cursor**: open the `.py` directly — with the Python + Jupyter
extensions every `# %%` cell gets a "Run Cell" link (pick the `.venv`
interpreter created by `uv sync`). No conversion needed.

**Classic Jupyter / JupyterLab**: generate a local `.ipynb` (stays untracked):

```bash
uvx jupytext --to ipynb notebooks/intro_tram_dag.py
```

For frequent notebook editing you can install jupytext into the venv instead of
using `uvx` each time: `uv sync --group notebooks`.

**Edit in a synced copy** (interactive notebook, `.py` stays the source of
truth): with the `notebooks` group installed, jupytext can keep a local `.ipynb`
paired to the `.py` so your interactive edits flow back into the tracked `.py`.

The cleanest way needs no `.ipynb` at all — in JupyterLab/Jupyter Notebook,
right-click the `.py` → *Open With* → *Notebook*. Edits save straight back to the
`.py`; there is nothing to clean up. (jupytext can also pair a real `.ipynb` to
the `.py` — `--set-formats ipynb,py:percent`, then `--sync` — if you prefer.)

The paired `.ipynb` stays git-ignored. Note that `--set-formats` adds `ipynb` to
the `.py` header — revert that one-line header change before committing (the
committed notebooks are paired to `py:percent` only).

**Headless check** (runs all cells top-to-bottom, plots suppressed):

```bash
MPLBACKEND=Agg uv run python notebooks/intro_tram_dag.py
```

**Regenerate the tracked Colab demo ipynb** after editing its `.py`:

```bash
uvx jupytext --to ipynb notebooks/demo_tram_dag_colab.py
```

(A fresh conversion contains no outputs, which is exactly the committed state.)

More on the format: [jupytext documentation](https://jupytext.readthedocs.io).
