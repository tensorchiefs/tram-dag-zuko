# Contributing

## Setup

```bash
uv sync                 # creates .venv from the pinned uv.lock
```

With [direnv](https://direnv.net/), `.envrc` activates `.venv` on `cd`.

## Tests

```bash
uv run pytest -q -m "not slow"          # the fast subset
uv run pytest -q                        # everything, incl. the long fits
uv run pytest tests/test_flow.py -q     # one file
```

Run `pytest` with no path: `testpaths` in `pyproject.toml` then picks up both
`tests/` and the per-area `experiments/*/tests/`.

`@pytest.mark.slow` marks the five longest fits, not every test that trains a
flow — some acceptance criteria (the `VC` recovery bar, the centering bias
reduction) train a flow deliberately in the fast subset, because a feature's
acceptance number is worth measuring on every run.

CI runs the fast subset on pull requests and on pushes to `main` and `dev-*`
(a feature branch triggers `pre-commit` and `experiments`, not `ci`), and the
full suite nightly.

What the suite guarantees, and where each reference number comes from, is
documented in [`tests/README.md`](tests/README.md). Two rules matter most:

- `data/` is a **contract**. A new seed or changed equations means a new
  folder, never an edit in place.
- Validate a new causal feature against a simulator's known truth, not
  just "it runs".

## Linting

```bash
uvx ruff check .        # report
uvx ruff format --diff  # what formatting would change
```

Rules live in `pyproject.toml`: `I`/`E`/`F`/`D`/`UP` at 88 columns, numpy
docstring convention. Docstrings are not required in `tests/`,
`experiments/` or `notebooks/`.

The hooks in `.pre-commit-config.yaml` are enforced by
`.github/workflows/pre-commit.yaml` on every push and pull request. Install
them locally so a push cannot fail on formatting:

```bash
pre-commit install --install-hooks -t pre-commit -t commit-msg -t pre-push
```

## Notebooks

`notebooks/*.py` are [jupytext](https://github.com/mwouts/jupytext)
percent-format files and are the source of truth. Edit those, never an
`.ipynb`. One `.ipynb` is tracked on purpose —
`demo_tram_dag_colab.ipynb`, because the README's Colab badge links to it —
and it is regenerated from the `.py`; see
[`notebooks/README.md`](notebooks/README.md).

## Conventions worth knowing

The implementation conventions that are easy to get wrong — latent-scale
signs, raw vs one-hot parent encoding, the log-space ordinal likelihood,
seeding — are documented in [`CLAUDE.md`](CLAUDE.md) and pinned by tests.
Read that before changing anything in `src/tramdag/`.
