# Contributing

## Setup

```bash
uv sync                 # creates .venv from the pinned uv.lock
```

With [direnv](https://direnv.net/), `.envrc` activates `.venv` on `cd`.

## Tests

```bash
uv run pytest tests/ -q -m "not slow"   # unit + contract tests (~2 min)
uv run pytest tests/ -q                 # everything, incl. the fits
uv run pytest tests/test_flow.py -q     # one file
```

Tests that train a flow are marked `@pytest.mark.slow`. CI runs the fast
subset on every push and pull request, and the full suite nightly.

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

`.pre-commit-config.yaml` exists but the hooks are **not installed yet**:
the ruff hooks reformat every file they touch, and the repository has not
had its one-off `ruff format` sweep. Installing them now would mix
formatting noise into unrelated diffs. Once the sweep has landed:

```bash
pre-commit install --install-hooks
```

## Notebooks

`notebooks/*.py` are [jupytext](https://github.com/mwouts/jupytext)
percent-format files and are the source of truth. Do not edit or commit
`.ipynb` — see [`notebooks/README.md`](notebooks/README.md).

## Conventions worth knowing

The implementation conventions that are easy to get wrong — latent-scale
signs, raw vs one-hot parent encoding, the log-space ordinal likelihood,
seeding — are documented in [`CLAUDE.md`](CLAUDE.md) and pinned by tests.
Read that before changing anything in `src/tramdag/`.
