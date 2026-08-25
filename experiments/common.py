"""The output plumbing every experiment area shares.

Experiments live in one directory per area — ``paper/`` (the replications of
arXiv:2503.16206), ``benchmarks/`` (training and machine speed) and ``misc/``
(everything else, currently the classical-MLE validation). ``paper`` and
``misc`` each own their ``data/``, ``ground_truth/``, ``results/``, ``tests/``
and whatever helpers only they need. ``benchmarks`` measures speed on the other
two's data, so it reads theirs and pins no ground truth of its own.

What is left here is the output layout the experiments workflow reads:
``results/<name>/`` with ``metrics.json``, ``report.md`` and ``plots/``.

Reading a script's YAML file is here; checking the section it yields is not.
:func:`load_variant` parses the file and hands the document to
:func:`tramdag.utils.config_section`, which enforces the exact key set — that
guarantee is worth having in one place, and it needs no YAML.

Every function takes the calling script's ``__file__``, so paths resolve
inside that script's own area with no directory names written in the code.

Run an experiment as a module, from ``experiments/``::

    uv run python -m paper.triangle atan-cs
    uv run python -m check paper triangle-atan-cs
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path

import yaml

from tramdag.utils import config_section


# %% public functions ------------------------------------------------------------------
def load_variant(script: str, variant: str) -> dict:
    """Read one variant's hyperparameters from the script's own YAML file.

    The file lists every value the experiment uses, one section per variant
    under ``variants:``. Shared values are written once under an anchor and
    merged with YAML's ``<<`` key, so the merge is visible in the file
    instead of happening in code.

    Parameters
    ----------
    script : str
        The calling script's ``__file__``. The config is the sibling file
        with the same stem and a ``.yaml`` suffix.
    variant : str
        Key under ``variants:``.

    Returns
    -------
    dict
        The variant's hyperparameters.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(script).resolve().with_suffix(".yaml")
    if not path.exists():
        raise FileNotFoundError(f"no config next to {Path(script).name}: {path}")
    document = yaml.safe_load(path.read_text())
    return config_section(document, "variants", variant)


def variants_of(script: str) -> list[str]:
    """Give the variant names the script's config file defines.

    ``argparse`` takes its choices from this, so adding a variant to the
    config file is enough to make it runnable.
    """
    path = Path(script).resolve().with_suffix(".yaml")
    document = yaml.safe_load(path.read_text())
    return sorted(document["variants"])


def make_output_dir(script: str, name: str) -> Path:
    """Create ``<area>/results/<name>/plots/`` and give the results directory."""
    out = Path(script).resolve().parent / "results" / name
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def save_metrics(out: Path, metrics: dict) -> None:
    """Write the numbers the ground-truth check reads."""
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))


def write_report(out: Path, title: str, metrics: dict, figures: list[str]) -> None:
    """Write ``report.md``: the metrics table and the figures.

    The experiments workflow posts this file as a commit comment, so the
    figure links are plain relative paths that ``cml comment`` resolves and
    uploads.

    Parameters
    ----------
    out : Path
        The experiment's results directory.
    title : str
        Heading of the report.
    metrics : dict
        Flat ``{name: value}`` mapping, as written to ``metrics.json``.
    figures : list[str]
        File names under ``plots/``, in the order they should appear.
    """
    lines = [f"## {title}", "", "| metric | value |", "|---|---|"]
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| `{name}` | {value:+.4f} |")
        else:
            lines.append(f"| `{name}` | {value} |")
    lines.append("")
    for figure in figures:
        lines += [f"![{figure}](plots/{figure})", ""]
    (out / "report.md").write_text("\n".join(lines))
