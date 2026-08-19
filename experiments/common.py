"""The output plumbing every experiment area shares.

Experiments live in one directory per area — ``paper/`` (the replications of
arXiv:2503.16206), ``benchmarks/`` (training and machine speed) and ``misc/``
(everything else, currently the classical-MLE validation). Each area owns its
own ``data/``, ``ground_truth/``, ``results/``, ``tests/`` and whatever
helpers only it needs.

What is left here is the output layout the experiments workflow reads:
``results/<name>/`` with ``metrics.json``, ``report.md`` and ``plots/``.

The strict config reader is **not** here: it moved into the framework as
:func:`tramdag.load_config`, because "a missing key must never become a
hidden default" is a guarantee worth having in one place, and useful to any
script that keeps its hyperparameters in a file.

Every function takes the calling script's ``__file__``, so paths resolve
inside that script's own area with no directory names written in the code.

Run an experiment as a module, from ``experiments/``::

    uv run python -m paper.triangle atan-cs
    uv run python -m check paper triangle-atan-cs
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def config_path(script: str) -> Path:
    """Give the YAML file belonging to a script: same directory, same stem.

    Parameters
    ----------
    script : str
        The calling script's ``__file__``.

    Returns
    -------
    Path
        The sibling ``.yaml`` file.
    """
    return Path(script).resolve().with_suffix(".yaml")


def variants_of(script: str) -> list[str]:
    """Give the variant names the script's config file defines.

    ``argparse`` takes its choices from this, so adding a variant to the
    config file is enough to make it runnable.
    """
    document = yaml.safe_load(config_path(script).read_text())
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
