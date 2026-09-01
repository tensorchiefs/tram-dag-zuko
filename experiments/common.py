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
:func:`load_variant` parses the file and picks the variant's section with
:func:`_config_section` below; that every key in it is read by the
script is what ``paper/tests/test_configs.py`` checks.

Every function takes the calling script's ``__file__``, so paths resolve
inside that script's own area with no directory names written in the code.

Run an experiment as a module, from ``experiments/``::

    uv run python -m paper.triangle atan-cs
    uv run python -m check paper triangle-atan-cs
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


# %% private functions------------------------------------------------------------------
def _config_section(document: dict, *keys: str) -> dict:
    """Pick a mapping out of a parsed configuration.

    Parsing is the caller's job — pass whatever ``yaml.safe_load``,
    ``json.load`` or ``tomllib.load`` returned. This descends through
    ``keys`` and gives the mapping found there as a shallow copy.

    Parameters
    ----------
    document : dict
        The parsed configuration.
    *keys : str
        Keys to descend through before the mapping is returned, for example
        ``"variants", "atan-cs"`` for a document that groups several
        variants. Without any key the document itself is used.

    Returns
    -------
    dict
        The selected mapping, as a shallow copy.

    Raises
    ------
    KeyError
        If one of ``keys`` is not present. The message lists what is
        available at that level.
    ValueError
        If a selected value is not a mapping.

    Examples
    --------
    >>> document = {"variants": {"fast": {"epochs": 5, "lr": 0.01}}}
    >>> _config_section(document, "variants", "fast")
    {'epochs': 5, 'lr': 0.01}
    """
    node = document
    for depth, key in enumerate(keys):
        if not isinstance(node, dict):
            # malformed config data, not a wrongly typed argument
            raise ValueError(  # noqa: TRY004
                f"{' -> '.join(keys[:depth]) or 'the document'} is "
                f"{type(node).__name__}, not a mapping"
            )
        if key not in node:
            raise KeyError(
                f"no '{key}' in {' -> '.join(keys[:depth]) or 'the document'}. "
                f"Available: {', '.join(sorted(map(str, node)))}"
            )
        node = node[key]

    where = " -> ".join(keys) or "the document"
    if not isinstance(node, dict):
        # malformed config data, not a wrongly typed argument
        raise ValueError(f"{where} is {type(node).__name__}, not a mapping")  # noqa: TRY004

    return dict(node)


def _variants_of(script: str) -> list[str]:
    """Give the variant names the script's config file defines.

    ``argparse`` takes its choices from this, so adding a variant to the
    config file is enough to make it runnable.
    """
    path = Path(script).resolve().with_suffix(".yaml")
    document = yaml.safe_load(path.read_text())
    return sorted(document["variants"])


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
    return _config_section(document, "variants", variant)


def cli(script: str, doc: str) -> str:
    """Parse an experiment script's command line and give the chosen variant.

    The one positional argument is the variant; its choices come from the
    script's YAML file and the description from the first line of its
    module docstring.
    """
    parser = argparse.ArgumentParser(description=doc.splitlines()[0])
    parser.add_argument(
        "variant",
        choices=_variants_of(script),
        help="which variant to run; hyperparameters live in the sibling YAML file",
    )
    return parser.parse_args().variant


def make_output_dir(script: str, name: str) -> Path:
    """Create ``<area>/results/<name>/plots/`` and give the results directory."""
    out = Path(script).resolve().parent / "results" / name
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def save_metrics(out: Path, metrics: dict) -> None:
    """Write the numbers the ground-truth check reads."""
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))


def _report_row(name: str, value, truths: dict) -> str:
    """One metric row; with truths, a fitted-vs-true row where one exists."""
    fmt = lambda v: f"{v:+.4f}" if isinstance(v, float) else f"{v}"  # noqa: E731
    if not truths:
        return f"| `{name}` | {fmt(value)} |"
    if name not in truths:
        return f"| `{name}` | {fmt(value)} |  |  |"
    err = abs(float(value) - float(truths[name]))
    return f"| `{name}` | {fmt(value)} | {fmt(truths[name])} | {err:.4f} |"


def write_report(
    out: Path, title: str, metrics: dict, figures: list[str], truths: dict | None = None
) -> None:
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
    truths : dict | None, optional
        ``{metric_name: true_value}`` — the DGP/analytic ground truth for the
        metrics that have one. Those rows gain a truth and an ``|err|``
        column, so the commit comment shows fitted-vs-true at a glance.
    """
    truths = truths or {}
    lines = [f"## {title}", ""]
    if truths:
        # no pipes in cell text: cml strips the backslash escapes, which
        # desyncs the header from the |---| separator row
        lines += ["| metric | value | DGP truth | abs. error |", "|---|---|---|---|"]
    else:
        lines += ["| metric | value |", "|---|---|"]
    lines += [_report_row(name, value, truths) for name, value in metrics.items()]
    lines.append("")
    for figure in figures:
        lines += [f"![{figure}](plots/{figure})", ""]
    (out / "report.md").write_text("\n".join(lines))
