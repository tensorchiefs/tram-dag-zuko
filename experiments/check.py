"""Compare an experiment's metrics against the committed ground truth.

The experiments workflow calls this after each run. Ground truth lives in
``<area>/ground_truth/<result-dir>.json`` as one entry per metric::

    {"_note": "what these numbers mean",
     "beta12": {"value": 2.0012, "atol": 0.05},
     "cs_curve_max_abs_err": {"max": 0.23}}

Two forms. ``{value, atol}`` is two-sided, for a quantity that should stay
where it is. ``{max}`` is an upper bound, for an **error measure**, where a
smaller number is a better fit rather than a drift and must not fail the run.

A key starting with an underscore is a note for the reader and is skipped.

Every entry needs its own tolerance, because torch results differ slightly
across operating systems and CPUs — a measured spread, not a hope. A metric
without an entry is reported and ignored; an entry without a metric is an
error, because that means the experiment stopped producing a number the
ground truth claims to check.

Usage (from ``experiments/``)::

    uv run python -m check paper triangle-atan-cs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# only areas that commit ground truth; the benchmarks are measured and
# written up in docs/, not checked against a recorded value
AREAS = ("paper", "misc")


def compare(area: str, name: str) -> tuple[list[str], list[str]]:
    """Compare one result directory against its ground truth.

    Parameters
    ----------
    area : str
        Experiment area: ``paper``, ``benchmarks`` or ``misc``.
    name : str
        Name of the results directory, for example ``"triangle-atan-cs"``.

    Returns
    -------
    tuple[list[str], list[str]]
        The failure lines and the informational lines.

    Raises
    ------
    FileNotFoundError
        If the metrics or the ground-truth file is missing.
    """
    metrics_path = HERE / area / "results" / name / "metrics.json"
    truth_path = HERE / area / "ground_truth" / f"{name}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics to check: {metrics_path}")
    if not truth_path.exists():
        raise FileNotFoundError(
            f"no ground truth for '{area}/{name}': {truth_path}. Write one from "
            "a reviewed run before wiring the experiment into CI."
        )

    metrics = json.loads(metrics_path.read_text())
    truth = json.loads(truth_path.read_text())

    failures, notes = [], []
    for metric, expected in truth.items():
        if metric.startswith("_"):
            continue  # a note for the reader, not a metric
        if metric not in metrics:
            failures.append(f"{metric}: the run produced no such metric")
            continue
        measured = metrics[metric]
        if "max" in expected:
            # an error measure: only exceeding it is a regression, because a
            # smaller error is a better fit, not a drifted one
            if measured > expected["max"]:
                failures.append(
                    f"{metric}: {measured:+.4f} exceeds its bound "
                    f"{expected['max']:+.4f}"
                )
            else:
                notes.append(f"{metric}: {measured:+.4f} (bound {expected['max']})")
            continue
        deviation = abs(measured - expected["value"])
        if deviation > expected["atol"]:
            failures.append(
                f"{metric}: {measured:+.4f} vs expected "
                f"{expected['value']:+.4f} (deviation {deviation:.4f} > "
                f"atol {expected['atol']})"
            )
        else:
            notes.append(f"{metric}: {measured:+.4f} (within {expected['atol']})")
    for metric in metrics:
        if metric not in truth:
            notes.append(f"{metric}: {metrics[metric]} (not checked)")
    return failures, notes


def main(area: str, name: str) -> int:
    """Print the comparison and give the process exit code."""
    failures, notes = compare(area, name)
    for note in notes:
        print(f"  ok   {note}")
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"\n{area}/{name}: {len(failures)} metric(s) outside tolerance")
        return 1
    print(f"\n{area}/{name}: all checked metrics within tolerance")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("area", choices=AREAS, help="experiment area")
    parser.add_argument("name", help="results directory name, e.g. triangle-atan-cs")
    args = parser.parse_args()
    sys.exit(main(args.area, args.name))
