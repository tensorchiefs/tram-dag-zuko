"""Compare an experiment's metrics against the committed ground truth.

The experiments workflow calls this after each run. Ground truth lives in
``ground_truth/<result-dir>.json`` as one entry per metric::

    {"_note": "what these numbers mean",
     "beta12": {"value": 2.0012, "atol": 0.05}}

A key starting with an underscore is a note for the reader and is skipped.

Every entry needs its own tolerance, because torch results differ slightly
across operating systems and CPUs — a measured spread, not a hope. A metric
without an entry is reported and ignored; an entry without a metric is an
error, because that means the experiment stopped producing a number the
ground truth claims to check.

Usage (from experiments/)::

    uv run python check.py triangle-atan-cs
"""

from __future__ import annotations

import argparse
import json
import sys

from common import GROUND_TRUTH, RESULTS


def compare(name: str) -> tuple[list[str], list[str]]:
    """Compare one result directory against its ground truth.

    Parameters
    ----------
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
    metrics_path = RESULTS / name / "metrics.json"
    truth_path = GROUND_TRUTH / f"{name}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics to check: {metrics_path}")
    if not truth_path.exists():
        raise FileNotFoundError(
            f"no ground truth for '{name}': {truth_path}. Write one from a "
            "reviewed run before wiring the experiment into CI."
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
        deviation = abs(metrics[metric] - expected["value"])
        if deviation > expected["atol"]:
            failures.append(
                f"{metric}: {metrics[metric]:+.4f} vs expected "
                f"{expected['value']:+.4f} (deviation {deviation:.4f} > "
                f"atol {expected['atol']})"
            )
        else:
            notes.append(
                f"{metric}: {metrics[metric]:+.4f} (within {expected['atol']})"
            )
    for metric in metrics:
        if metric not in truth:
            notes.append(f"{metric}: {metrics[metric]} (not checked)")
    return failures, notes


def main(name: str) -> int:
    """Print the comparison and give the process exit code."""
    failures, notes = compare(name)
    for note in notes:
        print(f"  ok   {note}")
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"\n{name}: {len(failures)} metric(s) outside tolerance")
        return 1
    print(f"\n{name}: all checked metrics within tolerance")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="results directory name, e.g. triangle-atan-cs")
    sys.exit(main(parser.parse_args().name))
