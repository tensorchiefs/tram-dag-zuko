"""Compare an experiment's metrics against the committed ground truth.

The experiments workflow calls this after each run. Ground truth lives in
``<area>/ground_truth/<result-dir>.json`` as one entry per metric::

    {"_note": "what these numbers mean",
     "beta12": {"value": 1.9825, "atol": 0.05},
     "cs_curve_max_abs_err": {"max": 0.23}}

Two forms. ``{value, atol}`` is two-sided, for a quantity that should stay
where it is. ``{max}`` is an upper bound, for an **error measure**, where a
smaller number is a better fit rather than a drift and must not fail the run.

A ``{max}`` bound is only useful in a band. Below **1.5x** its measurement it
fails on another machine for no reason (measured: one such bound passed at
0.028 here and failed CI at 0.113). Above **4x** it cannot catch a regression.
A bound outside the band is reported as ``note`` — not a failure, because a
tolerance is a judgement call, but visibly, so it gets re-pinned deliberately
rather than drifting. A bound that is *meant* to be wide carries a ``"why"``
string, which is printed in place of the note::

    "max_abs_diff_flow_vs_statsmodels": {
      "max": 0.25,
      "why": "the max is over a coefficient with 7 of 1275 observations: 0.028
              here, 0.113 on the CI runner"}

A ``"why"`` excuses width only. The *too tight* note always fires, because no
argument makes a bound below 1.5x its measurement survive another machine.

A ``{value, atol}`` center decays the other way: it keeps passing while
describing an older run, drifting through its tolerance until one machine
tips it over. A measurement past **half** its ``atol`` is reported the same
way, so the center gets re-pinned while it still passes.

A key starting with an underscore is a note for the reader and is skipped.

Every entry needs its own tolerance, because torch results differ slightly
across operating systems and CPUs — a measured spread, not a hope. A metric
without an entry is reported and ignored; an entry without a metric is an
error, because that means the experiment stopped producing a number the
ground truth claims to check.

Usage (from ``experiments/``)::

    uv run python -m check paper triangle-atan-cs
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# %% global variables ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
# only areas that commit ground truth; the benchmarks are measured and
# written up in docs/, not checked against a recorded value
AREAS = ("paper", "misc")


# %% private functions -----------------------------------------------------------------
def _load_json_pair(area: str, name: str) -> tuple[dict, dict]:
    """Load the run's metrics and its ground truth, or raise."""
    metrics_path = HERE / area / "results" / name / "metrics.json"
    truth_path = HERE / area / "ground_truth" / f"{name}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics to check: {metrics_path}")
    if not truth_path.exists():
        raise FileNotFoundError(
            f"no ground truth for '{area}/{name}': {truth_path}. Write one from "
            "a reviewed run before wiring the experiment into CI."
        )
    return json.loads(metrics_path.read_text()), json.loads(truth_path.read_text())


def _check_metric(name: str, metric: str, metrics: dict, expected: dict) -> list:
    """Check one ground-truth entry; give ``(kind, message)`` pairs.

    ``kind`` is ``"fail"``, ``"ok"`` or ``"note"`` (a bound outside the useful
    band, or a center past half its tolerance).
    """
    if metric not in metrics:
        return [
            (
                "fail",
                (
                    f"{metric}: the run produced no such metric. Either the "
                    f"experiment stopped computing it, or the entry belongs in "
                    f"neither ground_truth/{name}.json nor the run."
                ),
            )
        ]
    measured = metrics[metric]
    if "max" in expected:
        return _check_bound(metric, measured, expected)
    return _check_center(metric, measured, expected)


def _check_bound(metric: str, measured: float, expected: dict) -> list:
    """Check one ``{max}`` entry: an upper bound on an error measure.

    Only exceeding the bound is a regression, because a smaller error is a
    better fit, not a drifted one. A passing bound outside the useful band
    (1.5x to 4x the measurement) is noted, unless a ``"why"`` excuses its
    width; a bound below 1.5x is always noted.
    """
    bound = expected["max"]
    if measured > bound:
        return [("fail", f"{metric}: {measured:+.4f} exceeds its bound {bound:+.4f}")]
    label = f"bound {bound}" + (
        f", deliberately wide: {expected['why']}" if "why" in expected else ""
    )
    out = [("ok", f"{metric}: {measured:+.4f} ({label})")]
    if measured > 0:
        ratio = bound / abs(measured)
        if ratio > 4.0 and "why" not in expected:
            out.append(
                (
                    "note",
                    (
                        f"{metric}: bound {bound} is {ratio:.1f}x the measurement — "
                        "too loose to catch a regression"
                    ),
                )
            )
        elif ratio < 1.5:
            out.append(
                (
                    "note",
                    (
                        f"{metric}: bound {bound} is {ratio:.2f}x its measurement — "
                        "will fail on another machine"
                    ),
                )
            )
    return out


def _check_center(metric: str, measured: float, expected: dict) -> list:
    """Check one ``{value, atol}`` entry, two-sided; note a decayed center."""
    deviation = abs(measured - expected["value"])
    if deviation > expected["atol"]:
        return [
            (
                "fail",
                (
                    f"{metric}: {measured:+.4f} vs expected {expected['value']:+.4f} "
                    f"(deviation {deviation:.4f} > atol {expected['atol']})"
                ),
            )
        ]
    out = [("ok", f"{metric}: {measured:+.4f} (within {expected['atol']})")]
    if deviation > 0.5 * expected["atol"]:
        share = deviation / expected["atol"]
        out.append(
            (
                "note",
                (
                    f"{metric}: {measured:+.6f} sits {share:.0%} of the way to its "
                    "tolerance — the center describes an older run, re-pin it from "
                    "this one"
                ),
            )
        )
    return out


def compare(area: str, name: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Compare one result directory against its ground truth.

    Parameters
    ----------
    area : str
        Experiment area: ``paper`` or ``misc`` — the two that commit ground
        truth (see ``AREAS``).
    name : str
        Name of the results directory, for example ``"triangle-atan-cs"``.

    Returns
    -------
    tuple[list[str], list[str], list[str], list[str]]
        The failures, the metrics that passed, the metrics with no
        ground-truth entry, and the bounds outside the useful band.

    Raises
    ------
    FileNotFoundError
        If the metrics or the ground-truth file is missing.
    ValueError
        If the ground-truth file has no metric entry, so nothing was checked.
    """
    metrics, truth = _load_json_pair(area, name)

    results = [
        pair
        for metric, expected in truth.items()
        if not metric.startswith("_")  # a note for the reader, not a metric
        for pair in _check_metric(name, metric, metrics, expected)
    ]
    if not results:
        raise ValueError(f"ground_truth/{name}.json checks nothing: no metric entries")
    by_kind = {
        k: [m for kind, m in results if kind == k] for k in ("fail", "ok", "note")
    }
    failures, notes, loose = by_kind["fail"], by_kind["ok"], by_kind["note"]
    unchecked = [
        f"{metric}: {metrics[metric]}" for metric in metrics if metric not in truth
    ]
    return failures, notes, unchecked, loose


def main(area: str, name: str) -> int:
    """Print the comparison and give the process exit code."""
    failures, notes, unchecked, loose = compare(area, name)
    for note in notes:
        print(f"  ok   {note}")
    for item in unchecked:
        print(f"  --   {item} (no ground-truth entry)")
    for item in loose:
        print(f"  note {item}")
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"\n{area}/{name}: {len(failures)} metric(s) outside tolerance")
        return 1
    print(f"\n{area}/{name}: all checked metrics within tolerance")
    return 0


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("area", choices=AREAS, help="experiment area")
    parser.add_argument("name", help="results directory name, e.g. triangle-atan-cs")
    args = parser.parse_args()
    sys.exit(main(args.area, args.name))
