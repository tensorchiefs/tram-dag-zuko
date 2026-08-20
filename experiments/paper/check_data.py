"""Check that every frozen dataset still regenerates from its generator.

The CSVs under ``data/`` are a contract: an experiment's ground truth is
only meaningful if the data behind it cannot drift silently. This script
regenerates each one from the seed stored in its ``truth.json`` and
compares numerically.

The tolerance is 1e-9, not bit equality: numpy's transcendental functions
(``arctan``, ``sin``, ``**``) change in their last bits between releases
and CPU dispatch paths, which moves values by ~1e-16 while the data stays
the same data. Anything larger means the generator changed.

``magic-mrclean/ls`` is not listed: its generator left with the stroke
storyline, so that cohort is frozen input data with no generator to check
it against. Recover the generator from the ``pre-experiments-cut`` tag if
it ever needs regenerating.

Usage (from experiments/)::

    uv run python -m paper.check_data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from paper.simulations.carefl import Carefl4
from paper.simulations.triangle import TriangleContinuous, TriangleMixed
from paper.simulations.vaca import VacaTriangle

DATA = Path(__file__).resolve().parent / "data"
ATOL = 1e-9

# dataset directory -> (generator factory taking the truth dict, ...)
DATASETS = {
    "carefl": lambda truth: Carefl4(seed=truth["seed"]),
    "vaca": lambda truth: VacaTriangle(seed=truth["seed"]),
    "triangle/linear": lambda truth: TriangleContinuous(f="linear", seed=truth["seed"]),
    "triangle/atan": lambda truth: TriangleContinuous(f="atan", seed=truth["seed"]),
    "triangle/sin": lambda truth: TriangleContinuous(f="sin", seed=truth["seed"]),
    "triangle-mixed/linear": lambda truth: TriangleMixed(
        f="linear", seed=truth["seed"]
    ),
    "triangle-mixed/exp": lambda truth: TriangleMixed(f="exp", seed=truth["seed"]),
}


def worst_deviation(subdir: str) -> float:
    """Regenerate one frozen dataset and give its largest absolute deviation.

    ``experiments/paper/tests/test_generators.py`` asserts on this same
    function, so the CLI and the test cannot drift apart.

    Raises
    ------
    ValueError
        If the regenerated frame no longer has the frozen one's shape and
        columns — a changed generator, not a changed last bit.
    """
    directory = DATA / subdir
    truth = json.loads((directory / "truth.json").read_text())
    frozen = pd.read_csv(directory / "obs.csv")
    regenerated = DATASETS[subdir](truth).observational(truth["n_obs"])
    if list(frozen.columns) != list(regenerated.columns):
        raise ValueError(
            f"columns changed: {list(frozen.columns)} vs {list(regenerated.columns)}"
        )
    if len(frozen) != truth["n_obs"] or len(regenerated) != truth["n_obs"]:
        raise ValueError(
            f"row count changed: {len(frozen)} / {len(regenerated)} "
            f"against n_obs {truth['n_obs']}"
        )
    return max(
        float(
            np.abs(
                frozen[column].to_numpy(dtype=float)
                - regenerated[column].to_numpy(dtype=float)
            ).max()
        )
        for column in frozen.columns
    )


def main() -> int:
    """Check every dataset and give the process exit code."""
    failures = 0
    for name in DATASETS:
        deviation = worst_deviation(name)
        if deviation > ATOL:
            failures += 1
            print(f"  FAIL {name}: max |diff| = {deviation:.2e} > {ATOL:.0e}")
        else:
            print(f"  ok   {name}: max |diff| = {deviation:.2e}")

    if failures:
        print(f"\n{failures} dataset(s) no longer match their generator")
        return 1
    print(f"\nall {len(DATASETS)} frozen datasets regenerate within {ATOL:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
