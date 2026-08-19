"""Checks on the benchmark workloads.

A benchmark is only comparable if its workload does not move. Two things can
break that quietly, and both are pinned here: the standalone copy of the
bimodal DGP drifting away from the maintained generator, and a workload's
frozen CSV disappearing from the area that owns it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benchmarks.bench_training import all_ls_spec, stroke_data, vaca_data
from benchmarks.perf_machine import vaca_dgp

# a deliberate cross-area import: paper/ owns the generator, and this asserts
# that perf_machine.py's self-contained copy still reproduces it exactly
from paper.simulations.vaca import VacaTriangle

from tramdag import CausalFlowDAG


def test_standalone_dgp_matches_the_generator():
    """perf_machine.py carries its own DGP so it can be downloaded alone.

    It must stay bit-identical to the generator, including the order the
    noises are drawn in: a different order is a different sample, which would
    silently make `final_val_nll` incomparable with the committed
    cross-machine results in docs/perf/.
    """
    standalone = vaca_dgp(5_000)
    generator = VacaTriangle(seed=42).observational(5_000)
    assert list(standalone.columns) == list(generator.columns)
    for column in generator.columns:
        np.testing.assert_array_equal(
            standalone[column].to_numpy(), generator[column].to_numpy()
        )


def test_all_ls_spec_builds_and_is_classical():
    """The workload's spec is all linear shifts, so fit_classical accepts it.

    That is what makes its optimum the classical MLE, and therefore a fixed
    target the benchmark can measure time-to-accuracy against.
    """
    flow = CausalFlowDAG(all_ls_spec(), seed=0)
    assert flow.order[0] == "Age" and flow.order[-1] == "mRS_3m"
    observed, validation = stroke_data()
    assert validation is None  # full-data MLE fit
    assert len(observed) == 1275
    report = flow.fit_classical(observed, max_iter=5, verbose=False)
    assert np.isfinite(report["final_nll"])


def test_vaca_workload_reads_the_frozen_csv():
    train, val = vaca_data()
    assert len(train) + len(val) == 5000
    assert list(train.columns) == ["x1", "x2", "x3"]


@pytest.mark.parametrize(
    "path",
    [
        Path("misc") / "data" / "magic-mrclean" / "ls" / "obs.csv",
        Path("paper") / "data" / "vaca" / "obs.csv",
    ],
)
def test_workload_data_lives_where_the_benchmark_looks(path):
    """The benchmarks read frozen CSVs from the areas that own them."""
    full = Path(__file__).resolve().parents[2] / path
    assert full.exists(), f"missing benchmark workload data: {full}"
    assert len(pd.read_csv(full)) > 0
