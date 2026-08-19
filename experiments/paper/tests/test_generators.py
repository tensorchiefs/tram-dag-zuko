"""Checks on the paper DGPs themselves — the ground truth, not the flow.

These are the cheapest and most important tests in ``experiments/``: every
number the replications are scored against comes from these generators, so a
silent change here would look like a framework regression. They are numpy-only
and take seconds, which is why they run in the ordinary ``pytest`` alongside
the framework suite rather than only in the experiments workflow.

What the flow *does* with these DGPs is measured by the experiment scripts and
their committed ground truth (see ``paper/ground_truth/``), not here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from paper.check_data import DATASETS
from paper.simulations import REGISTRY
from paper.simulations.carefl import X_OBS, Carefl4
from paper.simulations.triangle import TriangleContinuous, TriangleMixed
from paper.simulations.vaca import VacaTriangle
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data"


def test_registry_lists_every_paper_dgp():
    assert set(REGISTRY) == {"triangle", "triangle-mixed", "vaca", "carefl"}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TriangleContinuous(f="atan"),
        lambda: TriangleMixed(f="exp"),
        lambda: VacaTriangle(),
        lambda: Carefl4(),
    ],
)
def test_do_clamps_and_propagates(factory):
    """An intervened variable is held fixed and its descendants react."""
    gen = factory()
    latents = gen.draw_latents(2000, np.random.default_rng(0))
    base = gen.simulate(latents=latents)
    intervened = gen.simulate(latents=latents, do={"x1": 0.5})
    assert (intervened["x1"] == 0.5).all()
    assert not np.allclose(intervened["x3"], base["x3"])


@pytest.mark.parametrize("f", ["linear", "atan", "sin"])
def test_triangle_continuous_is_a_tram(f):
    """The DGP is a transformation model: h(x) on samples is standard logistic.

    This is the identity the whole replication rests on — if it fails, the
    "true" coefficients the experiments compare against are not the truth.
    """
    gen = TriangleContinuous(f=f, seed=1)
    df = gen.observational(20_000)
    u2 = 5.0 * df["x2"] + 2.0 * df["x1"]
    u3 = 0.63 * df["x3"] - 0.2 * df["x1"] - gen.f_callable(df["x2"].to_numpy())
    assert stats.kstest(u2, "logistic").pvalue > 0.01
    assert stats.kstest(u3, "logistic").pvalue > 0.01


def test_triangle_mixed_pmf_matches_frequencies():
    """The analytic ordinal PMF agrees with simulated level frequencies."""
    gen = TriangleMixed(f="exp", seed=3)
    pmf = gen.true_pmf(np.array([0.5]), np.array([-0.4]))[0]
    assert pmf.shape == (4,) and abs(pmf.sum() - 1.0) < 1e-12
    simulated = gen.simulate(
        200_000, rng=np.random.default_rng(4), do={"x1": 0.5, "x2": -0.4}
    )
    frequencies = simulated["x3"].value_counts(normalize=True).sort_index().to_numpy()
    np.testing.assert_allclose(frequencies, pmf, atol=0.005)


def test_carefl_noise_recovery_is_exact():
    """Additive noise is recovered exactly, so counterfactuals are analytic."""
    gen = Carefl4(seed=5)
    latents = gen.draw_latents(1000, np.random.default_rng(6))
    df = gen.simulate(latents=latents)
    recovered = gen.abduct_noise(df)
    for name in ("x3", "x4"):
        np.testing.assert_allclose(recovered[name], latents[name], atol=1e-12)
    # a counterfactual at the observed value reproduces the observation
    unchanged = gen.true_counterfactual(X_OBS, {"x2": X_OBS["x2"]})
    assert abs(unchanged["x3"] - X_OBS["x3"]) < 1e-12


def test_vaca_analytic_moments():
    """The stored interventional means are the closed-form ones."""
    truth = VacaTriangle().true_moments(mc_n=200_000)
    mean_x1 = -0.25  # 0.5*(-2) + 0.5*1.5
    assert abs(truth["obs_mean"]["x1"] - mean_x1) < 0.02
    assert abs(truth["do_x2"]["0.0"]["mean_x3_analytic"] - mean_x1) < 1e-12
    assert abs(truth["do_x2"]["-3.0"]["mean_x3_analytic"] - (mean_x1 - 0.75)) < 1e-12


@pytest.mark.parametrize("subdir", sorted(DATASETS))
def test_frozen_csv_regenerates(subdir):
    """Each committed CSV still comes out of its generator, to 1e-9.

    Not bit equality: numpy's transcendental functions move their last bits
    between releases. ``paper/check_data.py`` runs the same comparison over
    every dataset from the command line.
    """
    directory = DATA / subdir
    truth = json.loads((directory / "truth.json").read_text())
    frozen = pd.read_csv(directory / "obs.csv")
    regenerated = DATASETS[subdir](truth).observational(truth["n_obs"])
    assert len(frozen) == truth["n_obs"]
    for column in frozen.columns:
        np.testing.assert_allclose(
            frozen[column].to_numpy(dtype=float),
            regenerated[column].to_numpy(dtype=float),
            atol=1e-9,
        )
