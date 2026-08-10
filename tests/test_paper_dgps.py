"""Tests for the TRAM-DAG paper DGPs (arXiv:2503.16206) and their flow recovery.

Fast tests pin the generators (schema, determinism, do-mechanics, the TRAM
identities of the triangle DGPs, the frozen-CSV contract, analytic ground truth).
Fit tests check that `CausalFlowDAG` recovers the paper's known truth: the
interpretable coefficients (Sec. 6), the complex-shift curve -f(x2), the VACA
interventional moments (App. C.1) and the CAREFL counterfactual curves (C.2).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from tramdag import CS, LS, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.simulations import (
    REGISTRY,
    Carefl4,
    TriangleContinuous,
    TriangleMixed,
    VacaTriangle,
)
from tramdag.simulations.carefl import X_OBS
from tramdag.simulations.triangle import THETA_MIXED

DATA = Path(__file__).resolve().parents[1] / "data"

FROZEN = [  # (data subdir, generator factory)
    ("triangle/linear", lambda s: TriangleContinuous(f="linear", seed=s)),
    ("triangle/atan", lambda s: TriangleContinuous(f="atan", seed=s)),
    ("triangle/sin", lambda s: TriangleContinuous(f="sin", seed=s)),
    ("triangle-mixed/linear", lambda s: TriangleMixed(f="linear", seed=s)),
    ("triangle-mixed/exp", lambda s: TriangleMixed(f="exp", seed=s)),
    ("vaca", lambda s: VacaTriangle(seed=s)),
    ("carefl", lambda s: Carefl4(seed=s)),
]


# ---------------------------------------------------------------- generators
def test_registry_contains_paper_dgps():
    for key in ["triangle", "triangle-mixed", "vaca", "carefl"]:
        assert key in REGISTRY


@pytest.mark.parametrize("subdir,factory", FROZEN)
def test_generator_reproducible(subdir, factory):
    a = factory(42).observational(500)
    b = factory(42).observational(500)
    pd.testing.assert_frame_equal(a, b)


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
    gen = factory()
    rng = np.random.default_rng(0)
    lat = gen.draw_latents(2000, rng)
    base = gen.simulate(latents=lat)
    inter = gen.simulate(latents=lat, do={"x1": 0.5})
    assert (inter["x1"] == 0.5).all()
    assert not np.allclose(inter["x3"], base["x3"])  # downstream reacts


@pytest.mark.parametrize("f", ["linear", "atan", "sin"])
def test_triangle_continuous_tram_identity(f):
    """The DGP is a TRAM: h2 and h3 evaluated on samples are standard logistic."""
    gen = TriangleContinuous(f=f, seed=1)
    df = gen.observational(20_000)
    u2 = 5.0 * df["x2"] + 2.0 * df["x1"]
    u3 = 0.63 * df["x3"] - 0.2 * df["x1"] - gen.f_callable(df["x2"].to_numpy())
    assert stats.kstest(u2, "logistic").pvalue > 0.01
    assert stats.kstest(u3, "logistic").pvalue > 0.01


def test_triangle_mixed_pmf_matches_frequencies():
    gen = TriangleMixed(f="exp", seed=3)
    pmf = gen.true_pmf(np.array([0.5]), np.array([-0.4]))[0]
    assert pmf.shape == (4,) and abs(pmf.sum() - 1.0) < 1e-12
    # MC check at fixed parents: clamp x1, x2 and compare level frequencies
    rng = np.random.default_rng(4)
    sim = gen.simulate(200_000, rng=rng, do={"x1": 0.5, "x2": -0.4})
    freq = sim["x3"].value_counts(normalize=True).sort_index().to_numpy()
    np.testing.assert_allclose(freq, pmf, atol=0.005)


def test_carefl_abduction_roundtrip_and_factual_cf():
    gen = Carefl4(seed=5)
    rng = np.random.default_rng(6)
    lat = gen.draw_latents(1000, rng)
    df = gen.simulate(latents=lat)
    eps = gen.abduct_noise(df)
    for k in ["x3", "x4"]:
        np.testing.assert_allclose(eps[k], lat[k], atol=1e-12)
    # cf at the observed value reproduces the factual observation
    cf = gen.true_counterfactual(X_OBS, {"x2": X_OBS["x2"]})
    assert abs(cf["x3"] - X_OBS["x3"]) < 1e-12


def test_vaca_analytic_moments():
    t = VacaTriangle().true_moments(mc_n=200_000)
    mu1 = -0.25  # 0.5*(-2) + 0.5*1.5
    assert abs(t["obs_mean"]["x1"] - mu1) < 0.02
    assert abs(t["do_x2"]["0.0"]["mean_x3_analytic"] - mu1) < 1e-12
    assert abs(t["do_x2"]["-3.0"]["mean_x3_analytic"] - (mu1 - 0.75)) < 1e-12


# ------------------------------------------------------------ frozen contract
@pytest.mark.parametrize("subdir,factory", FROZEN)
def test_frozen_csv_contract(subdir, factory):
    """Committed CSVs regenerate bit-identically from the stored seed."""
    vdir = DATA / subdir
    truth = json.loads((vdir / "truth.json").read_text())
    frozen = pd.read_csv(vdir / "obs.csv")
    regen = factory(truth["seed"]).observational(truth["n_obs"])
    assert len(frozen) == truth["n_obs"]
    for c in frozen.columns:
        np.testing.assert_allclose(
            frozen[c].to_numpy(dtype=float), regen[c].to_numpy(dtype=float), atol=1e-9
        )


# ------------------------------------------------------------------ fit tests
#
# Fit tests train on larger samples regenerated deterministically from the
# generators (paper scale; the frozen n=5000 CSVs are the data *contract*, but
# e.g. beta13 multiplies the low-variance x1 in [0.25, 0.73] and is too weakly
# identified at n=5000 for tight tolerances).
N_FIT = 20_000


def _fit(spec, df, epochs=(250, 80), seed=7) -> CausalFlowDAG:
    torch.manual_seed(seed)
    flow = CausalFlowDAG(spec)
    cut = int(len(df) * 0.9)
    tr, va = df.iloc[:cut], df.iloc[cut:]
    flow.fit(tr, va, epochs=epochs[0], learning_rate=1e-2, batch_size=512, verbose=0)
    flow.fit(tr, va, epochs=epochs[1], learning_rate=1e-3, batch_size=512, verbose=0)
    return flow


def _w(flow, node, parent) -> float:
    return float(flow.nodes[node].shifts[parent].weight.detach())


def _cs_curve(flow, node, parent, grid) -> np.ndarray:
    x = torch.as_tensor(grid, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        return flow.nodes[node].shifts[parent](x).detach().numpy()


@pytest.mark.slow
def test_triangle_linear_ls_recovers_coefficients():
    """Paper Sec. 6.1 / Fig. 14: beta12=2, beta13=-0.2, beta23=+0.3."""
    df = TriangleContinuous(f="linear", seed=42).observational(N_FIT, seed_offset=100)
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(terms=[LS("x1")]),
        "x3": ContinuousNode(terms=[LS("x1"), LS("x2")]),
    }
    flow = _fit(spec, df)
    assert abs(_w(flow, "x2", "x1") - 2.0) < 0.1
    assert abs(_w(flow, "x3", "x1") - (-0.2)) < 0.1
    assert abs(_w(flow, "x3", "x2") - 0.3) < 0.1


@pytest.mark.slow
def test_triangle_atan_cs_recovers_coefficients_and_curve():
    """Paper Sec. 6.1 / Fig. 7+15: LS coefficients + CS curve == -f + const."""
    gen = TriangleContinuous(f="atan", seed=42)
    df = gen.observational(N_FIT, seed_offset=100)
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(terms=[LS("x1")]),
        "x3": ContinuousNode(terms=[LS("x1"), CS("x2")]),
    }
    flow = _fit(spec, df)
    assert abs(_w(flow, "x2", "x1") - 2.0) < 0.1
    assert abs(_w(flow, "x3", "x1") - (-0.2)) < 0.1
    grid = np.linspace(*df["x2"].quantile([0.01, 0.99]).to_numpy(), 100)
    fitted = _cs_curve(flow, "x3", "x2", grid)
    true = gen.true_shift_curve(grid)
    rmse = np.sqrt(np.mean(((fitted - fitted.mean()) - (true - true.mean())) ** 2))
    assert rmse < 0.15


@pytest.mark.slow
def test_triangle_mixed_linear_ls_recovers_with_sign_flip():
    """Paper Sec. 6.2 / Fig. 19, in flow convention (ordinal shift subtracted):
    fitted weights -> -0.2 (x1) and +0.3 (x2); cutpoints -> (-2, 0.42, 1.02).
    """
    from tramdag.transforms import ordinal_cutpoints

    df = TriangleMixed(f="linear", seed=42).observational(N_FIT, seed_offset=100)
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(terms=[LS("x1")]),
        "x3": OrdinalNode(levels=4, terms=[LS("x1"), LS("x2")]),
    }
    flow = _fit(spec, df)
    assert abs(_w(flow, "x3", "x1") - (-0.2)) < 0.1
    assert abs(_w(flow, "x3", "x2") - 0.3) < 0.1
    with torch.no_grad():
        theta = ordinal_cutpoints(flow.nodes["x3"].intercept(1))[0, 1:-1].numpy()
    np.testing.assert_allclose(theta, THETA_MIXED, atol=0.15)


@pytest.mark.slow
def test_vaca_ci_flow_matches_interventional_moments():
    """Paper Sec. 5.1-5.2: an all-ci flow fits the bimodal DGP and reproduces
    E/sd of x3 under do(x2=a) (analytic truth from App. C.1).
    """
    truth = json.loads((DATA / "vaca" / "truth.json").read_text())
    df = VacaTriangle(seed=42).observational(N_FIT, seed_offset=100)
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(terms=[I("x1")]),
        "x3": ContinuousNode(terms=[I("x1", "x2")]),
    }
    flow = _fit(spec, df, epochs=(400, 120))
    # L1: bimodality of x1 is captured (the paper's headline vs CNF) — both
    # modes present: density mass on each side of the saddle near -0.3
    samp = flow.sample(20_000, seed=0)
    assert ((samp["x1"] < -1.0).mean() > 0.25) and ((samp["x1"] > 1.0).mean() > 0.25)
    for a in (-3.0, -1.0, 0.0):
        # do(x2=-3) pairs the lower x1 mode with a ~5-sigma-off x2: genuine
        # extrapolation beyond the observational manifold -> looser tolerance
        tol = 0.3 if a == -3.0 else 0.2
        ref = truth["do_x2"][str(a)]
        do_samp = flow.sample(20_000, do={"x2": a}, seed=1)
        assert abs(do_samp["x3"].mean() - ref["mean_x3_analytic"]) < tol, f"do(x2={a})"
        assert abs(do_samp["x3"].std() - ref["std_x3_analytic"]) < tol, f"do(x2={a})"


@pytest.mark.slow
def test_carefl_ci_flow_recovers_counterfactuals():
    """Paper Sec. 5.3 / Fig. 6 capability, scored robustly: individual
    counterfactuals of held-out rows vs. the analytic DGP truth. (The paper's
    single x_obs has a ~4-sigma abducted noise eps3, so a one-point test would
    hinge on sub-1% tail-CDF accuracy and be fragile across data draws — the
    faithful single-point sweep lives in experiments/paper_carefl.py.)
    """
    gen = Carefl4(seed=42)
    df = gen.observational(N_FIT, seed_offset=100)
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(),
        "x3": ContinuousNode(terms=[I("x1", "x2")]),
        "x4": ContinuousNode(terms=[I("x1", "x2")]),
    }
    flow = _fit(spec, df, epochs=(300, 100))
    rows = gen.observational(300, seed_offset=999)
    u = flow.abduct(rows)
    eps = gen.abduct_noise(rows)
    for a in (-1.5, 0.0, 1.5):
        cf_flow = flow.sample(do={"x2": a}, u=u)
        cf_true = gen.simulate(do={"x2": a}, latents=eps)
        mae3 = float(np.abs(cf_flow["x3"].to_numpy() - cf_true["x3"].to_numpy()).mean())
        cf_flow = flow.sample(do={"x1": a}, u=u)
        cf_true = gen.simulate(do={"x1": a}, latents=eps)
        mae4 = float(np.abs(cf_flow["x4"].to_numpy() - cf_true["x4"].to_numpy()).mean())
        assert mae3 < 0.3, f"do(x2={a}): MAE(x3_cf) = {mae3:.3f}"
        assert mae4 < 0.3, f"do(x1={a}): MAE(x4_cf) = {mae4:.3f}"
