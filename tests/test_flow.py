"""Unit tests for the tramdag causal flow."""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CS, LS, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.spec import validate_and_sort
from tramdag.transforms import (
    AffineUT,
    BernsteinUT,
    SplineUT,
    _expanding_bisection,
    ordinal_cutpoints,
    ordinal_log_prob,
    ordinal_pmf,
)

# %% global variables ------------------------------------------------------------------
torch.manual_seed(0)


# %% public functions ------------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted_flow():
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.normal(0, 2, n)
    y_lat = 0.8 * x + rng.logistic(size=n)
    y = np.digitize(y_lat, [-1.0, 0.5, 2.0]).astype(float)
    w = 0.5 * x - 0.3 * y + rng.logistic(size=n)
    df = pd.DataFrame({"X": x, "Y": y, "W": w})
    spec = {
        "X": ContinuousNode(),
        "Y": OrdinalNode(4, [LS("X")]),
        "W": ContinuousNode([LS("X"), LS("Y")]),
    }
    flow = CausalFlowDAG(spec)
    flow.fit(
        df.iloc[:2400],
        df.iloc[2400:],
        epochs=300,
        learning_rate=0.05,
        batch_size=600,
        verbose=0,
        seed=0,
    )
    return flow, df


# %% transforms ------------------------------------------------------------------------
@pytest.mark.parametrize("ut", [BernsteinUT(n_coeffs=12), SplineUT(bins=6), AffineUT()])
def test_univariate_roundtrip(ut):
    ut.set_range(-3.0, 7.0)
    n = 200
    theta = torch.randn(n, ut.n_params)
    x = torch.linspace(-6.0, 10.0, n)  # includes values outside the fitted range
    z0, ladj = ut.forward(theta, x)
    assert torch.isfinite(z0).all()
    assert torch.isfinite(ladj).all()
    x_rec = ut.inverse(theta, z0)
    assert torch.allclose(x_rec, x, atol=1e-3), (x_rec - x).abs().max()


def test_ordinal_log_prob_matches_pmf():
    theta = torch.randn(128, 6)
    shift = torch.randn(128)
    y = torch.randint(0, 7, (128,)).float()
    lp = ordinal_log_prob(theta, shift, y)
    p = ordinal_pmf(theta, shift)[torch.arange(128), y.long()]
    # below ~1e-4 the naive sigmoid-difference pmf hits float32 quantization
    # (eps ~ 1.2e-7 near sigmoid = 1); the log-space form stays accurate there
    ok = p > 1e-4
    assert ok.sum() > 100
    assert torch.allclose(lp[ok], torch.log(p[ok]), atol=2e-3)


def test_ordinal_log_prob_gradient_survives_saturation():
    """Regression: with raw sigmoid differences a saturated node (|shift| > ~17
    in float32) has exactly-zero gradients and can never recover. The log-space
    form must keep finite, non-zero gradients.
    """
    theta = torch.zeros(8, 1, requires_grad=True)  # binary node
    shift = torch.full((8,), -60.0, requires_grad=True)  # heavily saturated
    y = torch.ones(8)  # observed level has p ~ 0
    lp = ordinal_log_prob(theta, shift, y)
    assert torch.isfinite(lp).all()
    lp.sum().backward()
    assert torch.isfinite(theta.grad).all()
    assert torch.isfinite(shift.grad).all()
    assert theta.grad.abs().max() > 1e-3
    assert shift.grad.abs().max() > 1e-3


def test_ordinal_cutpoints_increasing_and_pmf_sums_to_one():
    theta = torch.randn(64, 6)  # 7 levels
    cuts = ordinal_cutpoints(theta)
    assert (cuts[:, 1:] >= cuts[:, :-1]).all()
    pmf = ordinal_pmf(theta, torch.randn(64))
    assert pmf.shape == (64, 7)
    assert (pmf >= 0).all()
    assert torch.allclose(pmf.sum(dim=1), torch.ones(64), atol=1e-5)


# %% dag -------------------------------------------------------------------------------


def test_topological_order():
    spec = {
        "C": OrdinalNode(3, [LS("A"), LS("B")]),
        "B": ContinuousNode([CS("A")]),
        "A": ContinuousNode(),
    }
    order = validate_and_sort(spec)
    assert order.index("A") < order.index("B") < order.index("C")


# %% flow logic ------------------------------------------------------------------------
def test_pmf_matches_do_sampling(fitted_flow):
    flow, _ = fitted_flow
    grid = pd.DataFrame({"X": [1.5]})
    pmf = flow.pmf(grid, node="Y")[0]
    s = flow.sample(40_000, do={"X": 1.5}, seed=11)
    emp = (
        s["Y"]
        .value_counts(normalize=True)
        .sort_index()
        .reindex(range(4), fill_value=0)
        .values
    )
    assert pmf.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.abs(pmf - emp).max() < 0.02


def test_abduction_roundtrip(fitted_flow):
    flow, df = fitted_flow
    sub = df.iloc[:300]
    u = flow.abduct(sub, seed=5)
    rec = flow.sample(u=u)
    assert np.abs(rec["X"].values - sub["X"].values).max() < 1e-3
    assert np.abs(rec["W"].values - sub["W"].values).max() < 1e-3
    assert (rec["Y"].values == sub["Y"].values).all()


def test_counterfactual_only_changes_descendants(fitted_flow):
    flow, df = fitted_flow
    sub = df.iloc[:300]
    u = flow.abduct(sub, seed=5)
    cf = flow.sample(do={"Y": 3}, u=u)
    # X is not a descendant of Y -> unchanged; Y is clamped; W may change
    assert np.abs(cf["X"].values - sub["X"].values).max() < 1e-3
    assert (cf["Y"].values == 3).all()


def test_log_prob_finite_and_decomposes(fitted_flow):
    flow, df = fitted_flow
    lp = flow.log_prob(df.iloc[:100])
    assert lp.shape == (100,)
    assert torch.isfinite(lp).all()
    per_node = flow.nll(df.iloc[:100])
    assert set(per_node) == {"X", "Y", "W"}
    assert float(-lp.mean()) == pytest.approx(sum(per_node.values()), abs=1e-4)


def test_save_load_roundtrip(tmp_path, fitted_flow):
    flow, df = fitted_flow
    p = tmp_path / "flow.pt"
    flow.save(p)
    flow2 = CausalFlowDAG.load(p)
    lp1 = flow.log_prob(df.iloc[:50])
    lp2 = flow2.log_prob(df.iloc[:50])
    assert torch.allclose(lp1, lp2, atol=1e-6)


# %% ls == ordered logit (MLE) ---------------------------------------------------------
def test_ls_node_equals_proportional_odds():
    """An all-ls ordinal node is exactly a proportional-odds model: the SGD fit
    must agree with the statsmodels MLE (same data) in coefficients and PMFs.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    rng = np.random.default_rng(42)
    n = 4000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    lat = 1.2 * x1 - 0.7 * x2 + rng.logistic(size=n)
    y = np.digitize(lat, [-1.0, 0.0, 1.5]).astype(float)
    df = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})

    spec = {
        "X1": ContinuousNode([I(transform="affine")]),
        "X2": ContinuousNode([I(transform="affine")]),
        "Y": OrdinalNode(4, [LS("X1"), LS("X2")]),
    }
    flow = CausalFlowDAG(spec)
    flow.fit(df, df, epochs=400, learning_rate=0.05, batch_size=1000, verbose=0, seed=1)

    res = OrderedModel(df["Y"].astype(int), df[["X1", "X2"]], distr="logit").fit(
        method="bfgs", disp=False
    )

    # our model: P(Y<=k) = sigmoid(theta_k - (w1 x1 + w2 x2)); statsmodels likewise
    w1 = float(flow.nodes["Y"].shifts["X1"].weight)
    w2 = float(flow.nodes["Y"].shifts["X2"].weight)
    assert w1 == pytest.approx(res.params["X1"], abs=0.05)
    assert w2 == pytest.approx(res.params["X2"], abs=0.05)

    # per-row PMFs agree
    pmf_flow = flow.pmf(df.iloc[:500], node="Y")
    pmf_sm = res.model.predict(
        res.params, exog=df[["X1", "X2"]].values[:500], which="prob"
    )
    assert np.abs(pmf_flow - pmf_sm).max() < 0.01


def test_bisection_warns_when_a_root_escapes_the_bracket():
    """A root beyond max_expand doublings is clipped and reported, not silent."""
    z = torch.tensor([0.5, 1e12])  # the second target is out of any reach
    lo, hi = torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])
    with pytest.warns(RuntimeWarning, match="1 latent value"):
        root = _expanding_bisection(lambda t: t, z, lo, hi, max_expand=5, iters=30)
    assert torch.isclose(root[0], torch.tensor(0.5), atol=1e-6)
    assert root[1] < 1e12  # clipped to the (expanded) bracket edge
