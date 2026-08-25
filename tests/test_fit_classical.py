"""Tests for CausalFlowDAG.fit_classical — deterministic float64 L-BFGS for
all-`ls` models.

Fast tests (guard, determinism, dtype round-trip, coefficient recovery) run
on PR CI; the statsmodels-equivalence and Adam-agreement tests are marked
`slow`. The DGP is the inline all-`ls` chain from conftest, whose outcome
node is a proportional-odds model by construction — which is what makes the
classical comparisons exact.
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CS, LS, CausalFlowDAG, ContinuousNode, I, OrdinalNode


# %% private functions -----------------------------------------------------------------
def _ls_spec() -> dict:
    """The in-class spec of the all-``ls`` chain: every edge a linear shift."""
    return {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([LS("x1")]),
        "t": OrdinalNode(2, [LS("x1"), LS("x2")]),
        "y": OrdinalNode(4, [LS("x1"), LS("x2"), LS("t")]),
    }


# %% public functions ------------------------------------------------------------------
# %% fast ------------------------------------------------------------------------------
def test_rejects_non_all_ls():
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode([CS("x1")])}
    flow = CausalFlowDAG(spec)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.standard_normal(50), "x2": rng.standard_normal(50)})
    with pytest.raises(ValueError, match="all-`ls`"):
        flow.fit_classical(df)


def test_rejects_ci_too():
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode([I("x1")])}
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.standard_normal(50), "x2": rng.standard_normal(50)})
    with pytest.raises(ValueError, match="requires an all-`ls` spec"):
        CausalFlowDAG(spec).fit_classical(df)


def test_same_seed_is_bit_identical(ls_chain):
    """No minibatching/shuffling -> deterministic given the same init."""
    obs = ls_chain["draw"](1500, 0)
    coefs = []
    for _ in range(2):
        torch.manual_seed(7)
        flow = CausalFlowDAG(_ls_spec())
        flow.fit_classical(obs, max_iter=100, verbose=False)
        coefs.append(flow.ls_coefficients()["y"]["t"].copy())
    np.testing.assert_array_equal(coefs[0], coefs[1])


def test_dtype_round_trip_and_usable(ls_chain):
    """Model is float32 before and after; usable for pmf/sample afterwards."""
    obs = ls_chain["draw"](1500, 1)
    torch.manual_seed(0)
    flow = CausalFlowDAG(_ls_spec())
    assert next(flow.parameters()).dtype == torch.float32
    rep = flow.fit_classical(obs, max_iter=75, verbose=False)
    assert next(flow.parameters()).dtype == torch.float32
    assert {"n_iter", "final_nll", "grad_norm", "coefficients", "seconds"} <= rep.keys()
    assert flow.pmf(obs.head(5), "y").shape == (5, 4)
    assert flow.sample(10, seed=0).shape == (10, 4)


def test_continuous_only_all_ls_recovers_the_true_shift(ls_chain):
    """An all-continuous all-ls spec fits and lands on the DGP coefficient."""
    df = ls_chain["draw"](4000, 2)[["x1", "x2"]]
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode([LS("x1")])}
    torch.manual_seed(0)
    flow = CausalFlowDAG(spec)
    rep = flow.fit_classical(df, max_iter=200, verbose=False)
    assert np.isfinite(rep["final_nll"])
    beta = float(flow.ls_coefficients()["x2"]["x1"][0])
    assert beta == pytest.approx(ls_chain["truth"]["beta_x2_x1"], abs=0.15)


def test_max_iter_and_history_size_reach_the_solver(ls_chain):
    """max_iter caps the L-BFGS iterations the report counts."""
    obs = ls_chain["draw"](800, 3)
    rep = CausalFlowDAG(_ls_spec(), seed=0).fit_classical(
        obs, max_iter=10, history_size=7, verbose=False
    )
    assert 0 < rep["n_iter"] <= 10
    assert rep["converged"] is False  # ten iterations cannot settle this fit
    with pytest.raises(TypeError):
        CausalFlowDAG(_ls_spec(), seed=0).fit_classical(obs, not_a_kwarg=1)


# %% slow ------------------------------------------------------------------------------
@pytest.mark.slow
def test_matches_statsmodels_mle(ls_chain):
    """fit_classical reaches the classical proportional-odds MLE.

    The outcome node of an all-``ls`` flow *is* an ordered-logit model, so
    this is an equality claim against independent software, not a
    tolerance-tuned similarity: every coefficient must match statsmodels'
    OrderedModel on the same design matrix.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    obs = ls_chain["draw"](4000, 4)
    torch.manual_seed(7)
    flow = CausalFlowDAG(_ls_spec())
    X = flow.design_matrix(obs, "y", drop_first=True)
    res = OrderedModel(obs["y"].astype(int), X, distr="logit").fit(
        method="bfgs", disp=False
    )
    flow.fit_classical(obs, verbose=False)
    coefs = flow.ls_coefficients()["y"]
    assert float(coefs["x1"][0]) == pytest.approx(res.params["x1"], abs=0.01)
    assert float(coefs["x2"][0]) == pytest.approx(res.params["x2"], abs=0.01)
    w_t = np.asarray(coefs["t"]).ravel()
    assert (w_t[1] - w_t[0]) == pytest.approx(res.params["t[1]"], abs=0.03)
    # and the classical MLE is near the DGP truth at this sample size
    tr = ls_chain["truth"]
    assert res.params["x1"] == pytest.approx(tr["beta_y_x1"], abs=0.15)
    assert res.params["t[1]"] == pytest.approx(tr["beta_y_t"], abs=0.20)


@pytest.mark.slow
def test_agrees_with_adam_mle(ls_chain):
    """Classical and (converged, no-early-stop) Adam reach the same optimum."""
    obs = ls_chain["draw"](2000, 5)
    torch.manual_seed(0)
    fa = CausalFlowDAG(_ls_spec())
    for ep, lr in [(3000, 1e-2), (1500, 1e-3)]:
        fa.fit(
            obs,
            epochs=ep,
            learning_rate=lr,
            batch_size=256,
            verbose=0,
            restore_best=False,
        )
    torch.manual_seed(0)
    fc = CausalFlowDAG(_ls_spec())
    fc.fit_classical(obs, verbose=False)
    for node, parent in [("y", "x1"), ("y", "x2"), ("t", "x1"), ("x2", "x1")]:
        a = float(fa.ls_coefficients()[node][parent][0])
        c = float(fc.ls_coefficients()[node][parent][0])
        assert a == pytest.approx(c, abs=0.03), f"{node}<-{parent}: {a} vs {c}"
