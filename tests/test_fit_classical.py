"""Tests for CausalFlowDAG.fit_classical — deterministic float64 L-BFGS for
all-`ls` models.

Fast tests (guard, determinism, dtype round-trip) run on PR CI; the
statsmodels-equivalence and Adam-agreement tests are marked `slow`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CS, LS, CausalFlowDAG, ContinuousNode, I, OrdinalNode

DATA = Path(__file__).resolve().parents[1] / "data"


def _stroke_ls_spec() -> dict:
    return {
        "Age": ContinuousNode(),
        "mRS_pre": OrdinalNode(6, [LS("Age")]),
        "NIHSSa": ContinuousNode([LS("Age"), LS("mRS_pre")]),
        "T": OrdinalNode(2, [LS("Age"), LS("mRS_pre"), LS("NIHSSa")]),
        "mRS_3m": OrdinalNode(7, [LS("Age"), LS("mRS_pre"), LS("NIHSSa"), LS("T")]),
    }


def _obs() -> pd.DataFrame:
    return pd.read_csv(DATA / "magic-mrclean" / "ls" / "obs.csv")


# ------------------------------------------------------------------ fast
def test_rejects_non_all_ls():
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode([CS("x1")])}
    flow = CausalFlowDAG(spec)
    df = pd.DataFrame({"x1": np.random.randn(50), "x2": np.random.randn(50)})
    with pytest.raises(ValueError, match="all-`ls`"):
        flow.fit_classical(df)


def test_rejects_ci_too():
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode([I("x1")])}
    with pytest.raises(ValueError):
        CausalFlowDAG(spec).fit_classical(
            pd.DataFrame({"x1": np.random.randn(50), "x2": np.random.randn(50)})
        )


def test_same_seed_is_bit_identical():
    """No minibatching/shuffling -> deterministic given the same init."""
    obs = _obs()
    coefs = []
    for _ in range(2):
        torch.manual_seed(7)
        flow = CausalFlowDAG(_stroke_ls_spec())
        flow.fit_classical(obs, max_iter=100, verbose=False)
        coefs.append(flow.ls_coefficients()["mRS_3m"]["T"].copy())
    np.testing.assert_array_equal(coefs[0], coefs[1])


def test_dtype_round_trip_and_usable():
    """Model is float32 before and after; usable for pmf/sample afterwards."""
    obs = _obs()
    torch.manual_seed(0)
    flow = CausalFlowDAG(_stroke_ls_spec())
    assert next(flow.parameters()).dtype == torch.float32
    rep = flow.fit_classical(obs, max_iter=75, verbose=False)
    assert next(flow.parameters()).dtype == torch.float32
    # report shape
    assert {"n_iter", "final_nll", "grad_norm", "coefficients", "seconds"} <= rep.keys()
    # still usable in float32
    assert flow.pmf(obs.head(5), "mRS_3m").shape == (5, 7)
    assert flow.sample(10, seed=0).shape == (10, 5)


def test_continuous_only_all_ls_runs():
    """All-continuous all-ls spec (vaca-style) is accepted and fits."""
    df = pd.read_csv(DATA / "vaca" / "obs.csv")
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([LS("x1")]),
        "x3": ContinuousNode([LS("x1"), LS("x2")]),
    }
    torch.manual_seed(0)
    flow = CausalFlowDAG(spec)
    rep = flow.fit_classical(df, max_iter=100, verbose=False)
    assert np.isfinite(rep["final_nll"])
    # x2 = -x1 + noise is a strong dependence -> large |shift| on the latent
    # scale (sign is on the log-odds scale, not the conditional-mean slope)
    assert abs(flow.ls_coefficients()["x2"]["x1"][0]) > 1.0


# ------------------------------------------------------------------ slow
@pytest.mark.slow
def test_matches_statsmodels_mle():
    """fit_classical reaches the classical proportional-odds MLE: well-identified
    outcome coefficients match statsmodels OrderedModel; the weakly-identified T
    matches within the same band used by test_flow_matches_r_reference.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    obs = _obs()
    X = pd.DataFrame(index=obs.index)
    X["Age"] = obs["Age"].values
    for k in range(6):
        X[f"mRS_pre_{k}"] = (obs["mRS_pre"].values == k).astype(float)
    X["NIHSSa"] = obs["NIHSSa"].values
    X["T"] = obs["T"].values
    X = X.drop(columns=["mRS_pre_0"])
    res = OrderedModel(obs["mRS_3m"].astype(int), X, distr="logit").fit(
        method="bfgs", disp=False
    )

    torch.manual_seed(7)
    flow = CausalFlowDAG(_stroke_ls_spec())
    flow.fit_classical(obs, verbose=False)
    n = flow.nodes["mRS_3m"]
    w_age = float(n.shifts["Age"].weight.detach())
    w_nih = float(n.shifts["NIHSSa"].weight.detach())
    w_t = n.shifts["T"].weight.detach().numpy().ravel()
    assert w_age == pytest.approx(res.params["Age"], abs=0.01)
    assert w_nih == pytest.approx(res.params["NIHSSa"], abs=0.01)
    assert (w_t[1] - w_t[0]) == pytest.approx(res.params["T"], abs=0.06)


@pytest.mark.slow
def test_agrees_with_adam_mle():
    """Classical and (converged, no-early-stop) Adam reach the same optimum."""
    obs = _obs()
    torch.manual_seed(0)
    fa = CausalFlowDAG(_stroke_ls_spec())
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
    fc = CausalFlowDAG(_stroke_ls_spec())
    fc.fit_classical(obs, verbose=False)
    for name, parent in [("mRS_3m", "Age"), ("mRS_3m", "NIHSSa"), ("NIHSSa", "Age")]:
        a = float(fa.nodes[name].shifts[parent].weight.detach())
        c = float(fc.nodes[name].shifts[parent].weight.detach())
        assert a == pytest.approx(c, abs=0.02), f"{name}<-{parent}: {a} vs {c}"
