"""Tests for the magic-mrclean synthetic cohort and the flow's recovery of its
known ground truth, plus a regression check against the committed R reference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CausalFlowDAG, ContinuousNode, OrdinalNode, term
from tramdag.simulations import MagicMrClean

DATA = Path(__file__).resolve().parents[1] / "data" / "magic-mrclean"


def _spec(style: str) -> dict:
    if style == "ls":
        t = {"Age": "ls", "mRS_pre": "ls", "NIHSSa": "ls", "T": "ls"}
    else:
        t = {"Age": "ci", "mRS_pre": "ls", "NIHSSa": "cs", "T": "ls"}
    return {
        "Age": ContinuousNode(transform="bernstein"),
        "mRS_pre": OrdinalNode(levels=6, terms=[term(t["Age"], "Age")]),
        "NIHSSa": ContinuousNode(
            transform="bernstein",
            terms=[term(t["Age"], "Age"), term(t["mRS_pre"], "mRS_pre")],
        ),
        "T": OrdinalNode(
            levels=2,
            terms=[
                term(t["Age"], "Age"),
                term(t["mRS_pre"], "mRS_pre"),
                term(t["NIHSSa"], "NIHSSa"),
            ],
        ),
        "mRS_3m": OrdinalNode(
            levels=7,
            terms=[
                term(t["Age"], "Age"),
                term(t["mRS_pre"], "mRS_pre"),
                term(t["NIHSSa"], "NIHSSa"),
                term(t["T"], "T"),
            ],
        ),
    }


# ----------------------------------------------------------------- generator
def test_generator_schema_and_ranges():
    gen = MagicMrClean(variant="nl", seed=7)
    df = gen.observational(2000)
    assert list(df.columns) == ["Age", "mRS_pre", "NIHSSa", "T", "mRS_3m"]
    assert df["Age"].between(20, 103).all()
    assert df["NIHSSa"].between(6, 42).all()
    assert set(df["T"].unique()) <= {0, 1}
    assert df["mRS_pre"].between(0, 5).all()
    assert df["mRS_3m"].between(0, 6).all()


def test_generator_reproducible():
    a = MagicMrClean(variant="nl", seed=7).observational(500)
    b = MagicMrClean(variant="nl", seed=7).observational(500)
    assert a.equals(b)


def test_rct_breaks_confounding():
    """In the RCT draw, T is independent of its parents (Bernoulli 1/2)."""
    gen = MagicMrClean(variant="nl", seed=7)
    rct = gen.rct(20000)
    assert rct["T"].mean() == pytest.approx(0.5, abs=0.03)
    # T uncorrelated with Age under randomization
    assert abs(np.corrcoef(rct["T"], rct["Age"])[0, 1]) < 0.05


def test_true_ate_positive_and_confounded():
    truth_nl = MagicMrClean(variant="nl", seed=7).true_ate(100_000)
    # known true benefit is positive but much smaller than the naive contrast
    assert truth_nl["true_ate"] > 0.04
    assert truth_nl["naive_obs_diff"] > truth_nl["true_ate"] + 0.1


def test_counterfactual_pair_shares_latents():
    gen = MagicMrClean(variant="nl", seed=7)
    fact, cf = gen.counterfactual_pair(1000, do={"T": 1})
    # non-descendants of T are untouched; T is clamped
    assert (fact["Age"] == cf["Age"]).all()
    assert (fact["NIHSSa"] == cf["NIHSSa"]).all()
    assert (cf["T"] == 1).all()


# ---------------------------------------------------------- frozen CSVs exist
@pytest.mark.parametrize("variant", ["ls", "nl"])
def test_frozen_csvs_present_and_match_truth(variant):
    base = DATA / variant
    obs = pd.read_csv(base / "obs.csv")
    rct = pd.read_csv(base / "rct.csv")
    truth = json.loads((base / "truth.json").read_text())
    assert len(obs) == truth["n_obs"]
    assert len(rct) == truth["n_rct"]
    assert truth["true_ate"] > 0


def _fit_and_ate(style, obs, rct, epochs=(1500, 500), restore_best=None):
    # the flexible model overfits the observational confounding at the MLE, so it
    # needs early-stopping regularization to recover the causal effect; the
    # constrained all-ls model does not. Default per-style accordingly.
    if restore_best is None:
        restore_best = style != "ls"
    n = len(obs)
    tr, va = obs.iloc[: int(0.85 * n)], obs.iloc[int(0.85 * n) :]
    torch.manual_seed(0)
    flow = CausalFlowDAG(_spec(style))
    flow.fit(
        tr,
        va,
        epochs=epochs[0],
        learning_rate=1e-2,
        batch_size=256,
        verbose=0,
        seed=0,
        restore_best=restore_best,
    )
    flow.fit(
        tr,
        va,
        epochs=epochs[1],
        learning_rate=1e-3,
        batch_size=256,
        verbose=0,
        restore_best=restore_best,
    )
    p0 = flow.pmf(rct, node="mRS_3m", do={"T": 0})[:, :3].sum(axis=1)
    p1 = flow.pmf(rct, node="mRS_3m", do={"T": 1})[:, :3].sum(axis=1)
    return flow, float((p1 - p0).mean())


def _fit_full_mle(style, obs, epochs=((4000, 1e-2), (2000, 1e-3), (1000, 1e-4))):
    """All-data, no early stopping: the exact training-data MLE."""
    torch.manual_seed(0)
    flow = CausalFlowDAG(_spec(style))
    for ep, lr in epochs:
        flow.fit(
            obs,
            epochs=ep,
            learning_rate=lr,
            batch_size=256,
            verbose=0,
            seed=0 if lr == 1e-2 else None,
            restore_best=False,
        )
    return flow


def _load(variant):
    base = DATA / variant
    cols = ["Age", "mRS_pre", "NIHSSa", "T", "mRS_3m"]
    return (
        pd.read_csv(base / "obs.csv")[cols],
        pd.read_csv(base / "rct.csv")[cols],
        json.loads((base / "truth.json").read_text()),
    )


# ----------------------------------------------- flow recovers the true ATE
@pytest.mark.slow
@pytest.mark.parametrize(
    "variant,style,tol",
    [
        ("ls", "ls", 0.04),  # all-ls DGP, all-ls model: exact up to finite-sample
        ("nl", "flexible", 0.04),  # nl DGP needs the flexible model to recover truth
    ],
)
def test_flow_recovers_true_ate(variant, style, tol):
    obs, rct, truth = _load(variant)
    _, ate = _fit_and_ate(style, obs, rct)
    assert ate == pytest.approx(truth["true_ate"], abs=tol)


@pytest.mark.slow
def test_nl_storyline_all_ls_underestimates_flexible_recovers():
    """The headline simulation result: on the heterogeneous-effect `nl` cohort the
    all-`ls` model (which cannot extrapolate tau(Age) from the older observational
    cohort to the younger trial) undershoots the true ATE, while the flexible
    ci/cs flow recovers it. Both massively de-confound the naive contrast."""
    obs, rct, truth = _load("nl")
    _, ate_ls = _fit_and_ate("ls", obs, rct)  # constrained -> no early stop
    _, ate_flex = _fit_and_ate(
        "flexible", obs, rct
    )  # flexible -> early-stop regularized
    true = truth["true_ate"]
    assert ate_ls < true - 0.015  # all-ls biased low (misspecified)
    assert abs(ate_flex - true) < abs(ate_ls - true)  # flexible closer to truth
    assert ate_flex == pytest.approx(true, abs=0.04)  # ...and close in absolute terms
    assert ate_ls < truth["naive_obs_diff"] - 0.1  # both far below the confounded naive


# ------------------------------------------- spot-on MLE (no early stopping)
@pytest.mark.slow
def test_all_ls_flow_is_exact_mle():
    """With restore_best=False and full-data convergence, an all-`ls` flow IS the
    classical MLE: its outcome-node coefficients match statsmodels OrderedModel
    (and, where present, the R polr reference) to within SGD tolerance.

    This is *only* achievable because fit() no longer early-stops by default --
    best-validation restoration would otherwise pin the fit off the train optimum."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    obs, _, _ = _load("ls")
    torch.manual_seed(0)
    flow = CausalFlowDAG(_spec("ls"))
    for ep, lr in [(4000, 1e-2), (2000, 1e-3), (1000, 1e-4)]:
        flow.fit(
            obs,
            epochs=ep,
            learning_rate=lr,
            batch_size=256,
            verbose=0,
            seed=0 if lr == 1e-2 else None,
            restore_best=False,
        )

    X = pd.DataFrame({"Age": obs["Age"], "NIHSSa": obs["NIHSSa"], "T": obs["T"]})
    for k in range(1, 6):
        X[f"mRS_pre_{k}"] = (obs["mRS_pre"] == k).astype(float)
    res = OrderedModel(obs["mRS_3m"].astype(int), X, distr="logit").fit(
        method="bfgs", disp=False
    )

    y = flow.nodes["mRS_3m"]
    w_age = float(y.shifts["Age"].weight.detach())
    w_nih = float(y.shifts["NIHSSa"].weight.detach())
    w_t = y.shifts["T"].weight.detach().numpy().ravel()
    assert w_age == pytest.approx(res.params["Age"], abs=0.01)
    assert w_nih == pytest.approx(res.params["NIHSSa"], abs=0.01)
    assert (w_t[1] - w_t[0]) == pytest.approx(res.params["T"], abs=0.02)


@pytest.mark.slow
def test_restore_best_changes_the_fit():
    """Guard the new default: restore_best=True (early stopping on a held-out
    split) lands at a different point than the converged MLE-style fit."""
    obs, _, _ = _load("ls")
    tr, va = obs.iloc[:1000], obs.iloc[1000:]
    fits = {}
    for rb in (False, True):
        torch.manual_seed(0)
        f = CausalFlowDAG(_spec("ls"))
        f.fit(
            tr,
            va,
            epochs=800,
            learning_rate=1e-2,
            batch_size=256,
            verbose=0,
            seed=0,
            restore_best=rb,
        )
        fits[rb] = float(f.nodes["mRS_3m"].shifts["T"].weight.detach().ravel()[1])
    assert fits[False] != fits[True]


# ------------------------------------------------ regression vs R reference
@pytest.mark.slow
@pytest.mark.parametrize("variant", ["ls", "nl"])
def test_flow_matches_r_reference(variant):
    """The all-ls flow must agree with the committed classical R fit (fit_ls.R)
    on the outcome-node coefficients and the ATE. Skips if R outputs are absent."""
    ref = DATA / variant / "ref_ls"
    if not (ref / "ate.csv").exists():
        pytest.skip(f"R reference not generated yet (run Rscript fit_ls.R {variant})")

    # like-for-like: both are the full-data all-ls MLE
    obs, rct, _ = _load(variant)
    flow = _fit_full_mle("ls", obs)
    ate_r = pd.read_csv(ref / "ate.csv")["ate"].iloc[0]
    p0 = flow.pmf(rct, node="mRS_3m", do={"T": 0})[:, :3].sum(axis=1)
    p1 = flow.pmf(rct, node="mRS_3m", do={"T": 1})[:, :3].sum(axis=1)
    ate_flow = float((p1 - p0).mean())
    assert ate_flow == pytest.approx(ate_r, abs=0.02)

    # outcome-node continuous-parent coefficients (Age, NIHSSa, T) vs polr
    coefs = pd.read_csv(ref / "coefficients.csv")
    y = coefs[coefs["node"] == "mRS_3m"].set_index("term")["estimate"]
    w_age = float(flow.nodes["mRS_3m"].shifts["Age"].weight.detach())
    w_nih = float(flow.nodes["mRS_3m"].shifts["NIHSSa"].weight.detach())
    w_t = flow.nodes["mRS_3m"].shifts["T"].weight.detach().numpy().ravel()
    assert w_age == pytest.approx(y["Age"], abs=0.03)
    assert w_nih == pytest.approx(y["NIHSSa"], abs=0.03)
    assert (w_t[1] - w_t[0]) == pytest.approx(y["T"], abs=0.06)
