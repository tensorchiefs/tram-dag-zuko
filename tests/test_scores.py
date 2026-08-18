"""Tests for per-observation scores + the effect-modifier scan (issue #29).

Acceptance: (1) per-parameter score sums ~ 0 at the fitted MLE; (2) exact
finite-difference agreement (float64, any parameter point); (3) the end-to-end
use case — on an SCM with a (X2, X3)-modified treatment effect and an inert X1,
the CUSUM of the treatment-coefficient scores flags X2 and X3 and not X1.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CS, LS, VC, CausalFlowDAG, ContinuousNode, OrdinalNode

RNG = np.random.default_rng(11)


def _hetero_df(n: int, seed: int = 11) -> pd.DataFrame:
    """Randomized-treatment SCM with an (X2, X3)-modified effect, inert X1:
    Y = (u - (0.8 X1 + X2 - 0.5 X3) - (-1 + 0.8 X2 - 0.6 X3) T) / 2, u logistic.
    The all-`ls` outcome model is correctly specified except for the effect
    heterogeneity — exactly the scan's target.
    """
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(size=(3, n))
    t = (rng.uniform(size=n) < 0.5).astype(float)
    u = rng.logistic(size=n)
    y = (u - (0.8 * x1 + x2 - 0.5 * x3) - (-1.0 + 0.8 * x2 - 0.6 * x3) * t) / 2.0
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "T": t, "Y": y})


def _ls_spec() -> dict:
    return {
        "X1": ContinuousNode(transform="affine"),
        "X2": ContinuousNode(transform="affine"),
        "X3": ContinuousNode(transform="affine"),
        "T": OrdinalNode(levels=2),
        "Y": ContinuousNode(terms=[LS("X1"), LS("X2"), LS("X3"), LS("T")]),
    }


@pytest.fixture(scope="module")
def mle_flow():
    df = _hetero_df(4000)
    flow = CausalFlowDAG(_ls_spec(), seed=0)
    flow.fit_classical(df, verbose=False)
    return flow, df


# ---------------------------------------------------------- acceptance 1: sums
def test_score_sums_vanish_at_mle(mle_flow):
    """At the MLE the score sums are ~0 per column (the defining property)."""
    flow, df = mle_flow
    psi = flow.scores(df, node="Y")
    assert set(psi.columns) == {"X1", "X2", "X3", "T[0]", "T[1]"}
    assert psi.shape == (len(df), 5)
    mean_abs = psi.sum(axis=0).abs() / len(df)
    assert (mean_abs < 1e-4).all(), mean_abs.to_dict()


def test_score_sums_vanish_at_mle_ordinal_outcome():
    """Same property for an ordinal outcome node."""
    rng = np.random.default_rng(4)
    n = 3000
    x = rng.normal(size=n)
    lat = 1.1 * x + rng.logistic(size=n)
    y = np.digitize(lat, [-1.0, 0.8]).astype(float)
    df = pd.DataFrame({"X": x, "Y": y})
    spec = {
        "X": ContinuousNode(transform="affine"),
        "Y": OrdinalNode(levels=3, terms=[LS("X")]),
    }
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit_classical(df, verbose=False)
    psi = flow.scores(df, node="Y")
    assert (psi.sum(axis=0).abs() / n < 1e-4).all()


# ------------------------------------------------ acceptance 2: exact FD check
def test_scores_match_finite_differences():
    """Analytic scores equal float64 central differences at an arbitrary
    (non-MLE) parameter point, for LS on continuous and ordinal parents, on
    continuous and ordinal outcomes, and for VC's beta0.
    """
    df = _hetero_df(200, seed=5)
    spec = {
        "X1": ContinuousNode(transform="affine"),
        "X2": ContinuousNode(transform="affine"),
        "X3": ContinuousNode(transform="affine"),
        "T": OrdinalNode(levels=2, terms=[LS("X1")]),
        "Y": ContinuousNode(terms=[LS("X1"), CS("X3"), VC("T", "X2", "X3")]),
    }
    flow = CausalFlowDAG(spec, seed=1)
    flow.fit(df, epochs=5, verbose=0, seed=1)  # any point works; move off init
    flow.double()
    try:
        for node, targets in [
            (
                "Y",
                {
                    "X1": flow.nodes["Y"].shifts["X1"].fc.weight,
                    "T": flow.nodes["Y"].shifts["T"].beta0,
                },
            ),
            ("T", {"X1": flow.nodes["T"].shifts["X1"].fc.weight}),
        ]:
            psi = flow.scores(df, node=node)
            needed = list(flow.nodes[node].parents) + [node]
            np_vals = {
                c: torch.as_tensor(df[c].to_numpy(dtype=np.float64)) for c in needed
            }
            for col, param in targets.items():
                h = 1e-6
                flat = param.data.view(-1)
                idx = 0  # first element of the parameter
                with torch.no_grad():
                    flat[idx] += h
                    lp_plus = flow.node_log_prob(np_vals, nodes=[node])[node]
                    flat[idx] -= 2 * h
                    lp_minus = flow.node_log_prob(np_vals, nodes=[node])[node]
                    flat[idx] += h
                fd = ((lp_plus - lp_minus) / (2 * h)).numpy()
                np.testing.assert_allclose(
                    psi[col].to_numpy(), fd, atol=1e-6, err_msg=f"{node}/{col}"
                )
    finally:
        flow.float()


# --------------------------------------------- acceptance 3: end-to-end CUSUM
def test_effect_modifier_scan_flags_true_modifiers(mle_flow):
    """The point of the feature: on the heterogeneous SCM the scan must flag
    the true modifiers X2 and X3 and NOT the inert X1 (issue #29).
    """
    flow, df = mle_flow
    scan = flow.effect_modifier_scan(df, node="Y", on="T")
    assert set(scan.index) == {"X1", "X2", "X3"}, scan.to_string()  # default cands
    assert bool(scan.loc["X2", "flag"]) and bool(scan.loc["X3", "flag"]), (
        scan.to_string()
    )
    assert not bool(scan.loc["X1", "flag"]), scan.to_string()
    # the modifiers clearly dominate the ranking
    assert scan.loc[["X2", "X3"], "stat"].min() > 2 * scan.loc["X1", "stat"]


def test_scan_null_is_quiet():
    """With a homogeneous effect nothing should be flagged (size sanity)."""
    rng = np.random.default_rng(7)
    n = 4000
    x1, x2 = rng.normal(size=(2, n))
    t = (rng.uniform(size=n) < 0.5).astype(float)
    y = (rng.logistic(size=n) - (x1 - 0.5 * x2) + 0.9 * t) / 2.0
    df = pd.DataFrame({"X1": x1, "X2": x2, "T": t, "Y": y})
    spec = {
        "X1": ContinuousNode(transform="affine"),
        "X2": ContinuousNode(transform="affine"),
        "T": OrdinalNode(levels=2),
        "Y": ContinuousNode(terms=[LS("X1"), LS("X2"), LS("T")]),
    }
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit_classical(df, verbose=False)
    scan = flow.effect_modifier_scan(df, node="Y", on="T", candidates=["X1", "X2"])
    assert not scan["flag"].any(), scan.to_dict()


def test_scores_on_vc_model_and_scan_column_resolution():
    """A VC treatment resolves to its own score column (`on` itself), and the
    scan runs on a VC model too.
    """
    df = _hetero_df(1500, seed=8)
    spec = {
        **_ls_spec(),
        "Y": ContinuousNode(terms=[LS("X1"), LS("X2"), LS("X3"), VC("T", "X2")]),
    }
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(df, epochs=40, verbose=0, seed=0)
    psi = flow.scores(df, node="Y")
    assert "T" in psi.columns  # beta0 score, not one-hot columns
    scan = flow.effect_modifier_scan(df, node="Y", on="T", candidates=["X2"])
    assert list(scan.index) == ["X2"]


# ------------------------------------------------------------------ error paths
def test_scores_error_paths(mle_flow):
    flow, df = mle_flow
    with pytest.raises(KeyError, match="unknown node"):
        flow.scores(df, node="nope")
    with pytest.raises(ValueError, match="only option"):
        flow.scores(df, node="Y", params="all")
    with pytest.raises(KeyError, match="missing column"):
        flow.scores(df.drop(columns=["Y"]), node="Y")
    with pytest.raises(ValueError, match="no LS or VC"):
        flow.scores(df, node="T")  # T is a source: no shift terms
    with pytest.raises(KeyError, match="no score column"):
        flow.effect_modifier_scan(df, node="Y", on="X9")
