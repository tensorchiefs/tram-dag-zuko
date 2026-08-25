"""Tests for issue #20 Option A: post-hoc mean-centered per-term decomposition
of an additive complex intercept (``flow.intercept_contributions``).

The decomposition must be (a) exact — baseline + summed contributions reproduce
the model's transform parameters; (b) sum-to-zero — each term's contribution has
zero column-mean over the centering data; and (c) purely post-hoc — it changes
nothing about the model or its outputs.
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import LS, CausalFlowDAG, ContinuousNode, I, OrdinalNode


# %% private functions -----------------------------------------------------------------
def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame(
        {"x1": x1, "x2": x2, "x3": x1 * 0.5 + x2 * 0.3 + rng.normal(size=n)}
    )


def _additive_ci_flow():
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(),
        "x3": ContinuousNode([I("x1", "x2", allow_interaction=False)]),
    }  # additive CI
    return CausalFlowDAG(spec, seed=1)


# %% public functions ------------------------------------------------------------------
# ------------------------------------------------ exactness of the decomposition
def test_baseline_plus_contributions_reproduces_theta():
    flow = _additive_ci_flow()
    df = _data()
    flow._set_ranges(df)  # not needed for theta, but exercises the realistic path

    res = flow.intercept_contributions("x3", df)
    # reconstruct theta from the centered components + baseline
    recon = res["baseline"][None, :] + sum(res["contributions"].values())

    # ground-truth theta straight from the model's intercept nets
    nd = flow.nodes["x3"]
    feats = flow._features(flow._tensorize(df))
    with torch.no_grad():
        theta = sum(
            net(torch.cat([feats[p] for p in grp], dim=1))
            for net, grp in zip(nd.intercept_nets, nd._intercept_groups, strict=True)
        )
    np.testing.assert_allclose(recon, theta.numpy(), rtol=1e-5, atol=1e-5)


# ------------------------------------------------------ sum-to-zero per parameter
def test_contributions_are_mean_centered():
    flow = _additive_ci_flow()
    df = _data()
    res = flow.intercept_contributions("x3", df)
    assert set(res["contributions"]) == {"x1", "x2"}
    for contrib in res["contributions"].values():
        # each column (transform parameter) averages to ~0 over the rows
        np.testing.assert_allclose(contrib.mean(axis=0), 0.0, atol=1e-6)


def test_shapes_and_parents():
    flow = _additive_ci_flow()
    df = _data(n=123)
    res = flow.intercept_contributions("x3", df)
    P = flow.nodes["x3"].ut.n_params
    assert res["baseline"].shape == (P,)
    for contrib in res["contributions"].values():
        assert contrib.shape == (123, P)
    assert res["parents"] == {"x1": ("x1",), "x2": ("x2",)}


# ------------------------------------------------------------------ ordinal node
def test_ordinal_additive_ci():
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(),
        "y": OrdinalNode(4, [I("x1", "x2", allow_interaction=False)]),
    }
    flow = CausalFlowDAG(spec, seed=2)
    df = _data()
    df["y"] = np.random.default_rng(3).integers(0, 4, len(df)).astype(float)
    res = flow.intercept_contributions("y", df)
    assert res["baseline"].shape == (3,)  # levels - 1 cutpoint params
    for contrib in res["contributions"].values():
        np.testing.assert_allclose(contrib.mean(axis=0), 0.0, atol=1e-6)


# --------------------------------------- transform-agnostic (spline / affine too)
@pytest.mark.parametrize(
    ("transform", "P"), [("bernstein", None), ("spline", 3 * 8 - 1), ("affine", 2)]
)
def test_works_for_any_continuous_transform(transform, P):
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(),
        "x3": ContinuousNode(
            [I("x1", "x2", allow_interaction=False, transform=transform)]
        ),
    }
    flow = CausalFlowDAG(spec, seed=5)
    df = _data()
    res = flow.intercept_contributions("x3", df)
    if P is not None:
        assert res["baseline"].shape == (P,)
    # exactness + sum-to-zero hold in unconstrained theta-space regardless of transform
    recon = res["baseline"][None, :] + sum(res["contributions"].values())
    nd = flow.nodes["x3"]
    feats = flow._features(flow._tensorize(df))
    with torch.no_grad():
        theta = sum(
            net(torch.cat([feats[p] for p in grp], dim=1))
            for net, grp in zip(nd.intercept_nets, nd._intercept_groups, strict=True)
        )
    np.testing.assert_allclose(recon, theta.numpy(), rtol=1e-5, atol=1e-5)
    for contrib in res["contributions"].values():
        np.testing.assert_allclose(contrib.mean(axis=0), 0.0, atol=1e-6)


# -------------------------------------------------------------------- guard rails
def test_raises_on_node_without_complex_intercept():
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([LS("x1")]),
    }  # shift only, no I-term
    flow = CausalFlowDAG(spec, seed=0)
    df = _data()
    with pytest.raises(ValueError, match="no complex-intercept"):
        flow.intercept_contributions("x2", df)
    # source node (unconditional simple intercept) also has nothing to decompose
    with pytest.raises(ValueError, match="no complex-intercept"):
        flow.intercept_contributions("x1", df)


def test_raises_on_unknown_node():
    flow = _additive_ci_flow()
    with pytest.raises(KeyError):
        flow.intercept_contributions("nope", _data())


def test_raises_on_missing_parent_column():
    flow = _additive_ci_flow()
    df = _data().drop(columns=["x2"])
    with pytest.raises(KeyError, match="missing intercept-parent"):
        flow.intercept_contributions("x3", df)


# ------------------------------------------------------- single joint complex CI
def test_joint_complex_intercept_single_component():
    # I("x1","x2") is one pooled network -> one centered component over both
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(),
        "x3": ContinuousNode([I("x1", "x2")]),
    }
    flow = CausalFlowDAG(spec, seed=4)
    df = _data()
    res = flow.intercept_contributions("x3", df)
    assert set(res["contributions"]) == {"x1+x2"}
    assert res["parents"] == {"x1+x2": ("x1", "x2")}
    recon = res["baseline"][None, :] + res["contributions"]["x1+x2"]
    nd = flow.nodes["x3"]
    feats = flow._features(flow._tensorize(df))
    with torch.no_grad():
        theta = nd.intercept(torch.cat([feats["x1"], feats["x2"]], dim=1))
    np.testing.assert_allclose(recon, theta.numpy(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------- purely post-hoc / pure
def test_does_not_mutate_model_outputs():
    flow = _additive_ci_flow()
    df = _data()
    before = flow.log_prob(df).detach().numpy().copy()
    params_before = [p.detach().clone() for p in flow.parameters()]
    flow.intercept_contributions("x3", df)
    after = flow.log_prob(df).detach().numpy()
    np.testing.assert_array_equal(before, after)
    for a, b in zip(params_before, flow.parameters(), strict=True):
        assert torch.equal(a, b)
