"""Tests for the term-formula notation (I/LS/CS) — construction, validation,
serialization, and the meta-adjacency view.
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
import torch

from tramdag import CS, LS, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.spec import spec_from_dict, spec_to_dict


# %% private functions -----------------------------------------------------------------
def _toy_df(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = -x1 + rng.normal(size=n)
    x3 = x1 + 0.25 * x2 + rng.normal(size=n)
    y = rng.integers(0, 4, size=n).astype(float)
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "Y": y})


def _terms_spec():
    return {
        "X1": ContinuousNode(),
        "X2": ContinuousNode([LS("X1")]),
        "X3": ContinuousNode([I("X1"), CS("X2")]),
        "Y": OrdinalNode(4, [LS("X3")]),
    }


# %% public functions ------------------------------------------------------------------
def test_terms_spec_builds_and_scores():
    """A term spec builds a flow whose per-node log-likelihood is finite."""
    flow = CausalFlowDAG(_terms_spec(), seed=0)
    per_node = flow.node_log_prob(flow._tensorize(_toy_df()))
    assert set(per_node) == {"X1", "X2", "X3", "Y"}
    for v in per_node.values():
        assert torch.isfinite(v).all()


def test_node_internal_structure_matches():
    flow = CausalFlowDAG(_terms_spec(), seed=0)
    assert flow.nodes["X3"].intercept.ci_parents == ["X1"]  # I("X1") -> ci-parent
    assert set(flow.nodes["X3"].shifts) == {"X2"}  # CS("X2") -> shift
    assert flow.nodes["X3"].parents == ("X1", "X2")  # ordered parent names


def test_ls_requires_exactly_one_parent():
    with pytest.raises(ValueError, match=r"LS\(\) takes exactly one parent"):
        LS("X1", "X2")


@pytest.mark.parametrize(
    ("term", "n_shift", "n_ci"), [(CS("X1", "X2"), 1, 0), (I("X1", "X2"), 0, 2)]
)
def test_joint_terms_build(term, n_shift, n_ci):
    """Joint (multi-parent) terms build one network over the parent group."""
    spec = {
        "X1": ContinuousNode(),
        "X2": ContinuousNode(),
        "X3": ContinuousNode([term]),
    }
    node = CausalFlowDAG(spec, seed=0).nodes["X3"]
    assert len(node.shifts) == n_shift  # joint CS -> a single shift module
    assert len(node.intercept.ci_parents) == n_ci  # joint I -> both parents pooled


def test_duplicate_parent_across_terms_raises():
    spec = {"X1": ContinuousNode(), "X3": ContinuousNode([LS("X1"), CS("X1")])}
    with pytest.raises(ValueError, match="more than one term"):
        CausalFlowDAG(spec)


def test_cycle_detected():
    spec = {"A": ContinuousNode([LS("B")]), "B": ContinuousNode([LS("A")])}
    with pytest.raises(ValueError, match="cycle"):
        CausalFlowDAG(spec)


def test_serialization_roundtrip_terms():
    flow = CausalFlowDAG(_terms_spec(), seed=0)
    spec2 = spec_from_dict(spec_to_dict(flow.spec))
    flow2 = CausalFlowDAG(spec2, seed=0)
    df = _toy_df()
    a = flow.node_log_prob(flow._tensorize(df))
    b = flow2.node_log_prob(flow2._tensorize(df))
    for k in a:
        assert torch.allclose(a[k], b[k]), k


def test_to_matrix_labels_every_effect_and_leaves_non_edges_empty():
    """The paper's meta-adjacency view: rows are parents, columns children."""
    spec = {
        "a": ContinuousNode(),
        "b": ContinuousNode(),
        "y": OrdinalNode(3, [I("a"), CS("b")]),  # one edge-owning term each
    }
    m = CausalFlowDAG(spec, seed=0).to_matrix()
    assert list(m.index) == list(m.columns)  # square, node-ordered
    assert m.loc["a", "y"] == "CI"  # an I term reads as CI
    assert m.loc["b", "y"] == "CS"
    assert m.loc["y", "a"] == ""  # no edge -> empty, not NaN
    assert m.loc["a", "b"] == ""

    joint = CausalFlowDAG(
        {**spec, "y": OrdinalNode(3, [CS("a", "b")])}, seed=0
    ).to_matrix()
    assert joint.loc["a", "y"] == "CS['a', 'b']"  # a joint term names its group


def test_ls_coefficients_shape_and_agreement_with_the_modules():
    """ls_coefficients is the public spelling of the LS weights.

    An ordinal parent contributes one weight per one-hot level, and only
    differences against level 0 are identified — so the raw vector has
    ``levels`` entries. Nodes without shift terms do not appear.
    """
    spec = {
        "x": ContinuousNode(),
        "t": OrdinalNode(3),
        "y": ContinuousNode([LS("x"), LS("t")]),
    }
    flow = CausalFlowDAG(spec, seed=0)
    coefs = flow.ls_coefficients()

    assert set(coefs) == {"y"}  # sources have no shift terms
    assert set(coefs["y"]) == {"x", "t"}
    assert coefs["y"]["x"].shape == (1,)
    assert coefs["y"]["t"].shape == (3,)  # one per ordinal level


def test_ordinal_node_refuses_a_basis():
    """An ordinal intercept is the cutpoint vector; I(transform=) has no meaning."""
    with pytest.raises(ValueError, match="cutpoint vector"):
        OrdinalNode(3, [I(transform="spline")])
    with pytest.raises(ValueError, match="cutpoint vector"):
        OrdinalNode(3, [I(n_coeffs=5)])
