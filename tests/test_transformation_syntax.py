"""The transformation syntax.

Every spelling must produce the same normalized term list — and, at a
fixed seed, the bit-identical model. A node takes at most one intercept
term with parents, and a VC term names its treatment by keyword.
"""

import pytest
import torch

from tramdag import CS, LS, VC, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.spec import spec_from_dict, spec_to_dict, validate_and_sort

# ----------------------------------------------------------- normalization


def test_sum_list_and_mixed_forms_are_identical():
    reference = ContinuousNode([I("x1"), LS("x2")])
    assert ContinuousNode(I("x1") + LS("x2")) == reference
    assert ContinuousNode([I("x1") + LS("x2")]) == reference
    assert ContinuousNode(transformation=[I("x1"), LS("x2")]) == reference


def test_sum_chains_flatten_in_order():
    node = ContinuousNode(I("a") + CS("b") + LS("c") + VC("b", t="t"))
    assert [t.effect for t in node.transformation] == ["I", "CS", "LS", "VC"]


def test_bare_i_and_single_term():
    assert ContinuousNode([I]).transformation == [I()]
    assert ContinuousNode(I).transformation == [I()]
    assert ContinuousNode(LS("x1")).transformation == [LS("x1")]


def test_ordinal_takes_transformation_positionally():
    node = OrdinalNode(4, I("x1") + LS("x2"))
    assert node.levels == 4
    assert node.transformation == [I("x1"), LS("x2")]


def test_junk_entries_are_rejected():
    with pytest.raises(TypeError, match="must be a Term"):
        ContinuousNode("LS(x)")
    with pytest.raises(TypeError, match="entries must be terms"):
        ContinuousNode([I("a"), "x"])


# ------------------------------------------------------- allow_interaction


def test_several_parented_i_terms_are_rejected():
    with pytest.raises(ValueError, match="allow_interaction=False"):
        ContinuousNode(I("a") + I("b"))
    with pytest.raises(ValueError, match="allow_interaction=False"):
        OrdinalNode(3, [I("a"), CS("c"), I("b")])


def test_multi_parent_i_stays_joint_by_default():
    assert ContinuousNode([I("a", "b")]).transformation == [I("a", "b")]


# ---------------------------------------------------------- transform on I


def test_i_transform_hoists_to_the_node():
    node = ContinuousNode([I("x1", transform="spline", transform_kwargs={"bins": 6})])
    assert node.transform == "spline"
    assert node.transform_kwargs == {"bins": 6}


def test_bare_i_can_carry_the_source_basis():
    assert ContinuousNode([I(transform="affine")]).transform == "affine"


def test_two_i_transforms_conflict():
    # a bare carrier plus a parented I is legal -- two bases are not
    with pytest.raises(ValueError, match="only one I term"):
        ContinuousNode(I(transform="spline") + I("a", transform="affine"))


def test_ordinal_rejects_i_transform():
    with pytest.raises(ValueError, match="ordinal"):
        OrdinalNode(3, [I("a", transform="spline")])


# ------------------------------------------------------- signature guards


def test_vc_treatment_is_keyword_only():
    with pytest.raises(TypeError):
        VC("T", "X2")  # the treatment is the keyword t=


# --------------------------------------------------- model-level identity


def _state_dicts_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_list_and_sum_build_the_identical_model():
    def build(spec):
        torch.manual_seed(7)
        return CausalFlowDAG(spec).state_dict()

    as_list = build(
        {
            "x1": ContinuousNode(),
            "x2": ContinuousNode([CS("x1")]),
            "y": OrdinalNode(3, [I("x1"), LS("x2")]),
        }
    )
    as_sum = build(
        {
            "x1": ContinuousNode(),
            "x2": ContinuousNode(CS("x1")),
            "y": OrdinalNode(3, I("x1") + LS("x2")),
        }
    )
    assert _state_dicts_equal(as_list, as_sum)


def test_additive_flag_builds_one_net_per_parent():
    spec = {
        "a": ContinuousNode(),
        "b": ContinuousNode(),
        "y": ContinuousNode([I("a", "b", allow_interaction=False)]),
    }
    flow = CausalFlowDAG(spec)
    node = flow.nodes["y"]
    assert node.intercept is None
    assert len(node.intercept_nets) == 2
    assert node._intercept_groups == [("a",), ("b",)]

    joint = CausalFlowDAG({**spec, "y": ContinuousNode([I("a", "b")])})
    assert joint.nodes["y"].intercept_nets is None
    assert joint.nodes["y"]._intercept_groups == [("a", "b")]


# ------------------------------------------------------------ persistence


def test_roundtrip_keeps_hoisted_transform_and_terms():
    spec = {
        "x1": ContinuousNode([I(transform="spline")]),
        "x2": ContinuousNode(I("x1") + CS("x1")),
    }
    with pytest.raises(ValueError, match="more than one"):
        validate_and_sort(spec)
    spec["x2"] = ContinuousNode(I("x1"))
    back = spec_from_dict(spec_to_dict(spec))
    assert back["x1"].transform == "spline"
    assert back["x2"].transformation == [I("x1")]


def test_saved_checkpoint_of_new_syntax_loads(tmp_path):
    spec = {"x": ContinuousNode(), "y": ContinuousNode(I("x") + LS("x"))}
    with pytest.raises(ValueError, match="more than one"):
        validate_and_sort(spec)
    spec["y"] = ContinuousNode(I("x"))
    flow = CausalFlowDAG(spec, seed=1)
    flow.save(tmp_path / "m.pt")
    loaded = CausalFlowDAG.load(tmp_path / "m.pt")
    assert loaded.spec["y"].transformation == [I("x")]


# ------------------------------------------------------------------ units


def test_units_reach_the_networks():
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(CS("x1", units=[16])),
        "y": ContinuousNode(I("x2", units=[4, 4]) + VC("x2", t="x1", units=[8])),
    }
    flow = CausalFlowDAG(spec)
    assert flow.nodes["x2"].shifts["x1"].net[0].out_features == 16
    assert flow.nodes["y"].intercept.net[0].out_features == 4
    assert flow.nodes["y"].shifts["x1"].net[0].out_features == 8


def test_units_survive_the_roundtrip():
    spec = {
        "a": ContinuousNode(),
        "b": ContinuousNode(CS("a", units=[16])),
    }
    back = spec_from_dict(spec_to_dict(spec))
    assert back["b"].transformation[0].units == (16,)


def test_vc_modifiers_are_positional_t_is_keyword():
    t = VC("X2", "X3", t="T")
    assert t.parents == ("T", "X2", "X3")  # internal layout: treatment first
    with pytest.raises(ValueError, match="cannot be both"):
        VC("T", t="T")
