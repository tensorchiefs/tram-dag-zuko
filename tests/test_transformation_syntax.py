"""The transformation syntax.

Every spelling must produce the same normalized term list — and, at a
fixed seed, the bit-identical model. A node takes at most one intercept
term with parents, and a VC term names its treatment by keyword.
"""

# %% imports ---------------------------------------------------------------------------
import json

import pytest
import torch

from tramdag import CI, CS, LS, SI, VC, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.spec import spec_from_dict, spec_to_dict, validate_and_sort


# %% private functions -----------------------------------------------------------------
def _state_dicts_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


# %% public functions ------------------------------------------------------------------


def test_sum_list_and_mixed_forms_are_identical():
    reference = ContinuousNode([I("x1"), LS("x2")])
    assert ContinuousNode(I("x1") + LS("x2")) == reference
    assert ContinuousNode(terms=[I("x1"), LS("x2")]) == reference


def test_sum_chains_flatten_in_order():
    node = ContinuousNode(I("a") + CS("b") + LS("c") + VC("b", t="t"))
    assert [t.effect for t in node.terms] == ["I", "CS", "LS", "VC"]


def test_bare_i_and_single_term():
    assert ContinuousNode([I]).terms == [I()]
    assert ContinuousNode(I).terms == [I()]
    # a shifts-only formula gets the implicit simple intercept, first
    assert ContinuousNode(LS("x1")).terms == [SI(), LS("x1")]
    assert ContinuousNode(LS("x1")) == ContinuousNode([SI(), LS("x1")])


def test_exactly_one_intercept_and_first():
    with pytest.raises(ValueError, match="exactly one intercept"):
        ContinuousNode([SI(), CI("a")])
    with pytest.raises(ValueError, match="comes first"):
        ContinuousNode(LS("a") + CI("b"))


def test_ordinal_takes_transformation_positionally():
    node = OrdinalNode(4, I("x1") + LS("x2"))
    assert node.levels == 4
    assert node.terms == [I("x1"), LS("x2")]


def test_junk_entries_are_rejected():
    for bad in ("LS(x)", [I("a"), "x"]):
        with pytest.raises(TypeError, match="built from terms"):
            ContinuousNode(bad)


def test_a_sum_nested_in_a_list_is_rejected():
    """A ``+`` sum is already flat, so a list of lists is a mistake.

    Both spellings on their own are fine; mixing them is what the error
    names, since it usually means someone expected ``+`` to combine list
    entries.
    """
    with pytest.raises(TypeError, match="do not nest"):
        ContinuousNode([I("x1") + LS("x2")])


def test_several_parented_i_terms_are_rejected():
    with pytest.raises(ValueError, match="allow_interaction=False"):
        ContinuousNode(I("a") + I("b"))
    with pytest.raises(ValueError, match="allow_interaction=False"):
        OrdinalNode(3, [I("a"), CS("c"), I("b")])


def test_multi_parent_i_stays_joint_by_default():
    assert ContinuousNode([I("a", "b")]).terms == [I("a", "b")]


def test_i_transform_hoists_to_the_node():
    node = ContinuousNode([I("x1", transform="spline", bins=6)])
    assert node.transform == "spline"
    assert node.transform_kwargs == {"bins": 6}


def test_i_dispatches_to_si_and_ci():
    """I is the fallback: no parents -> SI, parents -> CI."""
    assert I() == SI()
    assert I(transform="spline", bins=6) == SI(transform="spline", bins=6)
    assert I("a", "b", units=[4]) == CI("a", "b", units=[4])
    with pytest.raises(ValueError, match="at least one parent"):
        CI()


def test_bare_i_can_carry_the_source_basis():
    assert ContinuousNode([I(transform="affine")]).transform == "affine"


def test_two_i_transforms_conflict():
    # two intercepts is the error now, whatever they carry
    with pytest.raises(ValueError, match="exactly one intercept"):
        ContinuousNode(I(transform="spline") + I("a", transform="affine"))


def test_ordinal_rejects_i_transform():
    with pytest.raises(ValueError, match="ordinal"):
        OrdinalNode(3, [I("a", transform="spline")])


def test_vc_treatment_is_keyword_only():
    with pytest.raises(TypeError):
        VC("T", "X2")  # the treatment is the keyword t=


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
    assert len(node.intercept.nets) == 2
    assert node._intercept_groups == [("a",), ("b",)]
    assert node.ci_parents == ["a", "b"]  # flat, for introspection

    joint = CausalFlowDAG({**spec, "y": ContinuousNode([I("a", "b")])})
    assert not hasattr(joint.nodes["y"].intercept, "nets")  # one joint net
    assert joint.nodes["y"]._intercept_groups == [("a", "b")]
    assert joint.nodes["y"].ci_parents == ["a", "b"]

    single = CausalFlowDAG({**spec, "y": ContinuousNode([I("a")])})
    assert not hasattr(single.nodes["y"].intercept, "nets")  # one single net
    assert single.nodes["y"].ci_parents == ["a"]


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
    assert back["x2"].terms == [I("x1")]


def test_saved_checkpoint_of_new_syntax_loads(tmp_path):
    spec = {"x": ContinuousNode(), "y": ContinuousNode(I("x") + LS("x"))}
    with pytest.raises(ValueError, match="more than one"):
        validate_and_sort(spec)
    spec["y"] = ContinuousNode(I("x"))
    flow = CausalFlowDAG(spec, seed=1)
    flow.save(tmp_path / "m.pt")
    loaded = CausalFlowDAG.load(tmp_path / "m.pt")
    assert loaded.spec["y"].terms == [I("x")]


def test_every_option_survives_the_roundtrip():
    """One spec exercising every Term option, through the serializer."""
    spec = {
        "a": ContinuousNode([SI(transform="spline", bins=6)]),
        "b": ContinuousNode(),
        "t": OrdinalNode(2, [LS("a")]),
        "y": ContinuousNode(
            [
                I("a", "b", allow_interaction=False, units=[4, 4]),
                VC("b", t="t", penalty=2.5, center=True),
            ]
        ),
    }
    back = spec_from_dict(spec_to_dict(spec))
    assert back == spec
    i_term, vc_term = back["y"].terms
    assert (i_term.allow_interaction, i_term.units) == (False, (4, 4))
    assert (vc_term.penalty, vc_term.center) == (2.5, True)
    assert back["a"].transform_kwargs == {"bins": 6}


def test_malformed_serialized_spec_is_rejected():
    """spec_from_dict builds Terms directly, so validation is the only guard.

    A hand-edited or corrupted checkpoint can carry a term the
    constructors would have refused; validate_and_sort has to catch it.
    """

    def spec_with(term_dict):
        return {
            "a": {"kind": "continuous", "terms": []},
            "b": {"kind": "continuous", "terms": []},
            "y": {"kind": "continuous", "terms": [term_dict]},
        }

    two_parent_ls = {"effect": "LS", "parents": ["a", "b"], "options": {}}
    with pytest.raises(ValueError, match="exactly one parent"):
        validate_and_sort(spec_from_dict(spec_with(two_parent_ls)))

    unknown = {"effect": "XX", "parents": ["a"], "options": {}}
    with pytest.raises(ValueError, match="unknown term effect"):
        validate_and_sort(spec_from_dict(spec_with(unknown)))

    unknown_parent = {"effect": "LS", "parents": ["nope"], "options": {}}
    with pytest.raises(ValueError, match="unknown parent"):
        validate_and_sort(spec_from_dict(spec_with(unknown_parent)))


def test_short_aliases_are_the_definitions():
    """LS is linear_shift, and both spellings build the same spec."""
    import tramdag as td

    assert (
        td.intercept,
        td.linear_shift,
        td.complex_shift,
        td.varying_coefficient,
    ) == (I, LS, CS, VC)
    short = ContinuousNode(I("x") + LS("y") + CS("z") + VC("z", t="y"))
    long = ContinuousNode(
        td.intercept("x")
        + td.linear_shift("y")
        + td.complex_shift("z")
        + td.varying_coefficient("z", t="y")
    )
    assert short.terms == long.terms


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
    assert back["b"].terms[1].units == (16,)  # [0] is the canonical SI()


def test_vc_modifiers_are_positional_t_is_keyword():
    t = VC("X2", "X3", t="T")
    assert t.parents == ("T", "X2", "X3")  # internal layout: treatment first
    with pytest.raises(ValueError, match="cannot be both"):
        VC("T", t="T")


def test_a_pre_0_4_spec_says_it_is_too_old():
    """0.3 wrote a term's settings as sibling keys, not in "options".

    Without this the loader raises a bare KeyError('options') from inside the
    comprehension, which does not tell the reader their checkpoint is stale.
    """
    old_format = {
        "x1": {"kind": "continuous", "terms": []},
        "y": {
            "kind": "continuous",
            "terms": [{"effect": "VC", "parents": ["t", "x1"], "penalty": 2.5}],
        },
    }
    with pytest.raises(ValueError, match=r"predates 0\.4"):
        spec_from_dict(old_format)


def test_basis_arguments_apply_without_naming_the_basis():
    """SI(n_coeffs=40) must configure the default basis, not be ignored.

    The effective transform used to be read only from a term that also set
    `transform=`, so basis arguments on their own were silently dropped and
    the reader got the default order with no indication.
    """
    node = ContinuousNode([SI(n_coeffs=40)])
    assert node.transform == "bernstein"
    assert node.transform_kwargs == {"n_coeffs": 40}
    flow = CausalFlowDAG({"x": node}, seed=0)
    assert flow.nodes["x"].ut.n_params == 40


def test_a_source_node_is_canonical_too():
    """None normalizes to [SI()]: equal, hashable, terms[0] is the intercept."""
    assert ContinuousNode().terms == [SI()]
    assert OrdinalNode(3).terms == [SI()]
    assert ContinuousNode() == ContinuousNode([SI()])
    assert hash(ContinuousNode()) == hash(ContinuousNode([SI()]))
    assert len({ContinuousNode(), ContinuousNode([SI()]), OrdinalNode(3)}) == 2


def test_spec_survives_a_json_roundtrip():
    """Tuples come back as lists from json; spec_from_dict restores them."""
    spec = {
        "x": ContinuousNode([SI(transform="spline", bins=6)]),
        "t": OrdinalNode(2, [LS("x")]),
        "y": ContinuousNode([CI("x", units=[8, 8]), VC("x", t="t", units=[4])]),
    }
    back = spec_from_dict(json.loads(json.dumps(spec_to_dict(spec))))
    assert back == spec
    assert back["y"].terms[0].units == (8, 8)  # the CI intercept comes first
    assert hash(back["y"].terms[1]) == hash(spec["y"].terms[1])


def test_wrong_effect_option_errors_instead_of_defaulting():
    """A key another effect takes raises — no silent foreign defaults."""
    from tramdag.spec import _options

    with pytest.raises(ValueError, match="takes no option"):
        _options("CS", penalty=1.0)  # penalty is VC's
    with pytest.raises(AttributeError):
        _ = LS("x").penalty  # reading a foreign option refuses too
    d = {
        "x": {
            "kind": "continuous",
            "terms": [{"effect": "I", "parents": [], "options": {}}],
        },
        "y": {
            "kind": "continuous",
            "terms": [
                {"effect": "I", "parents": [], "options": {}},
                {"effect": "LS", "parents": ["x"], "options": {"pnealty": 1.0}},
            ],
        },
    }
    from tramdag import spec_from_dict

    with pytest.raises(ValueError, match="takes no option"):
        spec_from_dict(d)


def test_transform_accepts_a_custom_class():
    """``I(transform=<class>)`` builds a custom basis (pickle-only serialization)."""
    from tramdag.transforms import BernsteinUT

    spec = {"x": ContinuousNode([I(transform=BernsteinUT, n_coeffs=7)])}
    flow = CausalFlowDAG(spec, seed=0)
    assert isinstance(flow.nodes["x"].ut, BernsteinUT)
    assert flow.nodes["x"].ut.n_params == 7
