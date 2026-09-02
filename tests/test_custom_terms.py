"""Custom shift terms: ``fn_shift`` and ``register_term``.

The extension contract of 1.0: a callable (or ``nn.Module``) drops into the
additive shifts via ``fn_shift``; a whole new effect subclasses
``tramdag.terms.ShiftTerm`` and registers under its own name. Checkpoints
carry the effect NAME only, so loading a custom spec needs the class
registered first — and a lambda ``fn`` refuses to save.
"""

# %% imports ---------------------------------------------------------------------------
from typing import ClassVar

import numpy as np
import pytest
import torch
from torch import nn

from tramdag import CausalFlowDAG, ContinuousNode, Fn, fn_shift, register_term
from tramdag.spec import Term, _options
from tramdag.terms import _REGISTRY, ShiftTerm, get_term


# %% private functions -------------------------------------------------------------
def _double(features):
    """A module-level fn: picklable, so it survives save/load."""
    return 2.0 * features[:, 0]


def _two_node(term):
    return {"x1": ContinuousNode(), "x2": ContinuousNode([term])}


# %% public functions ------------------------------------------------------------------
def test_fn_shift_offsets_the_latent_and_round_trips(ls_chain, tmp_path):
    """A module-level fn shifts u by fn(x) exactly and survives a checkpoint."""
    df = ls_chain["draw"](600, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node(fn_shift("x1", fn=_double)), seed=0)
    flow.fit(df, epochs=5, batch_size=200)
    assert Fn is fn_shift
    # the shift really is fn (the grid read-out goes through shift_value)
    grid = np.linspace(-2, 2, 9)
    np.testing.assert_allclose(
        flow.shift_curve("x2", "x1", grid), 2.0 * grid, atol=1e-6
    )
    flow.save(tmp_path / "m.pt")
    loaded = CausalFlowDAG.load(tmp_path / "m.pt")
    assert torch.equal(loaded.log_prob(df), flow.log_prob(df))


def test_fn_shift_lambda_refuses_to_save(ls_chain, tmp_path):
    """A lambda fn fails at save() with the picklable-function message."""
    df = ls_chain["draw"](200, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node(fn_shift("x1", fn=lambda x: x[:, 0])), seed=0)
    flow.fit(df, epochs=1, batch_size=200)
    with pytest.raises(ValueError, match="module-level function"):
        flow.save(tmp_path / "m.pt")


def test_fn_shift_nn_module_trains(ls_chain):
    """An nn.Module fn registers as a submodule and its parameters move."""
    df = ls_chain["draw"](600, 0)[["x1", "x2"]]
    net = nn.Linear(1, 1)
    flow = CausalFlowDAG(_two_node(fn_shift("x1", fn=net)), seed=0)
    before = net.weight.detach().clone()
    flow.fit(df, epochs=10, batch_size=200, learning_rate=1e-2)
    assert not torch.equal(before, net.weight.detach())
    assert any(p is net.weight for p in flow.parameters())


def test_fn_shift_validates_its_arguments():
    """No parents and a non-callable fn both refuse loudly."""
    with pytest.raises(ValueError, match="at least one parent"):
        fn_shift(fn=_double)
    with pytest.raises(ValueError, match="must be callable"):
        fn_shift("x1", fn=3)


class _ScaledLS(ShiftTerm, nn.Module):
    """A minimal custom effect: w * x with a fixed scale, for the registry test."""

    effect = "SLS"
    slot = "shift"
    option_defaults: ClassVar[dict] = {"fn": None}  # the scale rides fn

    def __init__(self, scale: float):
        nn.Module.__init__(self)
        self.scale = scale
        self.w = nn.Parameter(torch.zeros(()))

    @classmethod
    def build(cls, term, spec):
        m = cls(scale=term.fn)  # smuggle the scale through the fn option
        m.key = term.parents[0]
        m.parents = tuple(term.parents)
        m.net_parents = ()
        return m

    def shift_value(self, node, feats):
        return self.scale * self.w * feats[self.parents[0]][:, 0]


def test_register_term_round_trip_and_collision(ls_chain):
    """A registered custom effect builds, fits and refuses re-registration;
    an unregistered effect name fails with the register_term pointer.
    """
    df = ls_chain["draw"](600, 0)[["x1", "x2"]]
    if "SLS" not in _REGISTRY:
        register_term(_ScaledLS)
    assert get_term("SLS") is _ScaledLS
    with pytest.raises(ValueError, match="already registered"):
        register_term(_ScaledLS)
    term = Term("SLS", ("x1",), _options("SLS", fn=3.0))
    flow = CausalFlowDAG(_two_node(term), seed=0)
    flow.fit(df, epochs=10, batch_size=200, learning_rate=1e-1)
    assert float(flow.nodes["x2"].shifts["x1"].w) != 0.0
    with pytest.raises(ValueError, match="register_term"):
        CausalFlowDAG(
            {"x1": ContinuousNode(), "x2": ContinuousNode([Term("NOPE", ("x1",), ())])}
        )


def test_custom_intercept_slot_refuses_instead_of_misbuilding(ls_chain):
    """A registered slot='intercept' custom effect must refuse loudly —
    normalization would otherwise drop it silently from the built model.
    """

    class _CustomI(ShiftTerm):  # slot lies on purpose; class shape irrelevant
        effect = "MYI"
        slot = "intercept"
        option_defaults: ClassVar[dict] = {}

    if "MYI" not in _REGISTRY:
        register_term(_CustomI)
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([Term("MYI", ("x1",), ())]),
    }
    with pytest.raises(ValueError, match="not supported yet"):
        CausalFlowDAG(spec)


class _PenShift(ShiftTerm, nn.Module):
    """A custom penalized term: the regularizer hook must reach the loss."""

    effect = "PEN"
    slot = "shift"
    option_defaults: ClassVar[dict] = {}

    def __init__(self):
        nn.Module.__init__(self)
        self.w = nn.Parameter(torch.zeros(()))
        self.calls = 0

    @classmethod
    def build(cls, term, spec):
        m = cls()
        m.key = term.parents[0]
        m.parents = tuple(term.parents)
        m.net_parents = ()
        return m

    def shift_value(self, node, feats):
        return self.w * feats[self.parents[0]][:, 0]

    @property
    def has_regularizer(self):
        return True

    def regularizer(self):
        self.calls += 1
        return self.w**2


def test_custom_regularizer_joins_the_loss(ls_chain):
    """Fit adds regularizer()/n — the hook, not VC's internals."""
    df = ls_chain["draw"](300, 0)[["x1", "x2"]]
    if "PEN" not in _REGISTRY:
        register_term(_PenShift)
    spec = {"x1": ContinuousNode(), "x2": ContinuousNode([Term("PEN", ("x1",), ())])}
    flow = CausalFlowDAG(spec, seed=0)
    flow.fit(df, epochs=2, batch_size=150)
    assert flow.nodes["x2"].shifts["x1"].calls >= 4  # every minibatch


def test_shift_curve_covers_fn_terms(ls_chain):
    """shift_curve evaluates through shift_value, so an Fn term works."""
    df = ls_chain["draw"](300, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node(fn_shift("x1", fn=_double)), seed=0)
    flow.fit(df, epochs=1, batch_size=150)
    grid = np.linspace(-1, 1, 5)
    np.testing.assert_allclose(
        flow.shift_curve("x2", "x1", grid), 2.0 * grid, atol=1e-6
    )
