"""Custom shift terms: ``fn_shift`` and the two-class effect contract.

The extension contract of 1.0: a callable (or ``nn.Module``) drops into the
additive shifts via ``fn_shift``; a whole new effect is a ``tramdag.Term``
subclass (its options and checks) plus a ``tramdag.terms.ShiftTerm`` subclass
declaring ``data =`` that term class. Subclassing is the registration:
checkpoints carry the effect NAME only, so loading a custom spec needs the
classes imported first — and a lambda ``fn`` refuses to save.
"""

# %% imports ---------------------------------------------------------------------------
import numpy as np
import pytest
import torch
from torch import nn

from tramdag import CausalFlowDAG, ContinuousNode, Fn, Term, fn_shift, spec_from_dict
from tramdag.terms import ShiftTerm, module_for


# %% private functions -------------------------------------------------------------
def _double(features):
    """A module-level fn: picklable, so it survives save/load."""
    return 2.0 * features[:, 0]


def _two_node(term):
    return {"x1": ContinuousNode(), "x2": ContinuousNode([term])}


# %% private classes -------------------------------------------------------------------
class SLS(Term):
    """A minimal custom effect: ``w * x`` with a fixed scale option."""

    scale: float = 1.0

    def __post_init__(self):
        """One parent, like LS."""
        if len(self.parents) != 1:
            raise ValueError("SLS() takes exactly one parent.")


class _ScaledLS(ShiftTerm, nn.Module):
    data = SLS

    def __init__(self, scale: float):
        nn.Module.__init__(self)
        self.scale = scale
        self.w = nn.Parameter(torch.zeros(()))

    @classmethod
    def build(cls, term, spec):
        m = cls(scale=term.scale)
        m.key = term.parents[0]
        m.parents = tuple(term.parents)
        m.net_parents = ()
        return m

    def shift_value(self, node, feats):
        return self.scale * self.w * feats[self.parents[0]][:, 0]


class PEN(Term):
    """A custom penalized term: the regularizer hook must reach the loss."""


class _PenShift(ShiftTerm, nn.Module):
    data = PEN

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


class Orphan(Term):
    """A term class no module builds."""


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


def test_custom_effect_builds_fits_and_round_trips(ls_chain, tmp_path):
    """A Term subclass plus a ShiftTerm with ``data =`` is a whole effect:
    it validates, builds, fits, serializes by name and loads back.
    """
    df = ls_chain["draw"](600, 0)[["x1", "x2"]]
    term = SLS("x1", scale=3.0)
    assert module_for(term) is _ScaledLS
    assert term.effect == "SLS"
    assert repr(term) == "SLS('x1', scale=3.0)"
    with pytest.raises(ValueError, match="exactly one parent"):
        SLS("x1", "x2")
    with pytest.raises(ValueError, match="takes no option"):
        SLS("x1", scael=3.0)
    flow = CausalFlowDAG(_two_node(term), seed=0)
    flow.fit(df, epochs=10, batch_size=200, learning_rate=1e-1)
    assert float(flow.nodes["x2"].shifts["x1"].w) != 0.0
    flow.save(tmp_path / "m.pt")
    loaded = CausalFlowDAG.load(tmp_path / "m.pt")
    assert loaded.spec == flow.spec
    assert torch.equal(loaded.log_prob(df), flow.log_prob(df))


def test_unknown_effect_and_orphan_term_fail_by_name():
    """A serialized effect no class carries, and a term class no module
    builds, both say what to define.
    """
    d = {
        "x1": {"kind": "continuous", "terms": []},
        "x2": {
            "kind": "continuous",
            "terms": [{"effect": "NOPE", "parents": ["x1"], "options": {}}],
        },
    }
    with pytest.raises(ValueError, match="unknown term effect 'NOPE'"):
        spec_from_dict(d)
    with pytest.raises(ValueError, match="data = Orphan"):
        CausalFlowDAG(_two_node(Orphan("x1")))


def test_custom_regularizer_joins_the_loss(ls_chain):
    """Fit adds regularizer()/n — the hook, not VC's internals."""
    df = ls_chain["draw"](300, 0)[["x1", "x2"]]
    flow = CausalFlowDAG(_two_node(PEN("x1")), seed=0)
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
