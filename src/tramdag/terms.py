"""The effect registry: one definition per term effect.

Each entry names its slot (``"intercept"`` or ``"shift"``) and owns the
validation that is specific to its effect — the arity rule of ``LS``, the
treatment/centering rules of ``VC``. ``spec.py`` consults the registry
lazily, so the generic checks (parents exist, ``input_transform`` shape,
edge ownership) stay there and run in the same order as before.

``register_term`` is the extension point for a custom effect; the built-ins
register themselves at import.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from torch import Tensor, nn

from .conditioners import ComplexShift, LinearShift, VaryingCoef

if TYPE_CHECKING:
    from .nodes import _Node
    from .spec import NodeSpec, Term

# %% global variables ------------------------------------------------------------------
_REGISTRY: dict[str, type[TermDef]] = {}


# %% public functions ------------------------------------------------------------------
def register_term(cls: type[TermDef]) -> type[TermDef]:
    """Register a term definition under its ``effect`` name; refuse collisions."""
    if cls.effect in _REGISTRY:
        raise ValueError(
            f"term effect {cls.effect!r} is already registered "
            f"(by {_REGISTRY[cls.effect].__name__})"
        )
    _REGISTRY[cls.effect] = cls
    return cls


def get_term(effect: str) -> type[TermDef]:
    """Give the registered definition of an effect (``KeyError`` if unknown)."""
    return _REGISTRY[effect]


# %% public classes --------------------------------------------------------------------
class TermDef:
    """One effect's definition: its slot and its effect-specific validation.

    ``check_arity`` runs before the generic parent checks (it carries the
    errors that must fire even for unknown parents); ``edge_parents`` runs
    after them and gives the parents that own an edge — every parent for
    the built-in intercept and shifts, only the treatment for ``VC``.
    """

    effect: ClassVar[str]
    slot: ClassVar[str]  # "intercept" | "shift"

    @staticmethod
    def check_arity(name: str, term: Term) -> None:
        """Reject a term whose parent count is wrong for its effect."""

    @staticmethod
    def edge_parents(name: str, term: Term, spec: dict[str, NodeSpec]) -> tuple:
        """Validate against the spec; give the edge-owning parents."""
        return term.parents


class ShiftTerm(TermDef):
    """A shift term's behavior hooks, mixed into its conditioner.

    A built term instance carries ``key`` (its ModuleDict key), ``parents``
    (the term's written parents) and ``net_parents`` (the parents whose
    columns feed its *network* — empty for ``LS``, the modifiers for
    ``VC``). ``build`` constructs the module exactly as the node used to,
    so state-dict paths and the seeded RNG stream stay bit-stable.
    """

    is_classical: ClassVar[bool] = False
    finalizes = False  # set per instance when a post-fit step is needed

    key: str
    parents: tuple
    net_parents: tuple

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> ShiftTerm:
        """Construct the term module from its spec Term."""
        raise NotImplementedError

    def shift_value(self, node: _Node, feats: dict, vc_ehat: dict | None) -> Tensor:
        """Give this term's contribution to the node's shift, shape ``(n,)``."""
        raise NotImplementedError

    def post_init(self) -> None:
        """Re-apply construction-time invariants after a global weight init."""

    @property
    def has_regularizer(self) -> bool:
        """``True`` when :meth:`regularizer` joins the training loss."""
        return False

    def regularizer(self) -> Tensor:
        """Give the term's penalty on the total-likelihood scale."""
        raise NotImplementedError

    def finalize(self, node: _Node, feats: dict) -> None:
        """Run the term's post-fit step (after the after-fit callbacks)."""


@register_term
class InterceptDef(TermDef):
    """``I``/``SI``/``CI`` — the transform-parameter slot."""

    effect = "I"
    slot = "intercept"


@register_term
class LSTerm(ShiftTerm, LinearShift):
    """``LS`` — one raw-unit coefficient per (single) parent."""

    effect = "LS"
    slot = "shift"
    is_classical = True

    @staticmethod
    def check_arity(name: str, term: Term) -> None:
        """Refuse any parent count but one."""
        if len(term.parents) != 1:
            raise ValueError(f"Node '{name}': LS term must have exactly one parent.")

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> LSTerm:
        """One weight per feature of the single parent; keyed by its name."""
        from .spec import feat_width

        m = cls(feat_width(spec, term.parents))
        m.key = term.parents[0]
        m.parents = tuple(term.parents)
        m.net_parents = ()
        return m

    def shift_value(self, node: _Node, feats: dict, vc_ehat: dict | None) -> Tensor:
        """Give the raw parent column times the weight — no input transform."""
        import torch

        return self(torch.cat([feats[p] for p in self.parents], dim=1))


@register_term
class CSTerm(ShiftTerm, ComplexShift):
    """``CS`` — a network shift over its parents."""

    effect = "CS"
    slot = "shift"

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> CSTerm:
        """One net over the concatenated parents; keyed 'a' or 'a+b'."""
        from .spec import feat_width

        ps = tuple(term.parents)
        m = cls(feat_width(spec, ps), units=term.units, activation=term.activation)
        m.key = ps[0] if len(ps) == 1 else "+".join(ps)
        m.parents = ps
        m.net_parents = ps
        return m

    def shift_value(self, node: _Node, feats: dict, vc_ehat: dict | None) -> Tensor:
        """Give the net over this term's (possibly input-transformed) features."""
        return self(node.net_input(feats, self.parents, self.key))


@register_term
class VCTerm(ShiftTerm, VaryingCoef):
    """``VC`` — ``beta(modifiers) * x_t``; only the treatment owns an edge."""

    effect = "VC"
    slot = "shift"

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> VCTerm:
        """Build the effect head over the modifiers; keyed by the treatment name."""
        from .spec import OrdinalNode, feat_width

        on, mods = term.parents[0], tuple(term.parents[1:])
        m = cls(
            feat_width(spec, mods),
            penalty=term.penalty,
            units=term.units,
            activation=term.activation,
        )
        m.key = on
        m.parents = tuple(term.parents)
        m.net_parents = mods
        m.mods = mods
        m.on_is_ord = isinstance(spec[on], OrdinalNode)
        m.vc_center = term.center
        m.finalizes = bool(mods)
        return m

    def shift_value(self, node: _Node, feats: dict, vc_ehat: dict | None) -> Tensor:
        """``beta(modifiers) * regressor``, with the centered-term guard."""
        if self.vc_center and (vc_ehat is None or self.key not in vc_ehat):
            raise RuntimeError(
                f"centered VC term on {self.key!r} needs e_hat. Internal "
                "callers must supply vc_ehat. Never evaluate a centered "
                "term without its propensity."
            )
        t = node.vc_column(self.group, feats, vc_ehat)
        mod_feat = node.net_input(feats, self.mods, self.key) if self.mods else None
        return self(t, mod_feat)

    def post_init(self) -> None:
        """Re-zero the head's output layer: ``beta(x) == beta0`` at start."""
        if self.net is not None:
            nn.init.zeros_(self.net[-1].weight)

    @property
    def has_regularizer(self) -> bool:
        """Penalized whenever the term has a head to shrink."""
        return self.net is not None and self.penalty > 0

    def regularizer(self) -> Tensor:
        """``penalty * ||b_theta weights||^2`` on the total-likelihood scale."""
        return self.penalty * self.l2()

    def finalize(self, node: _Node, feats: dict) -> None:
        """Re-split ``beta0``/``b_theta``: the head sums to zero over train."""
        self.recenter(node.net_input(feats, self.mods, self.key))

    @property
    def group(self):
        """Give the term as the legacy ``_VCGroup`` view (scores/read-outs)."""
        from .nodes import _VCGroup

        return _VCGroup(self.key, self.mods, self.on_is_ord, self.vc_center)

    @staticmethod
    def edge_parents(name: str, term: Term, spec: dict[str, NodeSpec]) -> tuple:
        """Check treatment, penalty and centering; the treatment owns the edge."""
        from .spec import OrdinalNode

        if not term.parents:
            raise ValueError(f"Node '{name}': VC term needs a treatment parent.")
        on = term.parents[0]
        if on in term.parents[1:]:
            raise ValueError(
                f"Node '{name}': VC treatment '{on}' cannot also be a modifier."
            )
        if term.penalty is None or term.penalty < 0:
            raise ValueError(f"Node '{name}': VC penalty must be >= 0.")
        on_node = spec[on]
        if isinstance(on_node, OrdinalNode) and on_node.levels != 2:
            raise ValueError(
                f"Node '{name}': VC treatment '{on}' is ordinal with "
                f"{on_node.levels} levels. Only a 2-level (binary) "
                "ordinal treatment is supported. Multi-level is a "
                "follow-up."
            )
        if term.center and not isinstance(on_node, OrdinalNode):
            raise ValueError(
                f"Node '{name}': VC(center=...) needs a binary ordinal "
                f"treatment, and '{on}' is continuous. E[T|x] centering "
                "is a follow-up."
            )
        if term.center and any(t.effect == "VC" and t.center for t in on_node.terms):
            raise ValueError(
                f"Node '{name}': treatment '{on}' carries a centered VC term "
                "itself; chained centering is not supported."
            )
        return (on,)
