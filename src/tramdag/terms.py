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

if TYPE_CHECKING:
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


@register_term
class InterceptDef(TermDef):
    """``I``/``SI``/``CI`` — the transform-parameter slot."""

    effect = "I"
    slot = "intercept"


@register_term
class LinearShiftDef(TermDef):
    """``LS`` — one raw-unit coefficient per (single) parent."""

    effect = "LS"
    slot = "shift"

    @staticmethod
    def check_arity(name: str, term: Term) -> None:
        """Refuse any parent count but one."""
        if len(term.parents) != 1:
            raise ValueError(f"Node '{name}': LS term must have exactly one parent.")


@register_term
class ComplexShiftDef(TermDef):
    """``CS`` — a network shift over its parents."""

    effect = "CS"
    slot = "shift"


@register_term
class VaryingCoefDef(TermDef):
    """``VC`` — ``beta(modifiers) * x_t``; only the treatment owns an edge."""

    effect = "VC"
    slot = "shift"

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
