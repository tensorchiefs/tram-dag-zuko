"""User-facing DAG specification.

A model is one dict ``{node_name: NodeSpec}``. Each node declares its
transformation ``h`` as an **additive formula of terms** — the first
positional argument, written as a list or as a ``+`` sum::

    "X3": ContinuousNode([I("X1"), CS("X2")])       # h = h_theta(x1) + g(x2)
    "X3": ContinuousNode(I("X1") + CS("X2"))        # the same, formula style

Each effect is a :class:`Term` subclass, called with the parent(s) it
depends on: :class:`Intercept`, :class:`LinearShift`, :class:`ComplexShift`,
:class:`VaryingCoefficient` and :class:`FnShift`. Their short names ``I``,
``LS``, ``CS``, ``VC``, ``Fn`` (and ``SI``/``CI`` for the two intercept
arities) are the notation of the docs and the paper and the same objects, so
use whichever reads better:

- :func:`I`  — *intercept* term: the parent(s) reshape the monotone transform
  (its Bernstein coefficients / ordinal cutpoints). ``I`` dispatches on its
  arguments: without parents it is the paper's simple intercept :func:`SI`
  (always present, optional to write — the bare names ``I`` and ``SI`` both
  work in a term list), with parents the complex intercept :func:`CI`.
  ``transform="spline"`` picks the class of the monotone transform for a
  continuous node; extra keyword arguments go straight to the transform
  class (``SI(transform="spline", bins=16)``).
- :func:`LS` — *linear shift*: ``beta * x`` (one interpretable weight), one parent.
- :func:`CS` — *complex shift*: an additive NN ``g(x)`` on the latent scale.
- :func:`VC` — *varying-coefficient shift*: ``beta(modifiers) * x_on`` with
  ``beta(x) = beta0 + b_theta(x)`` and ``b_theta`` a small, **penalized** network
  — a treatment-effect head with its own bias–variance budget (issue #28).

A formula holds **exactly one intercept term, first** — written, or added as
``SI()`` when the formula only lists shifts — so ``node.terms[0]`` is always
the intercept. The intercept slot sums in coefficient space; the shift slot
sums on the latent scale. "Joint vs additive" is argument grouping: a
multi-parent term such as
``CS("a","b")`` is one **joint** network over both parents (an interaction),
whereas ``CS("a") + CS("b")`` are two **additive** terms. For intercepts the
grouping is said explicitly: ``I("a", "b", allow_interaction=False)`` is the
additive intercept. Several ``I`` terms with parents on one node are an
error — the flag is the only way to say it, so a term list is always purely
additive on the latent scale.

What ``h`` looks like per transformation, for a continuous ``x3``:

======================================== =======================================
``terms=``                               ``u_3 = h(x_3 | pa)``
======================================== =======================================
``None`` / ``[I]``                       ``h_theta(x3)``
``[LS("X1")]``                           ``h_theta(x3) + beta*x1``
``[I("X1")]``                            ``h_theta(x1)(x3)``
``[CS("X1")]``                           ``h_theta(x3) + g_1(x1)``
``[LS("X1"), CS("X2")]``                 ``h_theta(x3) + beta*x1 + g_2(x2)``
``[CS("X1", "X2")]``                     ``h_theta(x3) + g_12(x1, x2)``
``[CS("X1"), CS("X2")]``                 ``h_theta(x3) + g_1(x1) + g_2(x2)``
``[I("X1", "X2")]``                      ``h_theta(x1,x2)(x3)``  (joint)
``[I("X1","X2", allow_interaction=       ``h_theta(x1)+theta(x2)(x3)``  (additive:
False)]``                                one net per parent, summed coefficients)
``[I, CS("X1"), VC("X2", t="T")]``       ``h_theta(x3) + g_1(x1) + beta(x2)*t``
======================================== =======================================

Each parent enters through exactly one *edge-owning* term (I/LS/CS parents,
and a VC term's treatment ``t``). VC **modifiers** are exempt:
``CS("x2")`` + ``VC("x2", t="T")`` is the intended pattern — ``x2`` acts
prognostically through the shift *and* modifies the treatment effect.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

# %% global variables ------------------------------------------------------------------
# bernstein, because zuko's spline extrapolates with a fixed slope outside
# [-B, B] while bernstein follows its own boundary derivative -- see the
# `transform` parameter of I.
DEFAULT_TRANSFORM = "bernstein"
INPUT_TRANSFORMS = ("minmax", "standardize")


# %% private functions -----------------------------------------------------------------
def _subclasses(cls) -> list[type]:
    """Give every subclass of ``cls``, transitively, in definition order."""
    out = []
    for sub in cls.__subclasses__():
        out.append(sub)
        out.extend(_subclasses(sub))
    return out


def _effect_class(effect: str) -> type[Term]:
    """Give the :class:`Term` subclass whose ``effect`` name this is.

    Raises
    ------
    ValueError
        If no term class carries the name — a custom effect must be
        imported before a spec naming it is loaded.
    """
    for cls in _subclasses(Term):
        if cls.effect == effect:
            return cls
    raise ValueError(
        f"unknown term effect '{effect}'. A custom effect is a tramdag.Term "
        "subclass; import it before the spec is built or loaded."
    )


def _tupled(value):
    """Take a serialized option value back to its canonical tuple form.

    Mappings become sorted tuple-of-pairs (for nested kwargs like
    ``transform_kwargs``); lists become tuples (for ``units``, ``parents``,
    etc.).
    """
    if isinstance(value, dict):
        return tuple(sorted((k, _tupled(v)) for k, v in value.items()))
    return tuple(_tupled(v) for v in value) if isinstance(value, list) else value


def _mapped(value):
    """Serialize one option value: kwargs tuples become mappings, tuples lists."""
    if (
        isinstance(value, tuple)
        and value
        and all(
            isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str)
            for v in value
        )
    ):
        return {k: _mapped(v) for k, v in value}
    return [_mapped(v) for v in value] if isinstance(value, tuple) else value


def _as_term(value) -> Term:
    """Take one entry of a formula to a :class:`Term`.

    The bare names ``I`` and ``SI`` stand for ``I()``, the simple-intercept
    baseline.

    Raises
    ------
    TypeError
        If the entry is neither a term nor the bare ``I``.
    """
    if value is Intercept or value is simple_intercept:
        return Intercept()
    if isinstance(value, Term):
        return value
    raise TypeError(
        "a transformation is built from terms (I/LS/CS/VC) — got "
        f"{type(value).__name__}. A '+' sum is already a flat list, so do "
        "not nest one inside another list: write either a list or a sum."
    )


def _normalize_terms(value):
    """Flatten a node's formula into its canonical term list.

    Accepted: ``None`` (a source node), one term, a ``+`` sum, the bare
    name ``I``, or a list of any of those. A ``+`` sum is already flat, so
    a list of lists is a mistake rather than a shape to flatten.

    The canonical form starts with the intercept: a formula written
    without one gets ``I()`` prepended, so ``terms[0]`` is always the
    intercept term. Exactly one intercept is allowed, and it must come
    first when written.

    Parameters
    ----------
    value : Term | list[Term] | None
        The formula as written.

    Returns
    -------
    list[Term]
        The canonical term list; a source node gives ``[I()]``.
    """
    if value is None:
        return [Intercept()]  # a source node: the free intercept alone
    written = value if isinstance(value, (list, tuple)) else [value]
    items = [_as_term(e) for e in written]
    intercept_at = [i for i, t in enumerate(items) if isinstance(t, Intercept)]
    if len(intercept_at) > 1:
        parented = [t for t in items if isinstance(t, Intercept) and t.parents]
        if len(parented) > 1:
            raise ValueError(
                "a formula takes exactly one intercept term. For an additive "
                "intercept write CI("
                + ", ".join(repr(p) for t in parented for p in t.parents)
                + ", allow_interaction=False), not several intercept terms."
            )
        raise ValueError(
            "a formula takes exactly one intercept term, and CI(...) already "
            "contains the baseline — drop the extra I/SI."
        )
    if not intercept_at:
        return [Intercept(), *items]  # canonical form: intercept first
    if intercept_at[0] != 0:
        raise ValueError(
            "the intercept term comes first: write "
            "I(...) + <shifts>, not the other way around."
        )
    return items


def _check_node(name: str, node: NodeSpec, spec: dict[str, NodeSpec]) -> None:
    """Validate one node against the spec: parents exist, each owns one edge.

    A term validates its own shape when it is built (arity, option
    values); what needs the spec — parents exist, the ``VC`` treatment and
    centering rules — runs here, through the term's ``edge_parents``.

    Raises
    ------
    ValueError
        If a parent is unknown, a term's spec-level rules fail, or a parent
        enters through more than one edge-owning term.
    """
    seen: set[str] = set()
    for term in node.terms:
        for p in term.parents:
            if p not in spec:
                raise ValueError(f"Node '{name}': unknown parent '{p}'.")
        for p in term.edge_parents(name, spec):
            if p in seen:
                raise ValueError(
                    f"Node '{name}': parent '{p}' appears in more than one "
                    "term. Each parent must enter through exactly one "
                    "edge-owning term. Only VC modifiers may repeat."
                )
            seen.add(p)


def _kahn_sort(spec: dict[str, NodeSpec]) -> list[str]:
    """Topologically sort the nodes with Kahn's algorithm.

    Dependencies are ``pa(x_i)``, the union of all term parents. Ready
    nodes are emitted in sorted batches, so the order is deterministic.

    Raises
    ------
    ValueError
        If the graph has a cycle.
    """
    remaining = {name: set(node_parents(node)) for name, node in spec.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(n for n, deps in remaining.items() if not deps)
        if not ready:
            raise ValueError(f"Graph has a cycle among: {sorted(remaining)}")
        for n in ready:
            order.append(n)
            del remaining[n]
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


# %% public functions ------------------------------------------------------------------
def feat_width(spec: dict[str, NodeSpec], parents) -> int:
    """Total feature width of the parents (ordinal one-hot, continuous raw)."""
    return sum(
        spec[p].levels if isinstance(spec[p], OrdinalNode) else 1 for p in parents
    )


def simple_intercept(**options) -> Intercept:
    """Build the simple-intercept baseline, the paper's SI: ``I()`` without parents.

    Parameters
    ----------
    **options
        As for :class:`Intercept`: ``transform`` and its keyword arguments.

    Returns
    -------
    Intercept
        The parentless intercept term.
    """
    return Intercept(**options)


def complex_intercept(*parents: str, **options) -> Intercept:
    """Build the complex intercept, the paper's CI: ``I(*parents)`` with parents.

    Parameters
    ----------
    *parents : str
        Parent names, at least one.
    **options
        As for :class:`Intercept`.

    Returns
    -------
    Intercept
        The parent-conditioned intercept term.

    Raises
    ------
    ValueError
        If no parent is given.
    """
    if not parents:
        raise ValueError(
            "complex_intercept() needs at least one parent. The parentless "
            "baseline is simple_intercept() / SI."
        )
    return Intercept(*parents, **options)


def node_parents(node: NodeSpec) -> list[str]:
    """Give the parent names that the terms of a node reference.

    Parameters
    ----------
    node : NodeSpec
        The node specification.

    Returns
    -------
    list[str]
        The parent names, ordered by first appearance, without
        duplicates.
    """
    seen: dict[str, None] = {}
    for term in node.terms:
        for p in term.parents:
            seen.setdefault(p, None)
    return list(seen)


def validate_and_sort(spec: dict[str, NodeSpec]) -> list[str]:
    """Validate the spec and return a topological ordering of the nodes.

    Edge ownership: every parent must enter through exactly one
    edge-owning term. Edge-owning are all parents of I/LS/CS terms and
    the ``t`` of a VC term. VC *modifiers* are exempt — they can repeat
    across terms, because a modifier typically also acts prognostically
    through a CS or LS term.

    Parameters
    ----------
    spec : dict[str, NodeSpec]
        The DAG specification.

    Returns
    -------
    list[str]
        The node names in topological order.

    Raises
    ------
    ValueError
        If a parent is unknown, a parent enters through more than one
        edge-owning term, a VC treatment is unsupported, or the graph has a
        cycle.
    """
    for name, node in spec.items():
        _check_node(name, node, spec)
    return _kahn_sort(spec)


def spec_to_dict(spec: dict[str, NodeSpec]) -> dict:
    """Give the serialized representation of a spec, for checkpoints.

    A term serializes as its effect name, its parents and the options that
    differ from their defaults — nothing else, so the form is canonical.
    The result is JSON- and YAML-safe: plain tuples become lists and
    nested kwargs tuples (``transform_kwargs``) become mappings, which is
    also how a hand-written YAML spec reads best; :func:`spec_from_dict`
    accepts both forms and turns them back, so a spec round-trips through
    ``json``/YAML as well as through ``torch.save`` — except when a term
    carries a *callable* (``input_transform``, ``fn``), which serializes
    only through pickle (``torch.save``) and only as a module-level
    function.

    Parameters
    ----------
    spec : dict[str, NodeSpec]
        The DAG specification.

    Returns
    -------
    dict
        The serialized spec. :func:`spec_from_dict` inverts it.
    """
    out = {}
    for name, node in spec.items():
        d = {
            "kind": node.kind,
            "terms": [
                {
                    "effect": t.effect,
                    "parents": list(t.parents),
                    "options": {k: _mapped(v) for k, v in t.options().items()},
                }
                for t in node.terms
            ],
        }
        if isinstance(node, OrdinalNode):
            d["levels"] = node.levels
        out[name] = d
    return out


def spec_from_dict(d: dict) -> dict[str, NodeSpec]:
    """Rebuild a spec from its serialized form.

    Each term is rebuilt through its class, so a stale or misspelled
    option key and a wrong arity fail here, by name.

    Parameters
    ----------
    d : dict
        The serialized spec, as produced by :func:`spec_to_dict`.

    Returns
    -------
    dict[str, NodeSpec]
        The node specification, keyed by node name.

    Raises
    ------
    ValueError
        If a term names an unknown effect or an option its effect does
        not take.
    """
    spec: dict[str, NodeSpec] = {}
    for name, nd in d.items():
        terms = []
        for t in nd["terms"]:
            cls = _effect_class(t["effect"])
            options = {k: _tupled(v) for k, v in t["options"].items()}
            try:
                terms.append(cls.from_serialized(tuple(t["parents"]), options))
            except ValueError as err:
                raise ValueError(f"node '{name}': {err}") from None
        if nd["kind"] == "continuous":
            spec[name] = ContinuousNode(terms or None)
        else:
            spec[name] = OrdinalNode(int(nd["levels"]), terms or None)
    return spec


# %% public classes --------------------------------------------------------------------
@dataclass(frozen=True, init=False, repr=False)
class Term:
    """One additive term of a node's transformation; each effect is a subclass.

    Terms add: ``I("a") + CS("b")`` is the same transformation as
    ``[I("a"), CS("b")]``. A term is frozen data — hashable, comparable,
    serializable by :func:`spec_to_dict` — and knows the spec-level rules
    of its effect (``edge_parents``, ``cells``, ``classical``). The module
    that trains it lives in :mod:`tramdag.terms` and declares which term
    class it builds (``data = CS``).

    Subclass to add an effect: the class name is the effect name (its
    serialized ``effect``), every annotated attribute with a default is an
    option, and ``__post_init__`` holds the construction-time checks::

        class Scaled(Term):
            scale: float = 1.0

    Attributes
    ----------
    parents : tuple[str, ...]
        Ordered parent names the term depends on. Empty only for the bare
        simple intercept ``I()``. For a :class:`VC` term, ``parents[0]`` is
        the treatment ``t`` and the rest are the effect modifiers; every
        other built-in term's parents all own their edges.
    """

    parents: tuple[str, ...] = ()

    effect: ClassVar[str] = "Term"
    cell_tag: ClassVar[str | None] = None  # to_matrix tag; None -> the effect

    def __init_subclass__(cls, **kwargs):
        """Make every subclass a frozen dataclass named after its effect.

        Raises
        ------
        TypeError
            If an option has no plain default (a ``default_factory`` would
            make the term unhashable and its serialization non-canonical).
        """
        super().__init_subclass__(**kwargs)
        cls.effect = cls.__dict__.get("effect", cls.__name__)
        dataclass(frozen=True, init=False, repr=False)(cls)  # decorates in place
        missing = [
            f.name for f in dataclasses.fields(cls) if f.default is dataclasses.MISSING
        ]
        if missing:
            raise TypeError(
                f"{cls.__name__}: option(s) {missing} need a plain default value."
            )

    def __init__(self, *parents: str, **options):
        names = self.option_names()
        unknown = sorted(set(options) - set(names))
        if unknown:
            raise ValueError(
                f"effect '{self.effect}' takes no option(s) {unknown}; "
                f"it takes {sorted(names)}."
            )
        object.__setattr__(self, "parents", tuple(parents))
        for f in dataclasses.fields(self):
            if f.name != "parents":
                object.__setattr__(self, f.name, options.get(f.name, f.default))
        self.__post_init__()

    def __post_init__(self) -> None:
        """Check the term's own shape; subclasses extend this."""
        value = getattr(self, "input_transform", None)
        if value is not None and not (callable(value) or value in INPUT_TRANSFORMS):
            raise ValueError(
                f"{self.effect}(): input_transform must be 'minmax', "
                f"'standardize' or a callable fn(x, train), got {value!r}."
            )
        units = getattr(self, "units", None)
        if units is not None:
            object.__setattr__(self, "units", tuple(units))

    @classmethod
    def option_names(cls) -> list[str]:
        """Give the option names this effect takes."""
        return [f.name for f in dataclasses.fields(cls) if f.name != "parents"]

    @classmethod
    def from_serialized(cls, parents: tuple[str, ...], options: dict) -> Term:
        """Rebuild a term from its serialized parents and options."""
        return cls(*parents, **options)

    def options(self) -> dict:
        """Give the options that differ from their defaults, by name."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "parents" and getattr(self, f.name) != f.default
        }

    @property
    def classical(self) -> bool:
        """Say whether the exact classical fit (``fit_classical``) handles this term."""
        return False

    def edge_parents(self, name: str, spec: dict[str, NodeSpec]) -> tuple[str, ...]:
        """Validate against the spec; give the parents that own an edge."""
        return self.parents

    def cells(self) -> list[tuple[str, str]]:
        """Give the term's adjacency cells as ``(parent, tag)`` pairs.

        A multi-parent term carries its parent group as a suffix.
        """
        tag = self.cell_tag or self.effect
        if len(self.parents) > 1:
            tag = f"{tag}{list(self.parents)}"
        return [(p, tag) for p in self.parents]

    def __repr__(self):
        """Show the call that builds the term: parents, then non-default options."""
        args = [repr(p) for p in self.parents]
        args += [f"{k}={v!r}" for k, v in self.options().items()]
        return f"{self.effect}({', '.join(args)})"

    def __add__(self, other: Term | list[Term]) -> list[Term]:
        """Concatenate into a plain term list."""
        if isinstance(other, Term):
            return [self, other]
        if isinstance(other, list):
            return [self, *other]
        return NotImplemented

    def __radd__(self, other: list[Term]) -> list[Term]:
        """Extend a term list from the right, for ``list + term`` chains."""
        if isinstance(other, list):
            return [*other, self]
        return NotImplemented


class Intercept(Term):
    """The intercept term ``I``: the parents reshape the monotone transform.

    Without parents it is the paper's simple intercept **SI** — one free
    parameter vector, the same for every row (the bare names ``I`` and
    ``SI`` in a term list both mean ``I()``). With parents it is the
    complex intercept **CI**: the transform parameters become a function
    of them. :func:`SI` and :func:`CI` are the two spellings with their
    arity checked.

    Parameters
    ----------
    *parents : str
        Parent names. Several parents form one **joint** network (an
        interaction) unless ``allow_interaction=False``.
    transform : str | type | None, optional
        Class of a continuous node's monotone transform: ``"bernstein"``
        (default), ``"spline"``, ``"affine"``, or a ``_ScaledUT`` subclass.
        Bernstein is the default because zuko's spline extrapolates outside
        ``[-B, B]`` with a *fixed* slope, independent of the fitted
        parameters, so the ~10% of data beyond the 5%/95% pre-scaling range
        is misweighted whenever the true tail slope differs; Bernstein
        extrapolates linearly along its own boundary derivative. An ordinal
        node accepts none, because its intercept is the cutpoint vector.
    **transform_kwargs
        Any other keyword goes straight to the transform class, for
        example ``I(transform="spline", bins=16)`` or ``I(n_coeffs=40)``. A
        serialized term passes them as one ``transform_kwargs`` mapping; a
        keyword written out wins over the same key inside that mapping.
    allow_interaction : bool, optional
        ``False`` makes a multi-parent term **additive**: one network per
        parent, their parameter vectors summed in coefficient space. A node
        takes at most one intercept term with parents — write an additive
        intercept with this flag, not with several intercept terms. Default
        ``True``: one joint network is what the reference implementations
        do.
    units : list[int] | tuple[int, ...] | None, optional
        Hidden layers of the term's network, for example ``units=[16]``.
        Default ``[8, 8]``, from the PyTorch reference — see
        :mod:`tramdag.conditioners`, which also explains why a paper
        replication sets this explicitly.
    activation : str | None, optional
        Activation of the network's hidden layers, by default the
        conditioners' ``relu``.
    input_transform : str | callable | None, optional
        ``"minmax"``, ``"standardize"`` or a callable ``fn(x, train)``
        applied per continuous parent column (``train`` is that column's
        raw training data, frozen at ``calibrate``). Parents only.

    Raises
    ------
    ValueError
        If a parentless term carries ``input_transform``, or
        ``allow_interaction=False`` comes with fewer than two parents (an
        interaction to disallow needs two).
    """

    effect = "I"
    cell_tag = "CI"

    transform: str | type | None = None
    transform_kwargs: tuple | None = None
    units: tuple[int, ...] | None = None
    activation: str | None = None
    input_transform: object = None
    allow_interaction: bool = True

    def __init__(self, *parents: str, transform_kwargs=None, **options):
        known = set(self.option_names())
        extra = {k: options.pop(k) for k in list(options) if k not in known}
        kwargs = {**dict(transform_kwargs or ()), **extra}
        super().__init__(
            *parents,
            transform_kwargs=tuple(sorted(kwargs.items())) or None,
            **options,
        )

    def __post_init__(self) -> None:
        """Refuse network options without parents, and a lone additive flag."""
        if not self.parents and self.input_transform is not None:
            raise ValueError(
                "a simple intercept has no network inputs — input_transform= "
                "belongs on CI/CS/VC terms."
            )
        if not self.allow_interaction and len(self.parents) < 2:
            raise ValueError(
                "allow_interaction=False makes a MULTI-parent intercept additive; "
                "with one parent there is no interaction to disallow — drop the "
                "argument."
            )
        super().__post_init__()

    @property
    def classical(self) -> bool:
        """Say yes only for a parentless ``I()`` — the simple baseline."""
        return not self.parents

    def __repr__(self):
        """Show the transform's keyword arguments as they were written."""
        opts = self.options()
        kwargs = dict(opts.pop("transform_kwargs", None) or ())
        args = [repr(p) for p in self.parents]
        args += [f"{k}={v!r}" for k, v in {**opts, **kwargs}.items()]
        return f"{self.effect}({', '.join(args)})"


class LinearShift(Term):
    """The linear shift ``LS``: ``beta * x``, one interpretable raw-unit coefficient.

    Parameters
    ----------
    *parents : str
        Exactly one parent name. The weight stays the interpretable
        raw-unit coefficient, so an LS takes no ``input_transform``.

    Raises
    ------
    ValueError
        If the parent count is not one.
    """

    effect = "LS"

    def __post_init__(self) -> None:
        """Refuse any parent count but one."""
        if len(self.parents) != 1:
            raise ValueError("LS() takes exactly one parent.")

    @property
    def classical(self) -> bool:
        """Say yes — an LS is a classical transformation-model coefficient."""
        return True


class ComplexShift(Term):
    """The complex shift ``CS``: an additive network ``g(x)`` on the latent scale.

    Parameters
    ----------
    *parents : str
        At least one parent name. Several parents feed one joint network;
        ``CS("a") + CS("b")`` are two additive terms instead.
    units : list[int] | tuple[int, ...] | None, optional
        Hidden layers, for example ``units=[16]``. Default
        ``[64, 128, 64]``, from the PyTorch reference — see
        :mod:`tramdag.conditioners`.
    activation : str | None, optional
        Activation of the hidden layers, by default the conditioners'
        ``relu``.
    input_transform : str | callable | None, optional
        As for :class:`I`.

    Raises
    ------
    ValueError
        If no parent is given.
    """

    effect = "CS"

    units: tuple[int, ...] | None = None
    activation: str | None = None
    input_transform: object = None

    def __post_init__(self) -> None:
        """Refuse a parentless network."""
        if not self.parents:
            raise ValueError("CS() needs at least one parent.")
        super().__post_init__()


class VaryingCoefficient(Term):
    """The varying-coefficient shift ``VC``: ``beta(modifiers) * x_t``.

    The treatment-effect term of issue #28: ``VC("X2", "X3", t="T")`` is
    ``(beta0 + b_theta(x2, x3)) * x_t``, with ``b_theta`` a small network
    whose weights carry the L2 ``penalty``. The fitting objective is the
    penalized NLL ``sum_i nll_i + penalty * ||b_theta weights||^2`` on the
    total-likelihood scale — a fixed Gaussian prior whose shrinkage
    vanishes as n grows. ``beta0`` is not penalized.

    The output of ``b_theta`` is zero-initialized and, after the fit,
    mean-centered over the training data, so ``beta0`` is the
    interpretable main effect on the log-odds scale — the classical
    ``Colr``/``LS`` reading when ``beta`` is constant. ``penalty -> inf``,
    or exactly zero modifiers, reduces the term to ``LS(t)``, so VC-vs-LS
    is a nested question. Read the fitted effect out with
    :meth:`CausalFlowDAG.varying_coef`.

    Unlike other effects, VC *modifiers* can also appear in the node's
    prognostic terms (``CS``/``LS``/``I``). Only ``t`` owns its edge.

    Parameters
    ----------
    *modifiers : str
        The effect modifiers — the covariates that enter ``b_theta``.
        Empty means a constant effect.
    t : str
        The treatment (required keyword). Must be a continuous node or a
        binary (2-level) ordinal node. The term is linear in ``x_t``.
    penalty : float, optional
        L2 weight on the ``b_theta`` weights, by default 1.0. Must be
        >= 0. 1.0 is the value at which ``tests/test_vc_term.py`` recovers
        the known ``beta(x)`` of the ``vc_hetero`` DGP at corr ~ 0.99. The
        penalty is on the total-NLL scale, so its effective strength moves
        with ``n``: raise it for small ``n`` or many modifiers.
    center : str | False, optional
        Propensity centering (issue #30), by default ``False``, which is
        bit-identical to the uncentered term. A string names the
        **training-frame column** holding the out-of-fold propensities
        ``P(t = 1 | pa_t)`` per row — compute them with any cross-fitted
        classifier OUTSIDE the flow and merge them as a column (in-sample
        values reintroduce the own-observation bias). The regressor becomes
        ``beta(x) * (x_t - e_hat(pa_t))`` — the Robinson/R-learner
        orthogonalization inside the likelihood; every query after the fit
        recomputes the propensity live from the flow's own treatment node.
        Requires a binary ordinal ``t``. ``docs/varying-coefficients.md``
        measures a 5-10x bias reduction from turning it on.
    units : list[int] | tuple[int, ...] | None, optional
        Hidden layers of ``b_theta``, by default ``[16]`` — see
        :class:`tramdag.conditioners.VaryingCoef` for why that size.
    activation : str | None, optional
        Activation of ``b_theta``'s hidden layers, by default the
        conditioners' ``relu``.
    input_transform : str | callable | None, optional
        As for :class:`I`, over the modifiers.

    Raises
    ------
    ValueError
        If ``t`` is also a modifier or if ``penalty`` is negative.

    Notes
    -----
    With ``center="col"``, training reads **out-of-fold** ``e_hat`` for
    every row from that column of the training frame — the DML
    cross-fitting requirement; in-sample centering can be *worse* than
    none. The values are frozen as data, so no gradient reaches the ``t``
    node from this node's loss. Inference (``log_prob``/``sample``/
    ``abduct``/``pmf``) recomputes ``e_hat`` from the flow's own fitted
    ``t`` node — the full-data fit, the standard DML train/predict split —
    and always re-derives ``x_t - e_hat`` under ``do``, never from a
    cache. With centering, ``beta0`` is the effect at the treatment margin
    (the observed propensities). The LS-nesting reading applies to the
    uncentered term only.
    """

    effect = "VC"

    penalty: float = 1.0
    center: str | bool = False
    units: tuple[int, ...] | None = None
    activation: str | None = None
    input_transform: object = None

    def __init__(self, *modifiers: str, t: str, **options):
        super().__init__(t, *modifiers, **options)

    def __post_init__(self) -> None:
        """Refuse a treatment that is also a modifier, and a negative penalty."""
        t, modifiers = self.parents[0], self.parents[1:]
        if t in modifiers:
            raise ValueError(
                f"VC(): '{t}' cannot be both the treatment (t) and a modifier."
            )
        if self.penalty is None or self.penalty < 0:
            raise ValueError(f"VC(): penalty must be >= 0, got {self.penalty}.")
        object.__setattr__(self, "penalty", float(self.penalty))
        super().__post_init__()

    @classmethod
    def from_serialized(cls, parents: tuple[str, ...], options: dict) -> Term:
        """Rebuild from the serialized parents: the treatment comes first."""
        return cls(*parents[1:], t=parents[0], **options)

    def edge_parents(self, name: str, spec: dict[str, NodeSpec]) -> tuple[str, ...]:
        """Check the treatment and the centering; only the treatment owns an edge.

        Raises
        ------
        ValueError
            If the centering column is malformed or collides with a node,
            the treatment is a multi-level ordinal, or centering meets a
            continuous or an itself-centered treatment.
        """
        on = self.parents[0]
        if self.center is not False and not isinstance(self.center, str):
            raise ValueError(
                f"Node '{name}': VC(center=) names the propensity COLUMN of "
                "the training frame (out-of-fold P(t=1|pa_t) per row), or is "
                f"False — got {self.center!r}. Cross-fit the propensities "
                "outside and merge them as a column."
            )
        if self.center and self.center in spec:
            raise ValueError(
                f"Node '{name}': the propensity column {self.center!r} "
                "collides with a node name."
            )
        on_node = spec[on]
        if isinstance(on_node, OrdinalNode) and on_node.levels != 2:
            raise ValueError(
                f"Node '{name}': VC treatment '{on}' is ordinal with "
                f"{on_node.levels} levels. Only a 2-level (binary) "
                "ordinal treatment is supported. Multi-level is a "
                "follow-up."
            )
        if self.center and not isinstance(on_node, OrdinalNode):
            raise ValueError(
                f"Node '{name}': VC(center=...) needs a binary ordinal "
                f"treatment, and '{on}' is continuous. E[T|x] centering "
                "is a follow-up."
            )
        if self.center and any(
            isinstance(t, VaryingCoefficient) and t.center for t in on_node.terms
        ):
            raise ValueError(
                f"Node '{name}': treatment '{on}' carries a centered VC term "
                "itself; chained centering is not supported."
            )
        return (on,)

    def cells(self) -> list[tuple[str, str]]:
        """Tag the treatment cell ``VC`` and the modifiers ``VCm``."""
        return [(self.parents[0], "VC")] + [(p, "VCm") for p in self.parents[1:]]

    def __repr__(self):
        """Show the modifiers, then ``t=``, then the non-default options."""
        args = [repr(p) for p in self.parents[1:]] + [f"t={self.parents[0]!r}"]
        args += [f"{k}={v!r}" for k, v in self.options().items()]
        return f"{self.effect}({', '.join(args)})"


class FnShift(Term):
    """The function shift ``Fn``: ``fn(features)`` joins the additive shifts.

    The cheapest custom term: ``fn`` takes the term's concatenated parent
    features ``(n, k)`` (continuous raw, ordinal one-hot — through
    ``input_transform`` when given) and returns the shift contribution,
    shape ``(n,)`` or ``(n, 1)``. A plain function is a fixed offset; an
    ``nn.Module`` registers as a submodule and trains with the flow.

    Checkpoints pickle ``fn``, so it must be a module-level function or an
    importable ``nn.Module`` — ``save()`` refuses a lambda. For a whole new
    effect (own options, penalty, side inputs) subclass :class:`Term` and
    :class:`tramdag.terms.ShiftTerm` instead.

    Parameters
    ----------
    *parents : str
        Parent node names feeding ``fn``.
    fn : callable | torch.nn.Module
        The shift function (required).
    input_transform : str | callable | None, optional
        As for :class:`CS`, by default None.

    Raises
    ------
    ValueError
        If no parent is given or ``fn`` is not callable.
    """

    effect = "Fn"

    fn: object = None
    input_transform: object = None

    def __post_init__(self) -> None:
        """Refuse a parentless term and a non-callable ``fn``."""
        if not self.parents:
            raise ValueError("fn_shift needs at least one parent.")
        if not callable(self.fn):
            # a domain error (a wrong option value), not a Python type error
            raise ValueError(  # noqa: TRY004
                f"fn_shift(fn=) must be callable, got {type(self.fn).__name__}."
            )
        super().__post_init__()


class ContinuousNode:
    """Continuous variable, modelled by a monotone 1-D transform + shifts.

    Parameters
    ----------
    terms : Term | list[Term] | None, optional
        The additive formula for ``h``: a list of terms, a ``+`` sum, a
        single term, or the bare ``I``. ``None`` (default) is a source node. The
        class of the monotone transform is chosen on the intercept term,
        ``I(..., transform="spline")``; the default is ``"bernstein"``.
    """

    kind = "continuous"

    def __init__(self, terms=None):
        self.terms = _normalize_terms(terms)
        # the arguments go straight to the transform class; if they are
        # wrong, that class says so — this layer does not second-guess it
        intercept = self.terms[0]
        self.transform = intercept.transform or DEFAULT_TRANSFORM
        self.transform_kwargs = dict(intercept.transform_kwargs or ())

    def __repr__(self):
        """Show the terms and the transform."""
        return f"ContinuousNode({self.terms!r}, transform={self.transform!r})"

    def __eq__(self, other):
        """Compare the terms; the transform is derived from them."""
        # transform/transform_kwargs are derived from the terms, so equal
        # term lists already imply an equal transform
        return isinstance(other, ContinuousNode) and self.terms == other.terms

    def __hash__(self):
        """Hash what ``__eq__`` compares, so nodes work in sets and as keys."""
        return hash((self.kind, tuple(self.terms)))


class OrdinalNode:
    """Ordinal variable with ``levels`` ordered classes, stored 0 to levels-1.

    An ordered logit models it: increasing cutpoints plus the shift terms.

    Parameters
    ----------
    levels : int
        Number of ordered classes.
    terms : Term | list[Term] | None, optional
        The additive formula, as for :class:`ContinuousNode`, by default
        ``None``.
    """

    kind = "ordinal"

    def __init__(self, levels: int, terms=None):
        self.levels = int(levels)
        if self.levels < 2:
            raise ValueError(f"ordinal levels must be >= 2, got {self.levels}.")
        self.terms = _normalize_terms(terms)
        intercept = self.terms[0]
        if intercept.transform or intercept.transform_kwargs:
            raise ValueError(
                "I(transform=...) is for continuous nodes. An ordinal node's "
                "intercept is the cutpoint vector, it has no transform to choose."
            )

    def __repr__(self):
        """Show the levels and the terms."""
        return f"OrdinalNode({self.levels}, {self.terms!r})"

    def __eq__(self, other):
        """Compare levels and terms."""
        return (
            isinstance(other, OrdinalNode)
            and self.levels == other.levels
            and self.terms == other.terms
        )

    def __hash__(self):
        """Hash what ``__eq__`` compares, so nodes work in sets and as keys."""
        return hash((self.kind, self.levels, tuple(self.terms)))


# kept after the classes: the union is evaluated at definition time
NodeSpec = ContinuousNode | OrdinalNode


# %% alias -----------------------------------------------------------------------------
# The short names are the notation of the docs and the paper, and the
# spelling nearly every caller uses; they are the classes above, unchanged.
I = intercept = Intercept  # noqa: E741 - ambiguous only out of context
SI = simple_intercept
CI = complex_intercept
LS = linear_shift = LinearShift
CS = complex_shift = ComplexShift
VC = varying_coefficient = VaryingCoefficient
Fn = fn_shift = FnShift
