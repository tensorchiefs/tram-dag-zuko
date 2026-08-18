"""User-facing DAG specification.

A model is one dict ``{node_name: NodeSpec}``. Each node declares its
transformation ``h`` as an **additive formula of terms** — the first
positional argument, written as a list or as a ``+`` sum::

    "X3": ContinuousNode([I("X1"), CS("X2")])       # h = h_theta(x1) + g(x2)
    "X3": ContinuousNode(I("X1") + CS("X2"))        # the same, formula style

Term constructors name the parent(s) a term depends on. Each has a
pythonic name and a short alias — ``intercept``/``I``,
``linear_shift``/``LS``, ``complex_shift``/``CS`` and
``varying_coefficient``/``VC`` — and the two spellings are the same
object, so use whichever reads better:

- :func:`I`  — *intercept* term: the parent(s) reshape the monotone transform
  (its Bernstein coefficients / ordinal cutpoints). ``I()`` with no parent — or
  the bare name ``I`` — is the simple-intercept baseline (always present,
  optional to write). ``I(..., transform="spline")`` picks the basis of the
  monotone transform for a continuous node.
- :func:`LS` — *linear shift*: ``beta * x`` (one interpretable weight), one parent.
- :func:`CS` — *complex shift*: an additive MLP ``g(x)`` on the latent scale.
- :func:`VC` — *varying-coefficient shift*: ``beta(modifiers) * x_on`` with
  ``beta(x) = beta0 + b_theta(x)`` and ``b_theta`` a small, **penalized** network
  — a treatment-effect head with its own bias–variance budget (issue #28).

The intercept slot sums in coefficient space; the shift slot sums on the latent
scale. "Joint vs additive" is argument grouping: a multi-parent term such as
``CS("a","b")`` is one **joint** network over both parents (an interaction),
whereas ``CS("a") + CS("b")`` are two **additive** terms. For intercepts the
grouping is said explicitly: ``I("a", "b", allow_interaction=False)`` is the
additive intercept. Several ``I`` terms with parents on one node are an
error — the flag is the only way to say it, so a term list is always purely
additive on the latent scale.

What ``h`` looks like per transformation, for a continuous ``x3``:

======================================== =======================================
``transformation=``                      ``u_3 = h(x_3 | pa)``
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
``[I, CS("X1"), VC("T", "X2")]``         ``h_theta(x3) + g_1(x1) + beta(x2)*t``
======================================== =======================================

Each parent enters through exactly one *edge-owning* term (I/LS/CS parents, and
a VC term's ``on``). VC **modifiers** are exempt: ``CS("x2")`` + ``VC("t", "x2")``
is the intended pattern — ``x2`` acts prognostically through the shift *and*
modifies the treatment effect.
"""

from __future__ import annotations

from dataclasses import dataclass

EFFECTS = ("I", "LS", "CS", "VC")


# defaults of the effect-specific options; a constructor value equal to its
# default stays out of ``Term.options``, so term equality is canonical
_OPTION_DEFAULTS = {
    "penalty": None,  # VC: L2 weight on b_theta
    "center": False,  # VC: propensity centering
    "center_folds": 5,  # VC: folds of the out-of-fold refits
    "transform": None,  # I: basis of the monotone transform
    "transform_kwargs": None,  # I: kwargs of the basis, as sorted pairs
    "units": None,  # I/CS/VC: hidden layers of the term's network
    "allow_interaction": True,  # I: one joint net, or one net per parent
}


def _options(**kwargs) -> tuple:
    """Canonicalize effect-specific options: sorted pairs, defaults dropped."""
    return tuple(sorted((k, v) for k, v in kwargs.items() if v != _OPTION_DEFAULTS[k]))


@dataclass(frozen=True)
class Term:
    """One additive term of a node's transformation.

    Terms add: ``I("a") + CS("b")`` is the same transformation as
    ``[I("a"), CS("b")]``. Build terms with the constructors :func:`I`,
    :func:`LS`, :func:`CS` and :func:`VC`, not directly.

    Attributes
    ----------
    effect : str
        One of ``"I"``, ``"LS"``, ``"CS"``, ``"VC"``.
    parents : tuple[str, ...]
        Ordered parent names the term depends on. Empty only for the bare
        simple-intercept ``I()``. For a ``VC`` term, ``parents[0]`` is the
        treatment (``on``) and the rest are the effect modifiers.
    options : tuple[tuple[str, object], ...]
        Effect-specific settings as canonical ``(key, value)`` pairs:
        sorted by key, defaults omitted. Attribute access serves them
        with their defaults, so ``term.penalty`` stays valid on every
        term. Keys: ``penalty``, ``center`` and ``center_folds`` (VC,
        see :func:`VC`); ``transform`` and ``transform_kwargs`` (I, the
        basis of the monotone transform, kwargs stored as sorted pairs);
        ``units`` (hidden layers of the term's network);
        ``allow_interaction`` (multi-parent I: one joint net or one net
        per parent).
    """

    effect: str
    parents: tuple[str, ...]
    options: tuple = ()  # canonical (key, value) pairs, see _OPTION_DEFAULTS

    def __getattr__(self, name: str):
        """Serve the effect-specific options, with their defaults."""
        if name in _OPTION_DEFAULTS:
            return dict(self.options).get(name, _OPTION_DEFAULTS[name])
        raise AttributeError(name)

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


def intercept(
    *parents: str,
    allow_interaction: bool = True,
    transform: str | None = None,
    transform_kwargs: dict | None = None,
    units: list[int] | tuple[int, ...] | None = None,
) -> Term:
    """Build an intercept term: the parents reshape the monotone transform.

    ``I`` is the exported alias of this function, the notation of the docs
    and the paper. A call with no parents, or the bare name ``I``, is the
    simple-intercept baseline.

    Parameters
    ----------
    *parents : str
        Parent names. With several parents the term is one **joint**
        network (an interaction).
    allow_interaction : bool, optional
        ``False`` makes a multi-parent term **additive** instead: one
        network per parent, their parameter vectors summed in coefficient
        space. A node takes at most one intercept term with parents —
        write an additive intercept with this flag, not with several ``I``
        terms. Default ``True``.
    transform : str | None, optional
        Basis of a continuous node's monotone transform: ``"bernstein"``
        (default), ``"spline"`` or ``"affine"``. At most one ``I`` term
        per node can set it. An ordinal node accepts none, because its
        intercept is the cutpoint vector.
    transform_kwargs : dict | None, optional
        Keyword arguments, forwarded to the basis.
    units : list[int] | tuple[int, ...] | None, optional
        Hidden layers of the term's network, for example ``units=[16]``
        for one hidden layer of 16 neurons. Default ``[8, 8]``.

    Returns
    -------
    Term
        The intercept term.
    """
    kw = tuple(sorted(transform_kwargs.items())) if transform_kwargs else None
    return Term(
        "I",
        tuple(parents),
        _options(
            transform=transform,
            transform_kwargs=kw,
            units=tuple(units) if units is not None else None,
            allow_interaction=bool(allow_interaction) or len(parents) < 2,
        ),
    )


def linear_shift(*parents: str) -> Term:
    """Build a linear-shift term ``beta * x``.

    ``LS`` is the exported alias of this function, the notation of the
    docs and the paper.

    Parameters
    ----------
    *parents : str
        Exactly one parent name.

    Returns
    -------
    Term
        The linear-shift term.

    Raises
    ------
    ValueError
        If the parent count is not one.
    """
    if len(parents) != 1:
        raise ValueError("LS() takes exactly one parent.")
    return Term("LS", tuple(parents))


def complex_shift(
    *parents: str, units: list[int] | tuple[int, ...] | None = None
) -> Term:
    """Build a complex-shift term: an additive MLP ``g(x)``.

    ``CS`` is the exported alias of this function, the notation of the
    docs and the paper.

    Parameters
    ----------
    *parents : str
        At least one parent name. Several parents feed one joint network.
    units : list[int] | tuple[int, ...] | None, optional
        Hidden layers, for example ``units=[16]``. Default
        ``[64, 128, 64]``.

    Returns
    -------
    Term
        The complex-shift term.

    Raises
    ------
    ValueError
        If no parent is given.
    """
    if not parents:
        raise ValueError("CS() needs at least one parent.")
    return Term("CS", tuple(parents), _options(units=tuple(units) if units else None))


def varying_coefficient(
    *modifiers: str,
    t: str,
    penalty: float = 1.0,
    center: bool | str = False,
    center_folds: int = 5,
    units: list[int] | tuple[int, ...] | None = None,
) -> Term:
    """Build a varying-coefficient shift term ``beta(modifiers) * x_t``.

    ``VC`` is the exported alias of this function, the notation of the
    docs and the paper.

    This is the treatment-effect term of issue #28:
    ``VC("X2", "X3", t="T")`` is ``(beta0 + b_theta(x2, x3)) * x_t``.

    ``beta(x) = beta0 + b_theta(x)``, with ``b_theta`` a small MLP whose
    weights carry the L2 ``penalty``. The fitting objective is the
    penalized NLL ``sum_i nll_i + penalty * ||b_theta weights||^2`` on the
    total-likelihood scale. That is a fixed Gaussian prior whose shrinkage
    vanishes as n grows. ``beta0`` is not penalized.

    The output of ``b_theta`` is zero-initialized and, after the fit,
    mean-centered over the training data. ``beta0`` is therefore the
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
        >= 0.
    center : bool | str, optional
        Propensity centering (issue #30), by default ``False``.
        ``True`` uses the **propensity-centered** regressor
        ``beta(x) * (x_t - e_hat(pa_t))`` — the Robinson/R-learner
        orthogonalization inside the likelihood. Requires a binary ordinal
        ``t``. A string takes the training-time cross-fitted propensity
        from that column of ``train_df`` instead.
    center_folds : int, optional
        Fold count for the out-of-fold refits under ``center=True``, by
        default 5. Must be >= 2.
    units : list[int] | tuple[int, ...] | None, optional
        Hidden layers of ``b_theta``, by default ``[16]``.

    Returns
    -------
    Term
        The varying-coefficient term.

    Raises
    ------
    ValueError
        If ``t`` is also a modifier, if ``penalty`` is negative, or if
        ``center_folds`` is below 2.

    Notes
    -----
    With ``center=True``, training uses **out-of-fold** ``e_hat``:
    ``center_folds``-fold refits of the ``t`` node only, the DML
    requirement — in-sample centering can be *worse* than none. The values
    are frozen as data, so no gradient reaches the ``t`` node from this
    node's loss. Inference (``log_prob``/``sample``/``abduct``/``pmf``)
    recomputes ``e_hat`` from the flow's own fitted ``t`` node — the
    full-data fit, the standard DML train/predict split — and always
    re-derives ``x_t - e_hat`` under ``do``, never from a cache. With
    centering, ``beta0`` is the effect at the treatment margin (the
    observed propensities). The LS-nesting reading applies to the
    uncentered term only.
    """
    if t in modifiers:
        raise ValueError(
            f"VC(): '{t}' cannot be both the treatment (t) and a modifier."
        )
    if penalty < 0:
        raise ValueError(f"VC(): penalty must be >= 0, got {penalty}.")
    if center_folds < 2:
        raise ValueError(f"VC(): center_folds must be >= 2, got {center_folds}.")
    return Term(
        "VC",
        (t, *modifiers),
        _options(
            penalty=float(penalty),
            center=center,
            center_folds=int(center_folds),
            units=tuple(units) if units else None,
        ),
    )


def term(effect: str, *parents: str, penalty: float | None = None) -> Term:
    """Build a :class:`Term` from an effect label.

    Use this when the effect type comes from data, for example when a
    study sweeps ``"LS"`` against ``"CS"``.

    Parameters
    ----------
    effect : str
        One of ``"I"``, ``"LS"``, ``"CS"``, ``"VC"``.
    *parents : str
        Parent names. For ``"VC"`` the first name is the treatment and
        the rest are the modifiers.
    penalty : float | None, optional
        L2 penalty, ``"VC"`` only. When omitted, ``VC`` uses its own
        default.

    Returns
    -------
    Term
        The term.

    Raises
    ------
    ValueError
        If the label is unknown, or if ``penalty`` is given for a
        non-``VC`` effect.
    """
    if penalty is not None and effect != "VC":
        raise ValueError(f"term(): penalty only applies to 'VC', not '{effect}'.")
    if effect == "I":
        return intercept(*parents)
    if effect == "LS":
        return linear_shift(*parents)
    if effect == "CS":
        return complex_shift(*parents)
    if effect == "VC":
        kw = {} if penalty is None else {"penalty": penalty}
        return varying_coefficient(*parents[1:], t=parents[0], **kw)
    raise ValueError(f"unknown term effect '{effect}'.")


def _normalize_transformation(value):
    """Flatten a transformation into a plain term list.

    Accepted: ``None``, a single :class:`Term`, a ``+`` sum, the bare name
    ``I``, or a list/tuple that mixes all of these.
    """
    if value is None:
        return None
    if value is intercept:
        value = [intercept()]
    elif isinstance(value, Term):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            "transformation must be a Term, a sum of terms, the bare I, or a "
            f"list of these — got {type(value).__name__}."
        )
    out: list[Term] = []
    for element in value:
        if element is intercept:
            element = intercept()
        if isinstance(element, (list, tuple)):  # a nested `+` sum
            out.extend(_normalize_transformation(element))
        elif isinstance(element, Term):
            out.append(element)
        else:
            raise TypeError(
                "transformation entries must be terms (I/LS/CS/VC) — got "
                f"{type(element).__name__}. A '+' between list entries "
                "does not combine them; write either a list or a sum."
            )
    return out


def _check_intercepts(terms, *, ordinal: bool):
    """Validate the intercept slot and read the basis choice off it.

    A node takes at most one ``I`` term with parents — an additive
    intercept is said with ``allow_interaction=False`` on one term, not by
    listing several — and at most one ``I`` term may name the basis.

    Parameters
    ----------
    terms : list[Term] | None
        The node's normalized term list.
    ordinal : bool
        ``True`` for an ordinal node, whose intercept is the cutpoint
        vector and therefore has no basis to choose.

    Returns
    -------
    tuple[str, dict]
        The effective ``(transform, transform_kwargs)`` of the node.

    Raises
    ------
    ValueError
        If several ``I`` terms carry parents, if several set a basis, or if
        an ordinal node sets one.
    """
    i_terms = [t for t in terms or [] if t.effect == "I"]
    parented = [t for t in i_terms if t.parents]
    if len(parented) > 1:
        raise ValueError(
            "a node takes at most one I term with parents. For an additive "
            "intercept write I("
            + ", ".join(repr(p) for t in parented for p in t.parents)
            + ", allow_interaction=False)."
        )
    carriers = [t for t in i_terms if t.transform]
    if len(carriers) > 1:
        raise ValueError(
            "only one I term per node may set transform=, got "
            f"{[t.transform for t in carriers]}."
        )
    if carriers and ordinal:
        raise ValueError(
            "I(transform=...) is for continuous nodes. An ordinal node's "
            "intercept is the cutpoint vector, it has no basis to choose."
        )
    if carriers:
        return carriers[0].transform, dict(carriers[0].transform_kwargs or ())
    return "bernstein", {}


class ContinuousNode:
    """Continuous variable, modelled by a monotone 1-D transform + shifts.

    Parameters
    ----------
    transformation : Term | list[Term] | None, optional
        The additive formula for ``h``: a list of terms, a ``+`` sum, a single
        term, or the bare ``I``. ``None`` (default) is a source node. The
        basis of the monotone transform is chosen on the intercept term,
        ``I(..., transform="spline")``; the default is ``"bernstein"``.
    """

    kind = "continuous"

    def __init__(self, transformation=None):
        self.transformation = _normalize_transformation(transformation)
        self.transform, self.transform_kwargs = _check_intercepts(
            self.transformation, ordinal=False
        )

    def __repr__(self):
        """Show the transformation and the basis."""
        return f"ContinuousNode({self.transformation!r}, transform={self.transform!r})"

    def __eq__(self, other):
        """Compare transformation, basis and basis kwargs."""
        # transform/transform_kwargs are derived from transformation, so
        # equal transformations already imply an equal basis
        return (
            isinstance(other, ContinuousNode)
            and self.transformation == other.transformation
        )


class OrdinalNode:
    """Ordinal variable with ``levels`` ordered classes, stored 0 to levels-1.

    An ordered logit models it: increasing cutpoints plus the shift terms.

    Parameters
    ----------
    levels : int
        Number of ordered classes.
    transformation : Term | list[Term] | None, optional
        The additive formula, as for :class:`ContinuousNode`, by default
        ``None``.
    """

    kind = "ordinal"

    def __init__(self, levels: int, transformation=None):
        self.levels = int(levels)
        self.transformation = _normalize_transformation(transformation)
        _check_intercepts(self.transformation, ordinal=True)

    def __repr__(self):
        """Show the levels and the transformation."""
        return f"OrdinalNode({self.levels}, {self.transformation!r})"

    def __eq__(self, other):
        """Compare levels and transformation."""
        return (
            isinstance(other, OrdinalNode)
            and self.levels == other.levels
            and self.transformation == other.transformation
        )


NodeSpec = ContinuousNode | OrdinalNode


def node_terms(node: NodeSpec) -> list[Term]:
    """Give the canonical term list of a node.

    Parameters
    ----------
    node : NodeSpec
        The node specification.

    Returns
    -------
    list[Term]
        The terms. Empty for a source node.
    """
    if node.transformation is None:
        return []
    return list(node.transformation)


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
    for term in node_terms(node):
        for p in term.parents:
            seen.setdefault(p, None)
    return list(seen)


def validate_and_sort(spec: dict[str, NodeSpec]) -> list[str]:
    """Validate the spec and return a topological ordering of the nodes.

    Edge ownership: every parent must enter through exactly one
    edge-owning term. Edge-owning are all parents of I/LS/CS terms and
    the ``on`` of a VC term. VC *modifiers* are exempt — they can repeat
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
        If a term is malformed, a parent is unknown, a parent enters
        through more than one edge-owning term, an ordinal node has fewer
        than 2 levels, a VC treatment is unsupported, or the graph has a
        cycle.
    """
    for name, node in spec.items():
        seen: set[str] = set()
        for term in node_terms(node):
            if term.effect not in EFFECTS:
                raise ValueError(f"Node '{name}': unknown term effect '{term.effect}'.")
            if term.effect == "LS" and len(term.parents) != 1:
                raise ValueError(
                    f"Node '{name}': LS term must have exactly one parent."
                )
            for p in term.parents:
                if p not in spec:
                    raise ValueError(f"Node '{name}': unknown parent '{p}'.")
            if term.effect == "VC":
                if not term.parents:
                    raise ValueError(
                        f"Node '{name}': VC term needs a treatment parent."
                    )
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
                owners = (on,)
            else:
                owners = term.parents
            for p in owners:
                if p in seen:
                    raise ValueError(
                        f"Node '{name}': parent '{p}' appears in more than one "
                        "term. Each parent must enter through exactly one "
                        "edge-owning term. Only VC modifiers may repeat."
                    )
                seen.add(p)
        if isinstance(node, OrdinalNode) and node.levels < 2:
            raise ValueError(f"Node '{name}': ordinal levels must be >= 2.")

    # Kahn's algorithm over pa(x_i) = union of all term parents
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


def spec_to_dict(spec: dict[str, NodeSpec]) -> dict:
    """Give the serialized representation of a spec, for checkpoints.

    ``Term.options`` is already canonical — sorted by key, defaults
    dropped — so a term serializes as its three fields and nothing else.

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
                    "options": dict(t.options),
                }
                for t in node_terms(node)
            ],
        }
        if isinstance(node, OrdinalNode):
            d["levels"] = node.levels
        out[name] = d
    return out


def spec_from_dict(d: dict) -> dict[str, NodeSpec]:
    """Rebuild a spec from its serialized form.

    Parameters
    ----------
    d : dict
        The serialized spec, as produced by :func:`spec_to_dict`.

    Returns
    -------
    dict[str, NodeSpec]
        The node specification, keyed by node name.
    """
    spec: dict[str, NodeSpec] = {}
    for name, nd in d.items():
        terms = [
            Term(
                t["effect"],
                tuple(t["parents"]),
                tuple(sorted(t["options"].items())),
            )
            for t in nd["terms"]
        ] or None
        if nd["kind"] == "continuous":
            spec[name] = ContinuousNode(terms)
        else:
            spec[name] = OrdinalNode(int(nd["levels"]), terms)
    return spec


# The short aliases are the notation of the docs and the paper, and the
# spelling nearly every caller uses; the long names above are their
# definitions, so `I is intercept` and the bare `I` sugar keeps working.
I = intercept  # noqa: E741 - ambiguous only out of context
LS = linear_shift
CS = complex_shift
VC = varying_coefficient
