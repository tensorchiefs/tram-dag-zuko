"""User-facing DAG specification.

A model is one dict ``{node_name: NodeSpec}``. Each node declares its
transformation ``h`` as an **additive formula of terms** — the first
positional argument, written as a list or as a ``+`` sum::

    "X3": ContinuousNode([I("X1"), CS("X2")])       # h = h_theta(x1) + g(x2)
    "X3": ContinuousNode(I("X1") + CS("X2"))        # the same, formula style

Term constructors name the parent(s) a term depends on. Each has a
pythonic name and a short alias — ``intercept``/``I``,
``simple_intercept``/``SI``, ``complex_intercept``/``CI``,
``linear_shift``/``LS``, ``complex_shift``/``CS`` and
``varying_coefficient``/``VC`` — and the two spellings are the same
object, so use whichever reads better:

- :func:`I`  — *intercept* term: the parent(s) reshape the monotone transform
  (its Bernstein coefficients / ordinal cutpoints). ``I`` dispatches on its
  arguments: without parents it is the paper's simple intercept :func:`SI`
  (always present, optional to write — the bare names ``I`` and ``SI`` both
  work in a term list), with parents the complex intercept :func:`CI`.
  ``transform="spline"`` picks the basis of the monotone transform for a
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

from dataclasses import dataclass

# %% global variables ------------------------------------------------------------------
# bernstein, because zuko's spline extrapolates with a fixed slope outside
# [-B, B] while bernstein follows its own boundary derivative -- see the
# `transform` parameter of simple_intercept.
DEFAULT_TRANSFORM = "bernstein"


# %% private functions -----------------------------------------------------------------
def _option_defaults(effect: str) -> dict:
    """Give an effect's option defaults from its registry entry."""
    from .terms import get_term

    return get_term(effect).option_defaults


def _options(effect: str, **kwargs) -> tuple:
    """Canonicalize one effect's options: sorted pairs, defaults dropped.

    A key the effect does not take raises — a wrong-effect option must
    error instead of silently answering with a default.
    """
    defaults = _option_defaults(effect)
    unknown = sorted(set(kwargs) - set(defaults))
    if unknown:
        raise ValueError(
            f"effect '{effect}' takes no option(s) {unknown}; "
            f"it takes {sorted(defaults)}."
        )
    return tuple(sorted((k, v) for k, v in kwargs.items() if v != defaults[k]))


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

    The bare name ``I`` stands for ``I()``, the simple-intercept baseline.

    Raises
    ------
    TypeError
        If the entry is neither a term nor the bare ``I``.
    """
    if value in (intercept, simple_intercept):
        return simple_intercept()
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
    without one gets ``SI()`` prepended, so ``terms[0]`` is always the
    intercept term. Exactly one intercept is allowed, and it must come
    first when written.

    Parameters
    ----------
    value : Term | list[Term] | None
        The formula as written.

    Returns
    -------
    list[Term]
        The canonical term list; a source node gives ``[SI()]``.
    """
    if value is None:
        return [simple_intercept()]  # a source node: the free intercept alone
    written = value if isinstance(value, (list, tuple)) else [value]
    items = [_as_term(e) for e in written]
    intercept_at = [i for i, t in enumerate(items) if t.effect == "I"]
    if len(intercept_at) > 1:
        parented = [t for t in items if t.effect == "I" and t.parents]
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
        return [simple_intercept(), *items]  # canonical form: intercept first
    if intercept_at[0] != 0:
        raise ValueError(
            "the intercept term comes first: write "
            "I(...) + <shifts>, not the other way around."
        )
    return items


def _intercept_basis(terms, *, ordinal: bool):
    """Read the basis choice off the intercept term.

    Normalization guarantees exactly one intercept, at ``terms[0]``, so the
    basis has exactly one possible carrier.

    Parameters
    ----------
    terms : list[Term] | None
        The node's canonical term list.
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
        If an ordinal node's intercept configures a basis.
    """
    intercept_term = terms[0] if terms else None
    configured = intercept_term is not None and (
        intercept_term.transform or intercept_term.transform_kwargs
    )
    if configured and ordinal:
        raise ValueError(
            "I(transform=...) is for continuous nodes. An ordinal node's "
            "intercept is the cutpoint vector, it has no basis to choose."
        )
    if not configured:
        return DEFAULT_TRANSFORM, {}
    # the arguments go straight to the transform class; if they are wrong,
    # that class says so — this layer does not second-guess it
    name = intercept_term.transform or DEFAULT_TRANSFORM
    return name, dict(intercept_term.transform_kwargs or ())


def feat_width(spec: dict[str, NodeSpec], parents) -> int:
    """Total feature width of the parents (ordinal one-hot, continuous raw)."""
    return sum(
        spec[p].levels if isinstance(spec[p], OrdinalNode) else 1 for p in parents
    )


def _check_term(name: str, term: Term, spec: dict[str, NodeSpec]) -> tuple[str, ...]:
    """Validate one term of a node and give its edge-owning parents.

    The effect-specific rules (LS arity, the VC treatment/centering rules)
    live on the term's registry entry (:mod:`tramdag.terms`); the generic
    checks — the effect is known, parents exist, ``input_transform`` is
    well-formed — run here, in the original order.

    Parameters
    ----------
    name : str
        Name of the node the term belongs to, for the error messages.
    term : Term
        The term to validate.
    spec : dict[str, NodeSpec]
        The DAG specification, to look up parent nodes.

    Returns
    -------
    tuple[str, ...]
        The edge-owning parents: all parents of an I/LS/CS term, only the
        treatment of a VC term.

    Raises
    ------
    ValueError
        If the effect is unknown, an LS term has not exactly one parent,
        a parent is unknown, or a VC term is malformed.
    """
    from .terms import get_term

    try:
        entry = get_term(term.effect)
    except KeyError:
        raise ValueError(
            f"Node '{name}': unknown term effect '{term.effect}'. A custom "
            "effect must be registered (tramdag.terms.register_term) before "
            "the spec is built or loaded."
        ) from None
    entry.check_arity(name, term)
    if entry.slot == "intercept" and term.effect != "I":
        # normalization keys intercepts on effect "I": a custom intercept
        # slot would silently never build — refuse instead
        raise ValueError(
            f"Node '{name}': custom intercept-slot effects are not supported "
            f"yet — '{term.effect}' registers slot='intercept'. Custom terms "
            "are shifts (tramdag.terms.ShiftTerm)."
        )
    for p in term.parents:
        if p not in spec:
            raise ValueError(f"Node '{name}': unknown parent '{p}'.")
    _check_input_transform(name, term)
    return entry.edge_parents(name, term, spec)


def _check_input_transform(name: str, term: Term) -> None:
    """Reject a malformed ``input_transform`` before anything is built.

    Allowed: ``None``, ``"minmax"``, ``"standardize"``, or a callable
    ``fn(x, train)`` applied per continuous parent column (``train`` is
    that column's raw training data, frozen at ``calibrate``).
    """
    # read the raw options: an effect that does not take the key (LS) must
    # still be caught here, not silently ignored at build time
    value = dict(term.options).get("input_transform")
    if value is None:
        return
    if not (callable(value) or value in ("minmax", "standardize")):
        raise ValueError(
            f"Node '{name}': input_transform must be 'minmax', 'standardize' "
            f"or a callable fn(x, train), got {value!r}."
        )


def _check_node(name: str, node: NodeSpec, spec: dict[str, NodeSpec]) -> None:
    """Validate one node: its terms, edge ownership, and ordinal levels.

    Parameters
    ----------
    name : str
        Name of the node, for the error messages.
    node : NodeSpec
        The node specification.
    spec : dict[str, NodeSpec]
        The DAG specification, to look up parent nodes.

    Raises
    ------
    ValueError
        If a term is malformed, a parent enters through more than one
        edge-owning term, or an ordinal node has fewer than 2 levels.
    """
    seen: set[str] = set()
    for term in node.terms:
        for p in _check_term(name, term, spec):
            if p in seen:
                raise ValueError(
                    f"Node '{name}': parent '{p}' appears in more than one "
                    "term. Each parent must enter through exactly one "
                    "edge-owning term. Only VC modifiers may repeat."
                )
            seen.add(p)
    if isinstance(node, OrdinalNode) and node.levels < 2:
        raise ValueError(f"Node '{name}': ordinal levels must be >= 2.")


def _kahn_sort(spec: dict[str, NodeSpec]) -> list[str]:
    """Topologically sort the nodes with Kahn's algorithm.

    Dependencies are ``pa(x_i)``, the union of all term parents. Ready
    nodes are emitted in sorted batches, so the order is deterministic.

    Parameters
    ----------
    spec : dict[str, NodeSpec]
        The (already validated) DAG specification.

    Returns
    -------
    list[str]
        The node names in topological order.

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
def simple_intercept(
    transform: str | None = None, input_transform=None, **transform_kwargs
) -> Term:
    """Build the simple-intercept baseline term — the paper's SI.

    ``SI`` is the exported alias of this function. The term's transform
    parameters are a free vector, the same for every observation.

    Parameters
    ----------
    transform : str | None, optional
        Basis of a continuous node's monotone transform: ``"bernstein"``
        (default), ``"spline"`` or ``"affine"``. Bernstein is the default
        because zuko's spline extrapolates outside ``[-B, B]`` with a *fixed*
        slope, independent of the fitted parameters, so the ~10% of data
        beyond the 5%/95% pre-scaling range is misweighted whenever the true
        tail slope differs; Bernstein extrapolates linearly along its own
        boundary derivative. At most one intercept term per node can set it.
        An ordinal node accepts none, because its intercept is the cutpoint
        vector.
    **transform_kwargs
        Forwarded to the transform class, for example
        ``SI(transform="spline", bins=16)``.

    Returns
    -------
    Term
        The intercept term.
    """
    if input_transform is not None:
        raise ValueError(
            "a simple intercept has no network inputs — input_transform= "
            "belongs on CI/CS/VC terms."
        )
    kw = tuple(sorted(transform_kwargs.items())) or None
    return Term("I", (), _options("I", transform=transform, transform_kwargs=kw))


def complex_intercept(
    *parents: str,
    allow_interaction: bool = True,
    units: list[int] | tuple[int, ...] | None = None,
    activation: str | None = None,
    input_transform=None,
    transform: str | None = None,
    **transform_kwargs,
) -> Term:
    """Build a complex-intercept term — the paper's CI.

    ``CI`` is the exported alias of this function. The parents reshape the
    monotone transform: its parameters become a function of them.

    Parameters
    ----------
    *parents : str
        Parent names, at least one. With several parents the term is one
        **joint** network (an interaction).
    allow_interaction : bool, optional
        ``False`` makes a multi-parent term **additive** instead: one
        network per parent, their parameter vectors summed in coefficient
        space. A node takes at most one intercept term with parents —
        write an additive intercept with this flag, not with several
        intercept terms. Default ``True``: one joint network is what the
        reference implementations do, and the additive form is the variant
        added here.
    units : list[int] | tuple[int, ...] | None, optional
        Hidden layers of the term's network, for example ``units=[16]``
        for one hidden layer of 16 neurons. Default ``[8, 8]``, from the
        PyTorch reference — see :mod:`tramdag.conditioners`, which also
        explains why a paper replication sets this explicitly.
    transform : str | None, optional
        Basis of the node's monotone transform, as for
        :func:`simple_intercept`.
    **transform_kwargs
        Forwarded to the transform class.

    Returns
    -------
    Term
        The intercept term.

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
    kw = tuple(sorted(transform_kwargs.items())) or None
    return Term(
        "I",
        tuple(parents),
        _options(
            "I",
            transform=transform,
            transform_kwargs=kw,
            units=tuple(units) if units is not None else None,
            activation=activation,
            input_transform=input_transform,
            allow_interaction=bool(allow_interaction) or len(parents) < 2,
        ),
    )


def intercept(*parents: str, **kwargs) -> Term:
    """Build an intercept term, dispatching on the arguments.

    ``I`` is the exported alias of this function, the notation of the docs
    and the paper. Without parents it is :func:`simple_intercept`; with
    parents it is :func:`complex_intercept`. The bare name ``I`` in a term
    list stands for ``I()``.

    Parameters
    ----------
    *parents : str
        Parent names, forwarded to the matching constructor.
    **kwargs
        Forwarded to the matching constructor.

    Returns
    -------
    Term
        The intercept term.
    """
    if parents:
        return complex_intercept(*parents, **kwargs)
    return simple_intercept(**kwargs)


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
    *parents: str,
    units: list[int] | tuple[int, ...] | None = None,
    activation: str | None = None,
    input_transform=None,
) -> Term:
    """Build a complex-shift term: an additive NN ``g(x)``.

    ``CS`` is the exported alias of this function, the notation of the
    docs and the paper.

    Parameters
    ----------
    *parents : str
        At least one parent name. Several parents feed one joint network.
    units : list[int] | tuple[int, ...] | None, optional
        Hidden layers, for example ``units=[16]``. Default
        ``[64, 128, 64]``, from the PyTorch reference — see
        :mod:`tramdag.conditioners`.

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
    return Term(
        "CS",
        tuple(parents),
        _options(
            "CS",
            units=tuple(units) if units else None,
            activation=activation,
            input_transform=input_transform,
        ),
    )


def varying_coefficient(
    *modifiers: str,
    t: str,
    penalty: float = 1.0,
    center: str | bool = False,
    units: list[int] | tuple[int, ...] | None = None,
    activation: str | None = None,
    input_transform=None,
) -> Term:
    """Build a varying-coefficient shift term ``beta(modifiers) * x_t``.

    ``VC`` is the exported alias of this function, the notation of the
    docs and the paper.

    This is the treatment-effect term of issue #28:
    ``VC("X2", "X3", t="T")`` is ``(beta0 + b_theta(x2, x3)) * x_t``.

    ``beta(x) = beta0 + b_theta(x)``, with ``b_theta`` a small NN whose
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
        >= 0. 1.0 is the value at which ``tests/test_vc_term.py`` recovers
        the known ``beta(x)`` of the ``vc_hetero`` DGP at corr ~ 0.99. The
        penalty is on the total-NLL scale, so its effective strength moves
        with ``n``: raise it for small ``n`` or many modifiers.
    center : str | False, optional
        Propensity centering (issue #30), by default ``False``, which is
        bit-identical to the uncentered term — so a plain ``VC`` stays what
        it was before centering existed, and every committed number keeps
        reproducing. ``docs/varying-coefficients.md`` measures a 5-10x bias
        reduction from turning it on, so turn it on for an effect estimate.
        A string names the **training-frame column** holding the
        out-of-fold propensities ``P(t = 1 | pa_t)`` per row — compute them
        with any cross-fitted classifier OUTSIDE the flow and merge them as
        a column (in-sample values reintroduce the own-observation bias).
        The regressor becomes ``beta(x) * (x_t - e_hat(pa_t))`` — the
        Robinson/R-learner orthogonalization inside the likelihood; every
        query after the fit recomputes the propensity live from the flow's
        own treatment node. Requires a binary ordinal ``t``.
    units : list[int] | tuple[int, ...] | None, optional
        Hidden layers of ``b_theta``, by default ``[16]`` — see
        :class:`tramdag.conditioners.VaryingCoef` for why that size.
    activation : str | None, optional
        Activation of ``b_theta``'s hidden layers, by default the
        conditioners' ``relu``.

    Returns
    -------
    Term
        The varying-coefficient term.

    Raises
    ------
    ValueError
        If ``t`` is also a modifier or if ``penalty`` is negative.

    Notes
    -----
    With ``center="col"``, training reads **out-of-fold** ``e_hat`` for
    every row from that column of the training frame — the DML
    cross-fitting requirement; in-sample centering can be *worse* than
    none. The values
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
    return Term(
        "VC",
        (t, *modifiers),
        _options(
            "VC",
            penalty=float(penalty),
            center=center,
            units=tuple(units) if units else None,
            activation=activation,
            input_transform=input_transform,
        ),
    )


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
        _check_node(name, node, spec)
    return _kahn_sort(spec)


def spec_to_dict(spec: dict[str, NodeSpec]) -> dict:
    """Give the serialized representation of a spec, for checkpoints.

    ``Term.options`` is already canonical — sorted by key, defaults
    dropped — so a term serializes as its three fields and nothing else.
    The result is JSON- and YAML-safe: plain tuples become lists and
    nested kwargs tuples (``transform_kwargs``) become mappings, which is
    also how a hand-written YAML spec reads best; :func:`spec_from_dict`
    accepts both forms and turns them back, so a spec round-trips through
    ``json``/YAML as well as through ``torch.save`` — except when a term
    carries a *callable* ``input_transform``, which serializes only
    through pickle (``torch.save``) and only as a module-level function.

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
                    "options": {k: _mapped(v) for k, v in t.options},
                }
                for t in node.terms
            ],
        }
        if isinstance(node, OrdinalNode):
            d["levels"] = node.levels
        out[name] = d
    return out


def _check_dict_options(name: str, t: dict) -> None:
    """Reject stale or misspelled option keys in a serialized term.

    An unknown key would silently break term equality against a freshly
    constructed spec. An unknown *effect* passes through — validate_and_sort
    carries the register_term message.
    """
    try:
        known = set(_option_defaults(t["effect"]))
    except KeyError:
        return
    unknown = sorted(set(t["options"]) - known)
    if unknown:
        raise ValueError(
            f"node '{name}': effect '{t['effect']}' takes no option(s) "
            f"{unknown} — a stale or misspelled key would silently break "
            "term equality."
        )


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
        for t in nd["terms"]:
            _check_dict_options(name, t)
        terms = [
            Term(
                t["effect"],
                tuple(t["parents"]),
                tuple(sorted((k, _tupled(v)) for k, v in t["options"].items())),
            )
            for t in nd["terms"]
        ] or None
        if nd["kind"] == "continuous":
            spec[name] = ContinuousNode(terms)
        else:
            spec[name] = OrdinalNode(int(nd["levels"]), terms)
    return spec


def fn_shift(*parents: str, fn, input_transform=None) -> Term:
    """Give a custom function shift: ``fn(features)`` joins the additive shifts.

    The cheapest custom term: ``fn`` takes the term's concatenated parent
    features ``(n, k)`` (continuous raw, ordinal one-hot — through
    ``input_transform`` when given) and returns the shift contribution,
    shape ``(n,)`` or ``(n, 1)``. A plain function is a fixed offset; an
    ``nn.Module`` registers as a submodule and trains with the flow.

    Checkpoints pickle ``fn``, so it must be a module-level function or an
    importable ``nn.Module`` — ``save()`` refuses a lambda. For a whole new
    effect (own validation, penalty, side inputs) subclass
    :class:`tramdag.terms.ShiftTerm` and ``register_term`` it instead.

    Parameters
    ----------
    *parents : str
        Parent node names feeding ``fn``.
    fn : callable | torch.nn.Module
        The shift function.
    input_transform : str | callable | None, optional
        As on :func:`complex_shift`, by default None.

    Returns
    -------
    Term
        The term, effect ``"Fn"``.

    Raises
    ------
    ValueError
        If no parent is given or ``fn`` is not callable.
    """
    if not parents:
        raise ValueError("fn_shift needs at least one parent.")
    if not callable(fn):
        # a domain error (a wrong option value), not a Python type error
        raise ValueError(  # noqa: TRY004
            f"fn_shift(fn=) must be callable, got {type(fn).__name__}."
        )
    return Term(
        "Fn", tuple(parents), _options("Fn", fn=fn, input_transform=input_transform)
    )


# %% public classes --------------------------------------------------------------------
@dataclass(frozen=True)
class Term:
    """One additive term of a node's transformation.

    Terms add: ``I("a") + CS("b")`` is the same transformation as
    ``[I("a"), CS("b")]``. Build terms with the constructors :func:`I`,
    :func:`LS`, :func:`CS`, :func:`VC` and :func:`Fn`, not directly.

    Attributes
    ----------
    effect : str
        A registered effect name: ``"I"``, ``"LS"``, ``"CS"``, ``"VC"``,
        ``"Fn"``, or a :func:`tramdag.terms.register_term` custom name.
    parents : tuple[str, ...]
        Ordered parent names the term depends on. Empty only for the bare
        simple-intercept ``I()``. For a ``VC`` term, ``parents[0]`` is the
        treatment (``on``) and the rest are the effect modifiers; every
        other built-in term's parents all own their edges.
    options : tuple[tuple[str, object], ...]
        Effect-specific settings as canonical ``(key, value)`` pairs:
        sorted by key, defaults omitted. Attribute access serves this
        effect's options with their defaults; a key another effect takes
        raises ``AttributeError`` instead of answering with a foreign
        default. Keys per effect: ``penalty`` and ``center`` (VC, see
        :func:`VC`); ``transform`` and ``transform_kwargs`` (I, the basis
        of the monotone transform, kwargs stored as sorted pairs);
        ``units`` and ``activation`` (the term's network); ``fn`` (Fn, the
        custom shift callable); ``input_transform`` (I/CS/VC/Fn: the
        network-input transform); ``allow_interaction`` (multi-parent I:
        one joint net or one net per parent).
    """

    effect: str
    parents: tuple[str, ...]
    options: tuple = ()  # canonical (key, value) pairs; defaults dropped

    def __getattr__(self, name: str):
        """Serve this effect's options, with their defaults.

        A key another effect takes raises AttributeError instead of
        answering with a foreign default. Reads ``effect`` from __dict__:
        pickle/deepcopy probe dunders on an empty instance, and going
        through attribute access again would recurse.
        """
        effect = self.__dict__.get("effect")
        if effect is None:
            raise AttributeError(name)
        try:
            defaults = _option_defaults(effect)
        except KeyError:
            defaults = {}
        if name in defaults:
            return dict(self.options).get(name, defaults[name])
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


class ContinuousNode:
    """Continuous variable, modelled by a monotone 1-D transform + shifts.

    Parameters
    ----------
    terms : Term | list[Term] | None, optional
        The additive formula for ``h``: a list of terms, a ``+`` sum, a
        single term, or the bare ``I``. ``None`` (default) is a source node. The
        basis of the monotone transform is chosen on the intercept term,
        ``I(..., transform="spline")``; the default is ``"bernstein"``.
    """

    kind = "continuous"

    def __init__(self, terms=None):
        self.terms = _normalize_terms(terms)
        self.transform, self.transform_kwargs = _intercept_basis(
            self.terms, ordinal=False
        )

    def __repr__(self):
        """Show the terms and the basis."""
        return f"ContinuousNode({self.terms!r}, transform={self.transform!r})"

    def __eq__(self, other):
        """Compare the terms; the basis is derived from them."""
        # transform/transform_kwargs are derived from the terms, so equal
        # term lists already imply an equal basis
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
        self.terms = _normalize_terms(terms)
        _intercept_basis(self.terms, ordinal=True)

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
# The short aliases are the notation of the docs and the paper, and the
# spelling nearly every caller uses; the long names above are their
# definitions, so `I is intercept` and the bare `I` sugar keeps working.
I = intercept  # noqa: E741 - ambiguous only out of context
SI = simple_intercept
CI = complex_intercept
LS = linear_shift
Fn = fn_shift
CS = complex_shift
VC = varying_coefficient
