"""User-facing DAG specification.

A model is one dict ``{node_name: NodeSpec}``. Each node declares its
transformation as an **additive formula of terms** (the native form), e.g.::

    "X3": ContinuousNode(terms=[I("X1"), CS("X2")])     # h = baseline + I(x1) + CS(x2)

Term constructors name the parent(s) a term depends on:

- :func:`I`  — *intercept* term: the parent(s) reshape the monotone transform
  (its Bernstein coefficients / ordinal cutpoints). ``I()`` with no parent is the
  implicit simple-intercept baseline (always present, optional to write).
- :func:`LS` — *linear shift*: ``beta * x`` (one interpretable weight), one parent.
- :func:`CS` — *complex shift*: an additive MLP ``g(x)`` on the latent scale.
- :func:`VC` — *varying-coefficient shift*: ``beta(modifiers) * x_on`` with
  ``beta(x) = beta0 + b_theta(x)`` and ``b_theta`` a small, **penalized** network
  — a treatment-effect head with its own bias–variance budget (issue #28).

The intercept slot sums in coefficient space; the shift slot sums on the latent
scale. "Joint vs additive" is just argument grouping — a multi-parent term such
as ``CS("a","b")`` is one **joint** network over both parents (an interaction),
whereas ``CS("a") + CS("b")`` are two **additive** terms.

Each parent enters through exactly one *edge-owning* term (I/LS/CS parents, and
a VC term's ``on``). VC **modifiers** are exempt: ``CS("x2")`` + ``VC("t", "x2")``
is the intended pattern — ``x2`` acts prognostically through the shift *and*
modifies the treatment effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# legacy dict term labels -> term effect (still accepted by ``term()`` and by the
# checkpoint loader, so old saved models keep loading)
_LEGACY = {"ls": "LS", "cs": "CS", "ci": "I"}

EFFECTS = ("I", "LS", "CS", "VC")


@dataclass(frozen=True)
class Term:
    """One additive term of a node's transformation.

    ``effect`` ∈ {"I", "LS", "CS", "VC"}; ``slot`` is "intercept" for ``I`` and
    "shift" for ``LS``/``CS``/``VC``. ``parents`` is the (ordered) tuple of parent
    names the term depends on — empty only for the bare simple-intercept ``I()``.
    For a ``VC`` term ``parents[0]`` is the treatment (``on``) and the rest are
    the effect modifiers; ``penalty`` is its L2 penalty weight (``None`` for
    every other effect).
    """

    effect: str
    slot: str
    parents: tuple[str, ...]
    penalty: float | None = None
    center: bool | str = False
    center_folds: int = 5


def I(*parents: str) -> Term:  # noqa: E743 - single-letter name is the intended notation
    """Intercept term — the parent(s) reshape the transform. ``I()`` = SI base."""
    return Term("I", "intercept", tuple(parents))


def LS(*parents: str) -> Term:
    """Linear shift ``beta * x`` — exactly one parent."""
    if len(parents) != 1:
        raise ValueError("LS() takes exactly one parent.")
    return Term("LS", "shift", tuple(parents))


def CS(*parents: str) -> Term:
    """Complex (MLP) shift — at least one parent."""
    if not parents:
        raise ValueError("CS() needs at least one parent.")
    return Term("CS", "shift", tuple(parents))


def VC(
    on: str,
    *modifiers: str,
    penalty: float = 1.0,
    center: bool | str = False,
    center_folds: int = 5,
) -> Term:
    """Build a varying-coefficient shift ``beta(modifiers) * x_on``.

    This is the treatment-effect term of issue #28.

    ``beta(x) = beta0 + b_theta(x)`` with ``b_theta`` a small MLP whose weights
    carry the L2 ``penalty``: the fitting objective is the penalized NLL
    ``sum_i nll_i + penalty * ||b_theta weights||^2`` (total-likelihood scale —
    a fixed Gaussian prior whose shrinkage vanishes as n grows; ``beta0``
    unpenalized). ``b_theta``'s output is zero-initialised and, after fitting,
    mean-centered over the training data, so ``beta0`` is the interpretable main
    effect (log-odds scale; the classical ``Colr``/``LS`` reading when ``beta``
    is constant). ``penalty -> inf`` — or ``modifiers=()`` exactly — reduces the
    term to ``LS(on)``, so VC-vs-LS is a nested question. Read the fitted effect
    out with :meth:`CausalFlowDAG.varying_coef`.

    ``on`` must be continuous or a binary (2-level) ordinal node; the term is
    linear in ``x_on``. Unlike other effects, VC *modifiers* may also appear in
    the node's prognostic terms (``CS``/``LS``/``I``) — only ``on`` owns its edge.

    ``center=True`` (issue #30) uses the **propensity-centered** regressor
    ``beta(x) * (x_on - e_hat(pa_on))`` — the Robinson/R-learner
    orthogonalization inside the likelihood; requires a binary ordinal ``on``.
    Training uses **out-of-fold** ``e_hat`` (``center_folds``-fold refits of the
    ``on`` node only — the DML requirement; in-sample centering can be *worse*
    than none), frozen as data so no gradient reaches the ``on`` node from this
    node's loss. Inference (``log_prob``/``sample``/``abduct``/``pmf``) recomputes
    ``e_hat`` from the flow's own fitted ``on`` node — the full-data fit, the
    standard DML train/predict split — and always re-derives ``x_on - e_hat``
    under ``do`` (never cached). ``center="colname"`` instead takes the
    training-time cross-fitted propensity from that column of ``train_df``.
    With centering, ``beta0`` is the effect at the treatment margin (the
    observed propensities); the LS-nesting reading applies to the uncentered
    term only.
    """
    if on in modifiers:
        raise ValueError(
            f"VC(): '{on}' cannot be both the treatment (on) and a modifier."
        )
    if penalty < 0:
        raise ValueError(f"VC(): penalty must be >= 0, got {penalty}.")
    if center_folds < 2:
        raise ValueError(f"VC(): center_folds must be >= 2, got {center_folds}.")
    return Term(
        "VC",
        "shift",
        (on, *modifiers),
        penalty=float(penalty),
        center=center,
        center_folds=int(center_folds),
    )


# explicit aliases (avoid confusion with the conditioner classes ComplexShift /
# ComplexIntercept, and give a non-single-letter option for I)
Intercept = I
LinShift = LS
CShift = CS


def term(effect: str, *parents: str, penalty: float | None = None) -> Term:
    """Build a :class:`Term` from an effect label.

    Use this when the effect type comes from data, for example when a study
    sweeps ``"ls"`` against ``"cs"``. The function accepts both the legacy
    labels ``"ls"``, ``"cs"`` and ``"ci"``, and the current labels ``"LS"``,
    ``"CS"``, ``"I"`` and ``"VC"``. ``penalty`` applies to ``"VC"`` only, and
    ``VC`` uses its own default when you omit it.
    """
    e = _LEGACY.get(effect.lower(), effect.upper())
    if penalty is not None and e != "VC":
        raise ValueError(f"term(): penalty only applies to 'VC', not '{effect}'.")
    if e == "I":
        return I(*parents)
    if e == "LS":
        return LS(*parents)
    if e == "CS":
        return CS(*parents)
    if e == "VC":
        return VC(*parents) if penalty is None else VC(*parents, penalty=penalty)
    raise ValueError(f"unknown term effect '{effect}'.")


@dataclass
class ContinuousNode:
    """Continuous variable, modelled by a monotone 1-D transform + shifts.

    Args:
        terms: additive formula, a list of :func:`I`/:func:`LS`/:func:`CS` terms
            (``None`` / omitted = a source node).
        transform: "bernstein" (TRAM-faithful), "spline" or "affine".
        transform_kwargs: forwarded to the transform.
    """

    terms: list[Term] | None = None
    transform: str = "bernstein"
    transform_kwargs: dict = field(default_factory=dict)
    kind: str = field(default="continuous", init=False)


@dataclass
class OrdinalNode:
    """Ordinal variable with ``levels`` ordered classes, stored 0 to levels-1.

    An ordered logit models it: increasing cutpoints plus the shift terms.
    """

    levels: int
    terms: list[Term] | None = None
    kind: str = field(default="ordinal", init=False)


NodeSpec = ContinuousNode | OrdinalNode


def node_terms(node: NodeSpec) -> list[Term]:
    """Canonical term list for a node (empty for a source node)."""
    return list(node.terms) if node.terms is not None else []


def node_parents(node: NodeSpec) -> list[str]:
    """Ordered, de-duplicated parent names referenced by a node's terms."""
    seen: dict[str, None] = {}
    for term in node_terms(node):
        for p in term.parents:
            seen.setdefault(p, None)
    return list(seen)


def validate_and_sort(spec: dict[str, NodeSpec]) -> list[str]:
    """Validate the spec and return a topological ordering of the nodes.

    Edge ownership: every parent must enter through exactly one edge-owning term
    (all parents of I/LS/CS terms; a VC term's ``on``). VC *modifiers* are exempt
    — they may repeat across terms (a modifier typically also acts prognostically
    through a CS/LS term).
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
                        f"{on_node.levels} levels; only binary (2-level) ordinal "
                        "treatments are supported (multi-level is a follow-up)."
                    )
                if term.center and not isinstance(on_node, OrdinalNode):
                    raise ValueError(
                        f"Node '{name}': VC(center=...) needs a binary ordinal "
                        f"treatment ('{on}' is continuous — E[T|x] centering is "
                        "a follow-up)."
                    )
                owners = (on,)
            else:
                owners = term.parents
            for p in owners:
                if p in seen:
                    raise ValueError(
                        f"Node '{name}': parent '{p}' appears in more than one term "
                        "(each parent must enter through exactly one edge-owning "
                        "term; only VC modifiers may repeat)."
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
    """JSON-serializable representation (for checkpoints)."""
    out = {}
    for name, node in spec.items():
        terms = [
            {
                "effect": t.effect,
                "parents": list(t.parents),
                **(
                    {
                        "penalty": t.penalty,
                        "center": t.center,
                        "center_folds": t.center_folds,
                    }
                    if t.effect == "VC"
                    else {}
                ),
            }
            for t in node_terms(node)
        ]
        d = {"kind": node.kind, "terms": terms}
        if isinstance(node, ContinuousNode):
            d["transform"] = node.transform
            d["transform_kwargs"] = dict(node.transform_kwargs)
        else:
            d["levels"] = node.levels
        out[name] = d
    return out


def _terms_from_dict(nd: dict) -> list[Term]:
    """Rebuild a term list from its serialized form.

    Both layouts are accepted: the current ``terms`` list and the legacy
    ``parents`` dict, so an old checkpoint still loads.
    """
    if "terms" in nd:
        ctor = {"I": I, "LS": LS, "CS": CS}
        return [
            VC(
                *t["parents"],
                penalty=t["penalty"],
                center=t.get("center", False),
                center_folds=t.get("center_folds", 5),
            )
            if t["effect"] == "VC"
            else ctor[t["effect"]](*t["parents"])
            for t in nd["terms"]
        ]
    # legacy checkpoint: {"parents": {parent: "ls"|"cs"|"ci"}}
    out: list[Term] = []
    for parent, label in nd.get("parents", {}).items():
        effect = _LEGACY[label]
        out.append(
            I(parent)
            if effect == "I"
            else (LS(parent) if effect == "LS" else CS(parent))
        )
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
        terms = _terms_from_dict(nd)
        if nd["kind"] == "continuous":
            spec[name] = ContinuousNode(
                terms=terms,
                transform=nd["transform"],
                transform_kwargs=dict(nd["transform_kwargs"]),
            )
        else:
            spec[name] = OrdinalNode(levels=int(nd["levels"]), terms=terms)
    return spec
