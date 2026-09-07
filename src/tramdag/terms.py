"""The term modules: one class per effect, built from the effect's spec class.

A spec term (:class:`tramdag.spec.Term` subclass — ``LS``, ``CS``, ``VC``,
``Fn``, ``I``) is frozen data and carries the spec-level rules; the module
here declares which term class it builds (``data = CS``) and owns the
runtime behaviour: ``build``, ``shift_value``/``theta_value``,
``post_init``, ``regularizer``, ``finalize``, ``score_columns`` and the
side-input contract. :func:`module_for` finds the module of a term by that
declaration, so subclassing is the whole registration.

A custom effect is two classes: a :class:`tramdag.spec.Term` subclass for
the options and checks, and a :class:`ShiftTerm` subclass with ``data =``
that term class, ``build`` and ``shift_value``.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from torch import Tensor, nn

from .conditioners import (
    ComplexIntercept,
    ComplexShift,
    LinearShift,
    SimpleIntercept,
    VaryingCoef,
)
from .spec import (
    CS,
    LS,
    VC,
    ContinuousNode,
    Fn,
    I,
    OrdinalNode,
    Term,
    _subclasses,
    feat_width,
)

if TYPE_CHECKING:
    import pandas as pd

    from .nodes import _Node
    from .spec import NodeSpec


# %% public functions ------------------------------------------------------------------
def module_for(term: Term) -> type[TermDef]:
    """Give the module class that builds ``term``.

    The class is the :class:`TermDef` subclass declaring ``data = type(term)``
    itself (not by inheritance).

    Raises
    ------
    ValueError
        If no module class declares the term's class.
    """
    for cls in _subclasses(TermDef):
        if cls.__dict__.get("data") is type(term):
            return cls
    raise ValueError(
        f"no module builds a {type(term).__name__} term. Subclass "
        f"tramdag.terms.ShiftTerm with `data = {type(term).__name__}` and "
        "implement build and shift_value."
    )


# %% private classes -------------------------------------------------------------------
class _InputTransform(nn.Module):
    """One term's frozen network-input transform.

    ``calibrate`` takes the statistics from the training rows once:
    ``"minmax"`` freezes per-column lo/hi, ``"standardize"`` mean/std, and a
    callable keeps the raw training columns and is applied per batch as
    ``fn(x, train)`` — so train statistics inside the callable are always the
    frozen training data, never the batch's.
    """

    def __init__(self, value, cols: tuple[str, ...]):
        super().__init__()
        self.kind = "callable" if callable(value) else value
        self.fn = value if callable(value) else None
        self.cols = cols  # the term's continuous parents, in parent order
        k = len(cols)
        if self.kind == "minmax":
            self.register_buffer("lo", torch.zeros(k))
            self.register_buffer("hi", torch.ones(k))
        elif self.kind == "standardize":
            self.register_buffer("mean", torch.zeros(k))
            self.register_buffer("std", torch.ones(k))
        else:  # callable: the raw train columns, shaped at calibrate
            self.register_buffer("train_cols", torch.zeros(0, k))

    def set_stats(self, cols: Tensor) -> None:
        """Freeze the statistics from the raw ``(n_train, k)`` train columns."""
        if self.kind == "minmax":
            self.lo.copy_(cols.min(0).values)
            self.hi.copy_(cols.max(0).values)
        elif self.kind == "standardize":
            self.mean.copy_(cols.mean(0))
            self.std.copy_(cols.std(0))
        else:
            self._buffers["train_cols"] = cols.detach().to(self.train_cols.device)

    def forward(self, x: Tensor, i: int) -> Tensor:
        """Transform one continuous parent column ``(n, 1)``."""
        if self.kind == "minmax":
            return (x - self.lo[i]) / (self.hi[i] - self.lo[i])
        if self.kind == "standardize":
            return (x - self.mean[i]) / self.std[i]
        return self.fn(x, self.train_cols[:, i : i + 1])


# %% private functions -----------------------------------------------------------------
def _attach_input_transform(m, term: Term, parents: tuple, spec: dict) -> None:
    """Register the term's input transform over its continuous parents.

    Ordinal one-hots pass through untransformed, so a term whose network
    parents are all ordinal carries none.
    """
    if term.input_transform is None:
        return
    cps = tuple(p for p in parents if isinstance(spec[p], ContinuousNode))
    if cps:
        m.add_module("_input_transform", _InputTransform(term.input_transform, cps))


# %% public classes --------------------------------------------------------------------
class TermDef:
    """What every term module shares: the input transform and its calibration.

    ``data`` names the :class:`tramdag.spec.Term` subclass the module
    builds; :func:`module_for` dispatches on it.
    """

    data: ClassVar[type[Term]]

    @property
    def input_transform(self):
        """The term's frozen network-input transform, or ``None``.

        Builds register one (``_attach_input_transform``) when the term
        declares ``input_transform=`` over continuous parents; a plain class
        attribute would shadow the registered submodule.
        """
        return (
            self._modules.get("_input_transform") if hasattr(self, "_modules") else None
        )

    def calibrate(self, train_df: pd.DataFrame) -> None:
        """Freeze this term's data-dependent state: the input-transform stats.

        ``CausalFlowDAG.calibrate`` calls this once per term; a term without
        an ``input_transform`` has nothing to freeze. The intercept slot
        overrides it with two extra arguments (its node's own column and
        transform), which the flow passes only there.
        """
        tr = self.input_transform
        if tr is None:
            return
        cols = torch.stack(
            [
                torch.as_tensor(train_df[p].to_numpy(dtype=np.float32).copy())
                for p in tr.cols
            ],
            dim=1,
        )
        tr.set_stats(cols)


class ShiftTerm(TermDef):
    """A shift term's behavior hooks, mixed into its conditioner.

    A built term instance carries ``key`` (its ModuleDict key), ``parents``
    (the term's written parents) and ``net_parents`` (the parents whose
    columns feed its *network* — empty for ``LS``, the modifiers for
    ``VC``); subclasses may add term-specific attributes (``VCTerm`` keeps
    ``mods``/``on_is_ord``/``center_col``). ``build`` constructs the module
    exactly as the node used to, so state-dict paths and the seeded RNG
    stream stay bit-stable.
    """

    scored: ClassVar[bool] = False  # True when score_columns gives coefficients
    finalizes = False  # set per instance when a post-fit step is needed

    key: str
    parents: tuple
    net_parents: tuple = ()  # the parents feeding the term's NETWORK

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> ShiftTerm:
        """Construct the term module from its spec Term."""
        raise NotImplementedError

    def shift_value(self, node: _Node, feats: dict) -> Tensor:
        """Give this term's contribution to the node's shift, shape ``(n,)``.

        ``feats`` holds the node's encoded parents plus this term's side
        columns (frozen from the data frame during training, injected live
        by the flow at query time).
        """
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

    def score_columns(self, node: _Node, flow, feats: dict, dlds) -> dict:
        """Give the per-observation score columns of this term's coefficients.

        Empty for a term with no interpretable coefficient (``CS``).
        """
        return {}

    def side_columns(self) -> tuple[str, ...]:
        """Name the data-frame columns this term needs beyond the parents.

        Training reads them from ``train_df`` like any other column (so the
        validation split and the minibatch slicing come for free).
        """
        return ()

    def check_column(self, node_name: str, col: str, values) -> None:
        """Validate one side column's values at fit time."""

    def live_side(self, flow, values: dict, n: int) -> dict:
        """Recompute the side columns from the fitted flow, at query time."""
        return {}

    def extra_columns(self, flow) -> list[str]:
        """List the extra columns queries must tensorize for :meth:`live_side`."""
        return []


class InterceptTerm(TermDef):
    """The intercept slot's behavior hooks, mixed into its module.

    A node has exactly one intercept term (normalization guarantees
    ``node.terms[0]``); it produces the transform parameters ``theta``.
    ``groups`` carries the parent groups — empty for a simple intercept,
    one tuple for a joint net, one per parent for an additive one — and
    ``ci_parents`` their flat order.
    """

    data = I
    has_marginal_start: ClassVar[bool] = False

    groups: list[tuple[str, ...]]
    ci_parents: list[str]

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec], n_params: int):
        """Construct the node's intercept module from its intercept Term."""
        if not term.parents:
            m = SITerm(n_params)
            m.groups, m.ci_parents = [], []
            return m
        groups = (
            [tuple(term.parents)]
            if term.allow_interaction
            else [(p,) for p in term.parents]
        )
        if len(groups) == 1:
            m = CITerm(
                feat_width(spec, groups[0]),
                n_params,
                units=term.units,
                activation=term.activation,
            )
        else:  # additive intercept: one net per parent, coefficients summed
            m = AdditiveCITerm(groups, n_params, spec, term.units, term.activation)
        m.groups = groups
        m.ci_parents = [p for grp in groups for p in grp]
        _attach_input_transform(m, term, tuple(term.parents), spec)
        return m

    def calibrate(self, train_df: pd.DataFrame, own=None, ut=None) -> None:
        """Freeze the input stats and set the transform's domain, once.

        ``own`` is the node's own training column and ``ut`` its monotone
        transform (``None`` for ordinal nodes — cutpoints have no domain):
        the train ``range_q``/``1 - range_q`` quantiles map onto the
        pre-scaled domain.
        """
        super().calibrate(train_df)
        if ut is None:
            return
        q = own.quantile([ut.range_q, 1.0 - ut.range_q])
        if q.iloc[1] <= q.iloc[0]:
            raise ValueError(
                f"node {own.name!r}: the {ut.range_q:.0%} and "
                f"{1 - ut.range_q:.0%} quantiles coincide at {q.iloc[0]}. A "
                "continuous node needs a spread of values; a level index "
                "needs OrdinalNode()."
            )
        ut.set_range(q.iloc[0], q.iloc[1])

    def theta_value(self, node: _Node, feats: dict, n: int) -> Tensor:
        """Give the transform parameters, shape ``(n, P)``."""
        raise NotImplementedError

    def marginal_start(self, theta: Tensor) -> None:
        """Set the calibrated marginal start (``has_marginal_start`` only)."""
        raise NotImplementedError


class SITerm(InterceptTerm, SimpleIntercept):
    """The free simple intercept: one theta vector, no parents."""

    has_marginal_start = True

    def theta_value(self, node: _Node, feats: dict, n: int) -> Tensor:
        """Broadcast the free theta over the batch."""
        return self(n)

    def marginal_start(self, theta: Tensor) -> None:
        """Start at the node's data marginal."""
        with torch.no_grad():
            self.theta.copy_(theta)


class CITerm(InterceptTerm, ComplexIntercept):
    """A single (possibly joint multi-parent) complex intercept net."""

    def theta_value(self, node: _Node, feats: dict, n: int) -> Tensor:
        """Run the one net over the joint parent features."""
        return self(node.net_input(feats, self.ci_parents, "@I"))


class AdditiveCITerm(InterceptTerm, nn.Module):
    """``allow_interaction=False``: one net per parent, outputs summed.

    Each parent reshapes the transform independently, in unconstrained
    coefficient space.
    """

    def __init__(self, groups, n_params: int, spec, units, activation):
        nn.Module.__init__(self)
        self.nets = nn.ModuleList(
            ComplexIntercept(
                feat_width(spec, grp), n_params, units=units, activation=activation
            )
            for grp in groups
        )

    def theta_value(self, node: _Node, feats: dict, n: int) -> Tensor:
        """Sum the per-parent nets in coefficient space."""
        return sum(
            net(node.net_input(feats, grp, "@I"))
            for net, grp in zip(self.nets, self.groups, strict=True)
        )


class LSTerm(ShiftTerm, LinearShift):
    """``LS`` — one raw-unit coefficient per (single) parent."""

    data = LS
    scored = True

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> LSTerm:
        """One weight per feature of the single parent; keyed by its name."""
        m = cls(feat_width(spec, term.parents))
        m.key = term.parents[0]
        m.parents = tuple(term.parents)
        return m

    def shift_value(self, node: _Node, feats: dict) -> Tensor:
        """Give the raw parent column times the weight — no input transform."""
        return self(torch.cat([feats[p] for p in self.parents], dim=1))

    def score_columns(self, node: _Node, flow, feats: dict, dlds) -> dict:
        """One column per weight: the parent (continuous) or its one-hot levels.

        ``d l_i / d beta = (d l_i / d s_i) * x_i`` — analytic and exact.
        """
        (parent,) = self.parents  # an LS term has exactly one parent
        psi = (dlds.unsqueeze(1) * feats[parent]).cpu().numpy()
        if isinstance(flow.spec[parent], OrdinalNode):
            return {f"{parent}[{k}]": psi[:, k] for k in range(psi.shape[1])}
        return {self.key: psi[:, 0]}


class CSTerm(ShiftTerm, ComplexShift):
    """``CS`` — a network shift over its parents."""

    data = CS

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> CSTerm:
        """One net over the concatenated parents; keyed 'a' or 'a+b'."""
        ps = tuple(term.parents)
        m = cls(feat_width(spec, ps), units=term.units, activation=term.activation)
        m.key = "+".join(ps)  # the parent itself for a single-parent term
        m.parents = ps
        m.net_parents = ps
        _attach_input_transform(m, term, ps, spec)
        return m

    def shift_value(self, node: _Node, feats: dict) -> Tensor:
        """Give the net over this term's (possibly input-transformed) features."""
        return self(node.net_input(feats, self.parents, self.key))


class VCTerm(ShiftTerm, VaryingCoef):
    """``VC`` — ``beta(modifiers) * x_t``; only the treatment owns an edge."""

    data = VC
    scored = True

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> VCTerm:
        """Build the effect head over the modifiers; keyed by the treatment name."""
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
        m.center_col = term.center or None
        m.finalizes = bool(mods)
        _attach_input_transform(m, term, mods, spec)
        return m

    def regressor(self, feats: dict) -> Tensor:
        """Give the ``(n, 1)`` column ``beta`` multiplies — the treatment, raw.

        The one-hot level-1 indicator for a binary ordinal treatment, the
        value itself for a continuous one; a centered term subtracts its
        propensity column (the Robinson regressor ``t - e_hat(x)``). It is
        also the score of ``beta0``, so :meth:`score_columns` reads it here.
        """
        if self.center_col and self.center_col not in feats:
            raise RuntimeError(
                f"centered VC term on {self.key!r} needs its propensity "
                f"column {self.center_col!r}. Internal callers inject it; "
                "never evaluate a centered term without its propensity."
            )
        t = feats[self.key][:, -1:] if self.on_is_ord else feats[self.key]
        if self.center_col:
            t = t - feats[self.center_col].view(-1, 1)
        return t

    def shift_value(self, node: _Node, feats: dict) -> Tensor:
        """``beta(modifiers) * regressor``, with the centered-term guard."""
        t = self.regressor(feats)
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

    def score_columns(self, node: _Node, flow, feats: dict, dlds) -> dict:
        """One column, keyed by the treatment: the ``beta0`` score.

        ``d s / d beta0`` is the term's own :meth:`regressor`, so forward
        and score share one definition by construction.
        """
        t = self.regressor(feats)
        return {self.key: (dlds * t.squeeze(-1)).cpu().numpy()}

    def side_columns(self) -> tuple[str, ...]:
        """Name the propensity column a centered term reads from the frame."""
        return (self.center_col,) if self.center_col else ()

    def check_column(self, node_name: str, col: str, values) -> None:
        """Propensities are probabilities."""
        if not ((values >= 0) & (values <= 1)).all():
            raise ValueError(
                f"column {col!r} (centered VC on node {node_name!r}) must "
                "hold probabilities in [0, 1]"
            )

    def live_side(self, flow, values: dict, n: int) -> dict:
        """Give the full-data propensity from the flow's own treatment node.

        Detached — no gradient reaches the treatment node from this node's
        loss — and derived from the current parent values, so
        ``do``-mutilated sampling centers with the intervened ``t`` (the DML
        prediction convention; training uses the frozen out-of-fold column).
        """
        if not self.center_col:
            return {}
        p1 = flow._binary_p1(flow.nodes[self.key], values, n).detach()
        return {self.center_col: p1}

    def extra_columns(self, flow) -> list[str]:
        """Give the treatment's parents (a treatment cannot be centered itself)."""
        return list(flow.nodes[self.key].parents) if self.center_col else []


class FnTerm(ShiftTerm, nn.Module):
    """``Fn`` — a user-supplied shift function over the parent features.

    A plain function contributes a fixed (non-trained) offset; an
    ``nn.Module`` registers as a submodule and trains with the flow. Built
    by :func:`tramdag.spec.fn_shift`.
    """

    data = Fn

    def __init__(self, fn):
        nn.Module.__init__(self)
        self.fn = fn  # an nn.Module registers as a submodule here

    @classmethod
    def build(cls, term: Term, spec: dict[str, NodeSpec]) -> FnTerm:
        """Wrap the callable; keyed like a CS ('a' or 'a+b')."""
        ps = tuple(term.parents)
        m = cls(term.fn)
        m.key = "+".join(ps)  # the parent itself for a single-parent term
        m.parents = ps
        m.net_parents = ps
        _attach_input_transform(m, term, ps, spec)
        return m

    def shift_value(self, node: _Node, feats: dict) -> Tensor:
        """Run ``fn`` on the term's features; accept ``(n,)`` or ``(n, 1)``."""
        out = self.fn(node.net_input(feats, self.parents, self.key))
        return out.squeeze(-1) if out.dim() > 1 else out
