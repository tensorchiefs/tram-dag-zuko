"""CausalFlowDAG — a single triangular normalizing flow on a user-defined DAG.

The flow maps iid standard-logistic latents ``U`` to the observed variables ``X``
in topological order; its Jacobian sparsity is exactly the DAG adjacency. The
joint log-likelihood decomposes per node, so one optimizer fits all nodes at once.

Causal queries:
    flow.sample(n)                    observational sampling
    flow.sample(n, do={"T": 1})       interventional sampling (graph mutilation)
    u = flow.abduct(df)               Pearl step 1 (latents from observations)
    flow.sample(do={"T": 1}, u=u)     Pearl steps 2+3 (counterfactuals)
    flow.pmf(df, node, do=...)        analytic per-row interventional PMF
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import copy
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .conditioners import (
    ComplexIntercept,
    ComplexShift,
    LinearShift,
    SimpleIntercept,
    VaryingCoef,
)
from .scores import effect_modifier_scan as _effect_modifier_scan
from .scores import node_scores as _node_scores
from .spec import (
    LS,
    ContinuousNode,
    I,
    NodeSpec,
    OrdinalNode,
    node_parents,
    node_terms,
    spec_from_dict,
    spec_to_dict,
    validate_and_sort,
)
from .transforms import (
    RANGE_Q,
    BernsteinUT,
    StandardLogistic,
    make_univariate_transform,
    ordinal_abduct,
    ordinal_log_prob,
    ordinal_marginal_init_theta,
    ordinal_pmf,
    ordinal_sample,
)
from .utils import machine_info

# %% global variables ------------------------------------------------------------------
__all__ = ["CausalFlowDAG"]

logger = logging.getLogger(__name__)


# %% private functions -----------------------------------------------------------------
def _slice_ehat(
    vc_ehat: dict[str, dict[str, Tensor]] | None, idx: Tensor
) -> dict[str, dict[str, Tensor]] | None:
    """Slice the frozen out-of-fold propensities down to one minibatch."""
    if vc_ehat is None:
        return None
    return {nm: {on: e[idx] for on, e in d.items()} for nm, d in vc_ehat.items()}


def _feat_width(spec: dict[str, NodeSpec], parents) -> int:
    """Total feature width of the parents (ordinal one-hot, continuous raw)."""
    return sum(
        spec[p].levels if isinstance(spec[p], OrdinalNode) else 1 for p in parents
    )


def _term_cells(term) -> list[tuple[str, str]]:
    """Give a term's adjacency cells as ``(parent, tag)`` pairs.

    A VC term tags its treatment cell ``"VC"`` and its modifiers ``"VCm"``.
    A multi-parent term carries its parent group as a suffix.
    """
    if term.effect == "VC":
        return [(term.parents[0], "VC")] + [(p, "VCm") for p in term.parents[1:]]
    tag = "CI" if term.effect == "I" else term.effect
    if len(term.parents) > 1:
        tag = f"{tag}{list(term.parents)}"
    return [(p, tag) for p in term.parents]


def _covered_by_classical(term) -> bool:
    """Say whether the exact classical fit handles this term.

    It handles an ``LS``, and a parentless ``I()`` — the simple-intercept
    baseline made explicit, for example as the carrier of ``transform=``.
    """
    return term.effect == "LS" or (term.effect == "I" and not term.parents)


# %% private classes -------------------------------------------------------------------
class _FitSchedule:
    """Per-node plateau decay and freezing bookkeeping for one ``fit`` call.

    The per-node losses have independent gradients, so per-node learning
    rates and freezing are exactly equivalent to independent per-node
    training. ``step`` runs once per epoch on the validation NLLs.
    """

    def __init__(
        self,
        order: list[str],
        schedule: str | None,
        learning_rate: float,
        plateau_patience: int,
        plateau_factor: float,
        freeze_patience: int | None,
        min_delta: float,
        min_lr: float | None = None,
    ):
        self.schedule = schedule
        self.lr = learning_rate
        self.min_lr = learning_rate * 1e-3 if min_lr is None else min_lr
        self.patience = plateau_patience
        self.factor = plateau_factor
        self.freeze_patience = freeze_patience
        self.min_delta = min_delta
        self.best = dict.fromkeys(order, float("inf"))
        self.bad = dict.fromkeys(order, 0)
        self.frozen: set[str] = set()

    def step(self, opt, val_per_node: dict[str, float], history: dict) -> None:
        """Decay plateaued learning rates and freeze converged nodes."""
        for g in opt.param_groups:
            name = g["node"]
            if name in self.frozen:
                continue
            if val_per_node[name] < self.best[name] - self.min_delta:
                self.best[name] = val_per_node[name]
                self.bad[name] = 0
            else:
                self.bad[name] += 1
            if (
                self.schedule == "plateau"
                and self.bad[name] > 0
                and self.bad[name] % self.patience == 0
            ):
                g["lr"] = max(g["lr"] * self.factor, self.min_lr)
            # under "plateau", only freeze nodes whose lr has already been
            # decayed substantially — otherwise a node can freeze while a
            # smaller lr would still make progress toward the optimum
            lr_decayed = self.schedule != "plateau" or g["lr"] <= self.lr * 1e-2 * (
                1 + 1e-9
            )
            if (
                self.freeze_patience is not None
                and lr_decayed
                and self.bad[name] >= self.freeze_patience
            ):
                self.frozen.add(name)
                history.setdefault("frozen", {}).setdefault(
                    name, len(history["val"])
                )  # 1-based global epoch


class _VCGroup(NamedTuple):
    """Bookkeeping for one VC term of a node.

    Attributes
    ----------
    on : str
        Name of the treatment node. The VC term owns this edge.
    mods : tuple[str, ...]
        Names of the effect-modifier nodes. Empty for a constant effect.
    on_is_ord : bool
        ``True`` when the treatment is a binary ordinal node. The raw
        treatment column is then the level-1 one-hot indicator.
    center : bool
        ``True`` centers the regressor with out-of-fold propensities from
        refits of the treatment node.
    folds : int
        Number of folds for those refits.
    """

    on: str
    mods: tuple[str, ...]
    on_is_ord: bool
    center: bool
    folds: int


class _Node(nn.Module):
    """One dimension of the flow: an intercept plus additive shift terms.

    The intercept produces the transform parameters ``theta``. The shift
    terms add up on the latent scale.

    Parameters
    ----------
    node : NodeSpec
        Specification of the node.
    spec : dict[str, NodeSpec]
        The full DAG specification. Needed for the parent feature widths.
    """

    def __init__(self, node: NodeSpec, spec: dict[str, NodeSpec]):
        super().__init__()
        self.kind = node.kind
        terms = node_terms(node)
        self.parents = tuple(node_parents(node))  # ordered parent names
        i_term = next((t for t in terms if t.effect == "I" and t.parents), None)
        if i_term is None:
            i_groups = []
        elif i_term.allow_interaction:
            i_groups = [tuple(i_term.parents)]
        else:  # additive intercept: one net per parent, coefficients summed
            i_groups = [(p,) for p in i_term.parents]
        self._intercept_groups = i_groups
        self.ci_parents = [
            p for grp in i_groups for p in grp
        ]  # flat, for introspection

        if isinstance(node, ContinuousNode):
            self.ut = make_univariate_transform(node.transform, **node.transform_kwargs)
            n_params = self.ut.n_params
            self.levels = None
        else:
            self.ut = None
            self.levels = node.levels
            n_params = node.levels - 1
        self._build_intercept(i_term, n_params, spec)
        self._build_shifts(terms, spec)

    def _build_intercept(self, i_term, n_params: int, spec: dict[str, NodeSpec]):
        """Build the intercept module(s) from the intercept groups.

        By group count: none -> the free SimpleIntercept theta_0; one (a
        single parent, or a joint multi-parent term) -> one ComplexIntercept
        that IS theta; several (``allow_interaction=False``) -> one net per
        parent, their outputs summed in unconstrained coefficient space, so
        each parent reshapes the transform independently.
        """
        i_groups = self._intercept_groups
        if not i_groups:
            self.intercept = SimpleIntercept(n_params)
            self.intercept_nets = None
        elif len(i_groups) == 1:
            self.intercept = ComplexIntercept(
                _feat_width(spec, i_groups[0]),
                n_params,
                units=i_term.units,
                activation=i_term.activation,
            )
            self.intercept_nets = None
        else:
            self.intercept = None
            self.intercept_nets = nn.ModuleList(
                ComplexIntercept(
                    _feat_width(spec, grp),
                    n_params,
                    units=i_term.units,
                    activation=i_term.activation,
                )
                for grp in i_groups
            )

    def _build_shifts(self, terms, spec: dict[str, NodeSpec]):
        """Build one shift network per LS/CS/VC term.

        Single-parent terms key the ModuleDict by the parent name (so
        ls_coefficients/introspection keep working); a joint CS over several
        parents keys by "a+b" and runs over their concatenated features. A VC
        term keys by its treatment (on) name — validation guarantees ``on``
        owns that edge — and carries (on, modifiers, on-is-ordinal) in
        ``_vc_groups``.
        """
        self.shifts = nn.ModuleDict()
        self._shift_groups: list[tuple[str, tuple[str, ...]]] = []
        self._vc_groups: list[_VCGroup] = []
        for term in terms:
            if term.effect == "VC":
                on, mods = term.parents[0], tuple(term.parents[1:])
                self.shifts[on] = VaryingCoef(
                    _feat_width(spec, mods),
                    penalty=term.penalty,
                    units=term.units,
                    activation=term.activation,
                )
                self._vc_groups.append(
                    _VCGroup(
                        on,
                        mods,
                        isinstance(spec[on], OrdinalNode),
                        term.center,
                        term.center_folds,
                    )
                )
            elif term.effect in ("LS", "CS"):
                ps = tuple(term.parents)
                key = ps[0] if len(ps) == 1 else "+".join(ps)
                feat_width = _feat_width(spec, ps)
                self.shifts[key] = (
                    LinearShift(feat_width)
                    if term.effect == "LS"
                    else ComplexShift(
                        feat_width, units=term.units, activation=term.activation
                    )
                )
                self._shift_groups.append((key, ps))

    def vc_column(self, g, feats: dict, vc_ehat: dict | None) -> Tensor:
        """Give the ``(n, 1)`` regressor a VC term multiplies its ``beta`` by.

        The treatment enters raw: the one-hot level-1 indicator for a binary
        ordinal treatment, the value itself for a continuous one. A centered
        term subtracts the propensity, which is the Robinson regressor
        ``t - e_hat(x)``. It is also the score of ``beta0``, so
        :mod:`tramdag.scores` reads it from here.
        """
        t = feats[g.on][:, -1:] if g.on_is_ord else feats[g.on]
        if g.center:
            t = t - vc_ehat[g.on].view(-1, 1)
        return t

    def _theta(self, feats: dict[str, Tensor], n: int) -> Tensor:
        """Evaluate the intercept: the transform parameters, shape ``(n, P)``."""
        if self.intercept_nets is not None:  # additive complex intercept
            return sum(
                net(torch.cat([feats[p] for p in grp], dim=1))
                for net, grp in zip(
                    self.intercept_nets, self._intercept_groups, strict=True
                )
            )
        if self.ci_parents:  # single or joint complex intercept
            return self.intercept(torch.cat([feats[p] for p in self.ci_parents], dim=1))
        return self.intercept(n)  # simple (free) intercept

    def _vc_shift(self, g, feats: dict, vc_ehat: dict | None) -> Tensor:
        """Give one VC term's contribution to the shift, shape ``(n,)``."""
        if g.center and (vc_ehat is None or g.on not in vc_ehat):
            raise RuntimeError(
                f"centered VC term on {g.on!r} needs e_hat. Internal "
                "callers must supply vc_ehat. Never evaluate a centered "
                "term without its propensity."
            )
        t = self.vc_column(g, feats, vc_ehat)
        mod_feat = torch.cat([feats[p] for p in g.mods], dim=1) if g.mods else None
        return self.shifts[g.on](t, mod_feat)

    def theta_shift(
        self, feats: dict[str, Tensor], n: int, vc_ehat: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute the transform parameters and the total shift of the node.

        Parameters
        ----------
        feats : dict[str, Tensor]
            Encoded parent features, keyed by parent name.
        n : int
            Batch size.
        vc_ehat : dict[str, Tensor] | None, optional
            Propensity ``e_hat(pa_on)`` per centered VC treatment, keyed by
            treatment name. Required whenever a term has ``center``.
            Training passes the frozen out-of-fold values. Inference passes
            the live full-fit values.

        Returns
        -------
        tuple[Tensor, Tensor]
            The transform parameters, shape ``(n, P)``, and the total
            shift, shape ``(n,)``.

        Raises
        ------
        RuntimeError
            If a centered VC term is evaluated without its propensity.
        """
        theta = self._theta(feats, n)
        shift = torch.zeros(n, dtype=theta.dtype, device=theta.device)
        for key, ps in self._shift_groups:
            feat = (
                feats[ps[0]]
                if len(ps) == 1
                else torch.cat([feats[p] for p in ps], dim=1)
            )
            shift = shift + self.shifts[key](feat)
        for g in self._vc_groups:
            shift = shift + self._vc_shift(g, feats, vc_ehat)
        return theta, shift


# %% public classes --------------------------------------------------------------------
class CausalFlowDAG(nn.Module):
    """A causal normalizing flow defined by a DAG specification.

    Parameters
    ----------
    spec : dict[str, NodeSpec]
        The DAG specification, ``{name: ContinuousNode | OrdinalNode}``.
    device : str, optional
        Torch device, by default ``"cpu"``.
    seed : int | None, optional
        If given, seeds the weight initialization deterministically
        (``torch.manual_seed`` runs before the nodes are built). Weight
        initialization happens at construction, so this is the one knob
        for a reproducible model. ``fit(seed=...)`` only seeds the
        minibatch shuffling.
    """

    def __init__(
        self, spec: dict[str, NodeSpec], device: str = "cpu", seed: int | None = None
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.spec = spec
        self.order = validate_and_sort(spec)
        self.nodes = nn.ModuleDict(
            {name: _Node(spec[name], spec) for name in self.order}
        )
        self.device = torch.device(device)
        self.history: dict = {"train": [], "val": [], "lr": [], "time": []}
        self.meta: dict = {}  # provenance attached at save() (machine, versions)
        self.vc_center_info: dict = {}  # OOF bookkeeping of centered VC terms (fit)
        self.to(self.device)

    def _encode_parent(self, name: str, values: Tensor) -> Tensor:
        """Encode the values of a node for use as a parent feature.

        This follows the original TRAM-DAG convention. A continuous parent stays
        raw, shape ``(n, 1)``. An ordinal parent is one-hot encoded, shape
        ``(n, levels)``.
        """
        node = self.spec[name]
        if isinstance(node, OrdinalNode):
            return torch.nn.functional.one_hot(
                values.long(), num_classes=node.levels
            ).to(values.dtype)
        return values.view(-1, 1)

    @property
    def _dtype(self) -> torch.dtype:
        """Current model dtype (float32 normally; float64 inside fit_classical)."""
        return next(self.parameters()).dtype

    @property
    def _np_dtype(self) -> type:
        """Numpy dtype that matches the current model dtype."""
        return np.float64 if self._dtype == torch.float64 else np.float32

    def _tensorize(
        self, df: pd.DataFrame, cols: list[str] | tuple[str, ...] | None = None
    ) -> dict[str, Tensor]:
        """DataFrame columns -> one ``(n,)`` tensor each, in the model dtype.

        ``cols=None`` takes every node, in topological order.
        """
        dtype = self._np_dtype
        return {
            c: torch.as_tensor(
                df[c].to_numpy(dtype=dtype, copy=True), device=self.device
            )
            for c in (self.order if cols is None else cols)
        }

    def _generator(self, seed: int | None) -> torch.Generator | None:
        """Give a seeded generator on this flow's device, or None for unseeded."""
        if seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(seed)

    @torch.no_grad()
    def _binary_p1(self, nd: _Node, values: dict[str, Tensor], n: int) -> Tensor:
        """Give ``P(node = 1 | parents)`` for a binary ordinal node.

        ``P(x <= 0) = sigmoid(theta_0 - s)``, so the answer is
        ``sigmoid(s - theta_0)``. No ``vc_ehat``: chained centering is refused
        at fit time, so a treatment node never carries a centered term itself.
        """
        feats = self._features({p: values[p] for p in nd.parents})
        theta, shift = nd.theta_shift(feats, n)
        return torch.sigmoid(shift - theta[:, 0])

    def _node(self, name: str) -> _Node:
        """Look a node up by name, with the same error everywhere."""
        if name not in self.nodes:
            raise KeyError(f"unknown node {name!r}")
        return self.nodes[name]

    def _features(self, values: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: self._encode_parent(name, vals) for name, vals in values.items()}

    def _vc_ehat_live(
        self, nd: _Node, values: dict[str, Tensor], n: int
    ) -> dict[str, Tensor] | None:
        """Recompute ``e_hat(pa_on) = P(on = 1 | pa_on)`` for the centered VC terms.

        The value comes from this flow's own fitted ``on`` node, as a full-data
        propensity fit. That is the DML prediction convention. Training uses
        frozen out-of-fold values instead, see :meth:`fit`.

        The result is detached, so no gradient reaches the ``on`` node from the
        loss of this node.

        The function derives the value from the current parent values, so
        ``do``-mutilated sampling uses ``t - e_hat(x)`` with the intervened ``t``
        and the observed ``x``. It never reads a cached value.
        """
        out = {}
        for g in nd._vc_groups:
            if not g.center:
                continue
            out[g.on] = self._binary_p1(self.nodes[g.on], values, n).detach()
        return out or None

    def _vc_ehat_columns(self, nd: _Node) -> list[str]:
        """List the extra columns needed for the centered VC terms of ``nd``.

        These are the columns beyond ``nd.parents``: the parents of the
        treatment nodes (which cannot be centered themselves, so one level
        is all there is).
        """
        cols = [p for g in nd._vc_groups if g.center for p in self.nodes[g.on].parents]
        return [c for c in dict.fromkeys(cols) if c not in nd.parents]

    def node_log_prob(
        self,
        values: dict[str, Tensor],
        nodes: list[str] | None = None,
        vc_ehat: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Compute the per-node log-likelihood contributions.

        Parameters
        ----------
        values : dict[str, Tensor]
            Raw node values, keyed by node name, each shape ``(n,)``.
        nodes : list[str] | None, optional
            Restrict the computation to these nodes. ``fit`` uses this to
            skip frozen nodes. That is valid because the per-node losses
            are independent. ``None`` (default) computes every node.
        vc_ehat : dict[str, dict[str, Tensor]] | None, optional
            Propensity override for centered VC terms, as
            ``{node: {on: e_hat}}``. ``fit`` passes the frozen
            **out-of-fold** values for the training rows. When omitted,
            the live full-fit propensity is recomputed from the flow's
            own treatment node.

        Returns
        -------
        dict[str, Tensor]
            One log-likelihood tensor per node, each shape ``(n,)``.
        """
        feats = self._features(values)
        n = next(iter(values.values())).shape[0]
        out = {}
        for name in self.order if nodes is None else nodes:
            node = self.nodes[name]
            ehat = (
                vc_ehat.get(name)
                if vc_ehat is not None
                else self._vc_ehat_live(node, values, n)
            )
            theta, shift = node.theta_shift(feats, n, vc_ehat=ehat)
            x = values[name]
            if node.kind == "continuous":
                u0, ladj = node.ut.forward(theta, x)
                u = u0 + shift
                out[name] = StandardLogistic.log_prob(u) + ladj
            else:
                out[name] = ordinal_log_prob(theta, shift, x)
        return out

    def log_prob(self, df: pd.DataFrame) -> Tensor:
        """Compute the joint log-likelihood per row.

        Parameters
        ----------
        df : pd.DataFrame
            Observations, one column per node.

        Returns
        -------
        Tensor
            ``log p(x)`` per row, shape ``(n,)``.
        """
        per_node = self.node_log_prob(self._tensorize(df))
        return torch.stack(list(per_node.values()), dim=0).sum(dim=0)

    def nll(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute the mean negative log-likelihood per node (a diagnostic).

        Parameters
        ----------
        df : pd.DataFrame
            Observations, one column per node.

        Returns
        -------
        dict[str, float]
            The mean NLL, keyed by node name.
        """
        with torch.no_grad():
            per_node = self.node_log_prob(self._tensorize(df))
        return {k: float(-v.mean()) for k, v in per_node.items()}

    def _set_ranges(self, train_df: pd.DataFrame, marginal_init: bool = True) -> None:
        """Map the train ``RANGE_Q``/1-``RANGE_Q`` quantiles onto the domain.

        This is the min-max scaling of the original implementation.

        ``marginal_init``: calibrated Bernstein init (see ``fit``). Applied only
        on the first fit (the same ``not ut._fitted`` guard as range-setting), so a
        multi-phase fit does not reset a partially-trained intercept.
        """
        for name in self.order:
            node = self.nodes[name]
            if node.kind == "continuous" and not node.ut._fitted:
                q = train_df[name].quantile([RANGE_Q, 1.0 - RANGE_Q])
                node.ut.set_range(q.iloc[0], q.iloc[1])
                if (
                    marginal_init
                    and isinstance(node.ut, BernsteinUT)
                    and isinstance(node.intercept, SimpleIntercept)
                ):
                    with torch.no_grad():
                        node.intercept.theta.copy_(node.ut.marginal_init_theta())
            elif (
                node.kind == "ordinal"
                and marginal_init
                and isinstance(node.intercept, SimpleIntercept)
                and not getattr(node.intercept, "_marginal_inited", False)
            ):
                # calibrate unconditional cutpoints to the marginal class log-odds
                counts = np.bincount(
                    train_df[name].to_numpy().astype(np.int64),
                    minlength=self.spec[name].levels,
                )
                with torch.no_grad():
                    node.intercept.theta.copy_(ordinal_marginal_init_theta(counts))
                node.intercept._marginal_inited = True

    def _best_store(self, restore_best: bool) -> dict | None:
        """Give the cross-call best-weights store, creating it on first use."""
        if not restore_best:
            return None
        if not hasattr(self, "_best"):
            self._best = {name: (float("inf"), None) for name in self.order}
        return self._best

    def _make_optimizer(self, learning_rate: float) -> torch.optim.Adam:
        """Build Adam with one parameter group per node, tagged by name."""
        return torch.optim.Adam(
            [
                {
                    "params": list(self.nodes[name].parameters()),
                    "lr": learning_rate,
                    "node": name,
                }
                for name in self.order
            ]
        )

    def _val_nll(self, val_vals: dict[str, Tensor]) -> dict[str, float]:
        """Give the mean validation NLL per node, for the epoch monitor."""
        with torch.no_grad():
            return {
                k: float(-v.mean()) for k, v in self.node_log_prob(val_vals).items()
            }

    def _vc_penalized(self) -> dict[str, list]:
        """List the VC effect heads whose L2 penalty joins the loss, per node."""
        return {
            name: [
                self.nodes[name].shifts[g.on]
                for g in self.nodes[name]._vc_groups
                if g.mods and self.nodes[name].shifts[g.on].penalty > 0
            ]
            for name in self.order
        }

    def _log_epoch(
        self,
        verbose: int,
        epoch: int,
        epochs: int,
        train_acc: dict[str, float],
        val_per_node: dict[str, float],
        frozen: set[str],
    ) -> None:
        """Emit the periodic progress record (INFO on the ``tramdag.flow`` logger)."""
        if not verbose or (epoch % verbose and epoch != epochs - 1):
            return
        logger.info(
            "[epoch %5d/%d] train NLL %.4f  val NLL %.4f%s",
            epoch + 1,
            epochs,
            sum(train_acc.values()),
            sum(val_per_node.values()),
            f"  frozen {sorted(frozen)}" if frozen else "",
        )

    def _end_epoch(
        self,
        *,
        opt: torch.optim.Optimizer,
        sched: _FitSchedule,
        best: dict | None,
        verbose: int,
        epoch: int,
        epochs: int,
        train_acc: dict[str, float],
        val_vals: dict[str, Tensor],
        t_start: tuple[float, float],
    ) -> bool:
        """Record one epoch, update the schedules, snapshot, and log.

        Returns ``True`` when every node froze, which stops the fit.
        """
        val_per_node = self._val_nll(val_vals)
        t0, t_offset = t_start
        self.history["train"].append(train_acc)
        self.history["val"].append(val_per_node)
        self.history["lr"].append(max(g["lr"] for g in opt.param_groups))
        # after the val pass, so an epoch's record includes its own monitoring
        self.history["time"].append(t_offset + time.perf_counter() - t0)
        sched.step(opt, val_per_node, self.history)
        if best is not None:
            self._snapshot_best(best, val_per_node)
        self._log_epoch(verbose, epoch, epochs, train_acc, val_per_node, sched.frozen)
        if len(sched.frozen) < len(self.order):
            return False
        if verbose:
            logger.info("[epoch %5d] all nodes frozen — stopping.", epoch + 1)
        return True

    def _fit_epoch(
        self,
        train_vals: dict[str, Tensor],
        prev_acc: dict[str, float],
        frozen: set[str],
        opt: torch.optim.Optimizer,
        batch_size: int,
        vc_ehat_train: dict[str, dict[str, Tensor]] | None,
        vc_penalized: dict[str, list],
    ) -> dict[str, float]:
        """Run one epoch of minibatch training over the non-frozen nodes.

        Returns the epoch-mean train NLL per node; a frozen node carries
        its last computed value forward. The VC penalty joins the loss but
        never the returned NLLs.
        """
        active = [name for name in self.order if name not in frozen]
        n = len(next(iter(train_vals.values())))
        perm = torch.randperm(n, device=self.device)
        acc = dict.fromkeys(active, 0.0)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = {k: v[idx] for k, v in train_vals.items()}
            per_node = self.node_log_prob(
                batch, nodes=active, vc_ehat=_slice_ehat(vc_ehat_train, idx)
            )
            node_nlls = {k: -v.mean() for k, v in per_node.items()}
            loss = torch.stack(list(node_nlls.values())).sum()
            for name in active:  # VC penalty joins the loss (never the history)
                for m in vc_penalized[name]:
                    loss = loss + m.penalty * m.l2() / n
            opt.zero_grad()
            loss.backward()
            opt.step()
            w = len(idx) / n
            for k, v in node_nlls.items():
                acc[k] += float(v.detach()) * w
        return {
            name: acc.get(name, prev_acc.get(name, float("nan"))) for name in self.order
        }

    def _snapshot_best(self, best: dict, val_per_node: dict[str, float]) -> None:
        """Deep-copy a node's weights whenever its validation NLL improves."""
        for name in self.order:
            if val_per_node[name] < best[name][0]:
                best[name] = (
                    val_per_node[name],
                    copy.deepcopy(self.nodes[name].state_dict()),
                )

    def _load_best_weights(self, best: dict) -> None:
        """Restore each node's best-validation snapshot, where one exists."""
        for name, (_, state) in best.items():
            if state is not None:
                self.nodes[name].load_state_dict(state)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        epochs: int | None = None,
        learning_rate: float = 1e-2,
        batch_size: int = 512,
        verbose: int = 50,
        seed: int | None = None,
        restore_best: bool = False,
        schedule: str | None = None,
        plateau_patience: int = 30,
        freeze_patience: int | None = None,
        min_delta: float = 1e-4,
        marginal_init: bool = True,
        vc_warm_start: bool = True,
        plateau_factor: float = 0.3,
        plateau_min_lr: float | None = None,
        vc_oof_fit: dict | None = None,
        epoch_callback=None,
    ) -> CausalFlowDAG:
        """Fit all nodes jointly by maximum likelihood.

        By default training keeps the **final** (converged) weights. An
        all-``ls`` model trained to convergence then reproduces the classical
        maximum-likelihood estimate exactly, and matches ``statsmodels`` and
        R ``polr``.

        The optimizer holds one parameter group per node. The joint NLL
        decomposes per node with independent gradients. Per-node learning
        rates and freezing are therefore exactly equivalent to independent
        per-node training.

        A second ``fit`` call continues the training, for example a second
        phase with a lower learning rate. Freezing state does not carry
        across calls.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data, one column per node.
        val_df : pd.DataFrame | None, optional
            Held-out set, used only for monitoring and for
            ``restore_best``, ``schedule="plateau"`` and
            ``freeze_patience``. If omitted, the training set supplies the
            validation metric.
        epochs : int
            Number of training epochs. **Required**: there is no defensible
            default. ``docs/training-speed.md`` measures fixed budgets going
            wrong in both directions on the repo's own workloads (stroke
            over-spends by 2.5x, vaca under-spends by 0.03 nats), so pick a
            budget for the data, or set a generous one and let
            ``schedule="plateau"`` with ``freeze_patience`` stop the fit.
        learning_rate : float, optional
            Adam learning rate, by default 1e-2 — the rate every row of the
            ``docs/training-speed.md`` benchmark runs at, and the one its
            recommended recipe uses. The paper replications in
            ``experiments/`` run at 1e-3, which is the paper's own value.
        batch_size : int, optional
            Minibatch size, by default 512. Measured: full-batch loses on
            time-to-target despite higher epoch throughput, and 16k batches
            only helped raw throughput at n=50k
            (``docs/training-speed.md``, finding 4).
        verbose : int, optional
            Emit a progress record every ``verbose`` epochs, by default 50.
            On the fits this package is for, minutes pass between epochs 0
            and 500, and silence is indistinguishable from a hang. These are
            INFO records on the ``tramdag.flow`` logger, so a caller sees
            them only after configuring logging
            (``logging.basicConfig(level=logging.INFO)``). 0 turns them off.
        seed : int | None, optional
            Seeds the minibatch shuffling only. Weight initialization
            happens at construction, see the class docstring.
        restore_best : bool, optional
            If True, snapshot each node's best-validation weights during
            training and restore them at the end. This is a mild
            early-stopping regularization and the convention of the
            original implementation. The snapshots persist on the model
            across ``fit`` calls, so a multi-phase fit restores the best
            epoch of *all* its phases; they are not saved in checkpoints.
            The fit is then *not* the training-data MLE, so leave it False
            for an exact classical comparison. Default False.
        schedule : str | None, optional
            ``None`` (default) keeps the learning rate constant, so a fit is
            reproducible from the rate and the budget alone; the measured
            recommendation for everyday fits is ``"plateau"``
            (``docs/training-speed.md``).
            ``"plateau"`` decays **per node**: when the validation NLL of
            a node did not improve by ``min_delta`` for
            ``plateau_patience`` epochs, its learning rate decreases by
            ``plateau_factor``, with floor ``1e-3 * learning_rate``.
        plateau_patience : int, optional
            Epochs without improvement before one plateau decay step, by
            default 30 — the value ``docs/training-speed.md`` recommends
            after measuring it against the hand-tuned two-phase schedule.
        freeze_patience : int | None, optional
            If set, a node whose validation NLL did not improve by
            ``min_delta`` for this many epochs is **frozen** — excluded
            from the loss and the backward pass. This is a real compute
            saving, because the per-node losses are independent. When
            every node is frozen the fit returns early. Freeze epochs are
            recorded in ``history["frozen"]``. Under ``schedule="plateau"``
            a node freezes only after its learning rate decayed to
            ``1e-2 * learning_rate`` or below.
        min_delta : float, optional
            Smallest validation improvement that counts, by default 1e-4 —
            an order of magnitude below the +1e-3/+5e-3 NLL tolerances any
            check in this repo applies, so it cannot mask a difference
            anyone measures.
        marginal_init : bool, optional
            Calibrate the intercept of each *unconditional* node to its
            marginal at initialization (default ``True``), instead of
            zuko's zero initialization. A Bernstein continuous node gets the
            linear map of the pre-scaled domain onto the standard-logistic
            5%/95% quantiles (the default is about 2.5x too steep). An
            ordinal node gets cutpoints at the empirical class log-odds
            (the default zeros are near-uniform). This is a pure
            initialization: the converged MLE is unchanged. Applied on the
            first fit only. Affects only ``SimpleIntercept`` nodes;
            conditional ``ci`` intercepts stay untouched. ``False`` keeps
            zuko's zero start.
        vc_warm_start : bool, optional
            If True (default), the ``beta0`` of each ``VC`` term is
            initialized from the classical all-``ls`` solution of its
            node's conditional (deterministic L-BFGS on a throwaway
            proxy). The penalized head then starts at the classical answer
            and only learns deviations. Applied once per term — a buffer
            that survives ``save``/``load`` guards against re-runs. Does
            nothing without VC terms.
        plateau_factor : float, optional
            Multiplier of one per-node plateau decay step, by default 0.3.
            Gentler than torch's own ``ReduceLROnPlateau`` (0.1), which
            matters here because the decay is per node off a noisy per-node
            validation curve: three 0.3 steps land near one 0.1 step, but a
            single spurious plateau costs a third of the rate instead of a
            tenth. Read only under ``schedule="plateau"``.
        plateau_min_lr : float | None, optional
            Absolute floor of the per-node learning rate under
            ``schedule="plateau"``. ``None`` (default) keeps the floor at
            ``1e-3 * learning_rate``; the paper's reference code uses an
            absolute ``1e-7``.
        vc_oof_fit : dict | None, optional
            Keyword overrides for the stage-1 out-of-fold proxy fits of
            centered VC terms, merged over the default
            ``{"epochs": 300, "learning_rate": 1e-2, "batch_size": 512}``.
            Nothing in this repo measures those three numbers; they are the
            pre-0.4 values, kept so centered fits stay comparable. They set
            the quality of ``e_hat``, which is what the centering buys, so
            override them if stage 1 underfits.
            Ignored when the treatment node is all-``ls`` — the
            deterministic :meth:`fit_classical` runs instead.
        epoch_callback : callable | None, optional
            ``epoch_callback(flow, epoch)`` runs after every epoch, once the
            validation pass, the schedules and the best-weight snapshot are
            done. This is how an experiment reads out coefficient trajectories
            from one continuous run, the way the reference's per-epoch Keras
            loop does. ``epoch`` counts from 1.

        Returns
        -------
        CausalFlowDAG
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``epochs`` is not given, if ``schedule`` is neither ``None``
            nor ``"plateau"``, or if ``batch_size`` is below 1.

        Notes
        -----
        For ``VC`` terms the objective is the **penalized** NLL on the
        total-likelihood scale. Each term adds
        ``penalty * ||b_theta weights||^2`` to the summed NLL, that is
        ``penalty * ||w||^2 / n_train`` to the mean loss. This is a fixed
        Gaussian prior: the shrinkage vanishes as n grows, the classical
        penalized-likelihood convention. ``beta0`` is not penalized. The
        recorded ``history`` NLLs stay pure likelihoods. After training,
        each ``b_theta`` is re-centered to mean zero over the training
        data. That preserves the function: the constant moves into
        ``beta0``.

        ``VC(center=...)`` terms run a stage-1 out-of-fold propensity
        computation before the loop (:meth:`_vc_oof_stage`). The training
        loss uses those frozen out-of-fold values. The epoch-level
        validation monitor, and every post-fit query, uses the live
        full-fit treatment node.
        """
        if epochs is None:
            raise ValueError(
                "fit() needs epochs=. There is no default budget: a fixed one "
                "over-spends on some workloads and under-spends on others "
                "(docs/training-speed.md). Set a budget, or a generous one "
                "with schedule='plateau' and freeze_patience= to stop early."
            )
        if schedule not in (None, "plateau"):
            raise ValueError(f"unknown schedule {schedule!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if seed is not None:
            torch.manual_seed(seed)
        self._set_ranges(train_df, marginal_init=marginal_init)
        if vc_warm_start:
            self._vc_warm_start(train_df)
        vc_penalized = self._vc_penalized()
        # stage 1 for centered VC terms (issue #30): frozen OUT-OF-FOLD e_hat
        # for the training rows — a plain tensor, so the Y-node loss has no
        # gradient path into the treatment node (per-node factorization intact).
        vc_ehat_train = self._vc_oof_stage(train_df, vc_oof_fit)

        train_vals = self._tensorize(train_df)
        val_vals = self._tensorize(val_df) if val_df is not None else train_vals

        opt = self._make_optimizer(learning_rate)
        best = self._best_store(restore_best)
        sched = _FitSchedule(
            self.order,
            schedule,
            learning_rate,
            plateau_patience,
            plateau_factor,
            freeze_patience,
            min_delta,
            plateau_min_lr,
        )
        t0 = time.perf_counter()
        t_offset = self.history["time"][-1] if self.history.get("time") else 0.0
        train_acc: dict[str, float] = {}

        for epoch in range(epochs):
            self.train()
            train_acc = self._fit_epoch(
                train_vals,
                train_acc,
                sched.frozen,
                opt,
                batch_size,
                vc_ehat_train,
                vc_penalized,
            )
            self.eval()
            if self._end_epoch(
                opt=opt,
                sched=sched,
                best=best,
                verbose=verbose,
                epoch=epoch,
                epochs=epochs,
                train_acc=train_acc,
                val_vals=val_vals,
                t_start=(t0, t_offset),
            ):
                break
            if epoch_callback is not None:
                epoch_callback(self, epoch + 1)

        if best is not None:  # restore per-node best-validation weights
            self._load_best_weights(best)
        self._recenter_vc(train_vals)
        self.eval()
        return self

    def _source_proxies(self, parents) -> dict[str, NodeSpec]:
        """Give a proxy spec whose parents are sources.

        A single-node proxy only has to reproduce one conditional, which the
        parent marginals cannot influence — so each parent collapses to a
        source node of the right kind.
        """
        out: dict[str, NodeSpec] = {}
        for p in parents:
            pn = self.spec[p]
            out[p] = (
                OrdinalNode(pn.levels)
                if isinstance(pn, OrdinalNode)
                else ContinuousNode([I(transform="affine")])
            )
        return out

    def _ls_proxy_spec(self, name: str) -> dict[str, NodeSpec]:
        """Give a throwaway all-``ls`` proxy spec of one node's conditional.

        Same kind and transform, every parent an LS term, every parent a
        source (their marginals cannot influence the conditional).
        """
        node_spec = self.spec[name]
        nd = self.nodes[name]
        proxy_spec = self._source_proxies(nd.parents)
        ls_terms = [LS(p) for p in nd.parents]
        if isinstance(node_spec, OrdinalNode):
            proxy_spec[name] = OrdinalNode(node_spec.levels, ls_terms)
        else:
            proxy_spec[name] = ContinuousNode(
                [
                    I(
                        transform=node_spec.transform,
                        **dict(node_spec.transform_kwargs),
                    ),
                    *ls_terms,
                ]
            )
        return proxy_spec

    def _vc_warm_start(self, train_df: pd.DataFrame) -> None:
        """Initialize every VC term's ``beta0`` from the classical solution.

        The value comes from the all-``ls`` solution of the node's conditional.
        Issue #28 recommends this warm start.

        A throwaway proxy of the node (same kind/transform, every parent an LS
        term, parent marginals irrelevant to the conditional because the joint
        NLL decomposes per node) is fitted with the deterministic
        :meth:`fit_classical`, and the ``on`` coefficient copied into ``beta0``
        (for a binary ordinal treatment, the identified one-hot difference
        ``w[1] - w[0]``). ``b_theta`` already starts at the zero function
        (zero-initialized output layer). Runs once per term — the
        ``warm_started`` buffer survives ``save``/``load``.
        """
        for name in self.order:
            nd = self.nodes[name]
            todo = [g for g in nd._vc_groups if not bool(nd.shifts[g[0]].warm_started)]
            if not todo:
                continue
            proxy = CausalFlowDAG(self._ls_proxy_spec(name), device=str(self.device))
            proxy.fit_classical(train_df[[*nd.parents, name]], verbose=False)
            for g in todo:
                w = proxy.nodes[name].shifts[g.on].weight.detach()
                b0 = float(w[-1] - w[0]) if g.on_is_ord else float(w[0])
                m = nd.shifts[g.on]
                with torch.no_grad():
                    m.beta0.fill_(b0)
                m.warm_started.fill_(True)

    @torch.no_grad()
    def _predict_p1(self, on: str, df: pd.DataFrame) -> np.ndarray:
        """Give ``P(on = 1 | pa_on)`` from this flow's ``on`` node.

        The treatment is binary ordinal, so the value is
        ``sigmoid(shift - theta_0)``.
        """
        nd = self.nodes[on]
        values = self._tensorize(df, nd.parents)
        return self._binary_p1(nd, values, len(df)).cpu().numpy()

    def _vc_oof_stage(
        self, train_df: pd.DataFrame, vc_oof_fit: dict | None = None
    ) -> dict[str, dict[str, Tensor]] | None:
        """Compute stage 1 of the two-stage centered-VC design, issue #30.

        The result holds the frozen training-time propensities, as
        ``{node: {on: (n,) tensor}}``.

        The values are **out-of-fold** — K refits of the treatment node
        only, each predicting its held-out fold (the DML cross-fitting
        requirement; in-sample e_hat reintroduces the own-observation bias
        and can be worse than no centering). Bookkeeping lands in
        ``self.vc_center_info[(node, on)]`` (``e_oof``, ``fold_id``,
        ``folds``, ``n``) so tests can assert the fold structure — a later
        "simplification" to in-sample e_hat fails CI.
        """
        jobs = [
            (name, g)
            for name in self.order
            for g in self.nodes[name]._vc_groups
            if g.center
        ]
        if not jobs:
            return None
        self.vc_center_info = {}
        np_dtype = self._np_dtype
        out: dict[str, dict[str, Tensor]] = {}
        rng_state = torch.get_rng_state()  # proxies reseed; keep fit reproducible
        try:
            for name, g in jobs:
                e, fold_id = self._vc_oof_propensity(
                    g.on, train_df, g.folds, vc_oof_fit
                )
                out.setdefault(name, {})[g.on] = torch.as_tensor(
                    e.astype(np_dtype), device=self.device
                )
                self.vc_center_info[(name, g.on)] = {
                    "folds": int(g.folds),
                    "fold_id": fold_id,
                    "e_oof": e.copy(),
                    "n": len(train_df),
                }
        finally:
            torch.set_rng_state(rng_state)
        return out

    def _vc_oof_propensity(
        self, on: str, train_df: pd.DataFrame, k: int, vc_oof_fit: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the out-of-fold ``P(on=1|pa_on)``.

        The function refits the ``on`` node K times, and only that node. Each
        refit uses a single-node proxy whose parents are sources, because their
        marginals cannot influence the conditional. Each refit then predicts the
        fold it never saw.

        A treatment with all-``ls`` terms uses :meth:`fit_classical`, which is
        deterministic and takes seconds. Any other treatment uses a
        fixed-budget Adam fit — ``vc_oof_fit`` overrides its keywords, see
        :meth:`fit`.
        """
        on_nd = self.nodes[on]
        if any(g.center for g in on_nd._vc_groups):
            raise NotImplementedError(
                f"treatment node {on!r} itself has a centered VC term. "
                "Chained centering is not supported."
            )
        node_spec = self.spec[on]
        proxy_spec = self._source_proxies(on_nd.parents)
        terms = node_terms(node_spec)
        proxy_spec[on] = OrdinalNode(2, terms)
        all_ls = all(map(_covered_by_classical, terms))
        cols = [*on_nd.parents, on]

        n = len(train_df)
        fold_id = np.random.default_rng(0).permutation(n) % k
        e = np.empty(n, dtype=np.float64)
        for j in range(k):
            proxy = CausalFlowDAG(proxy_spec, device=str(self.device), seed=0)
            held_in = train_df.iloc[fold_id != j][cols]
            if all_ls:
                proxy.fit_classical(held_in, verbose=False)
            else:
                fit_kw = {"epochs": 300, "learning_rate": 1e-2, "batch_size": 512}
                fit_kw.update(vc_oof_fit or {})
                proxy.fit(held_in, verbose=0, seed=0, restore_best=False, **fit_kw)
            e[fold_id == j] = proxy._predict_p1(on, train_df.iloc[fold_id == j])
        return e, fold_id

    @torch.no_grad()
    def _recenter_vc(self, values: dict[str, Tensor]) -> None:
        """Re-split every VC term so ``b_theta`` sums to zero over the train rows.

        The removed constant moves into ``beta0``, so the modelled function does
        not change.
        """
        feats: dict[str, Tensor] | None = None
        for name in self.order:
            nd = self.nodes[name]
            for g in nd._vc_groups:
                if not g.mods:
                    continue
                if feats is None:
                    feats = self._features(values)
                nd.shifts[g.on].recenter(torch.cat([feats[p] for p in g.mods], dim=1))

    @torch.no_grad()
    def varying_coef(
        self, node: str, data: pd.DataFrame, t: str | None = None
    ) -> np.ndarray:
        """Evaluate the fitted effect function ``beta(x)`` of a ``VC`` term.

        This is the first-class read-out of issue #28. The value comes in
        closed form from the fitted term, as ``beta0 + b_theta(modifiers)``.
        It is deterministic and needs no abduction. It is free of ``y``,
        because only the modifier columns of ``data`` are read. For a binary
        treatment it is identical to the abduction difference
        ``u(x, t=1, y) - u(x, t=0, y)``.

        The value lives on the latent, log-odds scale of the node. A
        continuous node adds it. An ordinal node subtracts it from the
        cutpoints.

        For a centered term (``center=...``) the form of the returned
        ``beta`` does not change. ``beta0`` then reads as the effect at the
        treatment margin, which is the observed propensities.

        Parameters
        ----------
        node : str
            Name of the node that carries the VC term.
        data : pd.DataFrame
            Rows at which to evaluate ``beta``. Must contain every modifier
            column of the term.
        t : str | None, optional
            Treatment name of the VC term. Optional when the node has
            exactly one VC term.

        Returns
        -------
        np.ndarray
            The ``beta`` values, shape ``(n,)``. Constant when the term has
            no modifiers.

        Raises
        ------
        KeyError
            If ``node`` is unknown, if the node has no VC term on ``t``, or
            if a modifier column is missing from ``data``.
        ValueError
            If the node has no VC term, or if ``t`` is omitted while the
            node has several VC terms.
        """
        nd = self._node(node)
        vcs = {g.on: g.mods for g in nd._vc_groups}
        if not vcs:
            raise ValueError(f"node {node!r} has no VC term.")
        if t is None:
            if len(vcs) > 1:
                raise ValueError(
                    f"node {node!r} has several VC terms ({sorted(vcs)}). "
                    "Pass t=<treatment name>."
                )
            t = next(iter(vcs))
        if t not in vcs:
            raise KeyError(
                f"node {node!r} has no VC term on {t!r} (has {sorted(vcs)})."
            )
        mods = vcs[t]
        mod_feat = None
        if mods:
            feats = self._features(self._tensorize(data, mods))
            mod_feat = torch.cat([feats[p] for p in mods], dim=1)
        return nd.shifts[t].beta(mod_feat, len(data)).cpu().numpy()

    def _is_all_ls(self) -> bool:
        return all(
            _covered_by_classical(term)
            for node in self.spec.values()
            for term in node_terms(node)
        )

    def ls_coefficients(self) -> dict[str, dict[str, np.ndarray]]:
        """Give the per-node linear-shift weights.

        For an all-``ls`` model these are the interpretable log-odds-ratio
        coefficients.

        Only ``LS`` terms have a weight to give. A node's ``CS`` and ``VC``
        shifts are networks, so they are skipped — reading them needs
        :meth:`varying_coef` or an evaluation of the network itself.

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            The weights, as ``{node: {parent: array}}``. A node without
            linear-shift terms is absent.
        """
        out: dict[str, dict[str, np.ndarray]] = {}
        for name in self.order:
            linear = {
                parent: module.weight.detach().cpu().numpy().ravel().copy()
                for parent, module in self.nodes[name].shifts.items()
                if isinstance(module, LinearShift)
            }
            if linear:
                out[name] = linear
        return out

    def to_matrix(self) -> pd.DataFrame:
        """Give the labeled adjacency matrix of term effects.

        This is the meta-adjacency view of the paper.

        Returns
        -------
        pd.DataFrame
            Rows are parents and columns are children. A cell holds the
            term tag: ``"LS"``, ``"CS"``, ``"CI"``, ``"VC"`` for a VC
            treatment, or ``"VCm"`` for a VC modifier. An empty cell means
            there is no edge. A multi-parent term carries its parent group
            as a suffix. When several terms share a cell, their tags join
            with ``"+"``.
        """
        m = pd.DataFrame("", index=list(self.order), columns=list(self.order))
        for child in self.order:
            for term in node_terms(self.spec[child]):
                for p, tag in _term_cells(term):  # a VC modifier may share its
                    cur = m.loc[p, child]  # cell with a prognostic term -> "+"
                    m.loc[p, child] = f"{cur}+{tag}" if cur else tag
        return m

    @torch.no_grad()
    def intercept_contributions(self, node: str, data: pd.DataFrame) -> dict:
        """Decompose a complex intercept into mean-centered per-term parts.

        The parts are contributions to the transform parameters of the node.
        Use them to plot additive partial effects.

        An additive complex intercept,
        ``CI("x1", "x2", allow_interaction=False)``, builds one
        network per ``I`` term and **sums their outputs in unconstrained
        parameter space**: ``theta(pa) = net_1(x1) + net_2(x2)``. The sum is
        identified, so every L1/L2/L3 query is correct. Each term's output,
        however, is identified only up to a constant — a constant moves
        freely between the nets. The *raw* per-term outputs are therefore
        not directly comparable.

        This method resolves the ambiguity with the usual additive-model
        (GAM) convention: a **sum-to-zero (mean-centering) constraint over
        the rows of** ``data``. Each term's contribution is centered to mean
        zero per parameter. The removed constants collect into a single
        ``baseline``. The decomposition is exact:

            ``theta(pa) = baseline + sum_terms contribution_term(pa)``

        ``baseline`` plus the uncentered row sum of the contributions
        reproduces the model's transform parameters. This is **post-hoc
        only**: it reads the fitted weights and changes nothing about the
        model or any frozen number (issue #20, Option A). Shift terms
        (``LS``/``CS``) are a separate, already-interpretable slot — see
        :meth:`ls_coefficients`.

        Parameters
        ----------
        node : str
            Name of a node with at least one complex-intercept (``I``) term
            that has parents.
        data : pd.DataFrame
            Rows over which to center and at which to evaluate the
            contributions. Must contain every intercept-parent column.

        Returns
        -------
        dict
            Three keys. ``"baseline"`` is the absorbed constant, a ``(P,)``
            array — the sum of the per-term means. ``P`` is the node's
            transform-parameter count: ``ut.n_params`` for a continuous
            node, ``levels - 1`` cutpoint parameters for an ordinal node.
            ``"contributions"`` is ``{term_label: (n, P) array}`` — each
            term's mean-centered contribution at each row, columns summing
            to about zero over the rows. ``term_label`` is the term's
            parents joined by ``"+"``. ``"parents"`` is
            ``{term_label: tuple(parent_names)}``.

        Raises
        ------
        KeyError
            If ``node`` is unknown, or if an intercept-parent column is
            missing from ``data``.
        ValueError
            If the node has no complex-intercept term with parents.

        Notes
        -----
        The contributions live in the transform's **unconstrained**
        parameter space, where the model sums the additive terms before the
        monotonicity constraint. They are exact partial effects on those
        parameters, but not, in general, an additive shift of the curve
        itself.
        """
        nd = self._node(node)
        groups = nd._intercept_groups
        if not groups:
            raise ValueError(
                f"node {node!r} has no complex-intercept (I) terms with parents. "
                "Its intercept is unconditional, so there is nothing to decompose."
            )
        missing = [p for p in nd.ci_parents if p not in data.columns]
        if missing:
            raise KeyError(f"data is missing intercept-parent column(s): {missing}")

        feats = self._features(self._tensorize(data, nd.ci_parents))
        # one net per group: the additive case stores them in intercept_nets;
        # a single (possibly joint) I-term is the lone `intercept` network.
        nets = (
            list(nd.intercept_nets) if nd.intercept_nets is not None else [nd.intercept]
        )

        contributions: dict[str, np.ndarray] = {}
        parents: dict[str, tuple] = {}
        baseline = None
        for net, grp in zip(nets, groups, strict=True):
            raw = net(torch.cat([feats[p] for p in grp], dim=1))  # (n, P)
            mean = raw.mean(dim=0, keepdim=True)  # (1, P)
            label = "+".join(grp)
            contributions[label] = (raw - mean).cpu().numpy()
            parents[label] = grp
            baseline = mean if baseline is None else baseline + mean
        return {
            "baseline": baseline.cpu().numpy().ravel(),
            "contributions": contributions,
            "parents": parents,
        }

    @torch.no_grad()
    def design_matrix(
        self, df: pd.DataFrame, node: str, *, drop_first: bool = False
    ) -> pd.DataFrame:
        """Encode a node's parents the way the flow feeds them to its shifts.

        A continuous parent stays raw in one column named after it. An
        ordinal parent becomes one column per level, named
        ``"{parent}[{k}]"`` — the same one-hot the flow builds internally.

        Use ``drop_first=True`` to get the design a classical reference
        expects (``statsmodels`` ``OrderedModel``, R ``polr``): with
        cutpoints the full one-hot is unidentified, so each ordinal parent's
        level-0 column drops out and its remaining coefficients read as
        differences against level 0 — exactly what ``w[k] - w[0]`` gives on
        the flow side.

        Parameters
        ----------
        df : pd.DataFrame
            Rows to encode. Must contain every parent column of ``node``.
        node : str
            Name of the node whose parents are encoded.
        drop_first : bool, optional
            Drop each ordinal parent's level-0 column, by default ``False``.

        Returns
        -------
        pd.DataFrame
            One column per encoded feature, indexed like ``df``.
        """
        nd = self._node(node)
        feats = self._features(self._tensorize(df, nd.parents))
        cols: dict[str, np.ndarray] = {}
        for p in nd.parents:
            arr = feats[p].cpu().numpy()
            if arr.shape[1] == 1:  # continuous parent: raw
                cols[p] = arr[:, 0]
            else:
                for k in range(1 if drop_first else 0, arr.shape[1]):
                    cols[f"{p}[{k}]"] = arr[:, k]
        return pd.DataFrame(cols, index=df.index)

    def fit_classical(
        self,
        train_df: pd.DataFrame,
        *,
        max_iter: int = 400,
        tol: float = 1e-9,
        history_size: int = 50,
        verbose: bool = True,
    ) -> dict:
        """Fit an all-``ls`` model the classical way.

        The fit uses full batches, float64, and L-BFGS with a strong-Wolfe
        line search. There are no minibatches, no schedule and no early
        stopping, so the fit is deterministic and bit-reproducible. It lands
        on the exact maximum-likelihood estimate and matches classical
        software, that is ``statsmodels`` ``OrderedModel`` and R ``polr`` or
        ``Colr``. It is much faster than minibatch Adam.

        This method is valid only when every edge is ``ls``, because each
        node-conditional is then a classical transformation model. Any other
        model raises. For a ``cs`` or ``ci`` model use :meth:`fit`, where
        the minibatch noise also regularizes the MLPs.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data, one column per node.
        max_iter : int, optional
            Upper limit on L-BFGS iterations, by default 400.
        tol : float, optional
            torch's ``tolerance_change``: the NLL (or parameter) change below
            which L-BFGS stops, by default 1e-9. Measured on the classical
            anchor: 1e-6 stops on a plateau step and leaves a rare one-hot
            level 0.24 off statsmodels; 1e-9 lands within 0.03, the same as
            running to the iteration cap.
        history_size : int, optional
            L-BFGS memory, by default 50.
        verbose : bool, optional
            Print a one-line summary, by default True.

        Returns
        -------
        dict
            A convergence report: ``converged``, ``n_iter``, ``final_nll``,
            ``grad_norm``, ``seconds``, and the fitted ``coefficients``
            from :meth:`ls_coefficients`.

        Raises
        ------
        ValueError
            If the spec has ``cs``, ``ci`` or ``vc`` terms.

        Notes
        -----
        float64 is a transient compute mode. The model is upcast for the
        fit, and ``self.double()`` converts the parameters and the range
        buffers of the transforms in one call. Afterwards the model returns
        to float32, so the stored model and ``save``/``load`` stay float32.
        Double precision is what lets the line search resolve the optimum
        cleanly.

        Convergence is torch's own: L-BFGS stops when the NLL or the
        parameters move by less than ``tol``, or when the gradient is
        below its ``tolerance_grad``. ``|grad|`` and individual
        coefficients do *not* settle to machine precision. A continuous
        node's Bernstein intercept, and weakly-identified directions such
        as rare one-hot levels or a flat treatment-effect ridge, keep
        drifting along near-zero-curvature valleys long after the
        likelihood and the well-identified coefficients reach the MLE.
        Correctness is therefore verified by comparison to classical
        software (see ``experiments/misc/validate_ls.py``), not by this flag.
        """
        if not self._is_all_ls():
            raise ValueError(
                "fit_classical requires an all-`ls` spec, that is every edge "
                "term 'ls'. This spec has cs, ci or vc terms. Use fit() for "
                "flexible models."
            )
        self._set_ranges(
            train_df, marginal_init=False
        )  # L-BFGS needs no calibrated start

        self.double()  # parameters + buffers (xmin/xmax) -> float64, one call
        t0 = time.perf_counter()
        try:
            vals = self._tensorize(train_df)
            self.train()
            opt = torch.optim.LBFGS(
                self.parameters(),
                lr=1.0,
                max_iter=max_iter,
                history_size=history_size,
                tolerance_grad=0.0,  # |grad| never settles on the flat ridges
                tolerance_change=tol,
                line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad()
                nll = torch.stack(
                    [-lp.mean() for lp in self.node_log_prob(vals).values()]
                ).sum()
                nll.backward()
                return nll

            opt.step(closure)
            n_iter = next(iter(opt.state.values()))["n_iter"]
            converged = n_iter < max_iter  # torch stopped on a tolerance
            with torch.no_grad():
                final_nll = float(
                    torch.stack(
                        [-lp.mean() for lp in self.node_log_prob(vals).values()]
                    ).sum()
                )
            grad_norm = float(
                torch.cat(
                    [
                        p.grad.reshape(-1)
                        for p in self.parameters()
                        if p.grad is not None
                    ]
                ).norm()
            )
            coefs = self.ls_coefficients()  # read while still float64
        finally:
            self.float()  # restore canonical float32 (lossy ~1e-7, harmless)
        self.eval()

        report = {
            "converged": converged,
            "n_iter": n_iter,
            "final_nll": final_nll,
            "grad_norm": grad_norm,
            "seconds": time.perf_counter() - t0,
            "coefficients": coefs,
        }
        if verbose:
            logger.info(
                "fit_classical: %d L-BFGS iters, NLL %.6f, %.2fs%s",
                n_iter,
                final_nll,
                report["seconds"],
                "" if converged else f"  (NLL still moving at {max_iter} iters)",
            )
        return report

    @torch.no_grad()
    def sample(
        self,
        n: int | None = None,
        *,
        do: dict[str, float] | None = None,
        u: pd.DataFrame | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Sample from the flow, with optional interventions.

        Parameters
        ----------
        n : int | None, optional
            Number of samples. Ignored if ``u`` is given.
        do : dict[str, float] | None, optional
            Interventions, as ``{node: value}``. An intervened node is
            clamped and its parent dependence removed (graph mutilation).
        u : pd.DataFrame | None, optional
            Latent variables, as returned by :meth:`abduct`. If given,
            they are pushed through the flow. Together with ``do`` this
            yields counterfactuals: Pearl's abduction, action, prediction.
        seed : int | None, optional
            If given, seeds the latent draw. Ignored if ``u`` is given.

        Returns
        -------
        pd.DataFrame
            The samples, one column per node.

        Raises
        ------
        ValueError
            If both ``n`` and ``u`` are omitted.
        """
        do = do or {}
        if u is not None:
            n = len(u)
            u_vals = self._tensorize(u)
        elif n is not None:
            gen = self._generator(seed)
            u_vals = {
                name: StandardLogistic.sample((n,), device=self.device, generator=gen)
                for name in self.order
            }
        else:
            raise ValueError("Provide either n or u.")

        values: dict[str, Tensor] = {}
        for name in self.order:
            if name in do:
                values[name] = torch.full(
                    (n,), float(do[name]), dtype=self._dtype, device=self.device
                )
                continue
            node = self.nodes[name]
            feats = self._features({p: values[p] for p in node.parents})
            # centered VC: e_hat(pa_on) is re-derived from the already-sampled
            # ancestor values — under do the regressor is t_do - e_hat(x), never
            # a cached training value
            theta, shift = node.theta_shift(
                feats, n, vc_ehat=self._vc_ehat_live(node, values, n)
            )
            u_val = u_vals[name]
            if node.kind == "continuous":
                values[name] = node.ut.inverse(theta, u_val - shift)
            else:
                values[name] = ordinal_sample(theta, shift, u_val)
        return pd.DataFrame({k: v.cpu().numpy() for k, v in values.items()})

    @torch.no_grad()
    def abduct(self, df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        """Recover the latent variables ``u`` from observations (Pearl step 1).

        A continuous node inverts exactly: ``u = h(x) + shift``. For an
        ordinal node the latent is only interval-identified, so it is
        sampled from the standard logistic truncated to the observed
        level's interval.

        Parameters
        ----------
        df : pd.DataFrame
            Observations, one column per node.
        seed : int | None, optional
            If given, seeds the truncated draw for the ordinal nodes.

        Returns
        -------
        pd.DataFrame
            The latents, one column per node, aligned with the rows of
            ``df``.
        """
        gen = self._generator(seed)
        values = self._tensorize(df)
        feats = self._features(values)
        n = len(df)
        u = {}
        for name in self.order:
            node = self.nodes[name]
            theta, shift = node.theta_shift(
                feats, n, vc_ehat=self._vc_ehat_live(node, values, n)
            )
            x = values[name]
            if node.kind == "continuous":
                u0, _ = node.ut.forward(theta, x)
                u[name] = u0 + shift
            else:
                u[name] = ordinal_abduct(theta, shift, x, generator=gen)
        return pd.DataFrame({k: v.cpu().numpy() for k, v in u.items()})

    @torch.no_grad()
    def pmf(
        self, df: pd.DataFrame, node: str, do: dict[str, float] | None = None
    ) -> np.ndarray:
        """Give the analytic class probabilities of an ordinal node.

        Parameters
        ----------
        df : pd.DataFrame
            Rows that supply the parent values of the node.
        node : str
            Name of the ordinal node.
        do : dict[str, float] | None, optional
            Column overrides, applied to ``df`` before the evaluation.

        Returns
        -------
        np.ndarray
            The class probabilities, shape ``(n, levels)``.

        Raises
        ------
        ValueError
            If ``node`` is continuous.
        """
        if not isinstance(self.spec[node], OrdinalNode):
            # a domain error (wrong node kind), not a Python type error
            raise ValueError(  # noqa: TRY004
                f"pmf() requires an ordinal node, '{node}' is continuous."
            )
        df_local = df.copy()
        for col, val in (do or {}).items():
            df_local[col] = val
        nd = self.nodes[node]
        cols = list(nd.parents) + self._vc_ehat_columns(nd)  # + e_hat inputs
        values = self._tensorize(df_local, cols)
        feats = self._features({p: values[p] for p in nd.parents})
        theta, shift = nd.theta_shift(
            feats, len(df_local), vc_ehat=self._vc_ehat_live(nd, values, len(df_local))
        )
        return ordinal_pmf(theta, shift).cpu().numpy()

    @torch.no_grad()
    def density(
        self,
        df: pd.DataFrame,
        node: str,
        grid,
        do: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Give the analytic conditional density of a continuous node on a grid.

        The continuous counterpart of :meth:`pmf`: for every row of ``df``
        the density ``p(node = g | parents)`` at each grid value ``g``, in
        closed form from the transform — no sampling.

        Parameters
        ----------
        df : pd.DataFrame
            Rows that supply the parent values of the node.
        node : str
            Name of the continuous node.
        grid : array-like
            Values of ``node`` at which to evaluate the density, shape ``(m,)``.
        do : dict[str, float] | None, optional
            Column overrides, applied to ``df`` before the evaluation.

        Returns
        -------
        np.ndarray
            The densities, shape ``(n, m)``.

        Raises
        ------
        ValueError
            If ``node`` is ordinal; use :meth:`pmf` for it.
        """
        if not isinstance(self.spec[node], ContinuousNode):
            # a domain error (wrong node kind), not a Python type error
            raise ValueError(  # noqa: TRY004
                f"density() requires a continuous node, '{node}' is ordinal; use pmf()."
            )
        df_local = df.copy()
        for col, val in (do or {}).items():
            df_local[col] = val
        nd = self.nodes[node]
        n = len(df_local)
        values = self._tensorize(df_local, list(nd.parents) + self._vc_ehat_columns(nd))
        feats = self._features({p: values[p] for p in nd.parents})
        theta, shift = nd.theta_shift(
            feats, n, vc_ehat=self._vc_ehat_live(nd, values, n)
        )
        y = torch.as_tensor(np.asarray(grid, dtype=self._np_dtype), device=self.device)
        m = y.numel()
        # one (row, grid value) pair per evaluation: rows repeat, the grid tiles
        u0, ladj = nd.ut.forward(theta.repeat_interleave(m, 0), y.repeat(n))
        log_p = StandardLogistic.log_prob(u0 + shift.repeat_interleave(m)) + ladj
        return log_p.exp().view(n, m).cpu().numpy()

    @torch.no_grad()
    def scores(self, df: pd.DataFrame, node: str) -> pd.DataFrame:
        """Give the per-observation scores ``psi_i = d l_i / d theta``.

        The scores belong to the interpretable shift coefficients of a node
        and are analytic and exact, see ``tramdag.scores`` (issue #29).

        At a fitted MLE each column sums to about zero. Order the rows by a
        covariate that truly modifies the treatment effect, and the
        cumulative sum of the treatment column drifts.
        :meth:`effect_modifier_scan` measures that drift.

        This is a pure read-out. It touches no fitting or sampling code
        path.

        Parameters
        ----------
        df : pd.DataFrame
            Observations. Must contain the node, its parents, and the
            propensity inputs of centered VC terms.
        node : str
            Name of the node whose coefficients are scored.

        Returns
        -------
        pd.DataFrame
            One column per coefficient, one row per observation. See
            :func:`tramdag.scores.node_scores` for the column naming.
        """
        return _node_scores(self, df, node)

    @torch.no_grad()
    def effect_modifier_scan(
        self, df: pd.DataFrame, node: str, t: str, candidates: list[str] | None = None
    ) -> pd.DataFrame:
        """Rank candidate effect modifiers with a fluctuation scan.

        Issue #29 describes the Zeileis-Hornik method. Each candidate
        covariate is ranked by how strongly the scores of the ``t``
        coefficient drift when the rows are ordered by it. A cheap
        all-``ls`` fit is enough, so this gives a measured shortlist for
        ``VC`` modifiers.

        Parameters
        ----------
        df : pd.DataFrame
            Observations, as for :meth:`scores`.
        node : str
            Name of the outcome node.
        t : str
            Name of the treatment whose coefficient is scanned.
        candidates : list[str] | None, optional
            Candidate covariates. Defaults to every column of ``df``
            except ``node`` and ``t``.

        Returns
        -------
        pd.DataFrame
            One row per candidate, sorted by ``stat`` descending, with
            columns ``stat``, ``p_value``, ``crit_5pct`` and ``flag``. See
            :func:`tramdag.scores.effect_modifier_scan`.
        """
        return _effect_modifier_scan(self, df, node, t, candidates=candidates)

    def save(self, path: str | Path) -> None:
        """Write the model, its history and its provenance to a checkpoint.

        The file holds the spec and the weights, the training ``history``,
        and a ``meta`` block with the tramdag version, the save time, the
        device, and the machine that trained the model. A cached run
        therefore stays self-describing: the file alone is enough to
        rebuild a training-curve plot or to compare timings.

        Parameters
        ----------
        path : str | Path
            Target file. Parent directories are created when missing.
        """
        from . import __version__  # lazy: circular through the package root

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "tramdag_version": __version__,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device": str(self.device),
            "machine": machine_info(),
        }
        torch.save(
            {
                "spec": spec_to_dict(self.spec),
                "state_dict": self.state_dict(),
                "history": self.history,
                "meta": meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> CausalFlowDAG:
        """Restore a model from a checkpoint.

        ``flow.history`` and ``flow.meta`` are refilled. A cached model can
        therefore still produce training and diagnostic plots, and can
        report the machine that trained it.

        Parameters
        ----------
        path : str | Path
            Checkpoint file written by :meth:`save`.
        device : str, optional
            Torch device to load onto, by default ``"cpu"``.

        Returns
        -------
        CausalFlowDAG
            The restored model, in eval mode.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        flow = cls(spec_from_dict(ckpt["spec"]), device=device)
        # A loaded model carries trained parameters, so both first-fit guards
        # must already be closed: re-fitting with marginal_init=True would
        # otherwise reset the intercept to the data marginal.
        for name in flow.order:
            node = flow.nodes[name]
            if node.kind == "continuous":
                node.ut._fitted = True
            elif isinstance(node.intercept, SimpleIntercept):
                node.intercept._marginal_inited = True
        flow.load_state_dict(ckpt["state_dict"])
        flow.history = ckpt["history"]
        flow.meta = ckpt["meta"]
        flow.eval()
        return flow
