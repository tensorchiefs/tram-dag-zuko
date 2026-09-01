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

import inspect
import pickle
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
    ContinuousNode,
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

# %% global variables ------------------------------------------------------------------
__all__ = ["CausalFlowDAG"]


# %% private functions -----------------------------------------------------------------
def _callback_list(cbs) -> list:
    """Normalize a ``fit`` callback argument: None, one callable, or a sequence."""
    if cbs is None:
        return []
    return [cbs] if callable(cbs) else list(cbs)


def _check_fit_sizes(epochs: int, batch_size: int) -> None:
    """Reject a non-positive epoch or batch budget before anything runs."""
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")


def _split_validation(
    train_df: pd.DataFrame,
    validation_data: pd.DataFrame | None,
    validation_split: float | None,
    verbose: int,
    vc_ehat: dict | None,
):
    """Resolve fit's validation arguments (the Keras rules).

    ``validation_split`` takes the LAST fraction of ``train_df`` as
    validation without shuffling, exactly like Keras — deterministic, no
    hidden RNG. ``vc_ehat`` rows are sliced with the same split, so the
    caller supplies propensities for the frame they passed.
    """
    if verbose < 0 or int(verbose) != verbose:
        raise ValueError(f"verbose must be a non-negative int, got {verbose!r}")
    if validation_split is None:
        return train_df, validation_data, vc_ehat
    if validation_data is not None:
        raise ValueError("pass validation_data OR validation_split, not both")
    if not 0.0 < validation_split < 1.0:
        raise ValueError(f"validation_split must be in (0, 1), got {validation_split}")
    cut = round(len(train_df) * (1.0 - validation_split))
    if cut < 1 or cut >= len(train_df):
        raise ValueError(
            f"validation_split={validation_split} leaves no rows on one side "
            f"of the {len(train_df)}-row frame"
        )
    if vc_ehat is not None:
        vc_ehat = {
            node: {t: np.asarray(e)[:cut] for t, e in d.items()}
            for node, d in vc_ehat.items()
        }
    return train_df.iloc[:cut], train_df.iloc[cut:], vc_ehat


def _check_callbacks(cbs: list, args: tuple[str, ...], name: str) -> None:
    """Reject a mis-registered callback before training, not hours after it.

    The classic slip is the instance/method swap (``RestoreBest`` itself in
    ``after_fit_callbacks`` instead of its ``restore``) — without this check
    that only raises after the last epoch, losing the whole run.
    """
    for cb in cbs:
        if not callable(cb):
            raise TypeError(f"{name}_callbacks entries must be callable, got {cb!r}")
        try:
            sig = inspect.signature(cb, follow_wrapped=False)
        except (TypeError, ValueError):  # a callable without a signature
            continue
        try:
            sig.bind(*[None] * len(args))
        except TypeError:
            raise TypeError(
                f"{name}_callbacks are called as cb({', '.join(args)}); "
                f"{cb!r} does not accept these arguments"
            ) from None


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


def _init_linear(m: nn.Linear, init: str) -> None:
    """Keras' two initializers on one linear layer: ``glorot`` or ``normal``."""
    if init == "glorot":
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    else:
        nn.init.normal_(m.weight, std=0.05)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=0.05)


def _is_classical_term(term) -> bool:
    """Say whether the exact classical fit handles this term.

    It handles an ``LS``, and a parentless ``I()`` — the simple-intercept
    baseline made explicit, for example as the carrier of ``transform=``.
    """
    return term.effect == "LS" or (term.effect == "I" and not term.parents)


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
        ``True`` centers the regressor with the caller's out-of-fold
        propensities (``fit(vc_ehat=)``).
    """

    on: str
    mods: tuple[str, ...]
    on_is_ord: bool
    center: bool


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

    def __init__(
        self,
        node: NodeSpec,
        spec: dict[str, NodeSpec],
    ):
        super().__init__()
        self.kind = node.kind
        terms = node_terms(node)
        self.parents = tuple(node_parents(node))  # ordered parent names
        self.continuous_parents = tuple(
            p for p in self.parents if isinstance(spec[p], ContinuousNode)
        )
        self.input_transforms = nn.ModuleDict()
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

    def _add_input_transform(self, key: str, term, parents, spec) -> None:
        """Register one term's ``input_transform`` under its net key.

        Keyed like the shift ModuleDict plus ``"@I"`` for the intercept term;
        identity until ``calibrate`` freezes the statistics. Only continuous
        parents transform — ordinal one-hots pass through.
        """
        if term.input_transform is None:
            return
        cps = tuple(p for p in parents if isinstance(spec[p], ContinuousNode))
        if cps:
            self.input_transforms[key] = _InputTransform(term.input_transform, cps)

    def _build_intercept(self, i_term, n_params: int, spec: dict[str, NodeSpec]):
        """Build the intercept module(s) from the intercept groups.

        By group count: none -> the free SimpleIntercept theta_0; one (a
        single parent, or a joint multi-parent term) -> one ComplexIntercept
        that IS theta; several (``allow_interaction=False``) -> one net per
        parent, their outputs summed in unconstrained coefficient space, so
        each parent reshapes the transform independently.
        """
        i_groups = self._intercept_groups
        if i_term is not None:
            self._add_input_transform("@I", i_term, i_term.parents, spec)
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
                self._add_input_transform(on, term, mods, spec)
                self._vc_groups.append(
                    _VCGroup(
                        on,
                        mods,
                        isinstance(spec[on], OrdinalNode),
                        term.center,
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
                if term.effect == "CS":
                    self._add_input_transform(key, term, ps, spec)
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

    def set_input_stats(self, train_df: pd.DataFrame) -> None:
        """Freeze every term's input-transform statistics (``calibrate``)."""
        for tr in self.input_transforms.values():
            # a constant column fails calibrate's quantile check on the
            # parent's own node first, so the statistics are well defined
            cols = torch.stack(
                [
                    torch.as_tensor(train_df[p].to_numpy(dtype=np.float32).copy())
                    for p in tr.cols
                ],
                dim=1,
            )
            tr.set_stats(cols)

    def net_input(self, feats: dict[str, Tensor], parents, key: str) -> Tensor:
        """Concatenate parent features for one term's network.

        ``key`` names the term ("@I" for the intercept, the shift key
        otherwise); a term with an ``input_transform`` gets its continuous
        parent columns transformed with the statistics frozen at
        ``calibrate``. Every network input goes through here — training and
        the read-outs (``varying_coef``, ``intercept_contributions``) alike —
        so the model seen at inference is the model that was fitted. Linear
        shifts and the VC treatment column are not network inputs and never
        pass through.
        """
        # no dict.get: nn.ModuleDict has no get()
        tr = self.input_transforms[key] if key in self.input_transforms else None  # noqa: SIM401
        cols = []
        for p in parents:
            x = feats[p]
            if tr is not None and p in tr.cols:
                x = tr(x, tr.cols.index(p))
            cols.append(x)
        return torch.cat(cols, dim=1)

    def _theta(self, feats: dict[str, Tensor], n: int) -> Tensor:
        """Evaluate the intercept: the transform parameters, shape ``(n, P)``."""
        if self.intercept_nets is not None:  # additive complex intercept
            return sum(
                net(self.net_input(feats, grp, "@I"))
                for net, grp in zip(
                    self.intercept_nets, self._intercept_groups, strict=True
                )
            )
        if self.ci_parents:  # single or joint complex intercept
            return self.intercept(self.net_input(feats, self.ci_parents, "@I"))
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
        mod_feat = self.net_input(feats, g.mods, g.on) if g.mods else None
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
            module = self.shifts[key]
            feat = (  # a linear shift stays raw: its weight is the paper's beta
                torch.cat([feats[p] for p in ps], dim=1)
                if isinstance(module, LinearShift)
                else self.net_input(feats, ps, key)
            )
            shift = shift + module(feat)
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
    init : str, optional
        Weight initialization of every linear layer (the LS weights and the
        CI/CS/VC networks): ``"torch"`` (default, ``nn.Linear``'s
        Kaiming-uniform), ``"glorot"`` — Keras' ``Dense`` default,
        glorot-uniform weights and zero biases, the paper's VACA/CAREFL
        scripts — or ``"normal"`` — Keras' ``RandomNormal``, N(0, 0.05^2)
        on weights and biases, the paper's triangle scripts. A VC head's
        output layer stays zero either way. Under the reference's full-batch
        protocol the choice decides the fit: VACA's ``do(x2)`` error at the
        config's seed is 0.52 / 0.33 / 0.13 with torch's init and
        0.098 / 0.159 / 0.026 with glorot (another draw: 0.035 / 0.006 /
        0.007; measured 2026-08-26). Stored in the checkpoint.
    """

    def __init__(
        self,
        spec: dict[str, NodeSpec],
        device: str = "cpu",
        seed: int | None = None,
        init: str = "torch",
    ):
        super().__init__()
        if init not in ("torch", "glorot", "normal"):
            raise ValueError(
                f"init must be 'torch', 'glorot' or 'normal', got {init!r}"
            )
        if seed is not None:
            torch.manual_seed(seed)
        self.spec = spec
        self.order = validate_and_sort(spec)
        self.init = init
        self.nodes = nn.ModuleDict(
            {name: _Node(spec[name], spec) for name in self.order}
        )
        self._apply_init(init)
        self.device = torch.device(device)
        # the data-dependent state (ranges, net min-max, calibrated start) is
        # taken once, by calibrate(); a checkpoint carries the flag
        self.register_buffer("calibrated", torch.tensor(False))
        self.history: dict = {"train": []}  # per-node mean train NLL per epoch
        self.meta: dict = {}  # provenance attached at save() (version, time)
        self.to(self.device)

    def _apply_init(self, init: str) -> None:
        """Re-initialize every linear layer the Keras way, if asked.

        ``"glorot"`` is Keras' ``Dense`` default (glorot-uniform weights, zero
        biases); ``"normal"`` is Keras' ``RandomNormal`` (N(0, 0.05^2)) on
        weights and biases, the initializer of the paper's triangle scripts.
        A VC head's output layer stays zero: ``beta(x) = beta0`` at the start
        is part of that term's design.
        """
        if init == "torch":
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_linear(m, init)
        for m in self.modules():
            if isinstance(m, VaryingCoef) and m.net is not None:
                nn.init.zeros_(m.net[-1].weight)

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

    def _to_frame(self, values: dict[str, Tensor]) -> pd.DataFrame:
        """Tensors -> DataFrame; an ordinal column goes back as a level index."""
        out = {}
        for k, v in values.items():
            arr = v.cpu().numpy()
            out[k] = arr.astype(np.int64) if self.nodes[k].kind == "ordinal" else arr
        return pd.DataFrame(out)

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
        by the spec, so a treatment node never carries a centered term itself.
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
            Restrict the computation to these nodes. A subset is exact
            because the per-node losses
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
        with torch.no_grad():
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

    def calibrate(
        self, train_df: pd.DataFrame, *, marginal_init: bool = True
    ) -> CausalFlowDAG:
        """Take the data-dependent state from the training rows, once.

        Per continuous node the transform's domain: the train ``RANGE_Q`` /
        ``1 - RANGE_Q`` quantiles map onto ``[-5, 5]`` (the min-max scaling
        of the original implementation, made robust to outliers). The
        statistics of every term-level ``input_transform=`` (minmax lo/hi,
        standardize mean/std, a callable's frozen train columns). With
        ``marginal_init`` a calibrated start:
        a Bernstein simple intercept starts at the data marginal instead of
        zuko's default (about 2.5x too steep), an ordinal simple intercept at
        the marginal class log-odds. The optimum is unchanged, the path to
        it is shorter (docs/training-speed.md).

        The first ``fit`` or ``fit_classical`` calls this when it has not run
        yet; a loaded model is already calibrated, and later fits on other rows
        reuse this state — data on a new scale needs a new flow. Calling it
        yourself is how to choose ``marginal_init=False`` (the paper
        replications do, to match a reference that has no such step). To
        re-apply the calibrated start later — a loaded or already-trained
        model — call :meth:`init_marginals` directly.

        Returns
        -------
        CausalFlowDAG
            ``self``.
        """
        if bool(self.calibrated):
            return self
        for name in self.order:
            node = self.nodes[name]
            if node.kind == "ordinal":
                self._check_levels(name, train_df)
            node.set_input_stats(train_df)
            if node.kind == "continuous":
                self._set_range(name, train_df)
        self.calibrated.fill_(True)
        if marginal_init:
            self.init_marginals(train_df)
        return self

    def init_marginals(self, train_df: pd.DataFrame) -> CausalFlowDAG:
        """Set every simple intercept to the marginal of its column — any time.

        The calibrated start, as an explicit step: a Bernstein simple
        intercept starts at the data marginal instead of zuko's default
        (about 2.5x too steep), an ordinal simple intercept at the marginal
        class log-odds; spline/affine intercepts and intercepts with parents
        are untouched. The optimum is unchanged, the path to it is shorter
        (docs/training-speed.md). ``calibrate(marginal_init=True)`` — and so
        the first ``fit`` — runs this once; unlike ``calibrate`` it is NOT
        guarded by the calibrated flag, so calling it on a loaded or
        already-trained flow **discards those intercepts' weights** and
        restarts them at the marginal. An uncalibrated flow takes its ranges
        from the same rows first. On a calibrated flow the Bernstein start
        comes from the stored range alone (the canonical map; the df is not
        read) — only ordinal intercepts re-read the rows.

        Returns
        -------
        CausalFlowDAG
            ``self``.
        """
        if not bool(self.calibrated):
            self.calibrate(train_df, marginal_init=False)
        for name in self.order:
            node = self.nodes[name]
            if isinstance(node.intercept, SimpleIntercept):
                if node.kind == "ordinal":
                    self._check_levels(name, train_df)
                self._marginal_start(name, train_df)
        return self

    def _set_range(self, name: str, train_df: pd.DataFrame) -> None:
        """Map the train ``RANGE_Q``/1-``RANGE_Q`` quantiles onto the domain."""
        q = train_df[name].quantile([RANGE_Q, 1.0 - RANGE_Q])
        if q.iloc[1] <= q.iloc[0]:
            raise ValueError(
                f"node {name!r}: the {RANGE_Q:.0%} and {1 - RANGE_Q:.0%} quantiles "
                f"coincide at {q.iloc[0]}. A continuous node needs a spread of "
                "values; a level index needs OrdinalNode()."
            )
        self.nodes[name].ut.set_range(q.iloc[0], q.iloc[1])

    def _marginal_start(self, name: str, train_df: pd.DataFrame) -> None:
        """Start a simple intercept at the node's data marginal."""
        node = self.nodes[name]
        if node.kind == "ordinal":
            counts = np.bincount(
                train_df[name].to_numpy().astype(np.int64),
                minlength=self.spec[name].levels,
            )
            theta = ordinal_marginal_init_theta(counts)
        elif isinstance(node.ut, BernsteinUT):
            theta = node.ut.marginal_init_theta()
        else:  # a spline or affine transform has no calibrated start
            return
        with torch.no_grad():
            node.intercept.theta.copy_(theta)

    def _check_levels(self, name: str, train_df: pd.DataFrame) -> None:
        """Reject an ordinal column that is not a level index of its node.

        ``bincount`` and the cutpoint likelihood both take the column as
        ``0..levels-1``; a 1-based or non-integer column would silently be
        truncated instead of failing.
        """
        levels = self.spec[name].levels
        v = train_df[name].to_numpy(dtype=np.float64)
        fractional = bool((v != np.round(v)).any())
        if fractional or v.min() < 0 or v.max() >= levels:
            raise ValueError(
                f"node {name!r}: an ordinal column holds the level indices "
                f"0..{levels - 1}, got values in [{v.min()}, {v.max()}]"
                f"{' (non-integer)' if fractional else ''}"
            )

    def _vc_ehat_train(
        self, train_df: pd.DataFrame, vc_ehat: dict | None
    ) -> dict[str, dict[str, Tensor]] | None:
        """Turn the user's stage-1 propensities into the training-time tensors.

        A centered ``VC`` term needs ``vc_ehat[node][t]``: ``P(t = 1 | pa_t)``
        for every training row, positionally aligned with ``train_df``,
        computed **out of fold** (the cross-fitting
        requirement of the DML design; in-sample values reintroduce the
        own-observation bias). How they are computed is the user's choice —
        a ``fit_classical`` on the treatment spec per fold, or any
        classifier. The training loss uses these frozen values; every query
        after the fit uses the live treatment node (:meth:`_vc_ehat_live`).
        """
        centered = {
            (name, g.on)
            for name in self.order
            for g in self.nodes[name]._vc_groups
            if g.center
        }
        given = {(node, on) for node, d in (vc_ehat or {}).items() for on in d}
        if centered != given:
            raise ValueError(
                "fit(vc_ehat=) must hold exactly the centered VC terms "
                f"{sorted(centered)} as {{node: {{t: P(t=1|pa_t) per row}}}}, "
                f"got {sorted(given)}. The values must be out of fold."
            )
        if not centered:
            return None
        n = len(train_df)
        out: dict[str, dict[str, Tensor]] = {}
        for node, on in centered:
            e = np.asarray(vc_ehat[node][on], dtype=self._np_dtype).reshape(-1)
            if len(e) != n:
                raise ValueError(
                    f"vc_ehat[{node!r}][{on!r}] has {len(e)} rows, not {n}"
                )
            if not ((e >= 0) & (e <= 1)).all():
                raise ValueError(
                    f"vc_ehat[{node!r}][{on!r}] must hold probabilities in [0, 1]"
                )
            out.setdefault(node, {})[on] = torch.as_tensor(e, device=self.device)
        return out

    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        epochs: int,
        learning_rate: float = 1e-2,
        batch_size: int = 512,
        validation_data: pd.DataFrame | None = None,
        validation_split: float | None = None,
        validation_batch_size: int | None = None,
        verbose: int = 0,
        seed: int | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        before_fit_callbacks=None,
        after_epoch_callbacks=None,
        after_fit_callbacks=None,
        vc_ehat: dict | None = None,
    ) -> CausalFlowDAG:
        """Fit all nodes jointly by maximum likelihood — one minibatch Adam loop.

        The joint NLL decomposes per node with independent gradients, so one
        optimizer over all parameters is the same as one per node. The loop
        keeps the **final** weights: an all-``ls`` model trained to
        convergence reproduces the classical maximum-likelihood estimate and
        matches ``statsmodels`` and R ``polr``. A second ``fit`` call
        continues the training. Everything else — validation monitoring,
        learning-rate schedules, early stopping, best-weight restoration,
        logging — is the caller's, through ``optimizer`` and the callback
        hooks; :mod:`tramdag.callbacks` ships the common recipes::

            from tramdag.callbacks import RestoreBest

            best = RestoreBest()
            flow.fit(
                train_df,
                epochs=4000,
                validation_data=val_df,
                verbose=50,
                after_epoch_callbacks=[best],
                after_fit_callbacks=[best.restore],
            )

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data, one column per node.
        epochs : int
            Number of passes over the data. There is no default: a fixed
            budget over-spends on some workloads and under-spends on others
            (docs/training-speed.md).
        learning_rate : float, optional
            Adam step size of the default optimizer, by default 1e-2.
            Ignored when ``optimizer`` is given.
        batch_size : int, optional
            Rows per gradient step, by default 512. ``len(train_df)`` is one
            full-batch step per epoch.
        validation_data : pd.DataFrame | None, optional
            Validation rows, one column per node. When given (or split off),
            the per-node validation NLL is computed after every epoch and
            appended to ``flow.history["val"]`` — once, centrally; the
            shipped callbacks read it there.
        validation_split : float | None, optional
            Keras' rule: the LAST fraction of ``train_df`` becomes the
            validation set, without shuffling, and only the remaining rows
            train (and calibrate — no leakage into the frozen statistics).
            Mutually exclusive with ``validation_data``.
        validation_batch_size : int | None, optional
            Chunk size of the validation pass, by default one full batch.
        verbose : int, optional
            0 (default) is silent. ``N >= 1`` prints one line every ``N``
            epochs and on the final epoch: epoch counter, summed train NLL,
            summed validation NLL when validation is configured. No
            progress bars.
        seed : int | None, optional
            Seeds torch's global RNG before the loop, for the minibatch
            shuffling. Weight initialization is seeded at construction
            (``CausalFlowDAG(spec, seed=...)``).
        optimizer : torch.optim.Optimizer | None, optional
            Any torch optimizer over ``flow.parameters()``; the default is
            ``Adam(lr=learning_rate)``. Build it yourself to attach a
            ``torch.optim.lr_scheduler`` or to continue with its state.
        before_fit_callbacks : callable | list[callable] | None, optional
            Each called as ``cb(flow, optimizer)`` once, after calibration
            and before the first epoch.
        after_epoch_callbacks : callable | list[callable] | None, optional
            Each called as ``cb(flow, epoch, optimizer)`` after every epoch
            (``epoch`` counts from 1), once the epoch's train NLLs are in
            ``flow.history["train"]``. All of them run each epoch; the fit
            stops after an epoch in which any returned ``True``
            (``len(flow.history["train"])`` says how many epochs ran). Use them
            for schedules, snapshots and coefficient trajectories —
            :mod:`tramdag.callbacks` ships ``RestoreBest`` and
            ``PerNodePlateau``, both reading ``history["val"]``.
        after_fit_callbacks : callable | list[callable] | None, optional
            Each called as ``cb(flow, optimizer)`` once, after the loop and
            **before** the VC re-centering — so a callback that swaps the
            weights (``RestoreBest.restore``) hands them to the re-centering.
        vc_ehat : dict | None, optional
            Out-of-fold propensities ``{node: {t: array}}`` for every centered
            ``VC`` term, one value per training row; required when the spec
            has one (see docs/varying-coefficients.md).

        Returns
        -------
        CausalFlowDAG
            ``self``, fitted, in eval mode.

        Raises
        ------
        ValueError
            If ``epochs`` or ``batch_size`` is below 1, ``verbose`` is
            negative, both validation arguments are given, the split leaves
            an empty side, or ``vc_ehat`` does not match the centered VC
            terms of the spec.
        TypeError
            If a callback does not accept its hook's arguments — checked
            before the first epoch, so a mis-registered callback cannot
            waste a run.

        Notes
        -----
        For ``VC`` terms the objective is the **penalized** NLL on the
        total-likelihood scale: each term adds
        ``penalty * ||b_theta weights||^2`` to the summed NLL, that is
        ``penalty * ||w||^2 / n_train`` to the mean loss — a fixed Gaussian
        prior whose shrinkage vanishes as n grows. ``beta0`` is not
        penalized, and ``history["train"]`` holds pure likelihoods. After
        the loop each ``b_theta`` is re-centered to mean zero over the
        training rows; the constant moves into ``beta0``, the function is
        unchanged.
        """
        _check_fit_sizes(epochs, batch_size)
        before = _callback_list(before_fit_callbacks)
        after_epoch = _callback_list(after_epoch_callbacks)
        after_fit = _callback_list(after_fit_callbacks)
        _check_callbacks(before, ("flow", "optimizer"), "before_fit")
        _check_callbacks(after_epoch, ("flow", "epoch", "optimizer"), "after_epoch")
        _check_callbacks(after_fit, ("flow", "optimizer"), "after_fit")
        if seed is not None:
            torch.manual_seed(seed)
        train_df, validation_data, vc_ehat = _split_validation(
            train_df, validation_data, validation_split, verbose, vc_ehat
        )
        # vc_ehat is validated BEFORE calibrate: a malformed dict must fail
        # while the flow is still untouched (calibrate sets ranges and the
        # marginal start — a half-mutated flow after an error would be worse
        # than no fit at all)
        ehat = self._vc_ehat_train(train_df, vc_ehat)
        self.calibrate(train_df)
        vals = self._tensorize(train_df)
        val_vals = (
            self._tensorize(validation_data) if validation_data is not None else None
        )
        opt = optimizer or torch.optim.Adam(self.parameters(), lr=learning_rate)
        penalized = [
            nd.shifts[g.on]
            for nd in self.nodes.values()
            for g in nd._vc_groups
            if g.mods and nd.shifts[g.on].penalty > 0
        ]
        for cb in before:
            cb(self, opt)
        for epoch in range(1, epochs + 1):
            self._epoch_pass(
                vals, ehat, opt, batch_size, penalized, val_vals, validation_batch_size
            )
            # every callback runs (a stop must not skip a monitoring one)
            stops = [bool(cb(self, epoch, opt)) for cb in after_epoch]
            self._log_epoch(epoch, epochs, verbose, stopped=any(stops))
            if any(stops):
                break
        for cb in after_fit:
            cb(self, opt)  # before the re-centering: restored weights re-center too
        self._recenter_vc(vals)
        self.eval()
        return self

    def _epoch_pass(
        self, vals, ehat, opt, batch_size, penalized, val_vals, validation_batch_size
    ) -> None:
        """Run one training epoch and, when configured, the validation pass."""
        self.train()
        self.history["train"].append(
            self._fit_epoch(vals, ehat, opt, batch_size, penalized)
        )
        self.eval()
        if val_vals is not None:
            self.history.setdefault("val", []).append(
                self._val_nll(val_vals, validation_batch_size)
            )

    def _log_epoch(self, epoch: int, epochs: int, verbose: int, stopped: bool):
        """Print one ``verbose`` progress line on the Nth and the final epoch."""
        last = stopped or epoch == epochs
        if not verbose or (epoch % verbose and not last):
            return
        line = f"epoch {epoch}/{epochs}"
        line += f"  train {sum(self.history['train'][-1].values()):.4f}"
        if self.history.get("val"):
            line += f"  val {sum(self.history['val'][-1].values()):.4f}"
        print(line)

    def _val_nll(
        self, vals: dict[str, Tensor], batch_size: int | None
    ) -> dict[str, float]:
        """Give the per-node mean validation NLL, chunked by validation batch size."""
        n = len(next(iter(vals.values())))
        chunk = batch_size or n
        acc = dict.fromkeys(self.order, 0.0)
        with torch.no_grad():
            for start in range(0, n, chunk):
                batch = {k: v[start : start + chunk] for k, v in vals.items()}
                weight = len(next(iter(batch.values()))) / n
                for k, v in self.node_log_prob(batch).items():
                    acc[k] += float(-v.mean()) * weight
        return acc

    def _fit_epoch(
        self,
        vals: dict[str, Tensor],
        ehat: dict[str, dict[str, Tensor]] | None,
        opt: torch.optim.Optimizer,
        batch_size: int,
        penalized: list,
    ) -> dict[str, float]:
        """One shuffled pass over the rows; give the epoch-mean train NLL per node."""
        n = len(next(iter(vals.values())))
        acc = dict.fromkeys(self.order, 0.0)
        for idx in torch.randperm(n, device=self.device).split(batch_size):
            batch = {k: v[idx] for k, v in vals.items()}
            per_node = self.node_log_prob(batch, vc_ehat=_slice_ehat(ehat, idx))
            nlls = {k: -v.mean() for k, v in per_node.items()}
            loss = torch.stack(list(nlls.values())).sum()
            for m in penalized:  # the penalty joins the loss, not the history
                loss = loss + m.penalty * m.l2() / n
            opt.zero_grad()
            loss.backward()
            opt.step()
            w = len(idx) / n
            for k, v in nlls.items():
                acc[k] += float(v.detach()) * w
        return acc

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
                nd.shifts[g.on].recenter(nd.net_input(feats, g.mods, g.on))

    @torch.no_grad()
    def varying_coef(
        self, df: pd.DataFrame, node: str, *, t: str | None = None
    ) -> np.ndarray:
        """Evaluate the fitted effect function ``beta(x)`` of a ``VC`` term.

        This is the first-class read-out of issue #28. The value comes in
        closed form from the fitted term, as ``beta0 + b_theta(modifiers)``.
        It is deterministic and needs no abduction. It is free of ``y``,
        because only the modifier columns of ``df`` are read. For a binary
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
        df : pd.DataFrame
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
            if a modifier column is missing from ``df``.
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
            feats = self._features(self._tensorize(df, mods))
            mod_feat = nd.net_input(feats, mods, t)
        return nd.shifts[t].beta(mod_feat, len(df)).cpu().numpy()

    def _is_classical(self) -> bool:
        return all(
            _is_classical_term(term)
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
    def intercept_contributions(self, df: pd.DataFrame, node: str) -> dict:
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
        the rows of** ``df``. Each term's contribution is centered to mean
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
        df : pd.DataFrame
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
            missing from ``df``.
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
        missing = [p for p in nd.ci_parents if p not in df.columns]
        if missing:
            raise KeyError(f"df is missing intercept-parent column(s): {missing}")

        feats = self._features(self._tensorize(df, nd.ci_parents))
        # one net per group: the additive case stores them in intercept_nets;
        # a single (possibly joint) I-term is the lone `intercept` network.
        nets = (
            list(nd.intercept_nets) if nd.intercept_nets is not None else [nd.intercept]
        )

        contributions: dict[str, np.ndarray] = {}
        parents: dict[str, tuple] = {}
        baseline = None
        for net, grp in zip(nets, groups, strict=True):
            raw = net(nd.net_input(feats, grp, "@I"))  # (n, P)
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
        the minibatch noise also regularizes the NNs.

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
        parameters move by less than ``tol`` (``tolerance_grad`` is set to
        0, so the gradient never ends the run). ``|grad|`` and individual
        coefficients do *not* settle to machine precision. A continuous
        node's Bernstein intercept, and weakly-identified directions such
        as rare one-hot levels or a flat treatment-effect ridge, keep
        drifting along near-zero-curvature valleys long after the
        likelihood and the well-identified coefficients reach the MLE.
        Correctness is therefore verified by comparison to classical
        software (see ``experiments/misc/validate_ls.py``), not by this flag.
        """
        if not self._is_classical():
            raise ValueError(
                "fit_classical requires an all-`ls` spec, that is every edge "
                "term 'ls'. This spec has cs, ci or vc terms. Use fit() for "
                "flexible models."
            )
        self.calibrate(train_df, marginal_init=False)  # L-BFGS needs no warm start

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
                torch.nn.utils.get_total_norm(
                    [p.grad for p in self.parameters() if p.grad is not None]
                )
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
        self.history["classical"] = {
            k: v for k, v in report.items() if k != "coefficients"
        }
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
        return self._to_frame(values)

    @torch.no_grad()
    def abduct(self, df: pd.DataFrame, *, seed: int | None = None) -> pd.DataFrame:
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
        return pd.DataFrame({k: v.cpu().numpy() for k, v in u.items()}, index=df.index)

    def _conditional(
        self, df: pd.DataFrame, node: str, do: dict[str, float] | None
    ) -> tuple[_Node, Tensor, Tensor, int]:
        """Evaluate one node's conditional at the rows of ``df``.

        ``do`` overrides columns before the parents are read, which is what
        makes :meth:`pmf` and :meth:`density` interventional. Gives the node,
        its transform parameters, its shift and the row count.
        """
        nd = self._node(node)
        df = df.assign(**(do or {}))
        n = len(df)
        values = self._tensorize(df, list(nd.parents) + self._vc_ehat_columns(nd))
        feats = self._features({p: values[p] for p in nd.parents})
        theta, shift = nd.theta_shift(
            feats, n, vc_ehat=self._vc_ehat_live(nd, values, n)
        )
        return nd, theta, shift, n

    @torch.no_grad()
    def pmf(
        self, df: pd.DataFrame, node: str, *, do: dict[str, float] | None = None
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
        _, theta, shift, _ = self._conditional(df, node, do)
        return ordinal_pmf(theta, shift).cpu().numpy()

    @torch.no_grad()
    def density(
        self,
        df: pd.DataFrame,
        node: str,
        grid,
        *,
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
        nd, theta, shift, n = self._conditional(df, node, do)
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
        self,
        df: pd.DataFrame,
        node: str,
        *,
        t: str,
        candidates: list[str] | None = None,
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
        and a ``meta`` block with the tramdag version, the save time and the
        device.

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
        }
        try:
            torch.save(
                {
                    "spec": spec_to_dict(self.spec),
                    "init": self.init,
                    "state_dict": self.state_dict(),
                    "history": self.history,
                    "meta": meta,
                },
                path,
            )
        except (pickle.PicklingError, AttributeError) as err:
            raise ValueError(
                "the spec does not serialize: a callable input_transform "
                "must be a picklable module-level function — use "
                "'minmax'/'standardize', or def the function at module level."
            ) from err

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> CausalFlowDAG:
        """Restore a model from a checkpoint.

        ``flow.history`` and ``flow.meta`` are refilled.

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
        flow = cls(
            spec_from_dict(ckpt["spec"]),
            device=device,
            init=ckpt.get("init", "torch"),
        )
        for name, t in ckpt["state_dict"].items():
            # a callable input_transform's train buffer is shaped at
            # calibrate; give it the checkpoint's shape before loading
            if not name.endswith(".train_cols"):
                continue
            buf = flow.get_buffer(name)
            if buf.shape != t.shape:
                mod_path, _, buf_name = name.rpartition(".")
                flow.get_submodule(mod_path).register_buffer(
                    buf_name, torch.empty_like(t)
                )
        flow.load_state_dict(ckpt["state_dict"])  # includes the `calibrated` flag
        flow.history = ckpt["history"]
        flow.meta = ckpt["meta"]
        flow.eval()
        return flow
