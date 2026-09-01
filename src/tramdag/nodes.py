"""The internal node model: one sub-model per variable.

`_Node` bundles a variable's intercept (transform parameters), monotone
transform and shift modules; `CausalFlowDAG` holds one per node and the DAG
lives in which parents each node reads.

Internal-but-stable surface
---------------------------
scores.py, callbacks recipes, the read-outs and the test suite read these
names by design; renaming any of them is an API change, not a cleanup:
``shifts``, ``intercept`` / ``intercept_nets`` / ``_intercept_groups``,
``_shift_groups``, ``_vc_groups``, ``ut``, ``input_transforms``,
``net_input``, ``theta_shift``, ``vc_column``.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

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
from .spec import (
    ContinuousNode,
    NodeSpec,
    OrdinalNode,
    node_parents,
)
from .transforms import make_univariate_transform


# %% private functions -----------------------------------------------------------------
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
        terms = node.terms
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
