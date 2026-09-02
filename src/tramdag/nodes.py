"""The internal node model: one sub-model per variable.

`_Node` bundles a variable's intercept (transform parameters), monotone
transform and shift modules; `CausalFlowDAG` holds one per node and the DAG
lives in which parents each node reads.

Internal-but-stable surface
---------------------------
scores.py, callbacks recipes, the read-outs and the test suite read these
names by design; renaming any of them is an API change, not a cleanup:
``shifts``, ``intercept`` (+ ``.nets``/``.groups``) / ``_intercept_groups``,
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

from .spec import (
    ContinuousNode,
    NodeSpec,
    node_parents,
)
from .transforms import (
    StandardLogistic,
    make_univariate_transform,
    ordinal_abduct,
    ordinal_log_prob,
    ordinal_marginal_init_theta,
    ordinal_sample,
)


# %% private functions -----------------------------------------------------------------
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
        self.input_transforms = nn.ModuleDict()
        if isinstance(node, ContinuousNode):
            self.ut = make_univariate_transform(node.transform, **node.transform_kwargs)
            n_params = self.ut.n_params
            self.levels = None
        else:
            self.ut = None
            self.levels = node.levels
            n_params = node.levels - 1
        self._build_intercept(terms[0], n_params, spec)
        self._build_shifts(terms, spec)

    def _add_input_transform(self, key: str, term, parents, spec) -> None:
        """Register one term's ``input_transform`` under its net key.

        Keyed like the shift ModuleDict plus ``"@I"`` for the intercept term;
        identity until ``calibrate`` freezes the statistics. Only continuous
        parents transform — ordinal one-hots pass through.
        """
        if getattr(term, "input_transform", None) is None:  # LS takes none
            return
        cps = tuple(p for p in parents if isinstance(spec[p], ContinuousNode))
        if cps:
            self.input_transforms[key] = _InputTransform(term.input_transform, cps)

    def _build_intercept(self, i_term, n_params: int, spec: dict[str, NodeSpec]):
        """Build the node's one intercept term module, via the registry.

        The term class decides its own shape: the free SimpleIntercept
        theta_0, one joint ComplexIntercept, or one net per parent summed in
        unconstrained coefficient space (``allow_interaction=False``).
        """
        from .terms import get_term

        if i_term.parents:
            self._add_input_transform("@I", i_term, i_term.parents, spec)
        self.intercept = get_term(i_term.effect).build(i_term, spec, n_params)

    @property
    def _intercept_groups(self) -> list[tuple[str, ...]]:
        """The intercept term's parent groups (legacy view for read-outs)."""
        return self.intercept.groups

    @property
    def ci_parents(self) -> list[str]:
        """The intercept term's flat parent order, for introspection."""
        return self.intercept.ci_parents

    def _build_shifts(self, terms, spec: dict[str, NodeSpec]):
        """Build one shift term module per LS/CS/VC term, via the registry.

        Each term class constructs itself exactly as this method used to
        (same widths, same order — the seeded RNG stream is pinned) and
        names its own ModuleDict key: the parent for single-parent terms,
        "a+b" for a joint CS, the treatment for a VC.
        """
        from .terms import ShiftTerm, VCTerm, get_term

        self.shifts = nn.ModuleDict()
        self._shift_groups: list[tuple[str, tuple[str, ...]]] = []
        for term in terms:
            entry = get_term(term.effect)
            if entry.slot != "shift":
                continue
            assert issubclass(entry, ShiftTerm), term.effect
            m = entry.build(term, spec)
            self.shifts[m.key] = m
            self._add_input_transform(m.key, term, m.net_parents, spec)
            if not isinstance(m, VCTerm):  # VC evaluates after the plain shifts
                self._shift_groups.append((m.key, m.parents))

    @property
    def _vc_groups(self) -> list[_VCGroup]:
        """The node's VC terms in the legacy group view (scores/read-outs)."""
        from .terms import VCTerm

        return [m.group for m in self.shifts.values() if isinstance(m, VCTerm)]

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
        from .terms import VCTerm

        theta = self.intercept.theta_value(self, feats, n)
        shift = torch.zeros(n, dtype=theta.dtype, device=theta.device)
        # plain shifts first, then the VC terms — today's accumulation order
        for key, _ps in self._shift_groups:
            shift = shift + self.shifts[key].shift_value(self, feats, vc_ehat)
        for m in self.shifts.values():
            if isinstance(m, VCTerm):
                shift = shift + m.shift_value(self, feats, vc_ehat)
        return theta, shift


# %% per-kind operations ---------------------------------------------------------------
# The ONLY continuous-vs-ordinal branches of the package live in these four
# adjacent functions. A third node kind earns a protocol; two stay an if/else
# in one place.
def kind_log_prob(node: _Node, theta: Tensor, shift: Tensor, x: Tensor) -> Tensor:
    """``log p(x | pa)`` from one node's transform parameters and shift."""
    if node.kind == "continuous":
        u0, ladj = node.ut.forward(theta, x)
        return StandardLogistic.log_prob(u0 + shift) + ladj
    return ordinal_log_prob(theta, shift, x)


def kind_sample(node: _Node, theta: Tensor, shift: Tensor, u: Tensor) -> Tensor:
    """Push one node's latent ``u`` forward to an observed value."""
    if node.kind == "continuous":
        return node.ut.inverse(theta, u - shift)
    return ordinal_sample(theta, shift, u)


def kind_abduct(
    node: _Node, theta: Tensor, shift: Tensor, x: Tensor, generator=None
) -> Tensor:
    """Recover one node's latent: exact (continuous) or truncated-sampled (ordinal)."""
    if node.kind == "continuous":
        u0, _ = node.ut.forward(theta, x)
        return u0 + shift
    return ordinal_abduct(theta, shift, x, generator=generator)


def kind_marginal_theta(node: _Node, column: np.ndarray):
    """Give the marginal-start theta of a simple intercept, or ``None``.

    Ordinal: the empirical class log-odds. Continuous: the transform's own
    calibrated start (``None`` for spline/affine — nothing to set).
    """
    if node.kind == "ordinal":
        counts = np.bincount(column.astype(np.int64), minlength=node.levels)
        return ordinal_marginal_init_theta(counts)
    return node.ut.marginal_init_theta()
