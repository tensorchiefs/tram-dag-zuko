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

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .fitting import _FitMixin
from .nodes import (
    _init_linear,
    _Node,
    kind_abduct,
    kind_log_prob,
    kind_marginal_theta,
    kind_sample,
)
from .readouts import _ReadoutsMixin
from .scores import effect_modifier_scan as _effect_modifier_scan
from .scores import node_scores as _node_scores
from .spec import (
    ContinuousNode,
    NodeSpec,
    OrdinalNode,
    spec_from_dict,
    spec_to_dict,
    validate_and_sort,
)
from .terms import ShiftTerm, get_term
from .transforms import (
    StandardLogistic,
    ordinal_pmf,
)

# %% global variables ------------------------------------------------------------------
__all__ = ["CausalFlowDAG"]


# %% public classes --------------------------------------------------------------------
class CausalFlowDAG(_FitMixin, _ReadoutsMixin, nn.Module):
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
        # calibrate() takes the data-dependent state once; checkpoints carry the flag
        self.register_buffer("calibrated", torch.tensor(False))
        self.history: dict = {"train": []}  # per-node mean train NLL per epoch
        self._fit_validated = (
            False  # True while the LAST fit validated (callbacks read it)
        )
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
            if isinstance(m, ShiftTerm):
                m.post_init()

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

        Raises
        ------
        KeyError
            If the frame lacks one of the columns, by name — a spec/data
            mismatch would otherwise surface deep inside a tensor op.
        """
        cols = self.order if cols is None else cols
        self._check_columns(df, cols)
        dtype = self._np_dtype
        return {
            c: torch.as_tensor(
                df[c].to_numpy(dtype=dtype, copy=True), device=self.device
            )
            for c in cols
        }

    @staticmethod
    def _check_columns(df: pd.DataFrame, cols) -> None:
        """Name the columns ``df`` lacks, before any tensor op would."""
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"the data frame lacks the column(s) {missing}; this needs "
                f"{list(cols)}, the frame has {list(df.columns)}"
            )

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
        ``sigmoid(s - theta_0)``. No side columns: chained centering is refused
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
        # spec columns only; side columns travel through _side_feats
        return {
            name: self._encode_parent(name, vals)
            for name, vals in values.items()
            if name in self.spec
        }

    def _side_feats(
        self, nd: _Node, values: dict[str, Tensor], n: int
    ) -> dict[str, Tensor]:
        """Give the node's side columns: frozen from ``values``, else live.

        Training frames carry them as ordinary columns; queries recompute
        them from the fitted flow (``ShiftTerm.live_side``).
        """
        out = {}
        for m in nd.shifts.values():
            cols = m.side_columns()
            if all(c in values for c in cols):
                out.update({c: values[c] for c in cols})
            else:
                out.update(m.live_side(self, values, n))
        return out

    def _query_side_columns(self, nd: _Node) -> list[str]:
        """List the extra columns a query needs to recompute live side inputs.

        These are the columns beyond ``nd.parents``: the parents of the
        treatment nodes (which cannot be centered themselves, so one level
        is all there is).
        """
        cols = [p for m in nd.shifts.values() for p in m.extra_columns(self)]
        return [c for c in dict.fromkeys(cols) if c not in nd.parents]

    def node_log_prob(
        self,
        values: dict[str, Tensor],
        nodes: list[str] | None = None,
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
            theta, shift = node.theta_shift(
                feats | self._side_feats(node, values, n), n
            )
            out[name] = kind_log_prob(node, theta, shift, values[name])
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

    def calibrate(self, train_df: pd.DataFrame) -> CausalFlowDAG:
        """Take the data-dependent state from the training rows, once.

        Every term calibrates itself: the intercept term maps its node's
        train ``range_q``/``1 - range_q`` quantiles (an intercept option,
        default 5%/95%) onto the transform's pre-scaled domain — ``0.0`` is
        the min-max scaling of the reference comparison scripts' ``scale_df``
        — and every term with an ``input_transform=`` freezes its statistics
        (minmax lo/hi, standardize mean/std, a callable's frozen train
        columns).

        The first ``fit`` or ``fit_classical`` calls this when it has not run
        yet; a loaded model is already calibrated, and later fits on other rows
        reuse this state — data on a new scale needs a new flow. Calibration
        never touches the weights: a calibrated start is a separate, always
        explicit step, :meth:`init_marginals`.

        Returns
        -------
        CausalFlowDAG
            ``self``.
        """
        if bool(self.calibrated):
            return self
        self._check_columns(train_df, self.order)
        for name in self.order:
            node = self.nodes[name]
            if node.kind == "ordinal":
                self._check_levels(name, train_df)
            node.intercept.calibrate(train_df, own=train_df[name], ut=node.ut)
            for term in node.shifts.values():
                term.calibrate(train_df)
        self.calibrated.fill_(True)
        return self

    def init_marginals(self, train_df: pd.DataFrame) -> CausalFlowDAG:
        """Set every simple intercept to the marginal of its column — any time.

        The calibrated start, always explicit — nothing runs it for you: a
        Bernstein simple intercept starts at the data marginal instead of
        zuko's default (about 2.5x too steep), an ordinal simple intercept
        at the marginal class log-odds; spline/affine intercepts and
        intercepts with parents are untouched. The optimum is unchanged, the
        path to it is shorter (docs/training-speed.md). It is NOT guarded by
        the calibrated flag, so calling it on a loaded or already-trained
        flow **discards those intercepts' weights** and restarts them at the
        marginal. An uncalibrated flow takes its ranges from the same rows
        first. On a calibrated flow the Bernstein start comes from the
        stored range alone (the canonical map; the df is not read) — only
        ordinal intercepts re-read the rows.

        Returns
        -------
        CausalFlowDAG
            ``self``.
        """
        if not bool(self.calibrated):
            self.calibrate(train_df)
        for name in self.order:
            node = self.nodes[name]
            if node.intercept.has_marginal_start:
                if node.kind == "ordinal":
                    self._check_levels(name, train_df)
                self._marginal_start(name, train_df)
        return self

    def _marginal_start(self, name: str, train_df: pd.DataFrame) -> None:
        """Start a simple intercept at the node's data marginal."""
        node = self.nodes[name]
        theta = kind_marginal_theta(node, train_df[name].to_numpy())
        if theta is None:  # a spline or affine transform has no calibrated start
            return
        node.intercept.marginal_start(theta)

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

    def _check_side_columns(self, train_df: pd.DataFrame) -> list[str]:
        """Check the terms' side columns in the frame; give their names.

        A centered ``VC`` needs its propensity column: ``P(t = 1 | pa_t)``
        per training row, computed **out of fold** (the cross-fitting
        requirement of the DML design; in-sample values reintroduce the
        own-observation bias). How it is computed is the caller's choice —
        a ``fit_classical`` on the treatment spec per fold, or any
        classifier — merged into ``train_df`` as an ordinary column. The
        training loss uses the frozen column; every query after the fit
        recomputes the value live from the treatment node.
        """
        cols: list[str] = []
        for name in self.order:
            for m in self.nodes[name].shifts.values():
                for col in m.side_columns():
                    if col not in train_df.columns:
                        raise ValueError(
                            f"the centered VC on node {name!r} needs its "
                            f"propensity column {col!r} in the training "
                            "frame — compute P(t=1|pa_t) out of fold and "
                            "merge it as a column."
                        )
                    m.check_column(name, col, train_df[col].to_numpy())
                    cols.append(col)
        return list(dict.fromkeys(cols))

    def _recenter_vc(self, values: dict[str, Tensor]) -> None:
        """Run every shift term's post-fit ``finalize`` (the VC re-centering).

        A VC term re-splits ``beta0``/``b_theta`` so the head sums to zero
        over the train rows; the modelled function does not change.
        """
        feats: dict[str, Tensor] | None = None
        for name in self.order:
            nd = self.nodes[name]
            for m in nd.shifts.values():
                if not m.finalizes:
                    continue
                if feats is None:
                    feats = self._features(values)
                m.finalize(nd, feats)

    def _is_classical(self) -> bool:
        """Say whether every term is one the exact classical fit handles."""
        return all(
            get_term(term.effect).term_is_classical(term)
            for node in self.spec.values()
            for term in node.terms
        )

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
            # under do, a centered VC re-derives t_do - e_hat(x); never cached
            theta, shift = node.theta_shift(
                feats | self._side_feats(node, values, n), n
            )
            values[name] = kind_sample(node, theta, shift, u_vals[name])
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
                feats | self._side_feats(node, values, n), n
            )
            u[name] = kind_abduct(node, theta, shift, values[name], generator=gen)
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
        values = self._tensorize(df, list(nd.parents) + self._query_side_columns(nd))
        feats = self._features({p: values[p] for p in nd.parents})
        theta, shift = nd.theta_shift(feats | self._side_feats(nd, values, n), n)
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
        self._node(node)  # the friendly unknown-node error, before the kind check
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
        self._node(node)  # the friendly unknown-node error, before the kind check
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
        column: str | None = None,
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
        column : str | None, optional
            Score column to scan instead of the ``t``-derived one — for a
            multi-level ordinal treatment's level contrast (``"t[2]"``).

        Returns
        -------
        pd.DataFrame
            One row per candidate, sorted by ``stat`` descending, with
            columns ``stat``, ``p_value``, ``crit_5pct`` and ``flag``. See
            :func:`tramdag.scores.effect_modifier_scan`.
        """
        return _effect_modifier_scan(
            self, df, node, t, candidates=candidates, column=column
        )

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
                "or fn_shift(fn=) must be a picklable module-level function "
                "— use 'minmax'/'standardize', or def the function (or the "
                "nn.Module class) at module level."
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
            init=ckpt["init"],
        )
        for name, t in ckpt["state_dict"].items():
            # a callable transform's train buffer takes the checkpoint's shape
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
