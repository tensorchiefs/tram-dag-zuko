"""Stateless read-outs: ``_ReadoutsMixin``, composed into ``CausalFlowDAG``.

Each read-out is defined here once and is an ordinary method of the flow.
`shift_curve` is the public replacement for reaching into
``flow.nodes[..].shifts[..]`` + ``net_input`` when plotting a fitted shift
against a grid.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .conditioners import LinearShift
from .terms import VaryingCoefficientTerm


# %% public classes --------------------------------------------------------------------
class _ReadoutsMixin:
    """The stateless read-outs, mixed into :class:`~tramdag.CausalFlowDAG`."""

    @torch.no_grad()
    def shift_curve(self, node: str, parent: str, grid) -> np.ndarray:
        """Evaluate one fitted shift term on a 1-D grid of parent values.

        The parent runs over ``grid`` with the term's other inputs absent, so the
        term must read exactly this one parent (an ``LS`` or one-parent ``CS``).
        Returns the shift values as a flat array — the curve a replication plots
        against the data-generating truth.
        """
        nd = self._node(node)
        if parent not in nd.shifts:
            raise KeyError(
                f"node {node!r} has no shift term keyed {parent!r}; "
                f"available: {sorted(nd.shifts)}"
            )
        x = torch.as_tensor(np.asarray(grid), dtype=torch.float32).view(-1, 1)
        # through the term's own evaluation, so Fn and custom terms work too
        curve = nd.shifts[parent].shift_value(nd, {parent: x})
        return curve.cpu().numpy().ravel()

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
        vcs = {
            m.key: m.mods
            for m in nd.shifts.values()
            if isinstance(m, VaryingCoefficientTerm)
        }
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
            for term in self.spec[child].terms:
                # a VC modifier may share its cell with an edge-owning term
                for p, tag in term.cells():
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
        groups = nd.intercept.groups
        if not groups:
            raise ValueError(
                f"node {node!r} has no complex-intercept (I) terms with parents. "
                "Its intercept is unconditional, so there is nothing to decompose."
            )
        missing = [p for p in nd.intercept.ci_parents if p not in df.columns]
        if missing:
            raise KeyError(f"df is missing intercept-parent column(s): {missing}")

        feats = self._features(self._tensorize(df, nd.intercept.ci_parents))
        # one net per group: the additive term holds them in .nets; a single
        # (possibly joint) I-term is itself the one network.
        nets = list(getattr(nd.intercept, "nets", None) or [nd.intercept])

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
