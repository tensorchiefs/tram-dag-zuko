"""Per-observation scores and the effect-modifier scan (issue #29).

The score psi_i = d l_i / d theta of a fitted model is the cheapest
effect-modifier detector we know (model-based recursive partitioning /
structural-change logic; Zeileis & Hornik 2007; Zeileis, Hothorn & Hornik 2008;
Dandl et al. 2024): at the MLE the scores sum to zero, but if the true effect
of a treatment *varies* with a covariate, the scores of the treatment
coefficient drift systematically when ordered by that covariate. Fitting the
cheap all-``ls`` model (seconds, ``fit_classical``) and scanning the scores
turns "which VC modifiers should I declare?" from a modeling guess into a
measured decision — *before* fitting anything expensive.

Because every shift coefficient enters the latent additively, the scores are
**analytic and exact** (no autograd): ``d l_i / d beta = (d l_i / d s_i) * x_i``
with the latent-scale derivative in closed form —

- continuous node (``z = h(x) + s``, standard-logistic latent):
  ``d l / d s = 1 - 2 sigmoid(z)``;
- ordinal node (``P(Y<=k) = sigmoid(theta_k - s)``):
  ``d l / d s = (sig'(l) - sig'(u)) / (sig(u) - sig(l))`` with ``l``/``u`` the
  observed level's shifted cutpoint bounds.

The public entry points are the ``CausalFlowDAG`` methods
:meth:`~tramdag.CausalFlowDAG.scores` and
:meth:`~tramdag.CausalFlowDAG.effect_modifier_scan`. Both delegate here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from .conditioners import LinearShift
from .spec import OrdinalNode
from .transforms import _bounds

__all__ = ["node_scores", "effect_modifier_scan", "sup_bb_pvalue"]

# 5% critical value of sup |Brownian bridge| (Kolmogorov distribution)
CRIT_5PCT = 1.3581


def _dl_ds(
    nd, feats: dict, x: torch.Tensor, n: int, vc_ehat: dict | None = None
) -> torch.Tensor:
    """Give ``d l_i / d s_i``, shape ``(n,)``.

    This is the closed-form derivative of the per-row log-likelihood with
    respect to the total shift of the node.
    """
    theta, shift = nd.theta_shift(feats, n, vc_ehat=vc_ehat)
    if nd.kind == "continuous":
        z0, _ = nd.ut.forward(theta, x)
        return 1.0 - 2.0 * torch.sigmoid(z0 + shift)
    lower, upper = _bounds(theta, shift, x)  # already include -s
    sl, su = torch.sigmoid(lower), torch.sigmoid(upper)
    return (sl * (1 - sl) - su * (1 - su)) / (su - sl)


def node_scores(flow, df: pd.DataFrame, node: str) -> pd.DataFrame:
    """Compute the per-observation scores of the interpretable coefficients.

    The scores cover every ``LS`` weight and the ``beta0`` of every ``VC``
    term. ``CS`` terms carry no interpretable coefficient, so this function
    skips them.

    Parameters
    ----------
    flow : CausalFlowDAG
        The fitted flow.
    df : pd.DataFrame
        Observations. Must contain the node, its parents, and the
        propensity inputs of centered VC terms.
    node : str
        Name of the node whose coefficients are scored.

    Returns
    -------
    pd.DataFrame
        One column per coefficient, shape ``(n, k)``, indexed like ``df``.
        A continuous ``LS`` parent gives one column, named after the
        parent. An ordinal ``LS`` parent gives one column per one-hot
        level, named ``"{parent}[{k}]"``. A ``VC`` term gives one column
        named after its treatment, which holds the ``beta0`` score. For a
        binary ordinal treatment that score belongs to the identified
        contrast of level 1 against level 0.

    Raises
    ------
    KeyError
        If ``node`` is unknown, or if a needed column is missing from
        ``df``.
    ValueError
        If the node has no ``LS`` or ``VC`` term.
    """
    if node not in flow.nodes:
        raise KeyError(f"unknown node {node!r}")
    nd = flow.nodes[node]
    ls_groups = [
        (key, ps)
        for key, ps in nd._shift_groups
        if isinstance(nd.shifts[key], LinearShift)
    ]
    if not ls_groups and not nd._vc_groups:
        raise ValueError(
            f"node {node!r} has no LS or VC terms. params='shift' scores need "
            "at least one interpretable shift coefficient."
        )

    needed = (
        list(nd.parents)
        + [node]  # scores are NOT y-free: l_i needs x
        + flow._vc_ehat_columns(nd)
    )  # + e_hat inputs of centered terms
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"data is missing column(s): {missing}")
    values = flow._tensorize(df, needed)
    feats = flow._features({p: values[p] for p in nd.parents})
    ehat = flow._vc_ehat_live(nd, values, len(df))
    dlds = _dl_ds(nd, feats, values[node], len(df), vc_ehat=ehat)

    cols: dict[str, np.ndarray] = {}
    for key, ps in ls_groups:
        feat = (
            feats[ps[0]] if len(ps) == 1 else torch.cat([feats[p] for p in ps], dim=1)
        )
        psi = (dlds.unsqueeze(1) * feat).cpu().numpy()
        if len(ps) == 1 and isinstance(flow.spec[ps[0]], OrdinalNode):
            for k in range(psi.shape[1]):
                cols[f"{ps[0]}[{k}]"] = psi[:, k]
        else:  # LS is single-parent, so a non-ordinal parent is one column
            cols[key] = psi[:, 0]
    for g in nd._vc_groups:
        t = feats[g.on][:, -1:] if g.on_is_ord else feats[g.on]
        if g.center:  # d s / d beta0 = t - e_hat(x)
            t = t - ehat[g.on].view(-1, 1)
        cols[g.on] = (dlds * t.squeeze(-1)).cpu().numpy()
    return pd.DataFrame(cols, index=df.index)


def sup_bb_pvalue(stat: float, terms: int = 100) -> float:
    """Give ``P(sup |Brownian bridge| > stat)``, the Kolmogorov series.

    Parameters
    ----------
    stat : float
        Observed supremum statistic.
    terms : int, optional
        Number of series terms, by default 100.

    Returns
    -------
    float
        The p-value, clipped to [0, 1].
    """
    if stat <= 0:
        return 1.0
    s = sum(
        (-1) ** (k + 1) * math.exp(-2.0 * k * k * stat * stat)
        for k in range(1, terms + 1)
    )
    return min(1.0, max(0.0, 2.0 * s))


def effect_modifier_scan(
    flow, df: pd.DataFrame, node: str, t: str, candidates: list[str] | None = None
) -> pd.DataFrame:
    """Scan the ``t``-coefficient scores for effect-modifier drift.

    For each candidate covariate ``c``, the scan orders the
    per-observation scores of the treatment coefficient by ``c`` and forms
    the scaled cumulative-sum process
    ``B_j = sum_{i<=j} psi_(i) / (sd(psi) * sqrt(n))``. Under parameter
    stability ``B`` converges to a Brownian bridge, so ``sup_j |B_j|`` has
    the Kolmogorov distribution (5% critical value 1.3581). A systematic
    drift — the true effect varying with ``c`` — inflates it. Covariates
    flagged here are the measured candidates for ``VC`` modifiers
    (Zeileis-Hornik fluctuation test).

    For heavily tied (few-level) candidates the ordering is only partial.
    Read the scan as a ranking diagnostic, not as an exact-size test.

    Parameters
    ----------
    flow : CausalFlowDAG
        The fitted flow.
    df : pd.DataFrame
        Observations, as for :func:`node_scores`.
    node : str
        Name of the outcome node.
    t : str
        Name of the treatment. Its scores column is ``t`` itself for a
        continuous parent or a VC term, and the identified level-1 column
        ``"{t}[1]"`` for a binary ordinal LS parent.
    candidates : list[str] | None, optional
        Candidate covariates. Defaults to every column of ``df`` except
        ``node`` and ``t``.

    Returns
    -------
    pd.DataFrame
        Indexed by candidate, sorted by ``stat`` descending, with columns
        ``stat``, ``p_value``, ``crit_5pct`` and ``flag``
        (``stat > crit_5pct``).

    Raises
    ------
    KeyError
        If no score column exists for ``t`` on ``node``.
    ValueError
        If the score column is constant.
    """
    psi_df = node_scores(flow, df, node)
    if t in psi_df.columns:
        col = t
    elif f"{t}[1]" in psi_df.columns and f"{t}[2]" not in psi_df.columns:
        col = f"{t}[1]"  # binary ordinal LS: the contrast
    else:
        raise KeyError(
            f"no score column for treatment {t!r} on node {node!r} "
            f"(have {list(psi_df.columns)})."
        )
    psi = psi_df[col].to_numpy()
    n = len(psi)
    sd = psi.std()
    if sd == 0:
        raise ValueError(f"score column {col!r} is constant. There is nothing to scan.")

    if candidates is None:
        candidates = [c for c in df.columns if c not in (node, t)]
    rows = {}
    for c in candidates:
        order = np.argsort(df[c].to_numpy(), kind="stable")
        b = np.cumsum(psi[order]) / (sd * math.sqrt(n))
        stat = float(np.abs(b).max())
        rows[c] = {
            "stat": stat,
            "p_value": sup_bb_pvalue(stat),
            "crit_5pct": CRIT_5PCT,
            "flag": stat > CRIT_5PCT,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    return out.sort_values("stat", ascending=False)
