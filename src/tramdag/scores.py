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

- continuous node (``u = h(x) + s``, standard-logistic latent):
  ``d l / d s = 1 - 2 sigmoid(u)``;
- ordinal node (``P(Y<=k) = sigmoid(theta_k - s)``):
  ``d l / d s = (sig'(l) - sig'(u)) / (sig(u) - sig(l))`` with ``l``/``u`` the
  observed level's shifted cutpoint bounds.

The public entry points are the ``CausalFlowDAG`` methods
:meth:`~tramdag.CausalFlowDAG.scores` and
:meth:`~tramdag.CausalFlowDAG.effect_modifier_scan`. Both delegate here.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from .transforms import ordinal_bounds

# %% global variables ------------------------------------------------------------------
__all__ = ["effect_modifier_scan", "node_scores", "sup_bb_pvalue"]

# 5% critical value of sup |Brownian bridge| (Kolmogorov distribution)
CRIT_5PCT = 1.3581


# %% private functions -----------------------------------------------------------------
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
    lower, upper = ordinal_bounds(theta, shift, x)  # already include -s
    sl, su = torch.sigmoid(lower), torch.sigmoid(upper)
    return (sl * (1 - sl) - su * (1 - su)) / (su - sl)


# %% public functions ------------------------------------------------------------------
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
    nd = flow._node(node)
    scored = [m for m in nd.shifts.values() if getattr(m, "scored", False)]
    if not scored:
        raise ValueError(
            f"node {node!r} has no LS or VC terms. Shift scores need "
            "at least one interpretable shift coefficient."
        )

    # not y-free: l_i needs x. Plus the e_hat inputs of centered terms.
    needed = [*nd.parents, node, *flow._vc_ehat_columns(nd)]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"data is missing column(s): {missing}")
    values = flow._tensorize(df, needed)
    feats = flow._features({p: values[p] for p in nd.parents})
    ehat = flow._vc_ehat_live(nd, values, len(df))
    dlds = _dl_ds(nd, feats, values[node], len(df), vc_ehat=ehat)

    cols: dict[str, np.ndarray] = {}
    for m in scored:
        cols.update(m.score_columns(nd, flow, feats, dlds, ehat))
    return pd.DataFrame(cols, index=df.index)


def sup_bb_pvalue(stat: float) -> float:
    """Give ``P(sup |Brownian bridge| > stat)``, the Kolmogorov series.

    Parameters
    ----------
    stat : float
        Observed supremum statistic.

    Returns
    -------
    float
        The p-value, clipped to [0, 1].
    """
    if stat <= 0:
        return 1.0  # the series alternates to 0.0 here, which is the wrong tail
    # 100 terms: the k-th is exp(-2k^2 stat^2), so past k ~ 10 it underflows
    s = sum(
        (-1) ** (k + 1) * math.exp(-2.0 * k * k * stat * stat) for k in range(1, 101)
    )
    return min(1.0, max(0.0, 2.0 * s))


def effect_modifier_scan(
    flow,
    df: pd.DataFrame,
    node: str,
    t: str,
    candidates: list[str] | None = None,
    column: str | None = None,
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
    column : str | None, optional
        Score column to scan, overriding the ``t``-derived choice — the
        way to scan one level contrast of a multi-level ordinal
        treatment (e.g. ``"t[2]"``), which has no single default column.

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
    if column is not None:
        if column not in psi_df.columns:
            raise KeyError(
                f"no score column {column!r} on node {node!r} "
                f"(have {list(psi_df.columns)})."
            )
        col = column
    elif t in psi_df.columns:
        col = t
    elif f"{t}[1]" in psi_df.columns and f"{t}[2]" not in psi_df.columns:
        col = f"{t}[1]"  # binary ordinal LS: the contrast
    else:
        raise KeyError(
            f"no score column for treatment {t!r} on node {node!r} "
            f"(have {list(psi_df.columns)}). For a multi-level ordinal "
            "treatment pass column= with the level contrast to scan."
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
