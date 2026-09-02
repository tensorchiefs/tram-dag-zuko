"""Stateless read-outs over a fitted flow.

Free functions taking the flow; `CausalFlowDAG` exposes each as a one-line
delegate method, which carries the public docstring. `shift_curve` is the
public replacement for reaching into ``flow.nodes[..].shifts[..]`` +
``net_input`` when plotting a fitted shift against a grid.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from .conditioners import LinearShift
from .terms import get_term

if TYPE_CHECKING:
    from .flow import CausalFlowDAG


# %% public functions ------------------------------------------------------------------
@torch.no_grad()
def shift_curve(flow: CausalFlowDAG, node: str, parent: str, grid) -> np.ndarray:
    """Evaluate one fitted shift term on a 1-D grid of parent values.

    The parent runs over ``grid`` with the term's other inputs absent, so the
    term must read exactly this one parent (an ``LS`` or one-parent ``CS``).
    Returns the shift values as a flat array — the curve a replication plots
    against the data-generating truth.
    """
    nd = flow.nodes[node]
    if parent not in nd.shifts:
        raise KeyError(
            f"node {node!r} has no shift term keyed {parent!r}; "
            f"available: {sorted(nd.shifts)}"
        )
    x = torch.as_tensor(np.asarray(grid), dtype=torch.float32).view(-1, 1)
    feat = nd.net_input({parent: x}, (parent,), parent)
    return nd.shifts[parent](feat).cpu().numpy().ravel()


@torch.no_grad()
def varying_coef(
    flow, df: pd.DataFrame, node: str, *, t: str | None = None
) -> np.ndarray:
    """varying_coef — the contract is on the flow method."""
    nd = flow._node(node)
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
        raise KeyError(f"node {node!r} has no VC term on {t!r} (has {sorted(vcs)}).")
    mods = vcs[t]
    mod_feat = None
    if mods:
        feats = flow._features(flow._tensorize(df, mods))
        mod_feat = nd.net_input(feats, mods, t)
    return nd.shifts[t].beta(mod_feat, len(df)).cpu().numpy()


def ls_coefficients(flow) -> dict[str, dict[str, np.ndarray]]:
    """ls_coefficients — the contract is on the flow method."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in flow.order:
        linear = {
            parent: module.weight.detach().cpu().numpy().ravel().copy()
            for parent, module in flow.nodes[name].shifts.items()
            if isinstance(module, LinearShift)
        }
        if linear:
            out[name] = linear
    return out


def to_matrix(flow) -> pd.DataFrame:
    """to_matrix — the contract is on the flow method."""
    m = pd.DataFrame("", index=list(flow.order), columns=list(flow.order))
    for child in flow.order:
        for term in flow.spec[child].terms:
            # a VC modifier may share its cell with an edge-owning term
            for p, tag in get_term(term.effect).cells(term):
                cur = m.loc[p, child]  # cell with a prognostic term -> "+"
                m.loc[p, child] = f"{cur}+{tag}" if cur else tag
    return m


@torch.no_grad()
def intercept_contributions(flow, df: pd.DataFrame, node: str) -> dict:
    """intercept_contributions — the contract is on the flow method."""
    nd = flow._node(node)
    groups = nd._intercept_groups
    if not groups:
        raise ValueError(
            f"node {node!r} has no complex-intercept (I) terms with parents. "
            "Its intercept is unconditional, so there is nothing to decompose."
        )
    missing = [p for p in nd.ci_parents if p not in df.columns]
    if missing:
        raise KeyError(f"df is missing intercept-parent column(s): {missing}")

    feats = flow._features(flow._tensorize(df, nd.ci_parents))
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
    flow, df: pd.DataFrame, node: str, *, drop_first: bool = False
) -> pd.DataFrame:
    """design_matrix — the contract is on the flow method."""
    nd = flow._node(node)
    feats = flow._features(flow._tensorize(df, nd.parents))
    cols: dict[str, np.ndarray] = {}
    for p in nd.parents:
        arr = feats[p].cpu().numpy()
        if arr.shape[1] == 1:  # continuous parent: raw
            cols[p] = arr[:, 0]
        else:
            for k in range(1 if drop_first else 0, arr.shape[1]):
                cols[f"{p}[{k}]"] = arr[:, k]
    return pd.DataFrame(cols, index=df.index)
