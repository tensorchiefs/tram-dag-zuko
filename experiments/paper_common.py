"""Shared helpers for the TRAM-DAG paper replication experiments.

The paper is arXiv:2503.16206. Run these scripts from the ``experiments/``
directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tramdag import CausalFlowDAG  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# paper training setup (App. C.3): n = 40000, 500 epochs, Adam lr 1e-3
PAPER_N = 40_000
PAPER_EPOCHS = 500
PAPER_LR = 1e-3
BATCH = 512


def results_dir(name: str) -> Path:
    out = REPO / "results" / name / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out.parent


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def fit_chunked(
    spec: dict,
    train_df,
    val_df,
    *,
    record=None,
    chunk: int = 10,
    epochs: int = PAPER_EPOCHS,
    lr: float = PAPER_LR,
    seed: int = 7,
) -> tuple[CausalFlowDAG, list[dict]]:
    """Fit in pieces of ``chunk`` epochs and call ``record(flow)`` after each.

    This produces the coefficient-against-epoch trajectories of paper
    Fig. 14, 15 and 19.
    """
    torch.manual_seed(seed)
    flow = CausalFlowDAG(spec)
    traj: list[dict] = []
    done = 0
    while done < epochs:
        e = min(chunk, epochs - done)
        flow.fit(
            train_df, val_df, epochs=e, learning_rate=lr, batch_size=BATCH, verbose=0
        )
        done += e
        if record is not None:
            traj.append({"epoch": done, **record(flow)})
    return flow, traj


def hist_overlay(ax, dgp, flow_samp, bins) -> None:
    """Draw the DGP histogram filled and the flow histogram stepped."""
    ax.hist(dgp, bins=bins, density=True, alpha=0.45, label="DGP")
    ax.hist(
        flow_samp,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.8,
        color="C3",
        label="flow",
    )


def finish(fig, path: Path) -> None:
    """Lay out, save at 150 dpi, and close."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ls_weight(flow: CausalFlowDAG, node: str, parent: str) -> float:
    return float(flow.nodes[node].shifts[parent].weight.detach())


def cs_curve(
    flow: CausalFlowDAG, node: str, parent: str, grid: np.ndarray
) -> np.ndarray:
    x = torch.as_tensor(grid, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        return flow.nodes[node].shifts[parent](x).detach().numpy()


def plot_trajectories(
    traj: list[dict], truths: dict[str, float], path: Path, title: str
) -> None:
    """Coefficient-vs-epoch plot with dashed true values (paper Fig. 14/15/19)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [t["epoch"] for t in traj]
    for i, key in enumerate(truths):
        ax.plot(epochs, [t[key] for t in traj], color=f"C{i}", label=key)
        ax.axhline(truths[key], color=f"C{i}", ls="--", lw=1)
    ax.set_xlabel("epoch"), ax.set_ylabel("coefficient")
    ax.set_title(title), ax.legend()
    finish(fig, path)


def plot_hist_grid(
    dgp_dfs: dict,
    flow_dfs: dict,
    columns: list[str],
    path: Path,
    title: str,
    ordinal: dict[str, int] | None = None,
) -> None:
    """Plot a grid of histograms, scenarios by variables.

    Each row is a scenario, for example ``Obs`` or ``do(x1=-1)``. Each column is
    a variable. The DGP histogram is filled and the flow histogram is stepped.
    This produces paper Fig. 16 and 17 right, Fig. 9 and Fig. 20.
    """
    ordinal = ordinal or {}
    rows = list(dgp_dfs)
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(3.2 * len(columns), 2.6 * len(rows)),
        squeeze=False,
    )
    for r, scen in enumerate(rows):
        for c, col in enumerate(columns):
            ax = axes[r][c]
            d, m = dgp_dfs[scen][col], flow_dfs[scen][col]
            if col in ordinal:
                levels = np.arange(ordinal[col])
                fd = d.value_counts(normalize=True).reindex(levels, fill_value=0)
                fm = m.value_counts(normalize=True).reindex(levels, fill_value=0)
                ax.bar(levels - 0.18, fd, width=0.36, alpha=0.6, label="DGP")
                ax.bar(
                    levels + 0.18, fm, width=0.36, alpha=0.8, color="C3", label="flow"
                )
                ax.set_xticks(levels)
            else:
                lo, hi = np.quantile(d, [0.001, 0.999])
                if hi - lo < 1e-9:  # do-clamped column: constant value
                    lo, hi = lo - 1.0, hi + 1.0
                hist_overlay(ax, d, m, np.linspace(lo, hi, 50))
            if r == 0:
                ax.set_title(col)
            if c == 0:
                ax.set_ylabel(scen)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    finish(fig, path)
