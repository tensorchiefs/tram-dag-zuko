"""Helpers shared by the paper replications.

The fitting loop with coefficient snapshots and the figure styles below are
specific to these experiments; what every area shares (config loading,
output directories, reports) lives in ``experiments/common.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless: the scripts only write files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tramdag import CausalFlowDAG


# ------------------------------------------------------------------ fitting
def split_train_val(df: pd.DataFrame, n_train: int, n_val: int) -> tuple:
    """Split positionally: the first rows train, the next ones validate.

    The split is positional, not random: the generators draw i.i.d. rows,
    so the first rows are already an unbiased sample, and a positional
    split keeps a run reproducible from the seed alone.

    Raises
    ------
    ValueError
        If the frame has fewer than ``n_train + n_val`` rows.
    """
    if len(df) < n_train + n_val:
        raise ValueError(f"need {n_train} + {n_val} rows, the sample has {len(df)}")
    return df.iloc[:n_train], df.iloc[n_train : n_train + n_val]


def fit_with_snapshots(
    spec: dict,
    train: pd.DataFrame,
    val: pd.DataFrame,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    init_seed: int,
    shuffle_seed: int,
    chunk_epochs: int,
    record=None,
) -> tuple[CausalFlowDAG, list[dict]]:
    """Fit in pieces and read out coefficients between them.

    Fitting in pieces of ``chunk_epochs`` epochs is what produces the
    coefficient-against-epoch trajectories of paper Fig. 14, 15 and 19.
    Consecutive ``fit`` calls continue from the current **weights** but not
    from the optimizer state, which is what makes the chunk size matter (see
    ``chunk_epochs``).

    Parameters
    ----------
    spec : dict
        The node specification.
    train, val : pd.DataFrame
        Training and validation rows.
    epochs : int
        Total number of epochs.
    learning_rate : float
        Adam learning rate.
    batch_size : int
        Minibatch size.
    init_seed : int
        Seeds the weight initialization, which happens at construction.
    shuffle_seed : int
        Seeds the minibatch shuffling of the first round. Later rounds
        continue the stream, so the whole trajectory is one training run.
    chunk_epochs : int
        Epochs per ``fit`` call. **This is a hyperparameter, not a reporting
        detail**: every call starts a fresh Adam, so the chunk size acts like
        a warm-restart schedule and changes where the fit lands. Measured on
        the VACA benchmark, 8 chunks of 50 reach an interventional mean 20x
        closer to the analytic value than one call of 400.
    record : callable | None, optional
        ``record(flow)`` gives a dict of numbers to store after each chunk.
        With ``None`` no snapshots are taken; the chunking is unchanged.

    Returns
    -------
    tuple[CausalFlowDAG, list[dict]]
        The fitted flow, and one ``{"epoch": ..., **record(flow)}`` entry
        per snapshot.
    """
    flow = CausalFlowDAG(spec, seed=init_seed)
    trajectory: list[dict] = []
    done = 0
    while done < epochs:
        this_round = min(chunk_epochs, epochs - done)
        flow.fit(
            train,
            val,
            epochs=this_round,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=0,
            seed=shuffle_seed if done == 0 else None,
        )
        done += this_round
        if record is not None:
            trajectory.append({"epoch": done, **record(flow)})
    return flow, trajectory


def ls_weight(flow: CausalFlowDAG, node: str, parent: str) -> float:
    """Give the node's linear-shift weight on that parent."""
    return float(flow.ls_coefficients()[node][parent][0])


def cs_curve(flow: CausalFlowDAG, node: str, parent: str, grid) -> np.ndarray:
    """Evaluate the node's complex-shift network on a grid of parent values."""
    x = torch.as_tensor(np.asarray(grid), dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        return flow.nodes[node].shifts[parent](x).detach().numpy()


# ------------------------------------------------------------------- plots
def finish(fig, path: Path) -> None:
    """Lay out, save at 150 dpi, and close."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hist_overlay(ax, dgp_values, flow_values, bins) -> None:
    """Draw the DGP histogram filled and the flow histogram stepped."""
    ax.hist(dgp_values, bins=bins, density=True, alpha=0.45, label="DGP")
    ax.hist(
        flow_values,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.8,
        color="C3",
        label="flow",
    )


def plot_trajectories(trajectory: list[dict], truths: dict, path: Path, title: str):
    """Coefficient-vs-epoch plot with dashed true values (paper Fig. 14/15/19)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [snapshot["epoch"] for snapshot in trajectory]
    for i, key in enumerate(truths):
        ax.plot(
            epochs, [snapshot[key] for snapshot in trajectory], color=f"C{i}", label=key
        )
        ax.axhline(truths[key], color=f"C{i}", ls="--", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("coefficient")
    ax.set_title(title)
    ax.legend()
    finish(fig, path)


def plot_cs_curve(grid, fitted, true, path: Path, title: str) -> float:
    """Overlay the fitted complex shift on the DGP's shift function.

    Both curves are anchored at the middle grid point, because a shift
    term is identified only up to an additive constant (it competes with
    the intercept). Gives the maximum absolute deviation after anchoring.
    """
    middle = len(grid) // 2
    anchored = fitted - fitted[middle] + true[middle]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(grid, true, "k-", lw=2, label="DGP  $-f(x_2)$")
    ax.plot(grid, anchored, "o", ms=3, color="C0", label="fitted CS")
    ax.set_xlabel("$x_2$")
    ax.set_ylabel("shift")
    ax.legend()
    ax.set_title(title)
    finish(fig, path)
    return float(np.max(np.abs(anchored - true)))


# complexipy: ignore - split planned in the complexity-reduction PR
def plot_hist_grid(
    dgp_samples: dict,
    flow_samples: dict,
    columns: list[str],
    path: Path,
    title: str,
    ordinal_levels: dict[str, int],
) -> None:
    """Plot a grid of histograms, scenarios by variables.

    Each row is a scenario, for example ``Obs`` or ``do(x1=-1)``; each
    column is a variable. The DGP histogram is filled and the flow
    histogram stepped. This produces paper Fig. 9, 16, 17 right and 20.

    Parameters
    ----------
    dgp_samples, flow_samples : dict
        ``{scenario: DataFrame}``, the same scenarios in both.
    columns : list[str]
        Variables to show, one column of panels each.
    path : Path
        Where to write the figure.
    title : str
        Figure title.
    ordinal_levels : dict[str, int]
        Number of levels per ordinal column, for example ``{"x3": 4}``.
        Those are drawn as level frequencies instead of histograms. Pass
        an empty dict when every column is continuous.
    """
    scenarios = list(dgp_samples)
    fig, axes = plt.subplots(
        len(scenarios),
        len(columns),
        figsize=(3.2 * len(columns), 2.6 * len(scenarios)),
        squeeze=False,
    )
    for row, scenario in enumerate(scenarios):
        for col, column in enumerate(columns):
            ax = axes[row][col]
            dgp_values = dgp_samples[scenario][column]
            flow_values = flow_samples[scenario][column]
            if column in ordinal_levels:
                levels = np.arange(ordinal_levels[column])
                dgp_freq = dgp_values.value_counts(normalize=True).reindex(
                    levels, fill_value=0
                )
                flow_freq = flow_values.value_counts(normalize=True).reindex(
                    levels, fill_value=0
                )
                ax.bar(levels - 0.18, dgp_freq, width=0.36, alpha=0.6, label="DGP")
                ax.bar(
                    levels + 0.18,
                    flow_freq,
                    width=0.36,
                    alpha=0.8,
                    color="C3",
                    label="flow",
                )
                ax.set_xticks(levels)
            else:
                low, high = np.quantile(dgp_values, [0.001, 0.999])
                if high - low < 1e-9:
                    # a do-clamped column is constant: give the panel a width
                    low, high = low - 1.0, high + 1.0
                hist_overlay(ax, dgp_values, flow_values, np.linspace(low, high, 50))
            if row == 0:
                ax.set_title(column)
            if col == 0:
                ax.set_ylabel(scenario)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    finish(fig, path)
