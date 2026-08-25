"""Helpers shared by the paper replications.

The fitting loop with coefficient snapshots and the figure styles below are
specific to these experiments; what every area shares (config loading,
output directories, reports) lives in ``experiments/common.py``.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless: the scripts only write files

import matplotlib.pyplot as plt
import numpy as np
import torch

from tramdag import CausalFlowDAG


# %% private functions -----------------------------------------------------------------
def _level_bars(ax, dgp_values, flow_values, n_levels: int) -> None:
    """Draw side-by-side level-frequency bars for one ordinal panel."""
    levels = np.arange(n_levels)
    dgp_freq = dgp_values.value_counts(normalize=True).reindex(levels, fill_value=0)
    flow_freq = flow_values.value_counts(normalize=True).reindex(levels, fill_value=0)
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


def _continuous_hist(ax, dgp_values, flow_values) -> None:
    """Draw one continuous panel, with bins from the DGP quantiles."""
    low, high = np.quantile(dgp_values, [0.001, 0.999])
    if high - low < 1e-9:
        # a do-clamped column is constant: give the panel a width
        low, high = low - 1.0, high + 1.0
    hist_overlay(ax, dgp_values, flow_values, np.linspace(low, high, 50))


# %% public functions ------------------------------------------------------------------
# ------------------------------------------------------------------ fitting
def fit_paper(flow: CausalFlowDAG, train, val, config: dict, record=None) -> list:
    """Fit the way the paper's R code does: one run, one optimizer, per-epoch read-out.

    ``summerof24/*.R`` calls Keras ``fit(epochs = 1)`` in a loop over one
    compiled model and reads the ``beta`` layer after every epoch, so the
    trajectory comes from a single continuous Adam run; ``comparison/utils.R``
    takes one full-batch step per epoch with a ReduceLROnPlateau on the
    validation NLL. Both are one ``fit`` call here: ``epoch_callback`` is the
    per-epoch read-out, and ``schedule``/``plateau_*`` carry the plateau rule.

    ``record(flow)``, when given, is stored after each epoch with the epoch
    count — the coefficient trajectories of paper Fig. 14, 15 and 19.
    """
    trajectory = []
    flow.fit(
        train,
        val,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        seed=config["shuffle_seed"],
        marginal_init=config["marginal_init"],
        schedule=config["schedule"],
        **(
            {
                "plateau_patience": config["plateau_patience"],
                "plateau_factor": config["plateau_factor"],
                "plateau_min_lr": config["plateau_min_lr"],
                "min_delta": config["min_delta"],
            }
            if config["schedule"] == "plateau"
            else {}
        ),
        verbose=0,
        epoch_callback=(
            None
            if record is None
            else lambda f, e: trajectory.append({"epoch": e, **record(f)})
        ),
    )
    return trajectory


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
                _level_bars(ax, dgp_values, flow_values, ordinal_levels[column])
            else:
                _continuous_hist(ax, dgp_values, flow_values)
    for ax, column in zip(axes[0], columns, strict=True):
        ax.set_title(column)
    for ax_row, scenario in zip(axes, scenarios, strict=True):
        ax_row[0].set_ylabel(scenario)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    finish(fig, path)
