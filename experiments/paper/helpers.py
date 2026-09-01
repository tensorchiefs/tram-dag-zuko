"""Helpers shared by the paper replications.

One fit call with a per-epoch read-out, the triangle spec pieces and the
figure styles below are specific to these experiments; what every area
shares (config loading, output directories, reports) lives in
``experiments/common.py``.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless: the scripts only write files

import matplotlib.pyplot as plt
import numpy as np
import torch

from tramdag import CS, LS, CausalFlowDAG, shift_curve


# %% public functions ------------------------------------------------------------------
def fit_paper(generator, spec: dict, config: dict, out: Path, record=None):
    """Fit the way the paper's R code does: one run, one optimizer, per-epoch read-out.

    ``summerof24/*.R`` calls Keras ``fit(epochs = 1)`` in a loop over one
    compiled model and reads the ``beta`` layer after every epoch, so the
    trajectory comes from a single continuous Adam run; ``comparison/utils.R``
    takes one full-batch step per epoch and reduces the learning rate of that
    one optimizer when the summed validation NLL plateaus
    (``update_learning_rate``: factor, patience, min_lr, strict ``<``). Both
    are one ``fit`` call here: the plateau rule is torch's own
    ``ReduceLROnPlateau`` on the summed validation NLL — global, like the
    reference — stepped from the epoch callback on ``history["val"]``
    (fit computes it), which is also where the coefficients are read.

    Train and validation are two separate draws, as in R. The reference has
    no calibrated start, so the flow is calibrated with ``marginal_init=False``.
    The fitted flow is saved to ``out / "flow.pt"``. ``record(flow)``, when
    given, is stored after each epoch with the epoch count — the coefficient
    trajectories of paper Fig. 14, 15 and 19.

    Returns
    -------
    tuple
        ``(flow, train, val, trajectory, fit_seconds)`` — the last is the
        wall-clock of the ``fit`` call alone, the CI runtime tripwire.
    """
    train = generator.observational(config["n_train"])
    val = generator.observational(config["n_val"], seed_offset=1)
    flow = CausalFlowDAG(spec, seed=config["init_seed"], init=config["init"])
    flow.calibrate(train, marginal_init=False)
    opt = torch.optim.Adam(flow.parameters(), lr=config["learning_rate"])
    plateau = None
    if config["schedule"] == "plateau":
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=config["plateau_factor"],
            # torch reduces once `bad > patience`, the reference at `bad == patience`
            patience=config["plateau_patience"] - 1,
            threshold=config["min_delta"],
            threshold_mode="abs",
            min_lr=config["plateau_min_lr"],
        )
    trajectory = []

    def epoch_end(f, epoch, _opt):
        if plateau is not None:
            plateau.step(sum(f.history["val"][-1].values()))
        if record is not None:
            trajectory.append({"epoch": epoch, **record(f)})

    t0 = time.perf_counter()
    flow.fit(
        train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        # per-epoch validation only where the protocol consumes it (the
        # plateau rule); the triangle scripts never computed it per epoch
        validation_data=val if plateau is not None else None,
        seed=config["shuffle_seed"],
        optimizer=opt,
        callbacks=epoch_end,
    )
    fit_seconds = round(time.perf_counter() - t0, 1)
    flow.save(out / "flow.pt")
    return flow, train, val, trajectory, fit_seconds


def shift_term(config: dict):
    """Give the x2 -> x3 term of a triangle spec: a linear or complex shift.

    Raises
    ------
    ValueError
        If ``shift`` is neither ``"ls"`` nor ``"cs"``.
    """
    shift = config["shift"]
    if shift == "ls":
        return LS("x2")
    if shift == "cs":
        return CS("x2", units=config["shift_units"], activation=config["activation"])
    raise ValueError(f"shift must be 'ls' or 'cs', got '{shift}'")


def snapshot(flow: CausalFlowDAG, shift: str) -> dict:
    """Read the triangle's linear-shift coefficients out of a flow mid-training."""
    values = {
        "beta12": ls_weight(flow, "x2", "x1"),
        "beta13": ls_weight(flow, "x3", "x1"),
    }
    if shift == "ls":
        values["beta23"] = ls_weight(flow, "x3", "x2")
    return values


def ls_weight(flow: CausalFlowDAG, node: str, parent: str) -> float:
    """Give the node's linear-shift weight on that parent."""
    return float(flow.ls_coefficients()[node][parent][0])


def true_coefficients(config: dict) -> dict:
    """Give the true linear-shift weights this variant can be scored on.

    In the flow's sign convention (an ordinal node subtracts its shift, so
    the paper's +0.2 / −0.3 read −0.2 / +0.3 here). ``beta23`` only has a
    true value for the linear DGP with an ``ls`` model; a nonlinear ``f``
    fitted linearly has no true weight, and a ``cs`` model has no weight.
    """
    truths = {"beta12": 2.0, "beta13": -0.2}
    if config["shift"] == "ls" and config["f"] == "linear":
        truths["beta23"] = 0.3
    return truths


def cs_curve_error(
    flow: CausalFlowDAG, generator, config: dict, out: Path, title
) -> float:
    """Plot the fitted x3 complex shift against the true curve, give the max error."""
    grid = np.linspace(config["grid_low"], config["grid_high"], config["grid_points"])
    fitted = shift_curve(flow, "x3", "x2", grid)
    return plot_cs_curve(
        grid,
        fitted=fitted,
        true=generator.true_shift_curve(grid),
        path=out / "plots" / "cs_curve.png",
        title=title,
    )


def compare_do_x1(
    generator, flow: CausalFlowDAG, config: dict, out: Path, ordinal_levels, title
) -> dict:
    """Compare the observational and ``do(x1)`` distributions, DGP vs flow.

    Writes ``plots/distributions.png`` (paper Fig. 9/16/17/20) and gives the
    ``do(x1)`` mean of x3 under both.
    """
    do_query = f"do(x1={config['do_x1']:+.0f})"
    n, seed = config["n_compare"], config["sample_seed"]
    dgp_samples = {
        "Obs": generator.observational(n, seed_offset=5),
        do_query: generator.interventional(n, {"x1": config["do_x1"]}),
    }
    flow_samples = {
        "Obs": flow.sample(n, seed=seed),
        do_query: flow.sample(n, do={"x1": config["do_x1"]}, seed=seed),
    }
    plot_hist_grid(
        dgp_samples,
        flow_samples,
        out / "plots" / "distributions.png",
        title,
        ordinal_levels,
    )
    dgp_mean = float(dgp_samples[do_query]["x3"].mean())
    flow_mean = float(flow_samples[do_query]["x3"].mean())
    return {
        "mean_x3_dgp_do_x1": dgp_mean,
        "mean_x3_flow_do_x1": flow_mean,
        "mean_x3_abs_err_do_x1": abs(flow_mean - dgp_mean),
    }


def finish(fig, path: Path) -> None:
    """Lay out, save at 150 dpi, and close."""
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def continuous_hist(ax, dgp, flow, bins: int = 50) -> None:
    """Draw one panel: the DGP histogram filled, the flow histogram stepped.

    The bins come from the DGP's 0.1%/99.9% quantiles.
    """
    low, high = np.quantile(dgp, [0.001, 0.999])
    if high - low < 1e-9:
        # a do-clamped column is constant: give the panel a width
        low, high = low - 1.0, high + 1.0
    edges = np.linspace(low, high, bins)
    ax.hist(dgp, bins=edges, density=True, alpha=0.45, label="DGP")
    ax.hist(
        flow,
        bins=edges,
        density=True,
        histtype="step",
        lw=1.8,
        color="C3",
        label="flow",
    )


def level_bars(ax, dgp_freq, flow_freq, labels=("DGP", "flow")) -> None:
    """Draw side-by-side level-frequency bars, one array per side."""
    levels = np.arange(len(dgp_freq))
    ax.bar(levels - 0.18, dgp_freq, width=0.36, alpha=0.6, label=labels[0])
    ax.bar(levels + 0.18, flow_freq, width=0.36, alpha=0.8, color="C3", label=labels[1])
    ax.set_xticks(levels)


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
        ``{scenario: DataFrame}``, the same scenarios in both; every
        column of the frames gets a column of panels.
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
    columns = list(next(iter(dgp_samples.values())))
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
                level_bars(
                    ax,
                    dgp_values.value_counts(normalize=True).reindex(
                        levels, fill_value=0
                    ),
                    flow_values.value_counts(normalize=True).reindex(
                        levels, fill_value=0
                    ),
                )
            else:
                continuous_hist(ax, dgp_values, flow_values)
    for ax, column in zip(axes[0], columns, strict=True):
        ax.set_title(column)
    for ax_row, scenario in zip(axes, scenarios, strict=True):
        ax_row[0].set_ylabel(scenario)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    finish(fig, path)
