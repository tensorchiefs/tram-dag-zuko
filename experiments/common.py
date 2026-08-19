"""Helpers shared by the TRAM-DAG replication experiments (arXiv:2503.16206).

Every experiment is one self-contained script next to this file, with the
same shape: imports, function definitions, a ``run`` function holding the
whole experiment, and a ``__main__`` block whose argparse call selects
*what* to run. **No hyperparameter has a default in code** — each script
reads its values from the YAML file of the same name, and this module's
:func:`load_config` refuses a config that is missing a key or carries an
unknown one.

Run an experiment from this directory::

    uv run python triangle.py atan-cs
    uv run python check.py triangle-atan-cs   # compare against ground truth
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: the scripts only write files

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from tramdag import CausalFlowDAG  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
GROUND_TRUTH = HERE / "ground_truth"


# ------------------------------------------------------------------ config
def load_config(script: str, variant: str, expected_keys: set[str]) -> dict:
    """Read one variant's hyperparameters from ``<script>.yaml``.

    The file lists every value the experiment uses, one section per
    variant under ``variants:``. Shared values are written once under an
    anchor and merged with YAML's ``<<`` key, so the merge is visible in
    the file instead of happening in code.

    Parameters
    ----------
    script : str
        Name of the calling script without extension, for example
        ``"triangle"``. The config file is ``<script>.yaml``.
    variant : str
        Key under ``variants:``.
    expected_keys : set[str]
        The keys the script reads. A config that does not match this set
        exactly is an error: a missing key would otherwise become a
        hidden default, an extra key a silently ignored setting.

    Returns
    -------
    dict
        The variant's hyperparameters.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If the variant is unknown.
    ValueError
        If the variant's keys do not match ``expected_keys``.
    """
    path = HERE / f"{script}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config for '{script}': {path}")

    document = yaml.safe_load(path.read_text())
    variants = document["variants"]
    if variant not in variants:
        raise KeyError(
            f"unknown variant '{variant}' in {path.name}. "
            f"Available: {', '.join(sorted(variants))}"
        )

    config = dict(variants[variant])
    missing = expected_keys - set(config)
    unknown = set(config) - expected_keys
    if missing or unknown:
        raise ValueError(
            f"{path.name}, variant '{variant}': "
            f"missing keys {sorted(missing)}, unknown keys {sorted(unknown)}"
        )
    return config


def variants_of(script: str) -> list[str]:
    """Give the variant names a script's config file defines."""
    document = yaml.safe_load((HERE / f"{script}.yaml").read_text())
    return sorted(document["variants"])


# ------------------------------------------------------------------ outputs
def make_output_dir(name: str) -> Path:
    """Create ``results/<name>/plots/`` and give the results directory."""
    out = RESULTS / name
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def save_metrics(out: Path, metrics: dict) -> None:
    """Write the numbers the CI ground-truth check reads."""
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))


def write_report(out: Path, title: str, metrics: dict, figures: list[str]) -> None:
    """Write ``report.md``: the metrics table and the figures.

    The experiments workflow posts this file as a commit comment, so the
    figure links are plain relative paths that ``cml comment`` resolves
    and uploads.

    Parameters
    ----------
    out : Path
        The experiment's results directory.
    title : str
        Heading of the report.
    metrics : dict
        Flat ``{name: value}`` mapping, as written to ``metrics.json``.
    figures : list[str]
        File names under ``plots/``, in the order they should appear.
    """
    lines = [f"## {title}", "", "| metric | value |", "|---|---|"]
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| `{name}` | {value:+.4f} |")
        else:
            lines.append(f"| `{name}` | {value} |")
    lines.append("")
    for figure in figures:
        lines += [f"![{figure}](plots/{figure})", ""]
    (out / "report.md").write_text("\n".join(lines))


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
    record_every: int,
    record=None,
) -> tuple[CausalFlowDAG, list[dict]]:
    """Fit in pieces and read out coefficients between them.

    Fitting in pieces of ``record_every`` epochs is what produces the
    coefficient-against-epoch trajectories of paper Fig. 14, 15 and 19.
    Consecutive ``fit`` calls continue from the current weights, so the
    trajectory is one training run, not several.

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
    record_every : int
        Epochs between snapshots.
    record : callable | None, optional
        ``record(flow)`` gives a dict of numbers to store for this
        snapshot. With ``None`` no snapshots are taken.

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
        this_round = min(record_every, epochs - done)
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
