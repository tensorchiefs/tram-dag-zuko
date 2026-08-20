"""Replicate the paper's continuous-triangle experiments (Sec. 6.1, App. C.3).

The DGP is ``x1 -> x2 -> x3 <- x1`` with every conditional a transformation
model, ``h(x3|x1,x2) = 0.63 x3 - 0.2 x1 - f(x2)``. Two model families are
fitted, chosen by the variant:

- ``ls`` — the x2 -> x3 edge is a linear shift. Correct only for the linear
  DGP, where the true weight is +0.3; for a nonlinear ``f`` the linear
  weight is the best linear approximation, so no true value is plotted.
- ``cs`` — the edge is a complex shift, which must converge to ``-f(x2)``
  up to an additive constant (paper Fig. 7 right).

Outputs: the coefficient trajectories (Fig. 14/15), the complex-shift
overlay (Fig. 7 right / 17 left / 18 right) and the observational plus
``do(x1)`` distributions (Fig. 16/17).

Usage (from experiments/)::

    uv run python triangle.py atan-cs
"""

from __future__ import annotations

import argparse

import numpy as np
from common import (
    load_variant,
    make_output_dir,
    save_metrics,
    variants_of,
    write_report,
)

from paper.helpers import (
    cs_curve,
    fit_with_snapshots,
    ls_weight,
    plot_cs_curve,
    plot_hist_grid,
    plot_trajectories,
    split_train_val,
)
from paper.simulations.triangle import TriangleContinuous
from tramdag import CS, LS, ContinuousNode

CONFIG_KEYS = {
    "f",
    "shift",
    "n_train",
    "n_val",
    "epochs",
    "learning_rate",
    "batch_size",
    "record_every",
    "dgp_seed",
    "init_seed",
    "shuffle_seed",
    "sample_seed",
    "n_compare",
    "do_x1",
    "grid_low",
    "grid_high",
    "grid_points",
}


def build_spec(shift: str) -> dict:
    """Give the DAG spec, with the x2 -> x3 edge as a linear or complex shift.

    Raises
    ------
    ValueError
        If ``shift`` is neither ``"ls"`` nor ``"cs"``.
    """
    if shift == "ls":
        x2_to_x3 = LS("x2")
    elif shift == "cs":
        x2_to_x3 = CS("x2")
    else:
        raise ValueError(f"shift must be 'ls' or 'cs', got '{shift}'")
    return {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([LS("x1")]),
        "x3": ContinuousNode([LS("x1"), x2_to_x3]),
    }


def true_coefficients(f: str, shift: str) -> dict:
    """Give the true linear-shift coefficients this variant can be scored on.

    ``beta23`` only has a true value for the linear DGP with an ``ls``
    model; a nonlinear ``f`` fitted linearly has no true weight, and a
    ``cs`` model has no weight at all.
    """
    truths = {"beta12": 2.0, "beta13": -0.2}
    if shift == "ls" and f == "linear":
        truths["beta23"] = 0.3
    return truths


def snapshot(flow, shift: str) -> dict:
    """Read the linear-shift coefficients out of a flow mid-training."""
    values = {
        "beta12": ls_weight(flow, "x2", "x1"),
        "beta13": ls_weight(flow, "x3", "x1"),
    }
    if shift == "ls":
        values["beta23"] = ls_weight(flow, "x3", "x2")
    return values


def run(variant: str) -> dict:
    """Run one variant end to end and give its metrics."""
    config = load_variant(__file__, variant, CONFIG_KEYS)
    out = make_output_dir(__file__, f"triangle-{variant}")
    figures = []

    generator = TriangleContinuous(f=config["f"], seed=config["dgp_seed"])
    sample = generator.observational(config["n_train"] + config["n_val"])
    train, val = split_train_val(sample, config["n_train"], config["n_val"])

    print(
        f"fitting triangle/{config['f']} with a {config['shift']} shift on "
        f"n={len(train)} for {config['epochs']} epochs "
        f"at lr {config['learning_rate']:g} ..."
    )
    flow, trajectory = fit_with_snapshots(
        build_spec(config["shift"]),
        train,
        val,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        init_seed=config["init_seed"],
        shuffle_seed=config["shuffle_seed"],
        record_every=config["record_every"],
        record=lambda flow: snapshot(flow, config["shift"]),
    )
    flow.save(out / "flow.pt")

    truths = true_coefficients(config["f"], config["shift"])
    plot_trajectories(
        trajectory,
        truths,
        out / "plots" / "coefficients.png",
        f"triangle/{config['f']}, {config['shift']} model — "
        "linear-shift coefficients (Fig. 14/15)",
    )
    figures.append("coefficients.png")

    metrics = {key: value for key, value in trajectory[-1].items() if key != "epoch"}
    metrics["val_nll_x3"] = float(flow.nll(val)["x3"])

    if config["shift"] == "cs":
        grid = np.linspace(
            config["grid_low"], config["grid_high"], config["grid_points"]
        )
        metrics["cs_curve_max_abs_err"] = plot_cs_curve(
            grid,
            fitted=cs_curve(flow, "x3", "x2", grid).ravel(),
            true=generator.true_shift_curve(grid),
            path=out / "plots" / "cs_curve.png",
            # which paper figure this is depends on f (7 right for atan,
            # 17 for the misspecified linear case, 18 for sin) —
            # PAPER_COVERAGE.md holds that mapping
            title=f"complex shift, DGP f = {config['f']}: fitted vs $-f(x_2)$",
        )
        figures.append("cs_curve.png")

    # L1 (observational fit) and L2 (interventional) distributions
    do_query = f"do(x1={config['do_x1']:+.0f})"
    dgp_samples = {
        "Obs": generator.observational(config["n_compare"], seed_offset=5),
        do_query: generator.interventional(
            config["n_compare"], {"x1": config["do_x1"]}
        ),
    }
    flow_samples = {
        "Obs": flow.sample(config["n_compare"], seed=config["sample_seed"]),
        do_query: flow.sample(
            config["n_compare"],
            do={"x1": config["do_x1"]},
            seed=config["sample_seed"],
        ),
    }
    plot_hist_grid(
        dgp_samples,
        flow_samples,
        ["x1", "x2", "x3"],
        out / "plots" / "distributions.png",
        f"triangle/{config['f']}, {config['shift']} model — L1/L2 (Fig. 16/17)",
        ordinal_levels={},
    )
    figures.append("distributions.png")

    metrics["mean_x3_dgp_do_x1"] = float(dgp_samples[do_query]["x3"].mean())
    metrics["mean_x3_flow_do_x1"] = float(flow_samples[do_query]["x3"].mean())

    save_metrics(out, metrics)
    write_report(
        out,
        f"triangle — f = {config['f']}, {config['shift']} shift "
        f"(paper Sec. 6.1, true beta12 = {truths['beta12']:+.1f}, "
        f"beta13 = {truths['beta13']:+.1f})",
        metrics,
        figures,
    )
    print(f"-> {out}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "variant",
        choices=variants_of(__file__),
        help="which DGP and model to run; hyperparameters live in triangle.yaml",
    )
    run(parser.parse_args().variant)
