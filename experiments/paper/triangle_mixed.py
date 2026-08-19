"""Replicate the paper's mixed-data triangle experiments (Sec. 6.2, App. C.4).

Same DAG as the continuous triangle, but ``x3`` is ordinal with 4 levels, so
its conditional is an ordered-logit model with cutpoints.

**Sign convention.** The paper ADDS the ordinal shift, the flow SUBTRACTS
it, so the fitted weights converge to the negated paper values: -0.2 on x1
and +0.3 on x2 (linear DGP). The App. C.4 odds-ratio check is free of that
convention: ``exp(beta12_hat)`` predicts how the odds of ``x2 <= c`` change
when x1 is raised by one unit in the DGP, and both sides are measured here.

Outputs: the coefficient trajectories (Fig. 19), the complex-shift overlay
for ``cs`` variants, the observational plus ``do(x1)`` distributions
(Fig. 9/20), and the odds-ratio check.

Usage (from experiments/)::

    uv run python triangle_mixed.py linear-ls
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from common import (
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
from paper.simulations.triangle import TriangleMixed
from tramdag import CS, LS, ContinuousNode, OrdinalNode, load_config

CONFIG = Path(__file__).with_suffix(".yaml")
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
    "levels",
    "odds_ratio_threshold",
    "odds_ratio_n",
    "odds_ratio_seed",
}


def build_spec(shift: str, levels: int) -> dict:
    """Give the DAG spec with an ordinal x3.

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
        "x3": OrdinalNode(levels, [LS("x1"), x2_to_x3]),
    }


def flow_convention_truths(f: str, shift: str) -> dict:
    """Give the expected fitted weights in the flow's sign convention."""
    truths = {"beta12": 2.0, "beta13_flow": -0.2}
    if shift == "ls" and f == "linear":
        truths["beta23_flow"] = 0.3
    return truths


def snapshot(flow, shift: str) -> dict:
    """Read the linear-shift coefficients out of a flow mid-training."""
    values = {
        "beta12": ls_weight(flow, "x2", "x1"),
        "beta13_flow": ls_weight(flow, "x3", "x1"),
    }
    if shift == "ls":
        values["beta23_flow"] = ls_weight(flow, "x3", "x2")
    return values


def odds_below(values, threshold: float) -> float:
    """Give the odds that a variable is at or below a threshold."""
    probability = float((values <= threshold).mean())
    return probability / (1.0 - probability)


def dgp_odds_ratio(generator, threshold: float, n: int, seed: int) -> float:
    """Measure the DGP's odds ratio for ``x2 <= threshold`` under ``x1 += 1``.

    The same latent draw is used with and without the shift, so the ratio
    is a clean intervention effect and not sampling noise.
    """
    rng = np.random.default_rng(seed)
    latents = generator.draw_latents(n, rng)
    observed = generator.simulate(latents=latents)
    shifted = generator.simulate(
        latents=latents, do={"x1": observed["x1"].to_numpy() + 1.0}
    )
    return odds_below(shifted["x2"], threshold) / odds_below(observed["x2"], threshold)


def run(variant: str) -> dict:
    """Run one variant end to end and give its metrics."""
    config = load_config(CONFIG, "variants", variant, require=CONFIG_KEYS)
    out = make_output_dir(__file__, f"triangle-mixed-{variant}")
    figures = []

    generator = TriangleMixed(f=config["f"], seed=config["dgp_seed"])
    sample = generator.observational(config["n_train"] + config["n_val"])
    train, val = split_train_val(sample, config["n_train"], config["n_val"])

    print(
        f"fitting triangle-mixed/{config['f']} with a {config['shift']} shift "
        f"on n={len(train)} for {config['epochs']} epochs ..."
    )
    flow, trajectory = fit_with_snapshots(
        build_spec(config["shift"], config["levels"]),
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

    truths = flow_convention_truths(config["f"], config["shift"])
    plot_trajectories(
        trajectory,
        truths,
        out / "plots" / "coefficients.png",
        f"triangle-mixed/{config['f']}, {config['shift']} model — "
        "coefficients in the flow's sign convention (Fig. 19)",
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
            title=f"complex shift on an ordinal node, DGP f = {config['f']}",
        )
        figures.append("cs_curve.png")

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
        f"triangle-mixed/{config['f']}, {config['shift']} model — L1/L2 (Fig. 9/20)",
        ordinal_levels={"x3": config["levels"]},
    )
    figures.append("distributions.png")

    # App. C.4: the interventional odds ratio, predicted vs measured
    threshold = config["odds_ratio_threshold"]
    predicted = float(np.exp(metrics["beta12"]))
    measured = dgp_odds_ratio(
        generator, threshold, config["odds_ratio_n"], config["odds_ratio_seed"]
    )
    metrics["odds_ratio_predicted"] = predicted
    metrics["odds_ratio_dgp"] = measured
    print(
        f"C.4 odds ratio for odds(x2 <= {threshold}) under do(x1 += 1): "
        f"predicted {predicted:.2f}, DGP {measured:.2f}, "
        f"theory exp(2) = {np.exp(2.0):.2f}"
    )

    save_metrics(out, metrics)
    write_report(
        out,
        f"triangle-mixed — f = {config['f']}, {config['shift']} shift "
        f"(paper Sec. 6.2; cutpoints {generator.theta.tolist()})",
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
        help="which DGP and model to run; hyperparameters live in triangle_mixed.yaml",
    )
    run(parser.parse_args().variant)
