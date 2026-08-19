"""Replicate the paper's CAREFL counterfactual benchmark (Sec. 5.3, App. C.2).

A 4-variable SCM with additive Laplace noise. Because the noise is additive
and the equations are known, the true counterfactuals are **analytic**: this
is the one benchmark where a flow's abduction can be scored against exact
values instead of a second sample.

Two things are measured. The paper's Fig. 6 curves at its single observation
``x_obs``, and — because that observation sits at a roughly 4-sigma abducted
noise value and is therefore a hard extrapolation — the mean absolute
counterfactual error over a sample of typical held-out rows, which is the
number to watch for regressions.

Usage (from experiments/)::

    uv run python carefl.py flexible
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import (
    finish,
    fit_with_snapshots,
    load_config,
    make_output_dir,
    save_metrics,
    split_train_val,
    variants_of,
    write_report,
)
from simulations.carefl import ALPHA_GRID, X_OBS, Carefl4

from tramdag import ContinuousNode, I

CONFIG_KEYS = {
    "n_train",
    "n_val",
    "epochs",
    "learning_rate",
    "batch_size",
    "polish_epochs",
    "polish_learning_rate",
    "dgp_seed",
    "init_seed",
    "shuffle_seed",
    "n_heldout",
    "heldout_seed",
    "alphas_scored",
}


def build_spec() -> dict:
    """Give the all-complex-intercept spec of the 4-variable SCM."""
    return {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(),
        "x3": ContinuousNode([I("x1", "x2")]),
        "x4": ContinuousNode([I("x1", "x2")]),
    }


def counterfactual_curve(flow, latents, do_variable, target, alphas) -> list[float]:
    """Push one abducted row through ``do(variable = alpha)`` for each alpha."""
    return [
        float(flow.sample(do={do_variable: alpha}, u=latents)[target].iloc[0])
        for alpha in alphas
    ]


def plot_curves(alphas, flow_x3, flow_x4, truth, path) -> dict:
    """Plot both counterfactual queries against the analytic truth (Fig. 6)."""
    panels = [
        (flow_x3, truth["x3_cf_do_x2"], "would $x_2$ have been $\\alpha$", "$x_3$"),
        (flow_x4, truth["x4_cf_do_x1"], "would $x_1$ have been $\\alpha$", "$x_4$"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (fitted, true, xlabel, ylabel) in zip(axes, panels):
        ax.plot(alphas, true, "-", color="C3", lw=2, label="DGP (analytic)")
        ax.plot(alphas, fitted, "o", ms=3, color="C0", label="flow")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(
        "CAREFL counterfactual queries at $x_{obs}$ = (2.00, 1.50, 0.81, -0.28) "
        "(Fig. 6)"
    )
    finish(fig, path)
    return {
        "fig6_max_abs_err_x3": float(
            np.max(np.abs(np.asarray(flow_x3) - truth["x3_cf_do_x2"]))
        ),
        "fig6_max_abs_err_x4": float(
            np.max(np.abs(np.asarray(flow_x4) - truth["x4_cf_do_x1"]))
        ),
    }


def heldout_errors(generator, flow, config) -> dict:
    """Score counterfactuals on typical rows, against the analytic truth.

    The DGP's noise is recovered exactly by ``abduct_noise``, so pushing the
    same rows through the mutilated SCM gives the true counterfactual of
    each row — the flow's abduction is compared against that.
    """
    rows = generator.observational(
        config["n_heldout"], seed_offset=config["heldout_seed"]
    )
    flow_latents = flow.abduct(rows)
    dgp_noise = generator.abduct_noise(rows)

    errors = {}
    for alpha in config["alphas_scored"]:
        for do_variable, target in [("x2", "x3"), ("x1", "x4")]:
            flow_values = flow.sample(do={do_variable: alpha}, u=flow_latents)[target]
            true_values = generator.simulate(
                do={do_variable: alpha}, latents=dgp_noise
            )[target]
            errors[f"cf_mae_{target}_do_{do_variable}_{alpha:+.1f}"] = float(
                np.abs(flow_values.to_numpy() - true_values.to_numpy()).mean()
            )
    for name, value in errors.items():
        print(f"{name}: {value:.3f}")
    return errors


def run(variant: str) -> dict:
    """Run the benchmark end to end and give its metrics."""
    config = load_config("carefl", variant, CONFIG_KEYS)
    out = make_output_dir(f"carefl-{variant}")

    generator = Carefl4(seed=config["dgp_seed"])
    sample = generator.observational(config["n_train"] + config["n_val"])
    train, val = split_train_val(sample, config["n_train"], config["n_val"])

    print(
        f"fitting the flexible flow on the CAREFL SCM, n={len(train)}: "
        f"{config['epochs']} epochs at lr {config['learning_rate']:g}, then "
        f"{config['polish_epochs']} at lr {config['polish_learning_rate']:g} ..."
    )
    flow, _ = fit_with_snapshots(
        build_spec(),
        train,
        val,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        init_seed=config["init_seed"],
        shuffle_seed=config["shuffle_seed"],
        record_every=config["epochs"],  # no snapshots needed here
    )
    flow.fit(
        train,
        val,
        epochs=config["polish_epochs"],
        learning_rate=config["polish_learning_rate"],
        batch_size=config["batch_size"],
        verbose=0,
    )
    flow.save(out / "flow.pt")

    paper_latents = flow.abduct(pd.DataFrame([X_OBS]))
    truth = generator.true_cf_curves()
    metrics = plot_curves(
        ALPHA_GRID,
        counterfactual_curve(flow, paper_latents, "x2", "x3", ALPHA_GRID),
        counterfactual_curve(flow, paper_latents, "x1", "x4", ALPHA_GRID),
        truth,
        out / "plots" / "cf_curves.png",
    )
    metrics.update(heldout_errors(generator, flow, config))
    metrics["val_nll_x3"] = float(flow.nll(val)["x3"])
    metrics["val_nll_x4"] = float(flow.nll(val)["x4"])

    save_metrics(out, metrics)
    write_report(
        out,
        "CAREFL counterfactual benchmark (paper Sec. 5.3; the DGP's "
        "counterfactuals are analytic, so these are exact errors)",
        metrics,
        ["cf_curves.png"],
    )
    print(f"-> {out}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "variant",
        choices=variants_of("carefl"),
        help="which model to run; hyperparameters live in carefl.yaml",
    )
    run(parser.parse_args().variant)
