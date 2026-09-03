"""Replicate the paper's CAREFL counterfactual benchmark (Sec. 5.3, App. C.2).

A 4-variable SCM with additive Laplace noise. Because the noise is additive
and the equations are known, the true counterfactuals are **analytic**: this
is the one benchmark where a flow's abduction can be scored against exact
values instead of a second sample.

The reference run (carefl_fig5.r, USE_EXTERNAL_DATA) trains on CAREFL's own
2500 rows — committed here under ``data/carefl-cf`` with the paper's
observation and the analytic truth curves, x3/x4 sd-standardized — so the
flow sees the paper's exact data and the Fig. 6 curves are comparable
point by point, against both the truth and CAREFL's own predictions.
Because one observation is a noisy yardstick, the mean absolute
counterfactual error over fresh held-out rows (scaled into the training
units) is measured next to it — the number to watch for regressions.

Usage (from experiments/)::

    uv run python -m paper.carefl flexible
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import cli, load_variant, make_output_dir, save_metrics, write_report

from paper.helpers import finish, fit_paper
from paper.simulations.carefl import Carefl4
from tramdag import CI, SI, ContinuousNode

# %% global variables ------------------------------------------------------------------
DATA = Path(__file__).resolve().parent / "data" / "carefl-cf"
COLUMNS = ["x1", "x2", "x3", "x4"]
# the curve files' grid: carefl_fig5.r's seq(-3, 2.9, 0.1)
ALPHAS = np.round(np.arange(-3.0, 3.0, 0.1), 4)


# %% public functions ------------------------------------------------------------------
def load_reference() -> dict:
    """Read the frozen reference data: training rows, observation, curves."""

    def curve(name: str) -> np.ndarray:
        values = pd.read_csv(DATA / name, header=None)[0].to_numpy()
        assert len(values) == len(ALPHAS)
        return values

    truth = json.loads((DATA / "truth.json").read_text())
    return {
        "train": pd.read_csv(DATA / "X.csv", header=None, names=COLUMNS),
        "x_obs": pd.read_csv(DATA / "xObs.csv", header=None, names=COLUMNS),
        "true_x3": curve("xCF_onX2_true.csv"),
        "true_x4": curve("xCF_onX1_true.csv"),
        "carefl_x3": curve("xCF_onX2_pred.csv"),
        "carefl_x4": curve("xCF_onX1_pred.csv"),
        "sds": {"x3": truth["sd_x3"], "x4": truth["sd_x4"]},
    }


def build_spec(config: dict) -> dict:
    """Give the all-complex-intercept spec of the 4-variable SCM.

    Every network and transform setting comes from the config, so nothing is
    inherited from a framework default.
    """
    basis = dict(
        transform=config["transform"],
        n_coeffs=config["n_coeffs"],
        range_q=config["range_q"],
    )
    net = dict(
        units=config["intercept_units"],
        activation=config["activation"],
        input_transform=config["input_transform"],
    )
    return {
        "x1": ContinuousNode([SI(**basis)]),
        "x2": ContinuousNode([SI(**basis)]),
        "x3": ContinuousNode([CI("x1", "x2", **basis, **net)]),
        "x4": ContinuousNode([CI("x1", "x2", **basis, **net)]),
    }


def counterfactual_curve(flow, latents, do_variable, target, alphas) -> list[float]:
    """Push one abducted row through ``do(variable = alpha)`` for each alpha."""
    return [
        float(flow.sample(do={do_variable: alpha}, u=latents)[target].iloc[0])
        for alpha in alphas
    ]


def plot_curves(flow_x3, flow_x4, ref, path) -> dict:
    """Plot both counterfactual queries against the truth and CAREFL (Fig. 6)."""
    panels = [
        (flow_x3, "x3", "carefl_x3", "would $x_2$ have been $\\alpha$"),
        (flow_x4, "x4", "carefl_x4", "would $x_1$ have been $\\alpha$"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (fitted, target, carefl, xlabel) in zip(axes, panels, strict=True):
        ax.plot(ALPHAS, ref[f"true_{target}"], "-", color="C3", lw=2, label="DGP")
        ax.plot(ALPHAS, ref[carefl], "o", ms=4, color="gold", label="CAREFL")
        ax.plot(ALPHAS, fitted, "o", ms=3, color="C0", label="flow")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"${target[0]}_{target[1]}$")
        ax.legend()
    x_obs = ", ".join(f"{value:.2f}" for value in ref["x_obs"].iloc[0])
    fig.suptitle(
        f"CAREFL counterfactual queries at $x_{{obs}}$ = ({x_obs})\n"
        "(Fig. 6; x3/x4 in CAREFL's sd-standardized units)"
    )
    finish(fig, path)
    return {
        "fig6_max_abs_err_x3": float(
            np.max(np.abs(np.asarray(flow_x3) - ref["true_x3"]))
        ),
        "fig6_max_abs_err_x4": float(
            np.max(np.abs(np.asarray(flow_x4) - ref["true_x4"]))
        ),
        "fig6_max_abs_err_x4_carefl": float(
            np.max(np.abs(ref["carefl_x4"] - ref["true_x4"]))
        ),
    }


def standardized(df: pd.DataFrame, sds: dict) -> pd.DataFrame:
    """Scale a raw-unit draw into the training data's units."""
    out = df.copy()
    for name, sd in sds.items():
        out[name] = out[name] / sd
    return out


def heldout_errors(generator, flow, sds, config) -> dict:
    """Score counterfactuals on typical rows, against the analytic truth.

    The DGP's noise is recovered exactly by ``abduct_noise``, so pushing the
    same rows through the mutilated SCM gives the true counterfactual of
    each row — the flow's abduction is compared against that, in the
    training data's sd-standardized units.
    """
    rows = generator.observational(
        config["n_heldout"], seed_offset=config["heldout_seed_offset"]
    )
    flow_latents = flow.abduct(standardized(rows, sds))
    dgp_noise = generator.abduct_noise(rows)

    errors = {}
    for alpha in config["alphas_scored"]:
        for do_variable, target in [("x2", "x3"), ("x1", "x4")]:
            flow_values = flow.sample(do={do_variable: alpha}, u=flow_latents)[target]
            true_values = (
                generator.simulate(do={do_variable: alpha}, latents=dgp_noise)[target]
                / sds[target]
            )
            errors[f"cf_mae_{target}_do_{do_variable}_{alpha:+.1f}"] = float(
                np.abs(flow_values.to_numpy() - true_values.to_numpy()).mean()
            )
    for name, value in errors.items():
        print(f"{name}: {value:.3f}")
    return errors


def run(variant: str) -> dict:
    """Run the benchmark end to end and give its metrics."""
    config = load_variant(__file__, variant)
    out = make_output_dir(__file__, f"carefl-{variant}")

    ref = load_reference()
    print(
        f"fitting the flexible flow on CAREFL's own rows, n={len(ref['train'])}: "
        f"{config['epochs']} epochs at lr {config['learning_rate']:g} ..."
    )
    # val = train, as in carefl_fig5.r: the plateau rule watches the train NLL
    flow, _, fit_seconds = fit_paper(
        ref["train"], ref["train"], build_spec(config), config, out
    )

    paper_latents = flow.abduct(ref["x_obs"])
    metrics = plot_curves(
        counterfactual_curve(flow, paper_latents, "x2", "x3", ALPHAS),
        counterfactual_curve(flow, paper_latents, "x1", "x4", ALPHAS),
        ref,
        out / "plots" / "cf_curves.png",
    )
    generator = Carefl4(seed=config["dgp_seed"])
    metrics.update(heldout_errors(generator, flow, ref["sds"], config))
    # the reference holds nothing out; a fresh standardized draw scores the fit
    heldout = standardized(
        generator.observational(config["n_val"], seed_offset=1), ref["sds"]
    )
    val_nll = flow.nll(heldout)
    metrics["val_nll_x3"] = float(val_nll["x3"])
    metrics["val_nll_x4"] = float(val_nll["x4"])
    metrics["fit_seconds"] = fit_seconds

    save_metrics(out, metrics)
    write_report(
        out,
        "CAREFL counterfactual benchmark (paper Sec. 5.3; trained on CAREFL's "
        "own 2500 rows, so Fig. 6 is comparable point by point — every error "
        "metric is an exact deviation from the DGP truth)",
        metrics,
        ["cf_curves.png"],
    )
    print(f"-> {out}")
    return metrics


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    run(cli(__file__, __doc__))
