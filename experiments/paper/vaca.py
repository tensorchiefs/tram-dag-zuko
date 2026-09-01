"""Replicate the paper's VACA/CNF benchmark (Sec. 5.1-5.2, App. C.1).

The DGP is a triangle whose source is a **bimodal** Gaussian mixture and
whose noise is Gaussian throughout — deliberately outside the flow's
logistic-latent family, so a flexible (all complex-intercept) TRAM-DAG has
to learn the shape rather than inherit it. This is the paper's headline L1
case that the default Causal Normalizing Flow fails to fit.

Outputs: the pairs plot of the observational joint (Fig. 4) and the
interventional densities ``p(x3 | do(x2 = a))`` (Fig. 5). The interventional
means are analytic — ``E[x3 | do(x2=a)] = E[x1] + 0.25 a`` — so the metrics
compare the flow against exact values, not against a second sample.

Usage (from experiments/)::

    uv run python -m paper.vaca flexible
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import matplotlib.pyplot as plt
from common import cli, load_variant, make_output_dir, save_metrics, write_report

from paper.helpers import continuous_hist, finish, fit_paper
from paper.simulations.vaca import DO_X2_VALUES, VacaTriangle
from tramdag import CI, SI, ContinuousNode


# %% private functions -----------------------------------------------------------------
def _scatter_panel(ax, observed, sampled, x: str, y: str, n_scatter: int) -> None:
    """Scatter DGP and flow samples of one variable pair."""
    ax.scatter(
        observed[x][:n_scatter],
        observed[y][:n_scatter],
        s=2,
        alpha=0.3,
        label="DGP",
    )
    ax.scatter(
        sampled[x][:n_scatter],
        sampled[y][:n_scatter],
        s=2,
        alpha=0.3,
        color="C3",
        label="flow",
    )


# %% public functions ------------------------------------------------------------------
def build_spec(config: dict) -> dict:
    """Give the all-complex-intercept spec: every conditional fully flexible.

    Every network and transform setting comes from the config, so nothing is
    inherited from a framework default.
    """
    basis = dict(transform=config["transform"], n_coeffs=config["n_coeffs"])
    net = dict(
        units=config["intercept_units"],
        activation=config["activation"],
        input_transform=config["input_transform"],
    )
    return {
        "x1": ContinuousNode([SI(**basis)]),
        "x2": ContinuousNode([CI("x1", **basis, **net)]),
        "x3": ContinuousNode([CI("x1", "x2", **basis, **net)]),
    }


def plot_pairs(observed, sampled, columns, bins, n_scatter, path):
    """Pairs plot: marginals on the diagonal, scatters off it (Fig. 4)."""
    k = len(columns)
    fig, axes = plt.subplots(k, k, figsize=(3 * k, 3 * k), squeeze=False)
    for row, row_column in enumerate(columns):
        for col, col_column in enumerate(columns):
            ax = axes[row][col]
            if row == col:
                continuous_hist(ax, observed[row_column], sampled[row_column], bins)
            else:
                _scatter_panel(ax, observed, sampled, col_column, row_column, n_scatter)
    for ax, column in zip(axes[-1], columns, strict=True):
        ax.set_xlabel(column)
    for ax_row, column in zip(axes, columns, strict=True):
        ax_row[0].set_ylabel(column)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("VACA triangle — observational joint, DGP vs flow (Fig. 4)")
    finish(fig, path)


def plot_interventional(generator, flow, config, truth, path) -> dict:
    """Interventional densities per do(x2) value (Fig. 5); give the mean errors."""
    fig, axes = plt.subplots(1, len(DO_X2_VALUES), figsize=(11, 3.2), sharey=True)
    errors = {}
    for ax, value in zip(axes, DO_X2_VALUES, strict=True):
        dgp = generator.interventional(config["n_compare"], {"x2": value})
        sampled = flow.sample(
            config["n_compare"], do={"x2": value}, seed=config["sample_seed"]
        )
        continuous_hist(ax, dgp["x3"], sampled["x3"], config["hist_bins"])
        ax.set_title(f"do($x_2$ = {value:+.0f})")
        ax.set_xlabel("$x_3$")

        analytic = truth["do_x2"][str(value)]["mean_x3_analytic"]
        flow_mean = float(sampled["x3"].mean())
        errors[f"mean_x3_flow_do_x2_{value:+.0f}"] = flow_mean
        errors[f"mean_x3_abs_err_do_x2_{value:+.0f}"] = abs(flow_mean - analytic)
        print(
            f"do(x2={value:+.0f}): E[x3] flow {flow_mean:+.3f} "
            f"vs analytic {analytic:+.3f}"
        )
    axes[0].legend()
    axes[0].set_ylabel("$p(x_3\\,|\\,do(x_2))$")
    fig.suptitle("VACA triangle — interventional distributions (Fig. 5)")
    finish(fig, path)
    return errors


def run(variant: str) -> dict:
    """Run the benchmark end to end and give its metrics."""
    config = load_variant(__file__, variant)
    out = make_output_dir(__file__, f"vaca-{variant}")

    generator = VacaTriangle(seed=config["dgp_seed"])
    truth = generator.true_moments(mc_n=0)  # analytic only, no Monte-Carlo draw

    print(
        f"fitting the flexible flow on the VACA triangle, n={config['n_train']}: "
        f"{config['epochs']} epochs at lr {config['learning_rate']:g} ..."
    )
    flow, train, val, _, fit_seconds = fit_paper(
        generator, build_spec(config), config, out
    )

    sampled = flow.sample(len(train), seed=config["sample_seed"])
    plot_pairs(
        train,
        sampled,
        ["x1", "x2", "x3"],
        config["hist_bins"],
        config["n_scatter"],
        out / "plots" / "pairs.png",
    )

    metrics = {"val_nll_x3": float(flow.nll(val)["x3"])}
    metrics.update(
        plot_interventional(
            generator, flow, config, truth, out / "plots" / "interventional.png"
        )
    )
    # L1: does the flow reproduce the bimodal source marginal?
    metrics["std_x1_flow"] = float(sampled["x1"].std())
    metrics["std_x1_abs_err"] = abs(metrics["std_x1_flow"] - truth["std_x1_analytic"])
    metrics["fit_seconds"] = fit_seconds

    save_metrics(out, metrics)
    write_report(
        out,
        "VACA / CNF benchmark (paper Sec. 5.1-5.2; bimodal source, Gaussian noise; "
        "interventional means are analytic)",
        metrics,
        ["pairs.png", "interventional.png"],
        truths={
            "std_x1_flow": truth["std_x1_analytic"],
            **{
                f"mean_x3_flow_do_x2_{v:+.0f}": truth["do_x2"][str(v)][
                    "mean_x3_analytic"
                ]
                for v in DO_X2_VALUES
            },
        },
    )
    print(f"-> {out}")
    return metrics


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    run(cli(__file__, __doc__))
