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

    uv run python -m paper.triangle_mixed linear-ls
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import cli, load_variant, make_output_dir, save_metrics, write_report

from paper.helpers import (
    compare_do_x1,
    cs_curve_error,
    finish,
    fit_paper,
    level_bars,
    plot_trajectories,
    shift_term,
    snapshot,
    true_coefficients,
)
from paper.simulations.triangle import TriangleMixed
from tramdag import LS, SI, ContinuousNode, OrdinalNode


# %% public functions ------------------------------------------------------------------
def build_spec(config: dict) -> dict:
    """Give the DAG spec with an ordinal x3.

    Every network and transform setting comes from the config, so the
    architecture is visible in one file. The ordinal node has no monotone
    basis — its intercept is the cutpoint vector.
    """
    basis = dict(transform=config["transform"], n_coeffs=config["n_coeffs"])
    return {
        "x1": ContinuousNode([SI(**basis)]),
        "x2": ContinuousNode([SI(**basis), LS("x1")]),
        "x3": OrdinalNode(config["levels"], [LS("x1"), shift_term(config)]),
    }


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


def counterfactual_pmf_from_flow(flow, factual, do, draws, seed, levels):
    """Average the flow's counterfactual level over repeated abductions.

    An ordinal latent is only interval-identified, so ``abduct`` draws it from
    the truncated logistic and one pass gives one *sample* of the
    counterfactual level. Averaging many passes turns that into the per-row
    distribution, which is the object the analytic truth can be compared to.
    The rows are tiled ``draws`` times, so it is one abduction and one sample.
    """
    n = len(factual)
    tiled = pd.concat([factual] * draws, ignore_index=True)
    latents = flow.abduct(tiled, seed=seed)
    sampled = flow.sample(do=do, u=latents)["x3"].to_numpy().astype(int)
    counts = np.zeros((n, levels))
    np.add.at(counts, (np.tile(np.arange(n), draws), sampled), 1.0)
    return counts / draws


def score_counterfactuals(
    flow, generator, config
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Score the flow's ordinal counterfactuals against what is identifiable.

    The paper's App. B (Fig. 10) makes the point that an individual
    counterfactual is not identified for a variable produced by interval
    censoring. So the flow is not scored against the realised counterfactual
    level — nothing could get that right — but against the exact
    counterfactual *distribution*.

    Two reference points come with the P(true level) score, because that score
    is **not** maximized by the truth. Scoring the analytic law itself gives
    ``E[p_true] = E[sum_i p_i^2]``: what a model that knew the identifiable
    distribution exactly would score. The largest *expected* score is
    ``E[max_i p_i]``, from always naming the modal level — a strictly worse
    *distribution* estimate that this score nonetheless rewards.

    Both are expectations, and this metric is one finite draw of ``cf_n`` rows,
    so neither is a per-run ceiling: on the ``linear`` DGP the mode predictor
    itself scores 0.829 against its own 0.806 expectation. Read them as
    reference points a run should sit between, and read
    ``cf_pmf_tv_vs_analytic`` as the metric that cannot be gamed by sharpening
    a prediction — a flow slightly above the analytic reference is sharper than
    the identifiable law, not better than it.
    """
    do = {"x1": config["do_x1"]}
    factual, realised = generator.counterfactual_pair(config["cf_n"], do)
    analytic = generator.true_counterfactual_pmf(factual, do)
    flow_pmf = counterfactual_pmf_from_flow(
        flow, factual, do, config["cf_draws"], config["cf_seed"], config["levels"]
    )

    rows = np.arange(len(factual))
    true_level = realised["x3"].to_numpy().astype(int)
    metrics = {
        # half the L1 distance between the two distributions, averaged
        "cf_pmf_tv_vs_analytic": float(
            0.5 * np.abs(flow_pmf - analytic).sum(axis=1).mean()
        ),
        # probability each assigns to the level that actually happened
        "cf_prob_true_level_flow": float(flow_pmf[rows, true_level].mean()),
        # what predicting the identifiable law itself scores, and the most
        # any prediction can score (by always naming the modal level)
        "cf_prob_true_level_analytic": float(analytic[rows, true_level].mean()),
        "cf_prob_true_level_mode_bound": float(analytic.max(axis=1).mean()),
    }
    print(
        f"ordinal counterfactuals ({config['cf_n']} rows, "
        f"{config['cf_draws']} abduction draws):"
    )
    print(
        f"  P(true level): flow {metrics['cf_prob_true_level_flow']:.3f}, "
        f"analytic law {metrics['cf_prob_true_level_analytic']:.3f}, "
        f"mode bound {metrics['cf_prob_true_level_mode_bound']:.3f}"
    )
    total_variation = metrics["cf_pmf_tv_vs_analytic"]
    print(f"  total-variation from the analytic law: {total_variation:.3f}")
    return metrics, flow_pmf, analytic


def plot_counterfactual_pmfs(flow_pmf, analytic, path, title):
    """Compare the two counterfactual distributions, averaged per level."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    level_bars(
        ax,
        analytic.mean(0),
        flow_pmf.mean(0),
        labels=("identifiable truth", "flow (averaged abductions)"),
    )
    ax.set_xlabel("counterfactual $x_3$")
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.legend(fontsize=8)
    finish(fig, path)


def run(variant: str) -> dict:
    """Run one variant end to end and give its metrics."""
    config = load_variant(__file__, variant)
    out = make_output_dir(__file__, f"triangle-mixed-{variant}")
    figures = ["coefficients.png"]

    generator = TriangleMixed(f=config["f"], seed=config["dgp_seed"])
    print(
        f"fitting triangle-mixed/{config['f']} with a {config['shift']} shift "
        f"on n={config['n_train']} for {config['epochs']} epochs ..."
    )
    train = generator.observational(config["n_train"])
    val = generator.observational(config["n_val"], seed_offset=1)
    flow, trajectory, fit_seconds = fit_paper(
        train,
        val,
        build_spec(config),
        config,
        out,
        record=partial(snapshot, shift=config["shift"]),
    )

    truths = true_coefficients(config)
    plot_trajectories(
        trajectory,
        truths,
        out / "plots" / "coefficients.png",
        f"triangle-mixed/{config['f']}, {config['shift']} model — "
        "coefficients in the flow's sign convention (Fig. 19)",
    )

    metrics = {key: value for key, value in trajectory[-1].items() if key != "epoch"}
    metrics["val_nll_x3"] = float(flow.nll(val)["x3"])
    metrics["fit_seconds"] = fit_seconds

    if config["shift"] == "cs":
        metrics["cs_curve_max_abs_err"] = cs_curve_error(
            flow,
            generator,
            config,
            out,
            f"complex shift on an ordinal node, DGP f = {config['f']}",
        )
        figures.append("cs_curve.png")

    metrics.update(
        compare_do_x1(
            generator,
            flow,
            config,
            out,
            ordinal_levels={"x3": config["levels"]},
            title=f"triangle-mixed/{config['f']}, {config['shift']} model — "
            "L1/L2 (Fig. 9/20)",
        )
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

    cf_metrics, flow_pmf, analytic_pmf = score_counterfactuals(flow, generator, config)
    metrics.update(cf_metrics)
    plot_counterfactual_pmfs(
        flow_pmf,
        analytic_pmf,
        out / "plots" / "counterfactual_pmf.png",
        f"ordinal counterfactuals under do(x1={config['do_x1']:+.0f})\n"
        "only a distribution is identified (App. B)",
    )
    figures.append("counterfactual_pmf.png")

    save_metrics(out, metrics)
    write_report(
        out,
        f"triangle-mixed — f = {config['f']}, {config['shift']} shift "
        f"(paper Sec. 6.2; cutpoints {generator.theta.tolist()})",
        metrics,
        figures,
        truths={
            **truths,
            "mean_x3_flow_do_x1": metrics["mean_x3_dgp_do_x1"],
            # C.4: both odds-ratio readings answer to the theory value e^2
            "odds_ratio_predicted": float(np.exp(2.0)),
            "odds_ratio_dgp": float(np.exp(2.0)),
            # what a model knowing the analytic law exactly would score
            "cf_prob_true_level_flow": metrics["cf_prob_true_level_analytic"],
        },
    )
    print(f"-> {out}")
    return metrics


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    run(cli(__file__, __doc__))
