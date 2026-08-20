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

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
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
    finish,
    fit_with_snapshots,
    ls_weight,
    plot_cs_curve,
    plot_hist_grid,
    plot_trajectories,
    split_train_val,
)
from paper.simulations.triangle import TriangleMixed
from tramdag import CS, LS, SI, ContinuousNode, OrdinalNode

CONFIG_KEYS = {
    "f",
    "shift",
    "transform",
    "n_coeffs",
    "n_train",
    "n_val",
    "epochs",
    "learning_rate",
    "batch_size",
    "chunk_epochs",
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
    "cf_n",
    "cf_draws",
    "cf_seed",
}
# only a complex shift has a network to configure
MLP_KEYS = {"shift_units", "activation"}


def build_spec(config: dict) -> dict:
    """Give the DAG spec with an ordinal x3.

    Every network and transform setting comes from the config, so the
    architecture is visible in one file. The ordinal node has no monotone
    basis — its intercept is the cutpoint vector.

    Raises
    ------
    ValueError
        If ``shift`` is neither ``"ls"`` nor ``"cs"``.
    """
    shift = config["shift"]
    if shift == "ls":
        x2_to_x3 = LS("x2")
    elif shift == "cs":
        x2_to_x3 = CS(
            "x2", units=config["shift_units"], activation=config["activation"]
        )
    else:
        raise ValueError(f"shift must be 'ls' or 'cs', got '{shift}'")
    basis = dict(transform=config["transform"], n_coeffs=config["n_coeffs"])
    return {
        "x1": ContinuousNode([SI(**basis)]),
        "x2": ContinuousNode([SI(**basis), LS("x1")]),
        "x3": OrdinalNode(config["levels"], [LS("x1"), x2_to_x3]),
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


def counterfactual_pmf_from_flow(flow, factual, do, draws, seed, levels):
    """Average the flow's counterfactual level over repeated abductions.

    An ordinal latent is only interval-identified, so ``abduct`` draws it from
    the truncated logistic and one pass gives one *sample* of the
    counterfactual level. Averaging many passes turns that into the per-row
    distribution, which is the object the analytic truth can be compared to.
    """
    counts = np.zeros((len(factual), levels))
    for draw in range(draws):
        latents = flow.abduct(factual, seed=seed + draw)
        sampled = flow.sample(do=do, u=latents)["x3"].to_numpy().astype(int)
        counts[np.arange(len(factual)), sampled] += 1.0
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
    distribution exactly would score. The attainable maximum is
    ``E[max_i p_i]``, reached by always naming the modal level — a strictly
    worse *distribution* estimate that this score nonetheless rewards. A flow
    slightly above the analytic reference is therefore sharper than the
    identifiable law, not better than it, and ``cf_pmf_tv_vs_analytic`` is the
    metric that cannot be gamed that way.
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
    levels = np.arange(flow_pmf.shape[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(
        levels - 0.18,
        analytic.mean(0),
        width=0.36,
        alpha=0.6,
        label="identifiable truth",
    )
    ax.bar(
        levels + 0.18,
        flow_pmf.mean(0),
        width=0.36,
        alpha=0.85,
        color="C3",
        label="flow (averaged abductions)",
    )
    ax.set_xticks(levels)
    ax.set_xlabel("counterfactual $x_3$")
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.legend(fontsize=8)
    finish(fig, path)


def run(variant: str) -> dict:
    """Run one variant end to end and give its metrics."""
    # an ls variant has no network, so its key set is smaller: read, then check
    config = load_variant(__file__, variant)
    keys = CONFIG_KEYS | (MLP_KEYS if config["shift"] == "cs" else set())
    config = load_variant(__file__, variant, keys)
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
        build_spec(config),
        train,
        val,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        init_seed=config["init_seed"],
        shuffle_seed=config["shuffle_seed"],
        chunk_epochs=config["chunk_epochs"],
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
