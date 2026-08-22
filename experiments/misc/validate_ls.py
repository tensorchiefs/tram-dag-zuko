"""Validate an all-linear-shift flow against the classical MLE.

The outcome node of an all-``ls`` flow *is* a proportional-odds model, so a
flow trained to convergence without early stopping must reproduce the
classical maximum-likelihood estimate — not approximately, but to the
precision of the optimizer. This experiment is the framework's external
correctness anchor: it fits the same data three ways and compares.

1. the flow, on the full dataset with ``restore_best=False`` so it sits at
   the training-data maximum likelihood;
2. ``statsmodels`` ``OrderedModel``, on the design matrix the flow builds;
3. R's ``MASS::polr`` / ``tram``, whose committed output is read from
   ``ref_ls/`` so no R installation is needed.

It then compares the analytic treatment effect (an average over
interventional PMFs) between the flow and statsmodels, and against the
known true effect of the synthetic cohort.

Usage (from experiments/)::

    uv run python -m misc.validate_ls adam
    uv run python -m misc.validate_ls classical
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from common import (
    load_variant,
    make_output_dir,
    save_metrics,
    variants_of,
    write_report,
)
from statsmodels.miscmodels.ordinal_model import OrderedModel

from tramdag import LS, SI, CausalFlowDAG, ContinuousNode, OrdinalNode

CONFIG_KEYS = {
    "cohort",
    "fitter",
    "phases",
    "batch_size",
    "init_seed",
    "shuffle_seed",
    "classical_max_iter",
    "good_outcome_levels",
    "transform",
    "n_coeffs",
    "levels_mrs_pre",
    "levels_treatment",
    "levels_outcome",
}


def build_spec(config: dict) -> dict:
    """Give the fully-connected all-``ls`` spec of the cohort.

    Every edge is a linear shift, which is what makes each node conditional a
    classical transformation model. The continuous nodes' basis comes from the
    config rather than from a framework default, so this file plus its YAML
    describe the model completely. The ordinal nodes have no basis: their
    intercept is the cutpoint vector.
    """
    basis = dict(transform=config["transform"], n_coeffs=config["n_coeffs"])
    return {
        "Age": ContinuousNode([SI(**basis)]),
        "mRS_pre": OrdinalNode(config["levels_mrs_pre"], [LS("Age")]),
        "NIHSSa": ContinuousNode([SI(**basis), LS("Age"), LS("mRS_pre")]),
        "T": OrdinalNode(
            config["levels_treatment"], [LS("Age"), LS("mRS_pre"), LS("NIHSSa")]
        ),
        "mRS_3m": OrdinalNode(
            config["levels_outcome"],
            [LS("Age"), LS("mRS_pre"), LS("NIHSSa"), LS("T")],
        ),
    }


def load_cohort(cohort: str) -> tuple:
    """Read the frozen cohort: observational rows, trial rows, truth, R reference."""
    base = Path(__file__).resolve().parent / "data" / cohort
    columns = ["Age", "mRS_pre", "NIHSSa", "T", "mRS_3m"]
    observed = pd.read_csv(base / "obs.csv")[columns]
    trial = pd.read_csv(base / "rct.csv")[columns].astype(
        {"NIHSSa": float, "mRS_pre": int, "T": int, "mRS_3m": int}
    )
    truth = json.loads((base / "truth.json").read_text())
    reference = pd.read_csv(base / "ref_ls" / "coefficients.csv")
    return observed, trial, truth, reference


def fit_flow(spec: dict, observed: pd.DataFrame, config: dict) -> CausalFlowDAG:
    """Fit the flow to the maximum likelihood, by the configured route.

    Raises
    ------
    ValueError
        If the configured fitter is unknown.
    """
    flow = CausalFlowDAG(spec, seed=config["init_seed"])
    if config["fitter"] == "classical":
        flow.fit_classical(
            observed, max_iter=config["classical_max_iter"], verbose=False
        )
    elif config["fitter"] == "adam":
        for phase, (epochs, learning_rate) in enumerate(config["phases"]):
            flow.fit(
                observed,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=config["batch_size"],
                verbose=0,
                restore_best=False,  # stay at the training-data MLE
                seed=config["shuffle_seed"] if phase == 0 else None,
            )
    else:
        raise ValueError(f"unknown fitter '{config['fitter']}'")
    return flow


def compare_coefficients(flow, statsmodels_result, reference) -> tuple[dict, list]:
    """Compare the outcome node's coefficients across the three fits.

    Only differences between one-hot levels are identified in a cutpoint
    model, so an ordinal parent is compared as ``w[k] - w[0]``.

    Two agreement metrics come out, because the coefficients are not equally
    well determined. ``max_abs_diff_named_coefs`` covers Age, NIHSSa and the
    treatment contrast — the three the paper reads — and holds to ~1e-3.
    ``max_abs_diff_flow_vs_statsmodels`` also covers the five ``mRS_pre``
    level contrasts, and level 5 has **7 of 1275 rows**: that coefficient is
    barely identified, so where two optimizers stop differs by ~0.1 between
    machines. It is kept as a sanity bound, not as a precision claim.
    """
    fitted = flow.ls_coefficients()["mRS_3m"]
    reference_outcome = reference[reference["node"] == "mRS_3m"].set_index("term")[
        "estimate"
    ]

    rows = [
        ("Age", float(fitted["Age"][0]), statsmodels_result.params["Age"]),
        ("NIHSSa", float(fitted["NIHSSa"][0]), statsmodels_result.params["NIHSSa"]),
        (
            "T (1 vs 0)",
            float(fitted["T"][1] - fitted["T"][0]),
            statsmodels_result.params["T[1]"],
        ),
    ]
    rows.extend(
        (
            f"mRS_pre_{level} (vs 0)",
            float(fitted["mRS_pre"][level] - fitted["mRS_pre"][0]),
            statsmodels_result.params[f"mRS_pre[{level}]"],
        )
        for level in range(1, 6)
    )

    metrics = {
        "coef_Age_flow": rows[0][1],
        "coef_Age_statsmodels": float(rows[0][2]),
        "coef_Age_r": float(reference_outcome["Age"]),
        "coef_NIHSSa_flow": rows[1][1],
        "coef_NIHSSa_statsmodels": float(rows[1][2]),
        "coef_NIHSSa_r": float(reference_outcome["NIHSSa"]),
        "coef_T_flow": rows[2][1],
        "coef_T_statsmodels": float(rows[2][2]),
        "coef_T_r": float(reference_outcome["T"]),
        "max_abs_diff_named_coefs": max(
            abs(flow_value - classical_value)
            for _, flow_value, classical_value in rows[:3]
        ),
        "max_abs_diff_flow_vs_statsmodels": max(
            abs(flow_value - classical_value) for _, flow_value, classical_value in rows
        ),
    }
    return metrics, rows


def treatment_effect(flow, statsmodels_result, trial, good_levels: int) -> dict:
    """Average the interventional effect on P(good outcome) over the trial rows.

    The effect is computed from per-row interventional PMFs under
    ``do(T=0)`` and ``do(T=1)``, for the flow and for the classical fit.
    """
    flow_pmf_untreated = flow.pmf(trial, node="mRS_3m", do={"T": 0})
    flow_pmf_treated = flow.pmf(trial, node="mRS_3m", do={"T": 1})
    flow_effect = float(
        (
            flow_pmf_treated[:, :good_levels].sum(axis=1)
            - flow_pmf_untreated[:, :good_levels].sum(axis=1)
        ).mean()
    )

    design_untreated = flow.design_matrix(
        trial.assign(T=0), "mRS_3m", drop_first=True
    ).values
    design_treated = flow.design_matrix(
        trial.assign(T=1), "mRS_3m", drop_first=True
    ).values
    classical_untreated = statsmodels_result.model.predict(
        statsmodels_result.params, exog=design_untreated, which="prob"
    )
    classical_treated = statsmodels_result.model.predict(
        statsmodels_result.params, exog=design_treated, which="prob"
    )
    classical_effect = float(
        (
            classical_treated[:, :good_levels].sum(axis=1)
            - classical_untreated[:, :good_levels].sum(axis=1)
        ).mean()
    )
    return {"ate_flow": flow_effect, "ate_statsmodels": classical_effect}


def run(variant: str) -> dict:
    """Run one variant end to end and give its metrics."""
    config = load_variant(__file__, variant, CONFIG_KEYS)
    out = make_output_dir(__file__, f"validate-ls-{variant}")

    observed, trial, truth, reference = load_cohort(config["cohort"])
    print(
        f"all-ls comparison on '{config['cohort']}' (N={len(observed)}) "
        f"using the {config['fitter']} fitter ..."
    )

    flow = fit_flow(build_spec(config), observed, config)
    flow.save(out / "flow.pt")

    # the design matrix comes from the flow, so both fits see the same encoding.
    # It reads no parameter -- only the parent encoding -- so the fitted flow
    # answers it as well as a fresh one would.
    design = flow.design_matrix(observed, "mRS_3m", drop_first=True)
    statsmodels_result = OrderedModel(
        observed["mRS_3m"].astype(int), design, distr="logit"
    ).fit(method="bfgs", disp=False)

    metrics, rows = compare_coefficients(flow, statsmodels_result, reference)
    print(f"\n{'coefficient':<22}{'flow':>10}{'statsmodels':>13}{'|diff|':>9}")
    for name, flow_value, classical_value in rows:
        print(
            f"{name:<22}{flow_value:>10.4f}{classical_value:>13.4f}"
            f"{abs(flow_value - classical_value):>9.4f}"
        )
    print(f"max |diff| = {metrics['max_abs_diff_flow_vs_statsmodels']:.4f}")

    metrics.update(
        treatment_effect(flow, statsmodels_result, trial, config["good_outcome_levels"])
    )
    metrics["ate_true"] = float(truth["true_ate"])
    metrics["ate_naive_observational"] = float(truth["naive_obs_diff"])
    metrics["ate_abs_diff_flow_vs_statsmodels"] = abs(
        metrics["ate_flow"] - metrics["ate_statsmodels"]
    )
    print(
        f"\nATE on the trial covariates: flow {metrics['ate_flow']:+.4f}, "
        f"statsmodels {metrics['ate_statsmodels']:+.4f}, "
        f"true {metrics['ate_true']:+.4f} "
        f"(naive observational contrast {metrics['ate_naive_observational']:+.4f})"
    )

    save_metrics(out, metrics)
    write_report(
        out,
        f"all-`ls` validation on {config['cohort']} — flow vs statsmodels vs R "
        f"({config['fitter']} fitter)",
        metrics,
        [],
    )
    print(f"-> {out}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "variant",
        choices=variants_of(__file__),
        help="which fitting route to use; hyperparameters live in validate_ls.yaml",
    )
    run(parser.parse_args().variant)
