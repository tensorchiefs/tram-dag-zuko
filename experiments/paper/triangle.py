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

    uv run python -m paper.triangle atan-cs
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
from common import cli, load_variant, make_output_dir, save_metrics, write_report

from paper.helpers import (
    compare_do_x1,
    cs_curve,
    fit_paper,
    plot_cs_curve,
    plot_trajectories,
    shift_term,
    snapshot,
)
from paper.simulations.triangle import TriangleContinuous
from tramdag import LS, SI, ContinuousNode


# %% public functions ------------------------------------------------------------------
def build_spec(config: dict) -> dict:
    """Give the DAG spec, with the x2 -> x3 edge as a linear or complex shift.

    Every network and transform setting is taken from the config rather than
    from a framework default, so the architecture is visible in one file.
    """
    basis = dict(transform=config["transform"], n_coeffs=config["n_coeffs"])
    return {
        "x1": ContinuousNode([SI(**basis)]),
        "x2": ContinuousNode([SI(**basis), LS("x1")]),
        "x3": ContinuousNode([SI(**basis), LS("x1"), shift_term(config)]),
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


def run(variant: str) -> dict:
    """Run one variant end to end and give its metrics."""
    config = load_variant(__file__, variant)
    out = make_output_dir(__file__, f"triangle-{variant}")
    figures = ["coefficients.png"]

    generator = TriangleContinuous(f=config["f"], seed=config["dgp_seed"])
    print(
        f"fitting triangle/{config['f']} with a {config['shift']} shift on "
        f"n={config['n_train']} for {config['epochs']} epochs "
        f"at lr {config['learning_rate']:g} ..."
    )
    flow, val, trajectory = fit_paper(
        generator,
        build_spec(config),
        config,
        out,
        record=lambda flow: snapshot(flow, config["shift"]),
    )

    truths = true_coefficients(config["f"], config["shift"])
    plot_trajectories(
        trajectory,
        truths,
        out / "plots" / "coefficients.png",
        f"triangle/{config['f']}, {config['shift']} model — "
        "linear-shift coefficients (Fig. 14/15)",
    )

    metrics = {key: value for key, value in trajectory[-1].items() if key != "epoch"}
    metrics["val_nll_x3"] = float(flow.nll(val)["x3"])

    if config["shift"] == "cs":
        grid = np.linspace(
            config["grid_low"], config["grid_high"], config["grid_points"]
        )
        metrics["cs_curve_max_abs_err"] = plot_cs_curve(
            grid,
            fitted=cs_curve(flow, "x3", "x2", grid),
            true=generator.true_shift_curve(grid),
            path=out / "plots" / "cs_curve.png",
            # which paper figure this is depends on f (7 right for atan,
            # 17 for the misspecified linear case, 18 for sin) —
            # PAPER_COVERAGE.md holds that mapping
            title=f"complex shift, DGP f = {config['f']}: fitted vs $-f(x_2)$",
        )
        figures.append("cs_curve.png")

    # L1 (observational fit) and L2 (interventional) distributions
    metrics.update(
        compare_do_x1(
            generator,
            flow,
            config,
            out,
            ordinal_levels={},
            title=f"triangle/{config['f']}, {config['shift']} model — "
            "L1/L2 (Fig. 16/17)",
        )
    )
    figures.append("distributions.png")

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


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    run(cli(__file__, __doc__))
