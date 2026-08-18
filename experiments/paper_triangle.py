"""Replicate the paper's continuous triangle experiments (Sec. 6.1, App. C.3).

Usage (from experiments/)::

    uv run python paper_triangle.py [linear|atan|sin] [ls|cs]

Defaults: ``atan cs`` (the complex-shift experiment, Fig. 7/15/16). Paper truth:
beta12 = 2, beta13 = -0.2 (+0.3 on x2 for the linear DGP); a fitted ``cs`` module
converges to -f(x2) up to an additive constant. Writes coefficient trajectories
(Fig. 14/15), the CS-curve overlay (Fig. 7 right / 17 left / 18 right), and
obs + do(x1=-1) distribution comparisons (Fig. 16/17) to
``results/paper-triangle-<f>-<model>/``.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from paper_common import (
    PAPER_N,
    cs_curve,
    finish,
    fit_chunked,
    ls_weight,
    plot_hist_grid,
    plot_trajectories,
    results_dir,
    save_json,
)

from tramdag import LS, ContinuousNode, term
from tramdag.simulations import TriangleContinuous

f_name = sys.argv[1] if len(sys.argv) > 1 else "atan"
model = sys.argv[2] if len(sys.argv) > 2 else "cs"
assert model in ("ls", "cs")

gen = TriangleContinuous(f=f_name, seed=42)
df = gen.observational(PAPER_N)
train, val = df.iloc[: int(PAPER_N * 0.9)], df.iloc[int(PAPER_N * 0.9) :]
out = results_dir(f"paper-triangle-{f_name}-{model}")

spec = {
    "x1": ContinuousNode(),
    "x2": ContinuousNode([LS("x1")]),
    "x3": ContinuousNode([term("LS", "x1"), term(model.upper(), "x2")]),
}

truths = {"beta12": 2.0, "beta13": -0.2}
if model == "ls":
    truths["beta23"] = 0.3 if f_name == "linear" else float("nan")


def record(flow):
    rec = {"beta12": ls_weight(flow, "x2", "x1"), "beta13": ls_weight(flow, "x3", "x1")}
    if model == "ls":
        rec["beta23"] = ls_weight(flow, "x3", "x2")
    return rec


print(
    f"fitting triangle/{f_name} with {model} model on n={PAPER_N} "
    "(paper: 500 epochs @ lr 1e-3) ..."
)
flow, traj = fit_chunked(spec, train, val, record=record)
flow.save(out / "flow.pt")

plot_trajectories(
    traj,
    {k: v for k, v in truths.items() if v == v},
    out / "plots" / "coefficients.png",
    f"triangle/{f_name}, {model} model — LS coefficients (Fig. 14/15)",
)

if model == "cs":  # CS-curve overlay (Fig. 7 right): both anchored at x2 = 0
    grid = np.linspace(-1.0, 1.0, 81)
    fitted, true = cs_curve(flow, "x3", "x2", grid), gen.true_shift_curve(grid)
    fitted = fitted - fitted[len(grid) // 2] + true[len(grid) // 2]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(grid, true, "k-", lw=2, label="DGP  $-f(x_2)$")
    ax.plot(grid, fitted, "o", ms=3, color="C0", label="fitted CS")
    ax.set_xlabel("$x_2$"), ax.set_ylabel("$-f(x_2)$"), ax.legend()
    ax.set_title(f"complex shift, DGP f = {f_name} (Fig. 7)")
    finish(fig, out / "plots" / "cs_curve.png")

# L1 + L2: observational and do(x1 = -1) distributions (Fig. 16/17)
n_show = 10_000
dgp = {
    "Obs": gen.observational(n_show, seed_offset=5),
    "do(x1=-1)": gen.interventional(n_show, {"x1": -1.0}),
}
fl = {
    "Obs": flow.sample(n_show, seed=0),
    "do(x1=-1)": flow.sample(n_show, do={"x1": -1.0}, seed=0),
}
plot_hist_grid(
    dgp,
    fl,
    ["x1", "x2", "x3"],
    out / "plots" / "distributions.png",
    f"triangle/{f_name}, {model} model — L1/L2 (Fig. 16)",
)

summary = {
    "f": f_name,
    "model": model,
    "n": PAPER_N,
    "coefficients": traj[-1],
    "truth": gen.paper_truth(),
    "val_nll": flow.nll(val),
}
save_json(out / "summary.json", summary)
print("final coefficients:", {k: round(v, 4) for k, v in traj[-1].items()})
print(f"-> {out}")
