"""Replicate the paper's mixed-data triangle experiments (Sec. 6.2, App. C.4).

Usage (from experiments/)::

    uv run python paper_triangle_mixed.py [linear|exp] [ls|cs]

Defaults: ``linear ls`` (Fig. 9/19 + the C.4 odds-ratio check). x3 is ordinal
(4 levels). Flow convention subtracts the ordinal shift while the paper adds it,
so the fitted weights converge to -0.2 (x1) and +0.3 (x2, linear DGP); the C.4
check is convention-free: e^beta12_hat predicts how odds(x2 <= -1) change when
x1 is increased by one unit in the DGP (paper: OR ~ 7.4).
Writes to ``results/paper-triangle-mixed-<f>-<model>/``.
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

from tramdag import CS, LS, ContinuousNode, OrdinalNode
from tramdag.simulations import TriangleMixed

f_name = sys.argv[1] if len(sys.argv) > 1 else "linear"
model = sys.argv[2] if len(sys.argv) > 2 else "ls"
assert model in ("ls", "cs")
SHIFT = {"ls": LS, "cs": CS}  # the x2 -> x3 term, chosen on the CLI

gen = TriangleMixed(f=f_name, seed=42)
df = gen.observational(PAPER_N)
train, val = df.iloc[: int(PAPER_N * 0.9)], df.iloc[int(PAPER_N * 0.9) :]
out = results_dir(f"paper-triangle-mixed-{f_name}-{model}")

spec = {
    "x1": ContinuousNode(),
    "x2": ContinuousNode([LS("x1")]),
    "x3": OrdinalNode(4, [LS("x1"), SHIFT[model]("x2")]),
}

truths = {"beta12": 2.0, "beta13_zuko": -0.2}  # ordinal sign flip (see docstring)
if model == "ls" and f_name == "linear":
    truths["beta23_zuko"] = 0.3


def record(flow):
    rec = {
        "beta12": ls_weight(flow, "x2", "x1"),
        "beta13_zuko": ls_weight(flow, "x3", "x1"),
    }
    if model == "ls":
        rec["beta23_zuko"] = ls_weight(flow, "x3", "x2")
    return rec


print(f"fitting triangle-mixed/{f_name} with {model} model on n={PAPER_N} ...")
flow, traj = fit_chunked(spec, train, val, record=record)
flow.save(out / "flow.pt")

plot_trajectories(
    traj,
    truths,
    out / "plots" / "coefficients.png",
    f"triangle-mixed/{f_name}, {model} model — flow-convention coefficients (Fig. 19)",
)

if model == "cs":  # fitted CS == -f(x2) + const, anchored at x2 = 0
    grid = np.linspace(-1.0, 1.0, 81)
    fitted, true = cs_curve(flow, "x3", "x2", grid), gen.true_shift_curve(grid)
    fitted = fitted - fitted[len(grid) // 2] + true[len(grid) // 2]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(grid, true, "k-", lw=2, label="DGP  $-f(x_2)$")
    ax.plot(grid, fitted, "o", ms=3, color="C0", label="fitted CS")
    ax.set_xlabel("$x_2$"), ax.legend()
    ax.set_title(f"complex shift on ordinal node, DGP f = {f_name}")
    finish(fig, out / "plots" / "cs_curve.png")

# L1 + L2 (Fig. 9 / 20): obs and do(x1 = -1), x3 as level frequencies
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
    f"triangle-mixed/{f_name}, {model} model — L1/L2 (Fig. 9/20)",
    ordinal={"x3": 4},
)

# --- C.4: predicted interventional odds ratio for odds(x2 <= -1) under x1 += 1
beta12_hat = ls_weight(flow, "x2", "x1")
rng = np.random.default_rng(99)
lat = gen.draw_latents(40_000, rng)
obs = gen.simulate(latents=lat)
shifted = gen.simulate(latents=lat, do={"x1": obs["x1"].to_numpy() + 1.0})
c = -1.0


def odds(x):
    p = float((x <= c).mean())
    return p / (1.0 - p)


or_dgp = odds(shifted["x2"]) / odds(obs["x2"])
or_pred = float(np.exp(beta12_hat))
print(f"C.4 odds-ratio check, odds(x2 <= {c}) under do(x1 += 1):")
print(
    f"  predicted e^beta12_hat = {or_pred:.2f}   DGP = {or_dgp:.2f}   "
    f"theory e^2 = {np.exp(2.0):.2f}"
)

summary = {
    "f": f_name,
    "model": model,
    "n": PAPER_N,
    "coefficients": traj[-1],
    "truth": gen.paper_truth(),
    "zuko_expectations": gen.zuko_expectations(),
    "c4_or_predicted": or_pred,
    "c4_or_dgp": or_dgp,
    "val_nll": flow.nll(val),
}
save_json(out / "summary.json", summary)
print("final coefficients:", {k: round(v, 4) for k, v in traj[-1].items()})
print(f"-> {out}")
