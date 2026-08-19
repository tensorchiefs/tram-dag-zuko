"""Replicate the paper's CAREFL counterfactual benchmark (Sec. 5.3, App. C.2).

Usage (from experiments/)::

    uv run python paper_carefl.py

Fits an all-``ci`` flow to the 4-variable Laplace SCM and reproduces Fig. 6:
the counterfactual curves for the paper's observation
x_obs = (2.00, 1.50, 0.81, -0.28) — (i) x3 had x2 been alpha, (ii) x4 had x1
been alpha — against the analytic DGP truth. Also reports the mean absolute
counterfactual error over 300 held-out rows (the robust pytest metric).
Writes to ``results/paper-carefl/``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paper_common import finish, fit_chunked, results_dir, save_json

from tramdag import ContinuousNode, I
from tramdag.simulations import Carefl4
from tramdag.simulations.carefl import ALPHA_GRID, X_OBS

N = 20_000
gen = Carefl4(seed=42)
df = gen.observational(N)
train, val = df.iloc[: int(N * 0.9)], df.iloc[int(N * 0.9) :]
out = results_dir("paper-carefl")

spec = {
    "x1": ContinuousNode(),
    "x2": ContinuousNode(),
    "x3": ContinuousNode([I("x1", "x2")]),
    "x4": ContinuousNode([I("x1", "x2")]),
}

print(f"fitting all-ci flow on the CAREFL SCM, n={N} ...")
flow, _ = fit_chunked(spec, train, val, epochs=300, lr=1e-2, chunk=50)
flow.fit(train, val, epochs=100, learning_rate=1e-3, batch_size=512, verbose=0)
flow.save(out / "flow.pt")

# --- Fig. 6: counterfactual curves at the paper's x_obs
u = flow.abduct(pd.DataFrame([X_OBS]))
truth = gen.true_cf_curves()
flow_x3 = [float(flow.sample(do={"x2": a}, u=u)["x3"].iloc[0]) for a in ALPHA_GRID]
flow_x4 = [float(flow.sample(do={"x1": a}, u=u)["x4"].iloc[0]) for a in ALPHA_GRID]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, fl, tr, xlab, ylab in [
    (
        axes[0],
        flow_x3,
        truth["x3_cf_do_x2"],
        "would $x_2$ have been $\\alpha$",
        "$x_3$",
    ),
    (
        axes[1],
        flow_x4,
        truth["x4_cf_do_x1"],
        "would $x_1$ have been $\\alpha$",
        "$x_4$",
    ),
]:
    ax.plot(ALPHA_GRID, tr, "-", color="C3", lw=2, label="DGP (analytic)")
    ax.plot(ALPHA_GRID, fl, "o", ms=3, color="C0", label="flow")
    ax.set_xlabel(xlab), ax.set_ylabel(ylab), ax.legend()
fig.suptitle(
    "CAREFL counterfactual queries at $x_{obs}$ = (2.00, 1.50, 0.81, -0.28) (Fig. 6)"
)
finish(fig, out / "plots" / "cf_curves.png")

# --- robust metric: counterfactual MAE over held-out rows
rows = gen.observational(300, seed_offset=999)
u_rows = flow.abduct(rows)
eps = gen.abduct_noise(rows)
mae = {}
for a in (-1.5, 0.0, 1.5):
    d3 = (
        flow.sample(do={"x2": a}, u=u_rows)["x3"].to_numpy()
        - gen.simulate(do={"x2": a}, latents=eps)["x3"].to_numpy()
    )
    d4 = (
        flow.sample(do={"x1": a}, u=u_rows)["x4"].to_numpy()
        - gen.simulate(do={"x1": a}, latents=eps)["x4"].to_numpy()
    )
    mae[str(a)] = {"x3_cf": float(np.abs(d3).mean()), "x4_cf": float(np.abs(d4).mean())}
    print(
        f"alpha={a:+.1f}: MAE x3_cf {mae[str(a)]['x3_cf']:.3f}   "
        f"x4_cf {mae[str(a)]['x4_cf']:.3f}   (300 held-out rows)"
    )

save_json(
    out / "summary.json",
    {
        "n": N,
        "x_obs": X_OBS,
        "cf_mae_heldout": mae,
        "fig6_max_abs_err_x3": float(
            np.max(np.abs(np.array(flow_x3) - truth["x3_cf_do_x2"]))
        ),
        "fig6_max_abs_err_x4": float(
            np.max(np.abs(np.array(flow_x4) - truth["x4_cf_do_x1"]))
        ),
        "val_nll": flow.nll(val),
    },
)
print(f"-> {out}")
