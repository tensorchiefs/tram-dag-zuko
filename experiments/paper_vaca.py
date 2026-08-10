"""Replicate the paper's VACA/CNF benchmark (Sec. 5.1-5.2, App. C.1).

Usage (from experiments/)::

    uv run python paper_vaca.py

An all-``ci`` flow is fitted to the bimodal Gaussian triangle. Outputs the
pairs plot (Fig. 4 — the headline: the flow captures the bimodal x1 that the
default CNF misses) and the interventional densities p(x3 | do(x2=a)) for
a in {-3, -1, 0} (Fig. 5), to ``results/paper-vaca/``.
"""

import matplotlib.pyplot as plt
import numpy as np
from paper_common import fit_chunked, results_dir, save_json

from tramdag import ContinuousNode, I
from tramdag.simulations import VacaTriangle
from tramdag.simulations.vaca import DO_X2_VALUES

N = 20_000
gen = VacaTriangle(seed=42)
df = gen.observational(N)
train, val = df.iloc[: int(N * 0.9)], df.iloc[int(N * 0.9):]
out = results_dir("paper-vaca")

spec = {"x1": ContinuousNode(),
        "x2": ContinuousNode(terms=[I("x1")]),
        "x3": ContinuousNode(terms=[I("x1", "x2")])}

print(f"fitting all-ci flow on the VACA triangle, n={N} ...")
flow, _ = fit_chunked(spec, train, val, epochs=400, lr=1e-2, chunk=50)
flow.fit(train, val, epochs=120, learning_rate=1e-3, batch_size=512, verbose=0)
flow.save(out / "flow.pt")

# --- Fig. 4: pairs plot (diag marginals, off-diag scatters) DGP vs flow
cols = ["x1", "x2", "x3"]
samp = flow.sample(len(df), seed=0)
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ci in enumerate(cols):
    for j, cj in enumerate(cols):
        ax = axes[i][j]
        if i == j:
            bins = np.linspace(df[ci].quantile(0.001), df[ci].quantile(0.999), 60)
            ax.hist(df[ci], bins=bins, density=True, alpha=0.45, label="DGP")
            ax.hist(samp[ci], bins=bins, density=True, histtype="step", lw=1.8,
                    color="C3", label="flow")
        else:
            ax.scatter(df[cj][:2000], df[ci][:2000], s=2, alpha=0.3, label="DGP")
            ax.scatter(samp[cj][:2000], samp[ci][:2000], s=2, alpha=0.3,
                       color="C3", label="flow")
        if i == 2:
            ax.set_xlabel(cj)
        if j == 0:
            ax.set_ylabel(ci)
axes[0][0].legend(fontsize=8)
fig.suptitle("VACA triangle — observational joint, DGP vs flow (Fig. 4)")
fig.tight_layout(), fig.savefig(out / "plots" / "pairs.png", dpi=150)
plt.close(fig)

# --- Fig. 5: interventional densities p(x3 | do(x2 = a))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
moments = {}
for ax, a in zip(axes, DO_X2_VALUES):
    truth = gen.interventional(50_000, {"x2": a})
    fl = flow.sample(50_000, do={"x2": a}, seed=1)
    bins = np.linspace(truth["x3"].quantile(0.001), truth["x3"].quantile(0.999), 60)
    ax.hist(truth["x3"], bins=bins, density=True, alpha=0.45, label="DGP")
    ax.hist(fl["x3"], bins=bins, density=True, histtype="step", lw=1.8,
            color="C3", label="flow")
    ax.set_title(f"do($x_2$ = {a:+.0f})"), ax.set_xlabel("$x_3$")
    moments[str(a)] = {"mean_dgp": float(truth["x3"].mean()),
                       "mean_flow": float(fl["x3"].mean()),
                       "std_dgp": float(truth["x3"].std()),
                       "std_flow": float(fl["x3"].std())}
    print(f"do(x2={a:+.0f}): E[x3] DGP {moments[str(a)]['mean_dgp']:+.3f}  "
          f"flow {moments[str(a)]['mean_flow']:+.3f}")
axes[0].legend(), axes[0].set_ylabel("$p(x_3\\,|\\,do(x_2))$")
fig.suptitle("VACA triangle — interventional distributions (Fig. 5)")
fig.tight_layout(), fig.savefig(out / "plots" / "interventional.png", dpi=150)
plt.close(fig)

save_json(out / "summary.json",
          {"n": N, "do_x2": moments, "val_nll": flow.nll(val)})
print(f"-> {out}")
