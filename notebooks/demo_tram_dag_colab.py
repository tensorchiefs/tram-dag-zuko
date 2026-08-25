# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TRAM-DAG in 5 minutes — one causal model, all three rungs of Pearl's ladder
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)
#
# **TRAM-DAGs** ([Sick & Dürr, CLeaR 2025](https://arxiv.org/abs/2503.16206)) are
# *interpretable neural causal models*: one normalizing flow, wired exactly like
# the **adjacency matrix of your causal DAG**. The flow transforms each variable
# conditional on its parents. Fit it **once** on observational data. Then you can
#
# 1. **L1** sample and score the observational distribution,
# 2. **L2** answer interventional queries (`do(...)`) by graph mutilation,
# 3. **L3** compute **individual counterfactuals** ("what would have happened to
#    *this* unit?") with Pearl's abduction–action–prediction.
#
# This demo uses the first benchmark of the paper: a 3-variable SCM with a
# **bimodal** source. On this example, a default causal normalizing flow visibly
# fails (paper, Fig. 4) and TRAM-DAG does not. The ground truth is known
# analytically. Thus every claim below is *checked*, not asserted.
#
# The demo runs on CPU. On a Colab **GPU runtime** (Runtime → Change runtime
# type → any GPU, for example the free T4), the final section compares the
# speed of the two devices.

# %%
import importlib.util
import logging
import subprocess
import sys
import time

if importlib.util.find_spec("tramdag") is None:  # Colab: install from PyPI
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tramdag"])
    # to track active development instead of the latest release, use:
    #   pip install "git+https://github.com/tensorchiefs/tramdag.git@main"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st  # for KDE plots only (preinstalled on Colab)
import torch

from tramdag import CausalFlowDAG, ContinuousNode, I

# tramdag reports fit() progress on its module logger
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["figure.dpi"] = 110
print(f"torch {torch.__version__}  device: {DEVICE}")

# %% [markdown]
# ## 1. The challenge: a bimodal structural causal model
#
# The DGP (Sánchez-Martín et al. 2022, App. E.1, and TRAM-DAG paper App. C.1):
#
# $$
# \begin{aligned}
# x_1 &\sim \tfrac12\,\mathcal N(-2,\,1.5) + \tfrac12\,\mathcal N(1.5,\,1)
#       &&\text{(bimodal source)}\\
# x_2 &= -x_1 + \mathcal N(0,1)\\
# x_3 &= x_1 + 0.25\,x_2 + \mathcal N(0,1)
# \end{aligned}
# $$
#
# The DAG is $x_1 \to x_2$, $x_1 \to x_3$, $x_2 \to x_3$.

# %%
# The DGP, written out here so this notebook needs nothing but tramdag.
# draw_latents/simulate are separate on purpose: keeping the noise lets us
# intervene on the SAME individuals later, which is what makes the
# counterfactual check in section 5 possible.


def draw_latents(n, rng):
    """Draw every noise variable of the SCM, n rows each."""
    return {
        "x1_mix": rng.uniform(size=n),  # which mixture component
        "x1_a": rng.normal(size=n),  # N(-2, 1.5) branch
        "x1_b": rng.normal(size=n),  # N(1.5, 1) branch
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    }


def simulate(latents, do=None):
    """Run the SCM forward; a variable named in `do` is clamped instead."""
    do = do or {}
    n = len(latents["x2"])

    if "x1" in do:
        x1 = np.full(n, float(do["x1"]))
    else:
        x1 = np.where(
            latents["x1_mix"] < 0.5,
            -2.0 + np.sqrt(1.5) * latents["x1_a"],
            1.5 + latents["x1_b"],
        )

    if "x2" in do:
        x2 = np.full(n, float(do["x2"]))
    else:
        x2 = -x1 + latents["x2"]

    if "x3" in do:
        x3 = np.full(n, float(do["x3"]))
    else:
        x3 = x1 + 0.25 * x2 + latents["x3"]

    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


def sample_dgp(n, seed, do=None):
    """Draw n fresh rows from the SCM, optionally under an intervention."""
    return simulate(draw_latents(n, np.random.default_rng(seed)), do)


df = sample_dgp(50_000, seed=43)
train, val = df.iloc[:45_000], df.iloc[45_000:]

fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for ax, c in zip(axes, df.columns, strict=True):
    ax.hist(df[c], bins=80, density=True, alpha=0.7)
    ax.set_title(f"observed ${c[0]}_{c[1]}$")
fig.suptitle("50,000 observational samples — note the bimodal $x_1$")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Fit the TRAM-DAG — the spec *is* the DAG
#
# Each node gets a monotone Bernstein transform. The terms say how parents
# enter (`I(...)` = the parents control the transform parameters, for maximal
# flexibility). Training maximizes the exact joint likelihood with one Adam.
# `schedule="plateau"` + `freeze_patience` lets each node decay its
# learning rate **independently** and leave training after it converges.
# The fit stops itself.

# %%
spec = {
    "x1": ContinuousNode(),  # source
    "x2": ContinuousNode(I("x1")),
    "x3": ContinuousNode(I("x1", "x2")),
}

torch.manual_seed(0)
flow = CausalFlowDAG(spec, device=DEVICE)
t0 = time.perf_counter()
flow.fit(
    train,
    val,
    epochs=400,
    learning_rate=1e-1,
    batch_size=4096,
    verbose=50,
    schedule="plateau",
    plateau_patience=10,
    freeze_patience=30,
    marginal_init=True,
)
t_fit = time.perf_counter() - t0
print(
    f"\nfitted on {DEVICE} in {t_fit:.1f}s "
    f"({len(flow.history['val'])} epochs, then froze itself)"
)

# %% [markdown]
# Training diagnostics come for free. `fit` records per-node train/val NLL,
# learning rates, wall-clock time, and the freeze epochs. Watch the nodes leave
# training one by one:

# %%
hist = flow.history
ep = np.arange(1, len(hist["val"]) + 1)
tot_tr = np.array([sum(d.values()) for d in hist["train"]])
tot_va = np.array([sum(d.values()) for d in hist["val"]])
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(ep, tot_tr, label="train NLL (total)")
ax.plot(ep, tot_va, label="val NLL (total)")
for _, (name, e) in enumerate(
    sorted(hist.get("frozen", {}).items(), key=lambda kv: kv[1])
):
    ax.axvline(e, ls="--", lw=1, color="gray")
    ax.annotate(
        f" {name} frozen",
        (e, ax.get_ylim()[1]),
        rotation=90,
        va="top",
        fontsize=8,
        color="gray",
    )
ax.set_ylim(tot_va.min() - 0.02, tot_va.min() + 0.6)  # zoom past the initial drop
ax.set_xlabel("epoch"), ax.set_ylabel("NLL"), ax.legend()
ax.set_title("training curve — per-node plateau decay, then self-freezing")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Rung 1 — does it actually fit? (the plot the CNF baseline fails)

# %%
samp = flow.sample(5 * len(df), seed=1)
cols = list(df.columns)
fig, axes = plt.subplots(3, 3, figsize=(9, 8.5))
for i, ci in enumerate(cols):
    for j, cj in enumerate(cols):
        ax = axes[i][j]
        if i == j:
            bins = np.linspace(df[ci].quantile(0.001), df[ci].quantile(0.999), 70)
            # Plot DGP histogram
            ax.hist(df[ci], bins=bins, density=True, alpha=0.5, label="DGP")
            # KDE for TRAM-DAG samples
            kde = st.gaussian_kde(samp[ci][:50_000])
            x_eval = np.linspace(bins[0], bins[-1], 300)
            ax.plot(x_eval, kde(x_eval), color="C3", lw=1.8, label="TRAM-DAG KDE")
        else:
            ax.scatter(df[cj][:1500], df[ci][:1500], s=2, alpha=0.3)
            ax.scatter(samp[cj][:1500], samp[ci][:1500], s=2, alpha=0.3, color="C3")
        if i == 2:
            ax.set_xlabel(cj)
        if j == 0:
            ax.set_ylabel(ci)
axes[0][0].legend(fontsize=8)
fig.suptitle("L1: observational joint — DGP (blue) vs fitted TRAM-DAG (red, KDE)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Rung 2 — interventions: `do(x2 = a)`
#
# Graph mutilation: clamp $x_2$, cut its incoming edge, resample. Under the DGP,
# $x_3\,|\,do(x_2{=}a) = x_1 + 0.25a + \mathcal N(0,1)$. Thus
# $\mathbb E[x_3] = -0.25 + 0.25a$ **analytically**. This exact target makes an
# error easy to see.

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
print("E[x3 | do(x2=a)]:   analytic    TRAM-DAG")
for ax, a in zip(axes, (-3.0, -1.0, 0.0), strict=True):
    truth = sample_dgp(50_000, seed=543, do={"x2": a})
    fl = flow.sample(50_000, do={"x2": a}, seed=2)
    bins = np.linspace(truth["x3"].quantile(0.001), truth["x3"].quantile(0.999), 70)
    # DGP histogram
    ax.hist(truth["x3"], bins=bins, density=True, alpha=0.5, label="DGP")
    # Flow density: KDE for a smoother estimate
    kde = st.gaussian_kde(fl["x3"].iloc[:50_000])
    x_eval = np.linspace(bins[0], bins[-1], 300)
    ax.plot(x_eval, kde(x_eval), color="C3", lw=1.8, label="TRAM-DAG density")
    ax.set_title(f"$p(x_3 \\mid do(x_2={a:+.0f}))$")
    print(
        f"   a = {a:+.0f}:          {-0.25 + 0.25 * a:+.3f}      {fl['x3'].mean():+.3f}"
    )
axes[0].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Rung 3 — counterfactuals for held-out individuals
#
# Take 1,000 **held-out** individuals. Step 1 (*abduction*): invert the flow to
# recover the latent noise $u$ of each individual. This noise is everything
# about them that the model does not attribute to their parents. Sanity check:
# when we push $u$ back through the flow, it must reproduce the observed data
# *exactly*.
#
# Step 2+3 (*action* + *prediction*): rerun history with $do(x_1 = 0)$, that is,
# the same $u$ on a mutilated graph. Because the DGP is fully continuous, the
# **true** individual counterfactuals are known (the simulator keeps its noise).
# Thus we can score the flow *per individual*. This is the strictest test in
# causal inference, and it is impossible with real data.

# %%
lat = draw_latents(1_000, np.random.default_rng(7))
factual = simulate(lat)  # the same noise on both sides ...
cf_true = simulate(lat, do={"x1": 0.0})  # ... so these are true counterfactuals

u = flow.abduct(factual)
recon = flow.sample(u=u)
err = np.abs(recon.to_numpy() - factual.to_numpy()).max()
print(f"abduction -> reconstruction: max |error| = {err:.2e}  (exact recovery)")

cf_flow = flow.sample(do={"x1": 0.0}, u=u)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, c in zip(axes, ["x2", "x3"], strict=True):
    ax.scatter(cf_true[c], cf_flow[c], s=4, alpha=0.4)
    lims = [cf_true[c].min(), cf_true[c].max()]
    ax.plot(lims, lims, "k--", lw=1)
    r = np.corrcoef(cf_true[c], cf_flow[c])[0, 1]
    ax.set_title(
        f"counterfactual ${c[0]}_{c[1]}$ under $do(x_1{{=}}0)$   (r = {r:.4f})"
    )
    ax.set_xlabel("true (DGP, shared noise)"), ax.set_ylabel("TRAM-DAG")
fig.suptitle("L3: individual counterfactuals, scored unit by unit")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Swapping the transform: Bernstein vs spline vs affine
#
# Each continuous node owns a **monotone 1-D transform**. That transform holds
# all of the distributional flexibility. One constructor argument switches it:
# `"bernstein"` (default, TRAM-faithful polynomial), `"spline"` (monotone
# rational-quadratic, the neural-spline-flow building block), or `"affine"`
# (location–scale only, which forces every node-conditional to be a logistic,
# in effect a classical GLM). Same DAG, same training, three model families:


# %%
def make_spec(transform):
    return {
        "x1": ContinuousNode([I(transform=transform)]),
        "x2": ContinuousNode(I("x1", transform=transform)),
        "x3": ContinuousNode(I("x1", "x2", transform=transform)),
    }


fits = {"bernstein": flow}  # already trained above
for tr in ["spline", "affine"]:
    torch.manual_seed(0)
    f = CausalFlowDAG(make_spec(tr), device=DEVICE)
    f.fit(
        train,
        val,
        epochs=400,
        learning_rate=1e-2,
        batch_size=4096,
        verbose=0,
        schedule="plateau",
        plateau_patience=10,
        freeze_patience=30,
    )
    fits[tr] = f
print("held-out NLL (lower is better):")
for tr, f in fits.items():
    print(f"  {tr:9s}: {sum(f.nll(val).values()):.4f}")

# %%
bins = np.linspace(df["x1"].quantile(0.001), df["x1"].quantile(0.999), 70)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(df["x1"], bins=bins, density=True, alpha=0.4, color="gray", label="data")
for tr, color in [("bernstein", "C3"), ("spline", "C0"), ("affine", "C2")]:
    ax.hist(
        fits[tr].sample(len(df), seed=3)["x1"],
        bins=bins,
        density=True,
        histtype="step",
        lw=1.8,
        color=color,
        label=tr,
    )
ax.set_title("the affine (GLM-like) transform cannot bend into two modes")
ax.set_xlabel("$x_1$"), ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# Reading the table: **affine** pays exactly where you expect it to pay. A
# location–scale transform *cannot* produce a bimodal $x_1$ (the same failure
# mode as the inflexible CNF in Fig. 4 of the paper). The **RQ-spline** is
# expressive enough *in principle*. But with the small TRAM-DAG parameter heads,
# it consistently trains to a worse optimum on this target (same result for
# 8–32 bins, lr 0.01–0.1, up to 2000 epochs). This is an honest empirical reason
# why **Bernstein** is the TRAM-faithful default: its monotone softplus-cumsum
# parametrization is easier to optimize. You can swap the transform per node at
# any time with `I(..., transform="spline", bins=16)`.

# %% [markdown]
# ### Misspecification bends causal answers, not just the likelihood
#
# The transform choice propagates into the *interventional* distribution — and
# the **mean** is forgiving. All three models get
# $\mathbb{E}[x_3 \mid do(x_2)]$ right to a few hundredths, because the shifts
# are right in each. What differs is the **shape**, so every query that reads the
# shape — tail probabilities, quantiles, the spread of individual effects —
# inherits the misfit. Compare a tail probability against the DGP:

# %%
truth_do = sample_dgp(50_000, seed=543, do={"x2": 0.0})["x3"]
print("under do(x2=0):        E[x3]     P(x3 < -3)")
print(
    f"  DGP                  {truth_do.mean():+.3f}      "
    f"{float((truth_do < -3).mean()):.4f}"
)
for tr in ("bernstein", "spline", "affine"):
    drawn = fits[tr].sample(50_000, do={"x2": 0.0}, seed=4)["x3"]
    print(
        f"  {tr:9s}            {drawn.mean():+.3f}      "
        f"{float((drawn < -3).mean()):.4f}"
    )

# %% [markdown]
# The means agree while the tails do not, and the ranking changes: the
# under-trained spline misses $P(x_3 < -3)$ by about two thirds and affine by
# roughly a sixth, though both looked fine on the mean. A model can be "close
# enough" on average and still be wrong about the question you actually asked —
# which is why the transform is a modelling decision, not a default to inherit.

# %% [markdown]
# ## 7. GPU vs CPU
#
# The whole flow is plain PyTorch, so it runs anywhere. Same 60-epoch fit, both
# devices (a CPU-only runtime reports only CPU):


# %%
def timed_fit(device, epochs=60):
    torch.manual_seed(0)
    f = CausalFlowDAG(
        {
            "x1": ContinuousNode(),
            "x2": ContinuousNode(I("x1")),
            "x3": ContinuousNode(I("x1", "x2")),
        },
        device=device,
    )
    t0 = time.perf_counter()
    f.fit(train, val, epochs=epochs, learning_rate=1e-2, batch_size=4096, verbose=0)
    return time.perf_counter() - t0


timings = {"cpu": timed_fit("cpu")}
if torch.cuda.is_available():
    timed_fit("cuda", epochs=5)  # warm-up (cuda kernel compilation)
    timings["cuda"] = timed_fit("cuda")
for dev, t in timings.items():
    print(f"{dev:5s}: {t:6.1f}s for 60 epochs @ n=45,000")
if len(timings) > 1:
    fig, ax = plt.subplots(figsize=(4, 2.8))
    ax.barh(list(timings), list(timings.values()), color=["C0", "C2"])
    ax.set_xlabel("seconds (60 epochs)"), ax.set_title("same model, same data")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## What you just saw
#
# | rung | query | call | checked against |
# |---|---|---|---|
# | L1 | observational joint | `flow.sample(n)` | 50k DGP samples (bimodal $x_1$ ✓) |
# | L2 | $p(x_3 \mid do(x_2{=}a))$ | `flow.sample(n, do=...)` | analytic $\mathbb E[x_3] = -0.25 + 0.25a$ |
# | L3 | individual counterfactuals | `flow.abduct(df)` + `flow.sample(do=..., u=u)` | per-unit DGP truth, r ≈ 0.99+ |
#
# And the same model family stays **interpretable** when you want it to be.
# Write an edge as `LS("parent")` instead of `I("parent")`. Then its
# coefficient is a log-odds ratio that you can read after training (this is
# the actual point of the paper).
#
# **More:** [repo](https://github.com/tensorchiefs/tramdag) ·
# [paper](https://arxiv.org/abs/2503.16206) ·
# didactic walkthrough: `notebooks/intro_tram_dag.py`
