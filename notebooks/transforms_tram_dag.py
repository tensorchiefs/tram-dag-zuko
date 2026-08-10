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
# # Choosing the transformation: Bernstein, spline, affine
#
# Every **continuous node** in a TRAM-DAG owns a monotone 1-D transformation
# $h$ that maps the node's value to the latent (log-odds) scale,
# $z = h(x) + \text{shifts}$. This notebook demonstrates the three built-in
# choices and when each one is the right tool — on the **same hand-built SCM as
# [`intro_tram_dag.py`](intro_tram_dag.py)**, so every claim is checked against
# known truth.
#
# | `ContinuousNode(transform=...)` | what it is | flexibility | parameters |
# |---|---|---|---|
# | `"bernstein"` (default) | Bernstein polynomial, strictly increasing via softplus-cumsum coefficients — the TRAM-faithful choice | any smooth monotone $h$ (order ↑ ⇒ arbitrarily flexible) | `n_coeffs` (default 20) |
# | `"spline"` | monotone rational-quadratic spline (the neural-spline-flow building block) | piecewise-rational monotone $h$ | `3·bins − 1` (default bins=8 → 23) |
# | `"affine"` | location–scale only, $h(x) = a\,x + b$ | none — forces the node-conditional to be a **logistic distribution** (a GLM-like baseline) | 2 |
#
# Extra arguments pass through `transform_kwargs`, e.g.
# `ContinuousNode(transform="bernstein", transform_kwargs={"n_coeffs": 40})`.
# (Ordinal nodes have no transform choice — their "transform" is the ordered
# cutpoint vector.)

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tramdag import CS, LS, CausalFlowDAG, ContinuousNode, OrdinalNode

plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## 1. The DGP — and what each $h$ must look like
#
# The intro notebook's SCM (logistic latents $z_i$):
#
# $$
# \begin{aligned}
# z_1 &= 1.2\,x_1 - 0.4               && \Rightarrow h_1 \text{ is exactly affine}\\
# z_2 &= 2\,x_2 + 1 \;(+\,1.5\,x_1)    && \Rightarrow h_2 \text{ is exactly affine}\\
# z_3 &= \sinh(x_3) \;(+\,0.8\,x_1 + \tfrac12 x_2^2) && \Rightarrow h_3 = \sinh \text{ — strongly nonlinear}\\
# P(Y\le k) &= \sigma(\vartheta_k - x_3), \quad \vartheta = (-2, 0, 1.5)
# \end{aligned}
# $$
#
# So this DAG contains both regimes at once: two nodes where the 2-parameter
# affine transform is *the correct model*, and one node where it cannot work.
# That makes the transform choice observable in the per-node likelihoods.


# %%
def simulate(n, rng):
    z = {k: rng.logistic(size=n) for k in "1234"}
    x1 = (z["1"] + 0.4) / 1.2
    x2 = (z["2"] - 1.5 * x1 - 1.0) / 2.0
    x3 = np.arcsinh(z["3"] - 0.8 * x1 - 0.5 * x2**2)
    cut = np.array([-2.0, 0.0, 1.5])[None, :] - 1.0 * x3[:, None]
    y = (z["4"][:, None] > cut).sum(axis=1)
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "Y": y.astype(float)})


rng = np.random.default_rng(1)
df = simulate(6000, rng)
train, val = df.iloc[:5400], df.iloc[5400:]

# %% [markdown]
# ## 2. Three flows, identical except for the transform
#
# Same DAG, same edges, same training (the self-stopping plateau schedule) —
# only `transform=` differs. `make_spec` shows how the choice is declared
# per node:


# %%
def make_spec(transform, **kwargs):
    tk = dict(transform=transform, transform_kwargs=kwargs)
    return {
        "X1": ContinuousNode(**tk),
        "X2": ContinuousNode(**tk, terms=[LS("X1")]),
        "X3": ContinuousNode(**tk, terms=[LS("X1"), CS("X2")]),
        "Y": OrdinalNode(levels=4, terms=[LS("X3")]),
    }


def fit(spec, seed=7):
    torch.manual_seed(seed)
    flow = CausalFlowDAG(spec)
    flow.fit(
        train,
        val,
        epochs=1500,
        learning_rate=1e-2,
        batch_size=512,
        verbose=0,
        schedule="plateau",
        plateau_patience=15,
        freeze_patience=60,
    )
    return flow


flows = {name: fit(make_spec(name)) for name in ["bernstein", "spline", "affine"]}

nll = pd.DataFrame({name: f.nll(val) for name, f in flows.items()}).T
nll["total"] = nll.sum(axis=1)
print("held-out NLL per node (lower is better):")
print(nll.round(4))

# %% [markdown]
# Read the table **per node**:
#
# - On `X1`/`X2`, **affine ties Bernstein** — affine is the *true* model there,
#   and the flexible Bernstein collapses onto it (flexibility costs nothing
#   when it isn't needed).
# - On `X3` the affine transform pays for its rigidity: it cannot bend into
#   $\sinh$ — worse NLL on exactly that node, nowhere else.
# - The **spline lags slightly everywhere**, even on the trivially-affine `X1`.
#   That has a structural reason (verified below the next figure): outside the
#   fitted 5%/95% range, zuko's RQ-spline extrapolates with a *fixed unit
#   slope*, whatever it learned inside — so the ~10% of data in the tails is
#   modeled with the wrong density slope whenever the true slope differs.
#   Bernstein extrapolates with its *boundary derivative* and has no such
#   constraint.


# %%
def fitted_h(flow, name, grid):
    node = flow.nodes[name]
    x = torch.as_tensor(grid, dtype=torch.float32)
    with torch.no_grad():
        z0, _ = node.ut.forward(node.intercept(len(grid)), x)
    return z0.detach().numpy()


grid = np.linspace(*df["X3"].quantile([0.01, 0.99]).to_numpy(), 200)
true_h = np.sinh(grid)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(grid, true_h - true_h.mean(), "k-", lw=2.5, label=r"true $h_3 = \sinh$")
for name, color in [("bernstein", "C3"), ("spline", "C0"), ("affine", "C2")]:
    h = fitted_h(flows[name], "X3", grid)
    ax.plot(grid, h - h.mean(), "--", lw=1.8, color=color, label=name)
ax.set_xlabel("$x_3$"), ax.set_ylabel("$h_3(x_3)$  (centered)"), ax.legend()
ax.set_title("the transform each flow learned for $X_3$")
fig.tight_layout()
plt.show()

# %% [markdown]
# Bernstein recovers $\sinh$ (up to the additive constant shared with the
# complex shift); the spline matches it inside the bulk of the data but is
# pinned to a fixed slope in the tails (see the table discussion); the affine
# fit is the best straight line through a hyperbolic sine — exactly the misfit
# the NLL table charged it for.
#
# > The tail behavior is easy to verify directly: evaluate a transform outside
# > its fitted range — the spline's tail slope equals the pre-scaling slope for
# > *any* parameters, while Bernstein's follows its boundary derivative. The
# > same effect explains why the spline trails Bernstein on the
# > [Colab demo's](demo_tram_dag_colab.py) bimodal target. This — robustness,
# > not expressiveness — is why `bernstein` is the default.

# %% [markdown]
# ## 3. Misspecification is not just a worse likelihood — it bends causal answers
#
# The transform choice propagates to the interventional distribution
# $p(x_3 \mid do(X_1{=}1))$. Instructively, the interventional **mean** is
# forgiving — the affine model's shifts are still correct, so it gets
# $\mathbb E[x_3]$ right. What it cannot get right is the **shape**: the true
# conditional is skewed ($\operatorname{asinh}$ of a logistic), while affine
# forces a symmetric logistic. Any query that depends on the shape — tail
# risks like $P(x_3 < -3 \mid do)$, quantiles, ITE distributions — inherits
# the error:


# %%
def dgp_do_x1(n, value, rng):
    z = {k: rng.logistic(size=n) for k in "234"}
    x1 = np.full(n, value)
    x2 = (z["2"] - 1.5 * x1 - 1.0) / 2.0
    x3 = np.arcsinh(z["3"] - 0.8 * x1 - 0.5 * x2**2)
    return x3


truth = dgp_do_x1(50_000, 1.0, np.random.default_rng(5))
bins = np.linspace(np.quantile(truth, 0.001), np.quantile(truth, 0.999), 60)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hist(truth, bins=bins, density=True, alpha=0.4, color="gray", label="DGP truth")
print("under do(X1=1):      E[X3]    P(X3 < -3)")
print(f"  truth             {truth.mean():+.3f}    {float((truth < -3).mean()):.4f}")
for name, color in [("bernstein", "C3"), ("spline", "C0"), ("affine", "C2")]:
    s = flows[name].sample(50_000, do={"X1": 1.0}, seed=2)["X3"]
    ax.hist(
        s, bins=bins, density=True, histtype="step", lw=1.8, color=color, label=name
    )
    print(f"  {name:9s}        {s.mean():+.3f}    {float((s < -3).mean()):.4f}")
ax.set_xlabel("$x_3$"), ax.legend()
ax.set_title("$p(x_3 \\mid do(X_1{=}1))$ under the three transforms")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Mix and match — the right transform per node
#
# The transform is a **per-node** choice. Since $h_1, h_2$ are truly affine and
# only $h_3$ needs flexibility, the honest minimal model uses affine where
# affine is correct:

# %%
spec_mixed = {
    "X1": ContinuousNode(transform="affine"),
    "X2": ContinuousNode(transform="affine", terms=[LS("X1")]),
    "X3": ContinuousNode(transform="bernstein", terms=[LS("X1"), CS("X2")]),
    "Y": OrdinalNode(levels=4, terms=[LS("X3")]),
}
flows["mixed"] = fit(spec_mixed)


def n_params(flow):
    return sum(p.numel() for p in flow.parameters())


cmp = pd.DataFrame(
    {
        name: {"val NLL (total)": sum(f.nll(val).values()), "parameters": n_params(f)}
        for name, f in flows.items()
    }
).T
print(cmp.round(4))

# %% [markdown]
# The mixed model matches the all-Bernstein likelihood while spending the
# flexible parameters only where the data needs them. (With Bernstein's
# `n_coeffs=20` per node the saving here is modest — the principle matters more
# at scale, or when you *want* a node to be a classical logistic model.)
#
# As a bonus, the affine nodes are directly interpretable: $h_1(x) = a x + b$
# means $X_1 \sim \text{Logistic}(-b/a,\, 1/a)$ — read location and scale
# straight off the parameters:

# %%
# extract slope/intercept empirically from the fitted transform (robust to
# the internal parametrization): h is affine, so two points determine it
h0, h1 = fitted_h(flows["mixed"], "X1", np.array([0.0, 1.0]))
a, b = h1 - h0, h0
print(f"fitted  h1(x) = {a:.3f}·x + {b:+.3f}   (truth: 1.2·x − 0.4)")
print(
    f"=> X1 ~ Logistic(loc={-b / a:.3f}, scale={1 / a:.3f})   "
    f"(truth: loc=0.333, scale=0.833)"
)

# %% [markdown]
# ## 5. `transform_kwargs`: how much flexibility do you need?
#
# Bernstein order (`n_coeffs`) and spline resolution (`bins`) control capacity.
# Sweep them and watch the $X_3$ node's held-out NLL — too few coefficients
# can't bend into $\sinh$, while beyond ~10 the extra capacity is free but idle:

# %%
sweeps = [
    ("bernstein", "n_coeffs", [3, 5, 10, 20, 40]),
    ("spline", "bins", [2, 4, 8, 16]),
]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for (name, kw, values), color in zip(sweeps, ["C3", "C0"]):
    nlls = [fit(make_spec(name, **{kw: v})).nll(val)["X3"] for v in values]
    ax.plot(values, nlls, "o-", color=color, label=f"{name} ({kw})")
    print(name, dict(zip(values, np.round(nlls, 4))))
ax.axhline(flows["bernstein"].nll(val)["X3"], color="gray", ls=":", lw=1)
ax.set_xscale("log"), ax.set_xlabel("capacity parameter")
ax.set_ylabel("X3 held-out NLL"), ax.legend()
ax.set_title("capacity sweep on the $\\sinh$ node")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - The transform is chosen **per node**: `ContinuousNode(transform=...,
#   transform_kwargs=...)`.
# - `affine` = classical logistic location–scale model: 2 parameters, fully
#   interpretable, and *exactly right* when the true transformation is linear —
#   the flexible transforms then collapse onto it at no benefit.
# - `bernstein` (default) and `spline` recover genuinely nonlinear
#   transformations like $\sinh$; misspecifying this (affine on $X_3$) shows up
#   in the per-node NLL *and* distorts interventional answers.
# - Capacity (`n_coeffs` / `bins`) saturates quickly here (~10 is enough for
#   $\sinh$); the default 20 is a comfortable ceiling.
# - The spline's weak spot is structural, not capacity: its tails extrapolate
#   with a fixed slope regardless of the fitted parameters, so the ~10% of data
#   outside the 5%/95% scaling range is misweighted whenever the true tail
#   slope differs. Bernstein extrapolates with its boundary derivative — hence
#   the default.
