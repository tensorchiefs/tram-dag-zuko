# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TRAM-DAG — a didactic introduction with `tramdag`
#
# *TRAM-DAGs* ([paper](https://arxiv.org/abs/2503.16206),
# [original R/Keras code](https://github.com/tensorchiefs/tram-dag)) are causal models.
# They use **structured transformation functions** to map a latent representation
# $U$ to the observed data $X$. For continuous variables they are **bijective causal
# models**. Thus one trained model answers all three rungs of Pearl's causal
# hierarchy:
#
# | rung | query | `tramdag` call |
# |---|---|---|
# | L1 association | $p(x)$, sampling | `flow.log_prob(df)`, `flow.sample(n)` |
# | L2 intervention | $p(x \mid do(x_j{=}a))$ | `flow.sample(n, do={...})`, `flow.pmf(df, node, do={...})` |
# | L3 counterfactual | "what would $x_i$ have been, had $x_j$ been $a$?" | `u=flow.abduct(df); flow.sample(do={...}, u=u)` |
#
# This notebook shows the model exactly in the paper notation. It builds a
# small data-generating process (DGP) **inside the model family**. It fits a
# `CausalFlowDAG` and verifies each claim against the known ground truth.
#
# (Notebook format and how to run it: see `notebooks/README.md`.)

# %% [markdown]
# ## 1. The model
#
# We assume a causal ordering of the variables. For each variable, a TRAM-DAG
# fits a monotone **transformation function** $h$ (bijective, monotone increasing).
# This function maps the observed value to a latent scale, conditional on the
# variable's parents:
#
# $$
# \begin{align*}
# u_1 &= h(x_1) \\
# u_2 &= h(x_2 \mid x_1)\\
# u_3 &= h(x_3 \mid x_1, x_2) \\
# \dots &\\
# u_p &= h(x_p \mid x_1, x_2, \dots, x_{p-1})
# \end{align*}
# $$
#
# This **observed → latent** map is the convention of the paper (Eq. 2,
# $F_{X\mid\mathrm{pa}}(x)=F_U\!\big(h(x\mid\mathrm{pa})\big)$) and of the code
# (in code: `u = h(x) + shift`). It is also the **training direction**: the model
# evaluates $h$ *directly* to score the likelihood, which is cheap. **Sampling**
# runs the inverse $x_i = h^{-1}(u_i \mid \mathrm{pa})$. This inverse has no
# closed form, so bracketed bisection solves it. That is the costlier direction.
#
# Together the $h$'s form one **triangular** flow. Each variable can depend only
# on a *subset* of its predecessors, its causal parents $\mathrm{pa}(x_i)$. Thus
# the Jacobian sparsity of the flow *is* the DAG.
#
# For the latents $u_1,\dots,u_p$ we assume a **standard logistic** distribution.
# This choice makes the fitted parameters interpretable: shifts on the
# latent scale are **log-odds ratios** (Section 6).
#
# ### The four components
#
# To keep a valid interpretation, we decompose the transformation **additively on
# the latent scale**. Each node's $h$ (observed → latent, as above) is
#
# $$
# u_i \;=\; h(x_i \mid \mathrm{pa}(x_i)) \;=\;
# \underbrace{h_{\boldsymbol{\vartheta}}(x_i)}_{\text{intercept}}
# \;+\; \underbrace{\textstyle\sum_j \beta_{ij}\, x_j}_{\text{linear shifts (LS)}}
# \;+\; \underbrace{\textstyle\sum_k g_{ik}(x_k)}_{\text{complex shifts (CS)}} ,
# $$
#
# with every causal parent in exactly one term. We use $x_5$ with parents
# $\mathrm{pa}(x_5) = \{x_1, x_2, x_4\}$ as the running example:
#
# * **Simple intercept (SI):** $h_{\boldsymbol{\vartheta}}(x_5)$ has *constant* parameters $\boldsymbol{\vartheta}$.
#   It is a flexible monotone baseline transformation (here: a Bernstein
#   polynomial), the same for every observation.
# * **Complex intercept (CI):** the parameters $\boldsymbol{\vartheta}$ of $h_{\boldsymbol{\vartheta}}(x_5)$ are
#   themselves a function of (a subset of) the parents. The whole transformation
#   bends with the parent, which permits interactions beyond additive shifts.
# * **Linear shift (LS):** $\beta_{51} x_1 + \beta_{52} x_2$. This gives one
#   interpretable number per parent.
# * **Complex shift (CS):** $g(x_4)$, an unrestricted (NN) function of the
#   parent. It stays *additive* on the latent scale.
#
# so that **sampling** (the inverse direction) must invert only the intercept.
# The shifts move to the other side:
#
# $$
# x_5 = h^{-1}(u_5 \mid x_1, x_2, x_4) = h_{\boldsymbol{\vartheta}}^{-1}\!\Big(u_5
# - \underbrace{\beta_{51} x_1 + \beta_{52} x_2}_{\text{LS}}
# - \underbrace{g(x_4)}_{\text{CS}}\Big).
# $$
#
# In `tramdag` each node declares its transformation as an **additive formula of
# terms**. The formula is the node's first argument, written as a list `[...]`
# or as a `+` sum. The constructors `I` (intercept), `LS` (linear shift), and
# `CS` (complex shift) build it. Each constructor names the parent(s) that its
# term depends on:
#
# | paper component | `tramdag` |
# |---|---|
# | SI — baseline $h_{\boldsymbol{\vartheta}}(x_i)$, constant $\boldsymbol{\vartheta}$ | automatic: every node owns a monotone transform (`bernstein` / `spline` / `affine`); with no intercept term its $\boldsymbol{\vartheta}$ is a free parameter vector |
# | CI — $\boldsymbol{\vartheta}$ depends on parents | `I("X1")` (several `I(...)` parents feed **one joint** network → interactions) |
# | LS — $\beta_{ij} x_j$ | `LS("X1")` (a single weight, no bias) |
# | CS — $g_{ik}(x_k)$ | `CS("X1")` (64-128-64 NN, additive) |
#
# `I(...)` dispatches on its arguments: no parents → a simple intercept (`SI`),
# parents → a complex intercept (`CI`). The explicit names work too.

# %% [markdown]
# ### Gallery: a transformation *is* an additive decomposition
#
# Read every spec line as a recipe for $h(x_i \mid \mathrm{pa})$. Each parent lands
# in **exactly one** term: the intercept (the *shape*) or one shift. The
# table shows the resulting decomposition for a single continuous target $X_3$:
#
# | formula (`terms`) | $u_3 = h(x_3 \mid \mathrm{pa})$ | what carries each parent |
# |---|---|---|
# | `None` / `[SI()]` (source) | $h_{\boldsymbol{\vartheta}}(x_3)$ | `SimpleIntercept` — $\boldsymbol{\vartheta}$ a free vector |
# | `[CI("X1")]` | $h_{\boldsymbol{\vartheta}(x_1)}(x_3)$ | `ComplexIntercept` — **no shift term**; $\boldsymbol{\vartheta}$ (the whole shape) bends with $x_1$ |
# | `[LS("X1")]` | $h_{\boldsymbol{\vartheta}}(x_3) + \beta\,x_1$ | `LinearShift` — **one number** $\beta$ |
# | `[CS("X1")]` | $h_{\boldsymbol{\vartheta}}(x_3) + g_1(x_1)$ | `ComplexShift` — additive NN $g_1$ |
# | `LS("X1") + CS("X2")` | $h_{\boldsymbol{\vartheta}}(x_3) + \beta x_1 + g_2(x_2)$ | one `LinearShift` + one `ComplexShift` (the model fitted below) |
# | `[CS("X1", "X2")]` | $h_{\boldsymbol{\vartheta}}(x_3) + g_{1,2}(x_1, x_2)$ | **one joint** `ComplexShift` — an interaction in the shift |
# | `CS("X1") + CS("X2")` | $h_{\boldsymbol{\vartheta}}(x_3) + g_1(x_1) + g_2(x_2)$ | two **additive** `ComplexShift`s |
# | `[I("X1", "X2", allow_interaction=False)]` | $h_{\boldsymbol{\vartheta}(x_1) + \boldsymbol{\vartheta}(x_2)}(x_3)$ | **additive** CI — one network per parent, coefficient vectors summed; each parent reshapes the transform independently |
# | `[I("X1", "X2")]` | $h_{\boldsymbol{\vartheta}(x_1,x_2)}(x_3)$ | **one joint** `ComplexIntercept` over both parents (they interact) |
#
# A list and a `+` sum are interchangeable: `[LS("X1"), CS("X2")]` ==
# `LS("X1") + CS("X2")`. You select the class of the monotone transform on the
# intercept term: `I("X1", transform="spline")`.
#
# For an **ordinal** target the intercept is not a Bernstein curve but the vector
# of ordered cutpoints $\vartheta_k(\mathrm{pa})$. The model **subtracts** the
# shift: $P(Y \le k \mid \mathrm{pa}) = \sigma\big(\vartheta_k - \text{shift}\big)$.
# `LS` and `CS` terms enter that shift exactly as above.
#
# The odd one out is **`I(...)`** (a complex intercept). It is the only term that
# is *not* an additive shift. It moves the parent into the intercept, so there is
# no separate summand and no single-number coefficient to read off (§6). That is
# the interpretability price when the transformation's *shape* depends on the
# parent.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tramdag import CS, LS, CausalFlowDAG, ContinuousNode, OrdinalNode

plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## 2. A hand-built DGP — *inside* the model family
#
# To verify everything against ground truth, we now build a small structural
# causal model **by hand**. It has logistic latents and has exactly the form
# above. This mirrors the construction in
# [`triangle_structured_continous.R`](https://github.com/tensorchiefs/tram-dag/blob/main/summerof24/triangle_structured_continous.R).
# The DAG is $X_1 \to X_2 \to X_3 \leftarrow X_1$ plus an ordinal outcome
# $X_3 \to Y$:
#
# $$
# \begin{aligned}
# u_1 &= h_1(x_1) = 1.2\,x_1 - 0.4
#   &&\Rightarrow\; x_1 = (u_1 + 0.4)/1.2 \\[2pt]
# u_2 &= h_2(x_2) + \beta_{21} x_1, \quad h_2(x) = 2x + 1,\; \beta_{21} = 1.5
#   &&\Rightarrow\; x_2 = (u_2 - 1.5\,x_1 - 1)/2 \\[2pt]
# u_3 &= h_3(x_3) + \beta_{31} x_1 + g(x_2), \quad h_3(x) = \sinh(x),\;
#       \beta_{31} = 0.8,\; g(x) = \tfrac12 x^2
#   &&\Rightarrow\; x_3 = \operatorname{asinh}\!\big(u_3 - 0.8\,x_1 - \tfrac12 x_2^2\big) \\[2pt]
# P(Y \le k) &= \sigma(\vartheta_k - \beta_{Y} x_3), \quad
#       \vartheta = (-2, 0, 1.5),\; \beta_{Y} = 1
#   &&\Rightarrow\; y = \#\{k : u_4 > \vartheta_k - \beta_Y x_3\}
# \end{aligned}
# $$
#
# These conventions are the TRAM conventions, and tests in this repo pin them:
#
# * continuous nodes: the model **adds** the shift on the latent scale,
#   $u = h(x) + \text{shift}$.
# * ordinal nodes: the model **subtracts** the shift inside the sigmoid,
#   $P(Y \le k) = \sigma(\vartheta_k - \text{shift})$, with increasing cutpoints
#   $\vartheta_k$ (an *ordered logit*). Because of this flip, a positive $\beta$
#   pushes $Y$ toward *higher* categories.
#
# $X_2$ enters $X_3$ through a quadratic **complex shift** $g(x_2)=\tfrac12 x_2^2$.
# A linear shift cannot represent this function. We will return to this later.

# %% [markdown]
# The figure shows the wiring at a glance. The label on each edge gives the term
# through which its parent enters:

# %%
fig, ax = plt.subplots(figsize=(7, 3.2))
pos = {"X1": (0, 1.0), "X2": (0, 0.0), "X3": (1.4, 0.5), "Y": (2.6, 0.5)}
edges = [
    ("X1", "X2", r"LS: $\beta_{21}=1.5$", (-0.32, 0.5)),
    ("X1", "X3", r"LS: $\beta_{31}=0.8$", (0.62, 0.93)),
    ("X2", "X3", r"CS: $g(x)=\frac{1}{2}x^2$", (0.62, 0.07)),
    ("X3", "Y", r"LS: $\beta_{Y}=1$", (2.0, 0.62)),
]
for src, dst, label, (lx, ly) in edges:
    ax.annotate(
        "",
        xy=pos[dst],
        xytext=pos[src],
        arrowprops=dict(
            arrowstyle="-|>", color="#333333", lw=1.4, shrinkA=16, shrinkB=16
        ),
    )
    ax.text(lx, ly, label, fontsize=10, ha="center", va="center")
for name, (x, y) in pos.items():
    ordinal = name == "Y"
    ax.scatter(
        x,
        y,
        s=900,
        facecolor="white" if not ordinal else "#dce8f4",
        edgecolor="#333333",
        zorder=3,
    )
    ax.text(
        x,
        y,
        f"${name[0]}_{name[1]}$" if len(name) > 1 else f"${name}$",
        ha="center",
        va="center",
        fontsize=13,
        zorder=4,
    )
ax.text(2.6, 0.24, "ordinal (4 levels)", fontsize=9, ha="center", color="#555555")
ax.set_xlim(-0.7, 3.1)
ax.set_ylim(-0.35, 1.35)
ax.axis("off")
ax.set_title("The DGP's DAG", fontsize=11)
plt.show()

# %%
TRUE = dict(b21=1.5, b31=0.8, bY=1.0, theta_Y=np.array([-2.0, 0.0, 1.5]))


def rlogis(rng, n):
    """Standard logistic draws."""
    u = rng.uniform(1e-9, 1 - 1e-9, size=n)
    return np.log(u) - np.log1p(-u)


def g_cs(x2):
    """The true complex shift of X2 on X3."""
    return 0.5 * x2**2


def simulate(n, rng, x1=None, u=None):
    """Sample from the SCM. `x1` overrides the source node (= do(X1)),
    `u` reuses given latents (= counterfactuals).
    """
    if u is None:
        u = {k: rlogis(rng, n) for k in ["u1", "u2", "u3", "u4"]}
    if x1 is None:
        x1 = (u["u1"] + 0.4) / 1.2
    else:
        x1 = np.full(n, float(x1))
    x2 = (u["u2"] - TRUE["b21"] * x1 - 1.0) / 2.0
    x3 = np.arcsinh(u["u3"] - TRUE["b31"] * x1 - g_cs(x2))
    cut = TRUE["theta_Y"][None, :] - TRUE["bY"] * x3[:, None]
    y = (u["u4"][:, None] > cut).sum(axis=1)
    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "Y": y.astype(float)})
    return df, u


rng = np.random.default_rng(1)
df, u_obs = simulate(6000, rng)  # keep the latents -> true counterfactuals
train_df, val_df = df.iloc[:5000], df.iloc[5000:]
df.describe().round(2)

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, (a, b) in zip(axes, [("X1", "X2"), ("X1", "X3"), ("X2", "X3")], strict=True):
    ax.scatter(df[a], df[b], s=3, alpha=0.25)
    ax.set_xlabel(a), ax.set_ylabel(b)
axes[2].set_title("the U-shape of the complex shift", fontsize=9)
fig.suptitle("Observational data from the hand-built SCM")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Specifying the DAG and fitting the flow
#
# The model spec *is* the labeled adjacency matrix, written per node. Each
# continuous node automatically gets its monotone baseline transformation
# (default: Bernstein polynomial with 20 coefficients, the TRAM-faithful choice).
# The edges declare how each parent enters.
#
#
# Fitting maximizes the joint likelihood. Because the flow is triangular, the
# negative log-likelihood **decomposes per node**
# ($\log p(x) = \sum_i \log p(x_i \mid \mathrm{pa}(x_i))$). Thus one Adam
# optimizer trains all nodes at once. `fit` keeps the final weights — for
# this unconfounded DGP the MLE is the right target. Progress printing is
# `fit`'s own `verbose=` (Keras-style; 0 = silent, N = every Nth epoch).

# %%
spec = {
    "X1": ContinuousNode(),
    "X2": ContinuousNode(LS("X1")),
    "X3": ContinuousNode(LS("X1") + CS("X2")),
    "Y": OrdinalNode(4, LS("X3")),
}

flow = CausalFlowDAG(
    spec, seed=1
)  # seed here too, for the Bernsteins' initial uniform knots
flow.fit(train_df, epochs=800, learning_rate=1e-2, batch_size=20000, verbose=200)
flow.fit(train_df, epochs=300, learning_rate=1e-3, batch_size=512)  # polish
flow.nll(val_df)

# %% [markdown]
# ## 4. Anatomy: the spec *is* the additive decomposition
#
# Before the per-node detail, the whole model at a glance. `flow.to_matrix()`
# labels every edge with the term that carries it — this is the paper's
# **meta-adjacency matrix** (Fig. 3), and it is generated from the fitted
# object rather than drawn by hand, so it cannot disagree with the model:

# %%
print(flow.to_matrix())

# %% [markdown]
# Rows are parents, columns children; an empty cell means no edge. Now read the
# same structure node by node, directly from the **fitted** flow. Two small helpers do the job. `describe_node` reports which
# network carries each parent (the structural view). `decompose_row` prints the
# actual numbers for one observation and verifies that they rebuild the per-node
# log-likelihood **exactly**. The equation $u = h_{\boldsymbol{\vartheta}}(x) + \sum \text{shifts}$ is an
# identity, not a picture. We run both on `X2` (an `LS` edge), `X3` (`LS` + `CS`),
# and the ordinal `Y` (shift **subtracted**).

# %%
from tramdag.transforms import (  # noqa: E402
    StandardLogistic,
    ordinal_cutpoints,
    ordinal_log_prob,
)


def describe_node(flow, name):
    """Structural view: the intercept module and each parent's term + network."""
    node = flow.nodes[name]
    n_params = node.ut.n_params if node.ut is not None else node.levels - 1
    print(f"{name}  ({node.kind})")
    if node.intercept.ci_parents:
        print(
            f"  intercept: ComplexIntercept({node.intercept.ci_parents}"
            f" -> {n_params} params)"
        )
    else:
        print(f"  intercept: SimpleIntercept({n_params} params)")
    for parent in node.intercept.ci_parents:
        print(f"    {parent:>3} -> I    (feeds the joint intercept above)")
    for parent, mod in node.shifts.items():
        eff = "LS" if type(mod).__name__ == "LinearShift" else "CS"
        print(f"    {parent:>3} -> {eff}   ({type(mod).__name__})")
    if not node.parents:
        print("    (source node — no parents)")


def decompose_row(flow, name, row_df):
    """Numeric view: print u = intercept + sum(shifts) for one row and check it
    reproduces flow.node_log_prob exactly.

    Anatomy on purpose: ``_tensorize``/``_features`` are library internals,
    used here to open the model up — user code never needs them.
    """
    node = flow.nodes[name]
    vals = flow._tensorize(row_df)
    feats = flow._features(vals)
    theta, shift = node.theta_shift(feats, len(row_df))
    parts = {p: node.shifts[p](feats[p]) for p in node.shifts}  # per-parent shift
    print(f"{name} = {float(vals[name][0]):+.3f}  ({node.kind})")
    if node.kind == "continuous":
        h0, ladj = node.ut.forward(theta, vals[name])
        terms = "  +  ".join(
            [f"h_theta(x)={float(h0[0]):+.3f}"]
            + [f"{p}={float(v[0]):+.3f}" for p, v in parts.items()]
        )
        u = h0 + shift
        print(f"  u = {terms}  =  {float(u[0]):+.3f}   (standard-logistic latent)")
        lp = StandardLogistic.log_prob(u) + ladj
    else:  # ordinal: cutpoints minus a subtracted shift
        cuts = ordinal_cutpoints(theta)[0, 1:-1].detach().numpy().round(3)
        terms = "  +  ".join(f"{p}={float(v[0]):+.3f}" for p, v in parts.items()) or "0"
        print(f"  cutpoints theta_k = {cuts}")
        print(
            f"  shift (SUBTRACTED) = {terms}   ->   P(Y<=k) = sigmoid(theta_k - shift)"
        )
        lp = ordinal_log_prob(theta, shift, vals[name])
    check = flow.node_log_prob(vals)[name]
    print(
        f"  log p(row) rebuilt = {float(lp[0]):+.4f}   node_log_prob = "
        f"{float(check[0]):+.4f}   match={bool(torch.allclose(lp, check))}\n"
    )


for nm in ["X2", "X3", "Y"]:
    describe_node(flow, nm)
print()
row0 = val_df.iloc[[0]]
for nm in ["X2", "X3", "Y"]:
    decompose_row(flow, nm, row0)

# %% [markdown]
# ## 5. Rung 1 — the observational distribution
#
# First sanity check: samples from the fitted flow must reproduce the joint
# observational distribution (including the ordinal outcome's marginal).

# %%
samp = flow.sample(len(df), seed=0)

fig, axes = plt.subplots(1, 4, figsize=(13, 3))
for ax, col in zip(axes[:3], ["X1", "X2", "X3"], strict=True):
    bins = np.linspace(df[col].min(), df[col].max(), 60)
    ax.hist(df[col], bins=bins, density=True, alpha=0.45, label="data")
    ax.hist(
        samp[col],
        bins=bins,
        density=True,
        histtype="step",
        lw=1.8,
        color="C3",
        label="flow",
    )
    ax.set_title(col)
levels = np.arange(4)
w = 0.35
axes[3].bar(
    levels - w / 2,
    df["Y"].value_counts(normalize=True).sort_index(),
    width=w,
    alpha=0.6,
    label="data",
)
axes[3].bar(
    levels + w / 2,
    samp["Y"].value_counts(normalize=True).sort_index(),
    width=w,
    color="C3",
    alpha=0.8,
    label="flow",
)
axes[3].set_title("Y"), axes[3].set_xticks(levels)
axes[0].legend()
fig.suptitle("L1: observational marginals, data vs. flow samples")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Single-number interpretable statistics
#
# Because the latents are standard logistic, every linear-shift weight is a
# **log-odds ratio**. For a continuous node ($u = h(x) + \beta\, x_{\text{pa}}$),
# a unit increase of the parent multiplies the odds of $\{X \le x\}$ by
# $e^\beta$, uniformly in $x$ (a proportional-odds / Colr-type effect). For the
# ordinal node the sign convention flips ($\sigma(\vartheta_k - \text{shift})$).
# Thus a positive $\beta$ moves $Y$ toward higher categories: $e^\beta$
# multiplies the odds of $\{Y > k\}$.
#
# These are *parameters of the fitted flow*. We can read them directly and
# compare them with the DGP constants:

# %%
# raw parameter access, to make the point; flow.ls_coefficients() is the
# one-call read-out for all of them
b21_hat = float(flow.nodes["X2"].shifts["X1"].weight.detach())
b31_hat = float(flow.nodes["X3"].shifts["X1"].weight.detach())
bY_hat = float(flow.nodes["Y"].shifts["X3"].weight.detach())


with torch.no_grad():
    theta_hat = ordinal_cutpoints(flow.nodes["Y"].intercept(1))[0, 1:-1].numpy()

print(f"beta_21 (X1 -> X2):  true {TRUE['b21']:+.3f}   fitted {b21_hat:+.3f}")
print(f"beta_31 (X1 -> X3):  true {TRUE['b31']:+.3f}   fitted {b31_hat:+.3f}")
print(f"beta_Y  (X3 -> Y):   true {TRUE['bY']:+.3f}   fitted {bY_hat:+.3f}")
print(f"cutpoints theta_Y:   true {TRUE['theta_Y']}   fitted {theta_hat.round(3)}")

# %% [markdown]
# The fit also recovers the flexible parts. The baseline transformation
# $\hat h_3$ must match $\sinh$, and the complex shift $\hat g$ must match
# $\tfrac12 x_2^2$. Each match holds **up to an additive constant**, because a
# constant can move freely between the intercept and a complex shift (only their
# sum is identified). Therefore we center both curves before the comparison.


# %%
def fitted_baseline(flow, name, grid):
    """h_hat(x) for a continuous node with constant (simple) intercept."""
    node = flow.nodes[name]
    x = torch.as_tensor(grid, dtype=torch.float32)
    with torch.no_grad():
        h0, _ = node.ut.forward(node.intercept(len(grid)), x)
    return h0.detach().numpy()


def fitted_cs(flow, name, parent, grid):
    """g_hat(parent) for a 'cs' edge."""
    x = torch.as_tensor(grid, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        return flow.nodes[name].shifts[parent](x).detach().numpy()


x3_grid = np.linspace(*df["X3"].quantile([0.01, 0.99]), 200)
x2_grid = np.linspace(*df["X2"].quantile([0.01, 0.99]), 200)
h3_hat, h3_true = fitted_baseline(flow, "X3", x3_grid), np.sinh(x3_grid)
g_hat, g_true = fitted_cs(flow, "X3", "X2", x2_grid), g_cs(x2_grid)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].plot(x3_grid, h3_true - h3_true.mean(), lw=2, label=r"true $\sinh(x)$")
axes[0].plot(x3_grid, h3_hat - h3_hat.mean(), "--", lw=2, label=r"fitted $\hat h_3$")
axes[0].set_title("baseline transformation of $X_3$"), axes[0].set_xlabel("$x_3$")
axes[1].plot(x2_grid, g_true - g_true.mean(), lw=2, label=r"true $\frac{1}{2}x^2$")
axes[1].plot(x2_grid, g_hat - g_hat.mean(), "--", lw=2, label=r"fitted $\hat g$")
axes[1].set_title("complex shift $X_2 \\to X_3$"), axes[1].set_xlabel("$x_2$")
for ax in axes:
    ax.legend()
fig.suptitle("Recovered transformation functions (centered)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Why the term choice matters: a deliberately misspecified model
#
# What happens if we declare the $X_2 \to X_3$ edge as a *linear* shift? The best
# a linear shift can do is the **average local slope** of $g$. The data mass sits
# around $E[x_2] \approx -0.75$, so the `LS` model finds
# $\hat\beta_{32} \approx E[g'(x_2)] = E[x_2] \approx -0.7$. That is the tangent
# of the U-shape, not the U-shape. The model loses the curvature. The per-node
# validation NLL makes the misfit measurable, and the interventional
# distributions in the next section are visibly wrong.

# %%
spec_ls = {
    "X1": ContinuousNode(),
    "X2": ContinuousNode(LS("X1")),
    "X3": ContinuousNode(LS("X1") + LS("X2")),  # <- CS replaced by LS
    "Y": OrdinalNode(4, LS("X3")),
}
flow_ls = CausalFlowDAG(spec_ls, seed=7)
# the same schedule as the cs model above, so the NLL comparison is fair
flow_ls.fit(train_df, epochs=800, learning_rate=1e-2, batch_size=20000)
flow_ls.fit(train_df, epochs=300, learning_rate=1e-3, batch_size=512)

print(
    f"misspecified beta_32 (X2 -> X3): "
    f"{float(flow_ls.nodes['X3'].shifts['X2'].weight.detach()):+.3f}"
)
print(
    f"val NLL of node X3:  cs model {flow.nll(val_df)['X3']:.4f}"
    f"   ls model {flow_ls.nll(val_df)['X3']:.4f}"
)

# %% [markdown]
# ## 7. Rung 2 — interventions: the do-operator
#
# `flow.sample(n, do={"X1": a})` performs **graph mutilation**: it clamps $X_1$
# to $a$ and discards the node's own mechanism (and latent). All downstream nodes
# react. Because we own the DGP, we can simulate the *true* interventional
# distribution and compare. We also show the misspecified all-`LS` model. It
# gets the interventional distribution of $X_3$ visibly wrong.

# %%
rng_iv = np.random.default_rng(123)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, a in zip(axes, [-1.0, 1.0], strict=True):
    truth, _ = simulate(20000, rng_iv, x1=a)
    fl = flow.sample(20000, do={"X1": a}, seed=5)
    fls = flow_ls.sample(20000, do={"X1": a}, seed=5)
    bins = np.linspace(truth["X3"].min(), truth["X3"].max(), 60)
    ax.hist(truth["X3"], bins=bins, density=True, alpha=0.4, label="DGP truth")
    ax.hist(
        fl["X3"],
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        color="C3",
        label="flow (cs)",
    )
    ax.hist(
        fls["X3"],
        bins=bins,
        density=True,
        histtype="step",
        lw=1.5,
        color="C7",
        ls=":",
        label="flow (all-ls)",
    )
    ax.set_title(f"$p(x_3 \\mid do(X_1 = {a:+.0f}))$"), ax.set_xlabel("$x_3$")
    print(
        f"E[X3 | do(X1={a:+.0f})]:  truth {truth['X3'].mean():+.3f}   "
        f"flow(cs) {fl['X3'].mean():+.3f}   flow(all-ls) {fls['X3'].mean():+.3f}"
    )
axes[0].legend()
fig.suptitle("L2: interventional distributions")
fig.tight_layout()
plt.show()

# %% [markdown]
# For the **ordinal** outcome, Monte Carlo is not necessary. The ordered-logit
# head gives the interventional PMF **analytically**,
# $P(Y = k \mid do(X_3 = a)) = \sigma(\vartheta_k - \beta_Y a) - \sigma(\vartheta_{k-1} - \beta_Y a)$.

# %%
a = 1.0
pmf_flow = flow.pmf(pd.DataFrame({"X3": [a]}), "Y")[0]
cdf_true = 1 / (1 + np.exp(-(np.r_[-np.inf, TRUE["theta_Y"], np.inf] - TRUE["bY"] * a)))
pmf_true = np.diff(cdf_true)
print("P(Y = k | do(X3 = 1)):")
print("  true:", pmf_true.round(4), "\n  flow:", pmf_flow.round(4))

# %% [markdown]
# For a **continuous** node the same closed form exists: `flow.density` gives
# $p(x_3 \mid do(X_1 = a), x_2)$ on a grid from the fitted transform, no sampling.
# The DGP tells us the truth: $u_3 = \sinh(x_3) + 0.8\,x_1 + \tfrac12 x_2^2$, so
# $p(x_3) = f_U\!\big(u_3\big)\cosh(x_3)$ with $f_U$ the standard-logistic density.

# %%
grid = np.linspace(-3.5, 3.5, 301)
row = pd.DataFrame({"X2": [0.0]})  # the parent we keep; X1 is set by do=
fig, ax = plt.subplots(figsize=(7, 3.2))
for a, color in ((-1.0, "C0"), (1.0, "C3")):
    dens = flow.density(row, "X3", grid, do={"X1": a})[0]
    u3 = np.sinh(grid) + 0.8 * a + 0.5 * 0.0**2
    truth_pdf = np.exp(-u3) / (1 + np.exp(-u3)) ** 2 * np.cosh(grid)
    ax.plot(
        grid, truth_pdf, color=color, lw=2.5, alpha=0.4, label=f"truth, $a={a:+.0f}$"
    )
    ax.plot(grid, dens, color=color, lw=1.2, label=f"flow.density, $a={a:+.0f}$")
ax.set_xlabel("$x_3$"), ax.set_ylabel("$p(x_3 \\mid do(X_1=a), x_2=0)$")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Rung 3 — counterfactuals: abduction → action → prediction
#
# Because the flow is **bijective in the continuous variables**, Pearl's three
# steps are exact:
#
# 1. **Abduction** — map the *factual* data to its latent (the training
#    direction): $u = h(x \mid \mathrm{pa})$ per node (`flow.abduct(df)`). Each row's $u$
#    is its individual "noise". It is everything about the unit that the model
#    does not attribute to the parents. (For ordinal nodes $u$ is only
#    interval-identified. Thus the flow draws it from the logistic truncated to
#    the observed level's interval.)
# 2. **Action** — mutilate the graph: `do={"X1": 0.0}`.
# 3. **Prediction** — push the *same* $u$ back through the mutilated flow:
#    `flow.sample(do=..., u=u)`.
#
# First check: we push the abducted latents through the flow with no action.
# This must reproduce the factual data exactly (level-exactly for $Y$).

# %%
u = flow.abduct(val_df, seed=11)
recon = flow.sample(u=u)
err = recon[["X1", "X2", "X3"]].to_numpy() - val_df[["X1", "X2", "X3"]].to_numpy()
print(f"max |reconstruction error| (continuous): {np.abs(err).max():.2e}")
print(f"Y level-exact: {(recon['Y'].to_numpy() == val_df['Y'].to_numpy()).mean():.1%}")

# %% [markdown]
# Now we ask the counterfactual: *"what would $X_2, X_3$ have been for **this**
# unit, had $X_1$ been 0?"*. The DGP kept every unit's true latents `u_obs`.
# Thus we can compute the **true individual counterfactuals** and compare them
# unit by unit. This is the strongest test on the ladder.

# %%
cf_flow = flow.sample(do={"X1": 0.0}, u=u)
u_val = {k: v[5000:] for k, v in u_obs.items()}
cf_true, _ = simulate(len(val_df), rng, x1=0.0, u=u_val)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, col in zip(axes, ["X2", "X3"], strict=True):
    ax.scatter(cf_true[col], cf_flow[col], s=5, alpha=0.4)
    lims = [cf_true[col].min(), cf_true[col].max()]
    ax.plot(lims, lims, "k--", lw=1)
    r = np.corrcoef(cf_true[col], cf_flow[col])[0, 1]
    rmse = float(np.sqrt(np.mean((cf_true[col] - cf_flow[col]) ** 2)))
    print(f"counterfactual {col}:  corr(truth, flow) = {r:.4f}   RMSE = {rmse:.3f}")
    ax.set_title(f"counterfactual {col}   (r = {r:.4f})")
    ax.set_xlabel("DGP truth"), ax.set_ylabel("flow")
fig.suptitle("L3: individual counterfactuals under $do(X_1 = 0)$, unit by unit")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Where to go from here
#
# * **Complex intercepts (`I(...)`)** — this is the one component not exercised
#   here. Declare `ContinuousNode(I("Age"))`, and the *parameters* of the
#   Bernstein transform become a function of the parent. Several `I(...)` parents
#   feed one joint network, that is, they can interact. The paper replications in
#   `experiments/` use `I(...)` heavily. Run
#   `uv run python -m paper.vaca flexible` (from `experiments/`) for an all-intercept flow on a
#   bimodal benchmark with analytic interventional truth.
# * **Early stopping vs. exact MLE** — this notebook's DGP has no unobserved
#   confounding, so the MLE (the final weights `fit` keeps) is the right
#   target. Under observational confounding, flexible (`I`/`CS`) models can
#   *overfit the confounding* at the MLE and need best-validation weights
#   (`tramdag.callbacks.EarlyStopping`) to recover
#   the causal effect. See `docs/fitting.md`.
# * **Validation against classical models** — an all-`LS` flow trained to
#   convergence *is* the classical proportional-odds MLE
#   (`experiments/misc/validate_ls.py` pins flow ≡ `statsmodels` ≡ R `polr`, and the
#   framework tests check the same identity on an inline DGP).
# * **Joint terms** — write several parents inside one term to model an
#   *interaction*. `CS("x1", "x2")` is a single shift network $g(x_1, x_2)$, and
#   `I("x1", "x2")` is a single intercept network over both parents. Separate
#   terms (`CS("x1") + CS("x2")`) stay additive. The grouping *is* the
#   joint/additive choice.
# * **Current limitations** (vs. the general formulation): the latent is fixed to
#   the standard logistic.

# %% [markdown]
# ---
# *This file is a jupytext percent notebook. If you prefer `.ipynb`, pair the
# file with `uvx jupytext --to ipynb notebooks/intro_tram_dag.py`. Keep the
# `.ipynb` out of git (see `.gitignore`).*
