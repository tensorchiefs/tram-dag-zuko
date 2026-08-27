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
# # Additive vs joint complex intercept — interpreting per-parent effects
#
# A node whose transform parameters depend on its parents (a **complex
# intercept**, `CI`) can group those parents two ways:
#
# - **joint** — `CI("x1", "x2")`: *one* network over both parents. It can
#   represent **interactions** (the effect of `x1` may depend on `x2`), but the
#   two parents are entangled in one black box. This is the default.
# - **additive** — `CI("x1", "x2", allow_interaction=False)`: *one network per
#   parent*, summed in unconstrained parameter space,
#   `theta(pa) = net_1(x1) + net_2(x2)`. Each parent reshapes the transform
#   **independently** — a separable, GAM-like structure.
#
# The grouping is said with the flag, not by writing two terms: a node takes at
# most **one** intercept term with parents, so `[CI("x1"), CI("x2")]` is an
# error. That keeps a term list purely additive on the latent scale.
#
# The additive form is the interpretable one: you can ask "what does `x1`
# *alone* do?". The catch (issue #20) is that the additive sum is identified only
# up to a constant moving between the nets, so the **raw** per-parent outputs are
# not comparable. `flow.intercept_contributions(data, node)` resolves this with a
# sum-to-zero (mean-centering) constraint and returns each parent's centered
# contribution.

# This notebook fits both models and shows the interpretability difference. The
# punchline is perhaps surprising: the two models fit the **data** almost
# identically — the difference is **structural**, visible only in parameter
# space, and that is precisely what `intercept_contributions` surfaces.

# %%
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tramdag import CI, CausalFlowDAG, ContinuousNode


def keep_best(flow, val):
    """The former ``restore_best`` as a callback: snapshot the best-validation
    weights; load them back after ``fit``.
    """
    best = {"nll": np.inf, "state": None}

    def cb(f, epoch, opt):
        nll = sum(f.nll(val).values())
        if nll < best["nll"]:
            best.update(nll=nll, state=copy.deepcopy(f.state_dict()))

    return best, cb


# %% [markdown]
# ## The data
#
# `x1, x2 ~ U(-2, 2)` and
# $x_3 = x_1 + x_2 + \gamma\,x_1 x_2 + \tfrac12\,\varepsilon$,
# $\varepsilon \sim \mathrm{Logistic}(0,1)$ — a genuine `x1·x2` interaction.

# %%
GAMMA = 0.6


def simulate(n, seed):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-2, 2, n)
    eps = rng.logistic(0, 1, n)
    x3 = x1 + x2 + GAMMA * x1 * x2 + 0.5 * eps
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


train = simulate(4000, seed=1)
val = simulate(1000, seed=2)
train.head()

# %% [markdown]
# ## Fit both models
#
# Same data, same node — only the grouping of the two `I` parents differs.


# %%
def make_flow(joint: bool):
    spec = {
        "x1": ContinuousNode(),
        "x2": ContinuousNode(),
        "x3": ContinuousNode(
            [CI("x1", "x2")] if joint else [CI("x1", "x2", allow_interaction=False)]
        ),
    }
    return CausalFlowDAG(spec, seed=0)


flow_joint = make_flow(joint=True)
flow_add = make_flow(joint=False)

for f in (flow_joint, flow_add):
    best, cb = keep_best(f, val)
    f.fit(train, epochs=1200, learning_rate=1e-2, callback=cb)
    f.load_state_dict(best["state"])

# Both fit the data well; the joint model is only marginally better on held-out
# likelihood. A flexible additive Bernstein intercept can *mimic* a lot of
# apparent interaction through its nonlinear transform, so you generally **cannot
# tell the two apart from the fitted distribution** — see the note below.
print("val NLL  joint   :", round(sum(flow_joint.nll(val).values()), 4))
print("val NLL  additive:", round(sum(flow_add.nll(val).values()), 4))

# %% [markdown]
# ## `intercept_contributions` — the exact, centered decomposition
#
# It returns each `I`-term's mean-centered (sum-to-zero) contribution to the
# transform parameters `theta`, plus the absorbed `baseline`. The centering is
# over the rows of the `data` you pass, and the decomposition is exact.

# %%
res_add = flow_add.intercept_contributions(train, "x3")
res_joint = flow_joint.intercept_contributions(train, "x3")

print("additive terms :", list(res_add["contributions"]))  # 'x1', 'x2' — separable
print("joint terms    :", list(res_joint["contributions"]))  # 'x1+x2' — inseparable

# exactness: baseline + sum of contributions reproduces theta; centering: ~0 means
nd = flow_add.nodes["x3"]
feats = flow_add._features(flow_add._tensorize(train))
with torch.no_grad():
    theta = sum(
        net(torch.cat([feats[p] for p in g], 1))
        for net, g in zip(
            nd.intercept_nets, nd._intercept_groups, strict=False
        )  # library internals: one net per group by construction, not checked here
    ).numpy()
recon = res_add["baseline"][None] + sum(res_add["contributions"].values())
print("max |reconstruction - theta| :", np.abs(recon - theta).max())
print(
    "per-term column means (≈0)   :",
    {k: float(np.abs(v.mean(0)).max()) for k, v in res_add["contributions"].items()},
)

# %% [markdown]
# ## The structural difference: separable curves vs an entangled cloud
#
# Pick the transform coefficient that varies most across the data and plot each
# term's centered contribution to it against a parent value.
#
# - **Additive** (`net_1` depends only on `x1`): its contribution to any
#   coefficient is a deterministic 1-D function of `x1` — the points collapse to
#   a **clean curve** (same for `x2`). These curves *are* the per-parent partial
#   effects: well-defined because the structure is separable.
# - **Joint** (one network over both parents): the single `x1+x2` component
#   depends on *both*, so plotted against `x1` alone it is a **cloud** whose
#   height is explained by the hidden `x2` (color). There is nothing to read off
#   per parent.

# %%
c1 = res_add["contributions"]["x1"]
c2 = res_add["contributions"]["x2"]
cj = res_joint["contributions"]["x1+x2"]
k = int((c1.var(0) + c2.var(0)).argmax())  # most-varying coefficient
x1v, x2v = train["x1"].values, train["x2"].values

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
o1, o2 = np.argsort(x1v), np.argsort(x2v)
axes[0].plot(x1v[o1], c1[o1, k], color="#1b9e77", lw=2.5, label="net(x1)  vs x1")
axes[0].plot(x2v[o2], c2[o2, k], color="#d95f02", lw=2.5, label="net(x2)  vs x2")
axes[0].axhline(0, color="0.7", lw=0.8)
axes[0].set_title("additive — separable per-parent curves")
axes[0].set_xlabel("parent value")
axes[0].set_ylabel(f"centered contribution to theta[{k}]")
axes[0].legend()

sc = axes[1].scatter(x1v, cj[:, k], c=x2v, s=8, alpha=0.6, cmap="coolwarm")
axes[1].axhline(0, color="0.7", lw=0.8)
axes[1].set_title("joint — vs x1 is a cloud (height set by x2)")
axes[1].set_xlabel("x1")
fig.colorbar(sc, ax=axes[1], label="x2")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Takeaway
#
# | you want… | use | what you get |
# |---|---|---|
# | a per-parent partial-effect plot ("what does `x1` do?") | **additive** `CI("x1", "x2", allow_interaction=False)` | `intercept_contributions` → exact, mean-centered, **separable** components |
# | interactions between parents in the transform | **joint** `CI("x1", "x2")` | one entangled network — flexible, **not** separable |
#
# Both give correct likelihoods and L1/L2/L3 causal queries — the choice is about
# *interpretability*, not correctness, and (as the near-equal NLLs show) you
# usually can't distinguish them from the fitted distribution alone. The
# separability that makes per-parent effects well-defined is a property of the
# **parameter space**, and `intercept_contributions` is what surfaces it. It is
# post-hoc: it reads the fitted weights and changes nothing about the model.
#
# *Caveat:* contributions live in the transform's **unconstrained** parameter
# space (where the additive terms are summed, before the monotonicity
# constraint), so they are exact partial effects on the parameters but not, in
# general, an additive split of the curve `h` itself.

# %%
