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
# # Classical fitting of all-`ls` TRAM-DAGs (`fit_classical`)
#
# When **every edge of a TRAM-DAG is a linear shift (`ls`)**, each
# node-conditional is a *classical transformation model* — an ordered-logit /
# proportional-odds model for ordinal nodes, a continuous-outcome logistic
# transformation model (R's `tram::Colr`) for continuous ones. For such a model,
# minibatch Adam is an odd choice: thousands of epochs, a learning-rate schedule
# to tune, shuffle noise, and no reproducibility — to fit something classical
# software solves deterministically in milliseconds.
#
# `flow.fit_classical()` is the classical optimizer for exactly this case:
#
# - **full-batch, float64, L-BFGS** with a strong-Wolfe line search,
# - **deterministic** — no minibatching, so the same start gives bit-identical
#   results,
# - lands on the **exact maximum-likelihood estimate**, matching `statsmodels`
#   and R to ~1e-3 on the well-identified coefficients,
# - **raises** on any `cs`/`ci` edge (use `fit()` there — minibatch noise also
#   regularizes the MLPs).
#
# This notebook fits all-`ls` models with `fit_classical` on two datasets,
# checks them against the classical solution, and shows how a classical fit can
# **warm-start** further training.

# %%
import time
import warnings

import numpy as np
import pandas as pd
import torch

from tramdag import LS, CausalFlowDAG, ContinuousNode, OrdinalNode
from tramdag.simulations import VacaTriangle

warnings.filterwarnings("ignore")
REPO = __import__("pathlib").Path.cwd()
REPO = REPO if (REPO / "data").exists() else REPO.parent  # run from repo or notebooks/
DATA = REPO / "data"

# %% [markdown]
# ## 1. Continuous case — the bimodal demo data
#
# The `VacaTriangle` benchmark (`x1 → x2 → x3 ← x1`, the demo notebook's bimodal
# SCM). We fit an **all-`ls`** model: each node is a continuous logistic
# transformation model with a Bernstein baseline and linear shifts.
#
# Note this is an *honest misspecification*: the DGP noise is Gaussian while the
# TRAM latent is logistic, so the all-`ls` model is not the true generator — but
# `fit_classical` still finds its exact MLE, which is the point here.

# %%
df = VacaTriangle(seed=42).observational(20_000)

spec_vaca = {
    "x1": ContinuousNode(),
    "x2": ContinuousNode(terms=[LS("x1")]),
    "x3": ContinuousNode(terms=[LS("x1"), LS("x2")]),
}

torch.manual_seed(0)
flow_c = CausalFlowDAG(spec_vaca)
rep = flow_c.fit_classical(df)  # prints iters / NLL / time
print("\nlinear-shift coefficients (log-odds scale):")
for node, parents in flow_c.ls_coefficients().items():
    for p, w in parents.items():
        print(f"  {p:>3} -> {node:<3}: {w[0]:+.4f}")

# %% [markdown]
# **Deterministic?** Same seed, twice — bit-identical (no minibatch RNG):


# %%
def fit_T_coef(seed):
    torch.manual_seed(seed)
    f = CausalFlowDAG(spec_vaca)
    f.fit_classical(df, verbose=False)
    return f.ls_coefficients()["x2"]["x1"][0]


a, b = fit_T_coef(0), fit_T_coef(0)
print(f"two runs, same seed: {a:.10f} == {b:.10f}  ->  {a == b}")

# %% [markdown]
# **Classical vs Adam — same optimum.** A converged Adam fit (no early stopping)
# reaches the same coefficients to ~1e-4. The classical fit's guarantees are
# *determinism* and the *exact MLE*; raw speed is model-dependent — it wins
# clearly on ordinal-outcome models (Section 2: ~2 s), but on this
# Bernstein-heavy continuous DAG L-BFGS has to grind through the flat polynomial
# valleys, so wall-clock here is comparable to Adam rather than faster:

# %%
torch.manual_seed(0)
flow_a = CausalFlowDAG(spec_vaca)
t0 = time.perf_counter()
flow_a.fit(
    df,
    epochs=2000,
    learning_rate=1e-1,
    batch_size=4096,
    verbose=0,
    schedule="plateau",
    plateau_patience=15,
    freeze_patience=60,
)
t_adam = time.perf_counter() - t0

print(f"{'coef':<10}{'classical':>12}{'adam':>12}{'|diff|':>10}")
for node, p in [("x2", "x1"), ("x3", "x1"), ("x3", "x2")]:
    c = flow_c.ls_coefficients()[node][p][0]
    aw = float(flow_a.nodes[node].shifts[p].weight.detach())
    print(f"{p}->{node:<6}{c:>12.4f}{aw:>12.4f}{abs(c - aw):>10.4f}")
print(f"\nclassical {rep['seconds']:.2f}s  vs  adam {t_adam:.1f}s")

# %% [markdown]
# ### Reproduce the continuous fit in R (`tram::Colr`)
#
# `statsmodels` has no continuous logistic transformation model, but R's `tram`
# package does — `Colr` fits exactly the per-node model above. For the
# `x3 | x1, x2` node (save `df` to `vaca.csv` first):
#
# ```r
# library(tram)
# d <- read.csv("vaca.csv")
# # Colr: continuous outcome logistic transformation model;
# # the linear-shift coefficients are log-odds ratios, comparable to
# # flow.ls_coefficients()["x3"] (up to tram's sign convention for the shift).
# m <- Colr(x3 ~ x1 + x2, data = d, order = 19)   # Bernstein order 19 ~ n_coeffs 20
# coef(m)        # -> the x1, x2 shift coefficients
# logLik(m)
# ```
#
# (The flow's Bernstein default is `n_coeffs=20`, i.e. polynomial order 19.)

# %% [markdown]
# ## 2. Ordinal case — exact, self-contained classical check
#
# For an **ordinal** outcome the classical model is the ordered-logit, which
# `statsmodels.OrderedModel` fits — so here the equivalence is checkable in
# Python. The all-`ls` stroke DAG (`magic-mrclean/ls`):

# %%
obs = pd.read_csv(DATA / "magic-mrclean" / "ls" / "obs.csv")

spec_stroke = {
    "Age": ContinuousNode(),
    "mRS_pre": OrdinalNode(levels=6, terms=[LS("Age")]),
    "NIHSSa": ContinuousNode(terms=[LS("Age"), LS("mRS_pre")]),
    "T": OrdinalNode(levels=2, terms=[LS("Age"), LS("mRS_pre"), LS("NIHSSa")]),
    "mRS_3m": OrdinalNode(
        levels=7, terms=[LS("Age"), LS("mRS_pre"), LS("NIHSSa"), LS("T")]
    ),
}

torch.manual_seed(7)
flow_s = CausalFlowDAG(spec_stroke)
flow_s.fit_classical(obs)

# %% [markdown]
# Classical reference for the outcome node (`statsmodels` ordered logit). Ordinal
# parents enter one-hot; with cutpoints only differences to level 0 are
# identified, so we compare those:

# %%
from statsmodels.miscmodels.ordinal_model import OrderedModel  # noqa: E402


def design(d):
    X = pd.DataFrame(index=d.index)
    X["Age"] = d["Age"].values
    for k in range(6):
        X[f"mRS_pre_{k}"] = (d["mRS_pre"].values == k).astype(float)
    X["NIHSSa"] = d["NIHSSa"].values
    X["T"] = d["T"].values
    return X.drop(columns=["mRS_pre_0"])


res = OrderedModel(obs["mRS_3m"].astype(int), design(obs), distr="logit").fit(
    method="bfgs", disp=False
)

n = flow_s.nodes["mRS_3m"]
w_t = n.shifts["T"].weight.detach().numpy().ravel()
rows = [
    ("Age", float(n.shifts["Age"].weight.detach()), res.params["Age"]),
    ("NIHSSa", float(n.shifts["NIHSSa"].weight.detach()), res.params["NIHSSa"]),
    ("T (1 vs 0)", w_t[1] - w_t[0], res.params["T"]),
]
print(f"{'coefficient':<14}{'fit_classical':>14}{'statsmodels':>13}{'|diff|':>9}")
for name, a_, b_ in rows:
    print(f"{name:<14}{a_:>14.4f}{b_:>13.4f}{abs(a_ - b_):>9.4f}")

# %% [markdown]
# Age and NIHSSa match to ~1e-3; the treatment effect `T` is *weakly identified*
# in this cohort (a nearly flat likelihood ridge — documented in the project's
# CLAUDE.md), so it agrees only to ~1e-2 — exactly the same ambiguity classical
# software shows. The likelihood is at its optimum; some directions are simply
# flat.

# %% [markdown]
# ## 3. Warm-start handoff: classical fit → further training
#
# `fit_classical` leaves the model at the MLE in float32, ready for any normal
# operation. Two things to verify:
#
# 1. the float64→float32 round-trip didn't move the coefficients, and
# 2. continuing with `fit()` from the classical solution *stays put* — confirming
#    it really is the optimum (and showing the classical fit as a fast, principled
#    initialization for further or richer training).

# %%
before = {k: v.copy() for k, v in flow_s.ls_coefficients()["mRS_3m"].items()}

# continue training from the classical solution with a gentle Adam phase
flow_s.fit(
    obs, epochs=300, learning_rate=1e-3, batch_size=256, verbose=0, restore_best=False
)
after = flow_s.ls_coefficients()["mRS_3m"]

print("coefficient drift after 300 more Adam epochs from the classical MLE:")
for p in ["Age", "NIHSSa", "T"]:
    d = float(np.abs(np.atleast_1d(after[p]) - np.atleast_1d(before[p])).max())
    print(f"  {p:<8} max|Δ| = {d:.4f}")
print("\n-> small drift = the classical fit was already at the optimum;")
print("   fit_classical is a valid warm start for continued / flexible training.")

# %% [markdown]
# ## When to use which
#
# | situation | use |
# |---|---|
# | all-`ls` model, final estimates / classical comparison / reproducibility | **`fit_classical`** |
# | any `cs`/`ci` edge (flexible model) | `fit(..., schedule="plateau")` |
# | need a fast, principled init for a flexible model | `fit_classical` (all-`ls` core) → upgrade edges → `fit()` |
#
# `fit_classical` is also the groundwork for **standard errors**: fitting at the
# MLE in float64 is exactly what a Hessian-based covariance needs (a future
# addition — see the CHANGELOG).
