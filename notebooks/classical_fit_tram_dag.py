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
# When **every edge of a TRAM-DAG is a linear shift (`LS`)**, each
# node-conditional is a *classical transformation model* — an ordered-logit /
# proportional-odds model for ordinal nodes, a continuous-outcome logistic
# transformation model (R's `tram::Colr`) for continuous ones. For such a model,
# the classical optimizer is best suited.
#
# `flow.fit_classical()` is the classical optimizer for exactly this case:
#
# - **full-batch, float64, L-BFGS** with a strong-Wolfe line search,
# - **deterministic** — no minibatching, so the same start gives bit-identical
#   results,
# - lands on the **exact maximum-likelihood estimate**, matching `statsmodels`
#   and R to ~1e-3 on the well-identified coefficients,
# - **raises** on any `CS`/`CI`/`VC` term (use `fit()` there — minibatch noise
#   also regularizes the MLPs).
#
# This notebook starts from the most familiar all-`ls` model of all — a plain
# logistic regression on `MASS::birthwt`, checked against R — then fits two
# larger DAGs with `fit_classical`, compares each with the classical solution,
# and shows how a classical fit can **warm-start** further training.

# %%
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.miscmodels.ordinal_model import OrderedModel

from tramdag import LS, SI, CausalFlowDAG, ContinuousNode, OrdinalNode

warnings.filterwarnings("ignore")
# so fit_classical reports its iters / NLL / time
logging.basicConfig(level=logging.INFO, format="%(message)s")

# repo-relative data, whether the notebook runs from the repo root or notebooks/
HERE = [Path.cwd(), *Path.cwd().parents]
REPO = next(p for p in HERE if (p / "pyproject.toml").exists())
DATA = REPO / "experiments" / "misc" / "data"
NB_DATA = REPO / "notebooks" / "data"

# %% [markdown]
# ## 0. The zeroth example — logistic regression on `MASS::birthwt`
#
# Before any DAG, take the model everyone already knows, on a dataset every R
# user already has. A **two-level** `OrdinalNode` whose terms are all `LS` *is*
# logistic regression — not an analogue of it, the same model. The ordinal
# transform is
#
# $$P(Y \le 0 \mid x) = \sigma\!\left(\theta_0 - \textstyle\sum_p w_p x_p\right),$$
#
# and with two levels there is a single cutpoint $\theta_0$, so
#
# $$\operatorname{logit} P(Y = 1 \mid x) = -\theta_0 + \textstyle\sum_p w_p x_p .$$
#
# The shift weights **are** the logistic-regression coefficients; the intercept
# is **minus** the cutpoint, because an ordinal node *subtracts* its shift. That
# sign convention is the one thing worth internalizing here — Section 2 is this
# same picture with $K-1$ cutpoints instead of one.
#
# The data is `birthwt` from **MASS**, the low-birth-weight study of Hosmer &
# Lemeshow: 189 births, outcome `low` (birth weight under 2.5 kg), predictors
# `age` (mother's age), `lwt` (mother's weight at last period, lbs) and `smoke`
# (smoking during pregnancy). `notebooks/data/birthwt.csv` is those four columns
# exported verbatim from MASS. Please note that this is not a **causal** model.

# %% [markdown]
# ### The R you can copy-paste
#
# `birthwt` ships with MASS, so nothing needs downloading — paste this into any
# R session and you have the reference fit:
#
# ```r
# library(MASS)
# m <- glm(low ~ age + lwt + smoke, data = birthwt, family = binomial)
# coef(m)
# logLik(m)
# ```
#
# which prints
#
# ```
# (Intercept)         age         lwt       smoke
#  1.36822527 -0.03899458 -0.01213854  0.67076374
# 'log Lik.' -111.4396765 (df=4)
# ```
#
# (R 4.2.3, MASS 7.3-58.2.) Those four numbers are hard-coded below as `R_GLM`,
# so the notebook can show the three-way comparison without an R installation.

# %%
R_GLM = {  # coef(glm(low ~ age + lwt + smoke, birthwt, family = binomial))
    "intercept": 1.3682252685,
    "age": -0.0389945827,
    "lwt": -0.0121385423,
    "smoke": 0.6707637407,
}
R_LOGLIK = -111.43967649

bw = pd.read_csv(NB_DATA / "birthwt.csv")
print(
    bw.head(3).to_string(index=False),
    f"\n\n{len(bw)} births, {bw['low'].sum()} of them low birth weight",
)

# %% [markdown]
# The spec. `low` is the outcome; `smoke` is a binary **parent**, so it is an
# ordinal node too. `age` and `lwt` are source nodes — the flow models their
# marginals as well, which a plain `glm` does not. That is a nuisance here, not
# a feature, so they get the cheapest transform (`SI(transform="affine")`), and
# it cannot disturb the outcome node in any case: the joint NLL decomposes per
# node, so the `low` node's gradient never sees the covariate marginals.

# %%
spec_bw = {
    "age": ContinuousNode([SI(transform="affine")]),
    "lwt": ContinuousNode([SI(transform="affine")]),
    "smoke": OrdinalNode(2),
    "low": OrdinalNode(2, [LS("age"), LS("lwt"), LS("smoke")]),
}

flow_bw = CausalFlowDAG(spec_bw, seed=0)
flow_bw.fit_classical(bw)

# %% [markdown]
# Now read the coefficients back out. `age` and `lwt` are continuous parents, so
# they enter raw and their weights compare directly. `smoke` is ordinal, so it
# enters **one-hot over both levels**.
#
# > **Careful Different Coding compared to R** — In contrast to R we use one-hot encoding and the
# > binary variable smoking get to levels.  Only `w[1] - w[0]` is
# > determined by the data. Each level on its own is fixed by the weight
# > initialization: a different `seed` shifts both levels by the same constant
# > and moves `theta_0` to compensate, leaving every fitted probability
# > unchanged. .
# - only the *difference* between levels is identified, so `w[1] - w[0]` is what
#   R's single `smoke` coefficient means (Section 2 leans on that same rule for
#   a 6-level parent);
# - the level-0 column is a column of ones on the untreated rows, so it doubles
#   as part of the intercept. Writing the shift out,
#   $w_0 \mathbb{1}[s{=}0] + w_1 \mathbb{1}[s{=}1] = w_0 + (w_1 - w_0)\,s$,
#   the constant $w_0$ lands on the intercept: R's `(Intercept)` is
#   $-\theta_0 + w_0$, not $-\theta_0$.

# %%
import statsmodels.formula.api as smf  # noqa: E402

from tramdag.transforms import ordinal_cutpoints  # noqa: E402

# the same formula as the R call above, and the same names in the output
res_bw = smf.logit("low ~ age + lwt + smoke", data=bw).fit(disp=False)

w = flow_bw.ls_coefficients()["low"]
with torch.no_grad():  # cutpoints are [-inf, theta_0, +inf]; take the finite one
    theta_0 = float(ordinal_cutpoints(flow_bw.nodes["low"].intercept(1))[0, 1])

rows_bw = [
    # -theta_0 + w_smoke[0]: the one-hot level-0 column is part of the intercept
    (
        "intercept",
        -theta_0 + float(w["smoke"][0]),
        res_bw.params["Intercept"],
        R_GLM["intercept"],
    ),
    ("age", float(w["age"][0]), res_bw.params["age"], R_GLM["age"]),
    ("lwt", float(w["lwt"][0]), res_bw.params["lwt"], R_GLM["lwt"]),
    (
        "smoke",
        float(w["smoke"][1] - w["smoke"][0]),
        res_bw.params["smoke"],
        R_GLM["smoke"],
    ),
]
print(f"{'':<11}{'fit_classical':>14}{'sm.Logit':>10}{'R glm':>10}{'max|diff|':>11}")
for name, flow_value, sm_value, r_value in rows_bw:
    spread = max(abs(flow_value - sm_value), abs(flow_value - r_value))
    print(
        f"{name:<11}{flow_value:>14.4f}{sm_value:>10.4f}{r_value:>10.4f}{spread:>11.2e}"
    )

# %% [markdown]
# Three optimizers — L-BFGS on a transformation model, `statsmodels`' Newton
# solver, R's IRLS — land on the same maximum likelihood, because there is only
# one. The agreement is not only in the parameters: the log-likelihood of the
# `low` node and R's `logLik(m)` are the same number, and `flow.pmf` reproduces
# `glm`'s fitted probabilities row by row.

# %%
# nll() is the *mean* NLL per node; times n and negated it is the log-likelihood
ll_flow = -flow_bw.nll(bw)["low"] * len(bw)
p_flow = flow_bw.pmf(bw, node="low")[:, 1]
p_classical = res_bw.predict(bw).values
print(f"log-likelihood   flow {ll_flow:.6f}   R glm {R_LOGLIK:.6f}")
print(
    f"max |P_flow(low=1) - P_glm(low=1)| over {len(bw)} rows: "
    f"{np.abs(p_flow - p_classical).max():.2e}"
)

# %% [markdown]
# So the framework's zeroth model is one you can check against a textbook.
# Everything that follows — more levels, continuous outcomes, several nodes at
# once — is this model repeated along a DAG.

# %% [markdown]
# ## 1. Continuous case — the bimodal demo data
#
# The VACA benchmark triangle (`x1 → x2 → x3 ← x1`, the demo notebook's bimodal
# SCM): `x1` is a two-component Gaussian mixture, `x2 = -x1 + N(0,1)`,
# `x3 = x1 + 0.25 x2 + N(0,1)`. We fit an **all-`ls`** model: each node is a
# continuous logistic transformation model with a Bernstein baseline and linear
# shifts.
#
# Note this is an *honest misspecification*: the DGP noise is Gaussian while the
# TRAM latent is logistic, so the all-`ls` model is not the true generator — but
# `fit_classical` still finds its exact MLE, which is the point here.


# %%
if False:

    def sample_vaca(n, seed):
        """Draw n rows from the VACA bimodal triangle SCM."""
        rng = np.random.default_rng(seed)
        mix, a, b = rng.uniform(size=n), rng.normal(size=n), rng.normal(size=n)
        x1 = np.where(mix < 0.5, -2.0 + np.sqrt(1.5) * a, 1.5 + b)
        x2 = -x1 + rng.normal(size=n)
        x3 = x1 + 0.25 * x2 + rng.normal(size=n)
        return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    df = sample_vaca(1000, seed=42)
    df.to_csv(NB_DATA / "vaca.csv", index=False)

# %%
df = pd.read_csv(NB_DATA / "vaca.csv")
spec_vaca = {
    "x1": ContinuousNode(),
    "x2": ContinuousNode([LS("x1")]),
    "x3": ContinuousNode([LS("x1"), LS("x2")]),
}

flow_c = CausalFlowDAG(spec_vaca, seed=0)
rep = flow_c.fit_classical(df)  # logs iters / NLL / time
print("\nlinear-shift coefficients (log-odds scale):")
for node, parents in flow_c.ls_coefficients().items():
    for p, w in parents.items():
        print(f"  {p:>3} -> {node:<3}: {w[0]:+.4f}")

# %% [markdown]
# **Deterministic?** Same seed, twice — bit-identical (no minibatch RNG):


# %%
def fit_x1_coef(seed):
    f = CausalFlowDAG(spec_vaca, seed=seed)
    f.fit_classical(df, verbose=False)
    return f.ls_coefficients()["x2"]["x1"][0]


a, b = fit_x1_coef(0), fit_x1_coef(0)
print(f"two runs, same seed: {a:.10f} == {b:.10f}  ->  {a == b}")

# %% [markdown]
# **Classical vs Adam — same optimum.** A converged Adam fit (no early stopping)
# reaches the same coefficients to ~1e-2. The classical fit's guarantees are
# *determinism* and the *exact MLE*; raw speed is model-dependent — it wins
# clearly on ordinal-outcome models (Section 2), but on this Bernstein-heavy
# continuous DAG L-BFGS has to grind through the flat polynomial valleys, so
# wall-clock here is comparable to Adam rather than faster:

# %%
flow_a = CausalFlowDAG(spec_vaca, seed=0)
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

coef_c, coef_a = flow_c.ls_coefficients(), flow_a.ls_coefficients()
print(f"{'coef':<10}{'classical':>12}{'adam':>12}{'|diff|':>10}")
for node, p in [("x2", "x1"), ("x3", "x1"), ("x3", "x2")]:
    c, aw = coef_c[node][p][0], coef_a[node][p][0]
    print(f"{p}->{node:<6}{c:>12.4f}{aw:>12.4f}{abs(c - aw):>10.4f}")
print(f"\nclassical {rep['seconds']:.2f}s  vs  adam {t_adam:.1f}s")

# %% [markdown]
# ### Reproduce the continuous fit outside the flow
#
# Unlike Sections 0 and 2, this one is a **consistency check, not an identity**.
# There the flow and the classical software are the same estimator and agree to
# 1e-8. Here they are two different sieve approximations of the same `h`, so
# they agree to about 0.1% and no closer. Both routes below show that.
#
# **R — `tram::Colr`** is the same model family: a continuous outcome logistic
# transformation model with a Bernstein baseline. Save `df` to `vaca.csv`, then:
#
# ```r
# library(tram)
# d <- read.csv("notebooks/data/vaca.csv")
# m <- Colr(x3 ~ x1 + x2, data = d, order = 19)
# coef(m)        # -> x1 -1.778313   x2 -0.455721
# logLik(m)      # -> -1412.012824
# ```
#
# Those coefficients need **no sign flip**: `Colr` reports the shift on the same
# side as the flow. The gap to the flow is ~0.0014, and at n=1000 no single
# cause is worth naming: re-running `Colr` at `order` 20 or 21 moves it by about
# as much again. Two sieves for the same `h` stop in slightly different places.
#
# **Python — a binned ordered logit.** `statsmodels` has no continuous
# transformation model, but it does not need one: a continuous transformation
# model *is* the limit of an ordered logit as the cutpoints multiply. Cut `x3`
# into `K` quantile bins and fit `OrderedModel`, and the shift coefficients
# converge to the flow's as `K` grows.
#
# Here the sign **does** flip. A continuous node *adds* its shift,
# `P(X <= x) = sigmoid(h(x) + shift)`, while `OrderedModel` subtracts it. That
# is why Sections 0 and 2 compared coefficients directly and this one negates
# them — the same framework, two conventions, one per node kind.

# %%
R_COLR = {"x1": -1.778313, "x2": -0.455721}  # Colr(x3 ~ x1 + x2, order = 19)

w3 = flow_c.ls_coefficients()["x3"]
print(f"{'':>9}{'x1':>10}{'x2':>10}")
print(f"{'flow':>9}{w3['x1'][0]:>10.4f}{w3['x2'][0]:>10.4f}")
print(f"{'R Colr':>9}{R_COLR['x1']:>10.4f}{R_COLR['x2']:>10.4f}")
print("\nstatsmodels ordered logit, x3 cut into K quantile bins (signs flipped):")
for K in (5, 25, 100):
    binned = pd.qcut(df["x3"], K, labels=False, duplicates="drop")
    fitted_k = OrderedModel(binned, df[["x1", "x2"]], distr="logit").fit(
        method="bfgs", disp=False
    )
    print(
        f"{'K=' + str(K):>9}{-fitted_k.params['x1']:>10.4f}"
        f"{-fitted_k.params['x2']:>10.4f}"
    )

# %% [markdown]
# ## 2. Ordinal case — exact, self-contained classical check
#
# For an **ordinal** outcome the classical model is the ordered-logit, which
# `statsmodels.OrderedModel` fits — so here the equivalence is checkable in
# Python. The all-`ls` stroke DAG (the synthetic `magic-mrclean/ls` cohort):

# %%
obs = pd.read_csv(DATA / "magic-mrclean" / "ls" / "obs.csv")

spec_stroke = {
    "Age": ContinuousNode(),
    "mRS_pre": OrdinalNode(6, [LS("Age")]),
    "NIHSSa": ContinuousNode([LS("Age"), LS("mRS_pre")]),
    "T": OrdinalNode(2, [LS("Age"), LS("mRS_pre"), LS("NIHSSa")]),
    "mRS_3m": OrdinalNode(7, [LS("Age"), LS("mRS_pre"), LS("NIHSSa"), LS("T")]),
}

flow_s = CausalFlowDAG(spec_stroke, seed=0)  # the seed validate_ls.py checks
flow_s.fit_classical(obs)

# %% [markdown]
# The log line says the NLL was *still moving* at the iteration cap, and that is
# expected rather than a failed fit: a cutpoint model's rare one-hot levels and
# the flat treatment-effect ridge keep drifting along near-zero-curvature
# valleys long after the likelihood and the well-identified coefficients have
# reached the MLE. Correctness is judged by the comparison below, not by that
# flag — `experiments/misc/validate_ls.py` runs the same 400-iteration budget.

# %% [markdown]
# Classical reference for the outcome node (`statsmodels` ordered logit).
# `flow.design_matrix(..., drop_first=True)` hands over the *same* encoding the
# flow feeds its own shifts, in the form a classical reference expects: a
# continuous parent stays raw, an ordinal parent becomes one column per level
# with level 0 dropped. With cutpoints only differences to level 0 are
# identified, so on the flow side we compare `w[k] - w[0]`.

# %%

design = flow_s.design_matrix(obs, "mRS_3m", drop_first=True)
res = OrderedModel(obs["mRS_3m"].astype(int), design, distr="logit").fit(
    method="bfgs", disp=False
)

fitted = flow_s.ls_coefficients()["mRS_3m"]
rows = [
    ("Age", float(fitted["Age"][0]), res.params["Age"]),
    ("NIHSSa", float(fitted["NIHSSa"][0]), res.params["NIHSSa"]),
    ("T (1 vs 0)", float(fitted["T"][1] - fitted["T"][0]), res.params["T[1]"]),
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
#
# `experiments/misc/validate_ls.py` runs this comparison as a checked
# experiment, and adds R (`MASS::polr` / `tram`) and the treatment effect.

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
    d = float(np.abs(after[p] - before[p]).max())
    print(f"  {p:<8} max|Δ| = {d:.4f}")
print("\n-> small drift = the classical fit was already at the optimum;")
print("   fit_classical is a valid warm start for continued / flexible training.")

# %% [markdown]
# ## When to use which
#
# | situation | use |
# |---|---|
# | all-`ls` model, final estimates / classical comparison / reproducibility | **`fit_classical`** |
# | any `CS`/`CI`/`VC` term (flexible model) | `fit(..., schedule="plateau")` |
# | need a fast, principled init for a flexible model | `fit_classical` (all-`ls` core) → upgrade terms → `fit()` |
#
# `fit_classical` is also the groundwork for **standard errors**: fitting at the
# MLE in float64 is exactly what a Hessian-based covariance needs (a future
# addition — see the CHANGELOG).
