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
# # Heterogeneous treatment effects: the `VC` term
#
# Most causal questions are not "does the treatment help?" but "**whom** does
# it help, and by how much?". A TRAM-DAG answers that with a
# **varying-coefficient** term, which gives one edge of the DAG a treatment
# effect that depends on covariates:
#
# $$\text{shift} \;\mathrel{+}=\; \beta(\text{modifiers})\cdot x_t,
#   \qquad \beta(x) = \beta_0 + b_\Theta(x)$$
#
# `beta0` is the constant part — an interpretable log-odds ratio, exactly as a
# linear shift would give — and `b_Theta` is a deliberately small, **penalized**
# network that bends it per individual.
#
# This notebook builds a DGP whose effect function we know exactly, fits the
# model, and scores the recovered `beta(x)` against the truth. It then shows
# the two things the term exists for that a plain flexible shift does not give:
# a read-out you can plot, and a **propensity-centered** variant that survives
# confounding.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tramdag import CS, LS, VC, CausalFlowDAG, ContinuousNode, I, OrdinalNode
from tramdag.callbacks import RestoreBest

plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## 1. A DGP with a known effect function
#
# Three covariates, a **confounded** treatment (`X1` and `X2` push assignment),
# and an outcome whose conditional is exactly a transformation model:
#
# $$h(y) + g(x) + \beta(x)\,t = U,\qquad U\sim\text{Logistic}(0,1)$$
#
# with $h(y) = 2y$, a nonlinear prognostic part
# $g(x) = \tfrac12 X_1^2 + X_2 - \tfrac12 X_3$, and the effect function
#
# $$\beta(x) = -1 + 0.8\,X_2 - 0.6\,X_3 .$$
#
# `X2` is deliberately *both* a confounder and an effect modifier — the case
# where a naive estimate fails hardest.

# %%
B0, B2, B3 = -1.0, 0.8, -0.6


def true_beta(df):
    """The effect function the model has to recover, on the latent scale."""
    return B0 + B2 * df["X2"].to_numpy() + B3 * df["X3"].to_numpy()


def simulate(n, seed):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (rng.normal(size=n) for _ in range(3))
    t = (rng.logistic(size=n) > -(0.4 * x1 + 0.4 * x2)).astype(float)
    g = 0.5 * x1**2 + x2 - 0.5 * x3
    beta = B0 + B2 * x2 + B3 * x3  # the same expression true_beta reports
    y = (rng.logistic(size=n) - g - beta * t) / 2.0
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "T": t, "Y": y})


train, val, test = simulate(4500, 1), simulate(500, 2), simulate(2000, 3)
print(f"treated fraction: {train['T'].mean():.2f}")
print(
    f"true effect ranges from {true_beta(test).min():+.2f} to {true_beta(test).max():+.2f}"
)

# %% [markdown]
# ## 2. Which covariates modify the effect? (`effect_modifier_scan`)
#
# Before committing to a set of modifiers, ask the data. Fit the **cheap
# all-`ls` model** — seconds, deterministic — and look at the per-observation
# scores of the treatment coefficient. If the effect really were constant,
# those scores would carry no structure in any covariate; a CUSUM statistic per
# covariate turns that into a ranking.


# %%
def cheap_all_ls():
    return {
        "X1": ContinuousNode(),
        "X2": ContinuousNode(),
        "X3": ContinuousNode(),
        "T": OrdinalNode(2, [LS("X1"), LS("X2")]),
        "Y": ContinuousNode([LS("X1"), LS("X2"), LS("X3"), LS("T")]),
    }


screen = CausalFlowDAG(cheap_all_ls(), seed=0)
screen.fit_classical(train)
print(screen.effect_modifier_scan(train, "Y", t="T"))

# %% [markdown]
# `X2` and `X3` are the true modifiers and both flag — but so does `X1`, which
# does **not** enter $\beta(x)$ at all. That is not a bug, and it is worth
# understanding before you trust the shortlist: the statistic measures how
# unstable the *cheap model* is along a covariate, and instability has two
# sources — a coefficient that truly varies, and a **prognostic part the cheap
# model gets wrong**. Here $g$ contains $\tfrac12 X_1^2$, which a linear term
# cannot represent.
#
# The demonstration: re-generate with a *linear* prognostic $X_1$, change
# nothing else, and `X1` drops out of the shortlist.


# %%
def simulate_linear_x1(n, seed):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (rng.normal(size=n) for _ in range(3))
    t = (rng.logistic(size=n) > -(0.4 * x1 + 0.4 * x2)).astype(float)
    g = 0.5 * x1 + x2 - 0.5 * x3  # linear, so the cheap model fits it exactly
    y = (rng.logistic(size=n) - g - (B0 + B2 * x2 + B3 * x3) * t) / 2.0
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "T": t, "Y": y})


well_specified = simulate_linear_x1(4500, 1)
screen2 = CausalFlowDAG(cheap_all_ls(), seed=0)
screen2.fit_classical(well_specified)
print(screen2.effect_modifier_scan(well_specified, "Y", t="T"))

# %% [markdown]
# Read the scan as a **screening** step, then: it shortlists covariates worth
# giving to the effect head, and a flag can also mean "your prognostic part is
# wrong here". Both are things you want to know.

# %% [markdown]
# ## 3. The spec: prognostic part and effect head are separate
#
# `CS("X1", "X2", "X3")` absorbs the prognostic signal — as flexible as you
# like — while `VC("X2", "X3", t="T")` carries the effect. `X2` and `X3` appear
# twice on purpose: prognostically through the shift, and as modifiers through
# the head. Only the treatment named by `t=` owns its edge.

# %%
spec = {
    "X1": ContinuousNode(),
    "X2": ContinuousNode(),
    "X3": ContinuousNode(),
    "T": OrdinalNode(2, [LS("X1"), LS("X2")]),
    "Y": ContinuousNode([CS("X1", "X2", "X3"), VC("X2", "X3", t="T")]),
}
flow = CausalFlowDAG(spec, seed=0)
best = RestoreBest(val)
flow.fit(
    train,
    epochs=300,
    learning_rate=1e-2,
    batch_size=512,
    seed=0,
    after_epoch_callbacks=best,
    after_fit_callbacks=best.restore,
)
print(flow.to_matrix())

# %% [markdown]
# The adjacency view above is the paper's *meta-adjacency matrix*: every edge
# labelled with the term that carries it. `VC` marks the treatment edge and
# `VCm` the modifiers, which is how you can see at a glance that `X2` enters
# `Y` twice.

# %% [markdown]
# ## 4. Reading the effect out
#
# `varying_coef` gives $\beta(x)$ per row. It is **deterministic** and does not
# look at the outcome — it is a property of the fitted model, not of the rows'
# `Y` values.

# %%
beta_hat = flow.varying_coef(test, "Y")
beta_true = true_beta(test)
corr = float(np.corrcoef(beta_hat, beta_true)[0, 1])
beta0 = float(flow.nodes["Y"].shifts["T"].beta0)

print(f"corr(beta_hat, beta_true) = {corr:.3f}")
print(f"beta0 = {beta0:+.3f}   (true constant part {B0:+.1f})")
print(f"mean |error| = {np.abs(beta_hat - beta_true).mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].scatter(beta_true, beta_hat, s=6, alpha=0.35)
lims = [min(beta_true.min(), beta_hat.min()), max(beta_true.max(), beta_hat.max())]
axes[0].plot(lims, lims, "k--", lw=1)
axes[0].set_xlabel(r"true $\beta(x)$")
axes[0].set_ylabel(r"fitted $\hat\beta(x)$")
axes[0].set_title(f"recovery, corr = {corr:.3f}")

order = np.argsort(test["X2"].to_numpy())
axes[1].plot(test["X2"].to_numpy()[order], beta_true[order], "k-", lw=2, label="true")
axes[1].scatter(test["X2"], beta_hat, s=6, alpha=0.3, color="C0", label="fitted")
axes[1].set_xlabel("$X_2$")
axes[1].set_ylabel(r"$\beta$")
axes[1].set_title(r"$\beta$ against a modifier")
axes[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# The scatter against `X2` alone is a band rather than a line, because the true
# effect also depends on `X3` — exactly as it should be.

# %% [markdown]
# ## 5. The read-out is an identity, not a summary
#
# For a binary treatment, $\beta(x)$ *equals* the difference of the abducted
# latents between the two arms, with the outcome held fixed. That is a
# definition the implementation must satisfy, and it does, to float precision:

# %%
u1 = flow.abduct(test.assign(T=1.0), seed=0)["Y"].to_numpy()
u0 = flow.abduct(test.assign(T=0.0), seed=0)["Y"].to_numpy()
print(f"max |beta(x) - (u(T=1) - u(T=0))| = {np.abs(beta_hat - (u1 - u0)).max():.2e}")

# %% [markdown]
# ## 6. Confounding: why `center=True` exists
#
# Everything above had a prognostic part flexible enough to absorb $g$. Real
# models are misspecified, and when the misfit correlates with **treatment
# assignment**, the effect head absorbs the confounding instead of the effect.
#
# The configuration where this bites hardest is deliberately simple: one
# covariate, a strong propensity $e(x)=\sigma(2x)$, a constant true effect
# $\tau = -1$, and a quadratic prognostic part fitted with a linear term.
# `center=True` replaces $t$ by $t - \hat e(x)$ using **cross-fitted**
# (out-of-fold) propensities, so the head sees the part of the treatment that
# the covariates do not explain. Stage 1 — the propensities — is yours: any
# classifier, predicted out of fold, handed to `fit(vc_ehat=)`. Here, five
# classical fits of the treatment spec.

# %%
TAU = -1.0


def confounded(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.logistic(size=n) > -2.0 * x).astype(float)  # strong confounding
    y = (rng.logistic(size=n) - 1.2 * x**2 - TAU * t) / 2.0  # quadratic prognostic
    return pd.DataFrame({"X": x, "T": t, "Y": y})


c_train, c_val, c_test = confounded(5400, 0), confounded(600, 7), confounded(3000, 1000)

# stage 1 of the centered design, the caller's job: cross-fitted P(T=1|X), each
# fold predicted by a treatment model that never saw it (the DML requirement)
fold_id = np.random.default_rng(0).permutation(len(c_train)) % 5
e_oof = np.empty(len(c_train))
for j in range(5):
    t_spec = {
        "X": ContinuousNode([I(transform="affine")]),
        "T": OrdinalNode(2, [LS("X")]),
    }
    proxy = CausalFlowDAG(t_spec, seed=0)
    proxy.fit_classical(c_train.iloc[fold_id != j][["X", "T"]])
    e_oof[fold_id == j] = proxy.pmf(c_train.iloc[fold_id == j], "T")[:, 1]

for center in (False, True):
    spec_c = {
        "X": ContinuousNode([I(transform="affine")]),
        "T": OrdinalNode(2, [LS("X")]),
        # linear prognostic term, though the truth is quadratic
        "Y": ContinuousNode([LS("X"), VC("X", center=center, t="T")]),
    }
    fc = CausalFlowDAG(spec_c, seed=0)
    best = RestoreBest(c_val)
    fc.fit(
        c_train,
        epochs=250,
        learning_rate=1e-2,
        batch_size=512,
        seed=0,
        after_epoch_callbacks=best,
        after_fit_callbacks=best.restore,
        vc_ehat={"Y": {"T": e_oof}} if center else None,
    )
    b = fc.varying_coef(c_test, "Y")
    print(
        f"center={center!s:5s}  mean |beta - tau| = {np.abs(b - TAU).mean():.3f}"
        f"   mean beta = {b.mean():+.3f}  (true tau {TAU:+.1f})"
    )

# %% [markdown]
# Without centering the confounding swallows the effect almost entirely — the
# average estimate lands near zero when the truth is $-1$. With centering it
# recovers most of it, and the mean absolute error falls by roughly a factor of
# four. Centering costs nothing when the prognostic part is adequate, which is
# why it is worth reaching for whenever assignment is not random.

# %% [markdown]
# ## What to take away
#
# | you want | use | read it with |
# |---|---|---|
# | one interpretable effect | `LS("T")` | `ls_coefficients()` |
# | an effect that varies with covariates | `VC(*modifiers, t="T")` | `varying_coef(df, node)` |
# | the same under confounding + a misspecified prognostic part | `VC(..., center=True)` | the same read-out |
# | a shortlist of modifiers before you commit | `effect_modifier_scan` on a cheap all-`ls` fit | its `flag` column, read as screening |
#
# The alternative of writing `CS("T", "X2", "X3")` is equally *expressive* —
# any shift decomposes into arms — but nothing in the likelihood rewards a
# smooth difference between them, so the implied effect is the difference of
# two unregularized networks. On this task class that reduced form measures
# corr ≈ 0.5 against the truth. The `VC` term exists to put the
# regularization where the question is. See
# [`docs/varying-coefficients.md`](../docs/varying-coefficients.md) for the
# semantics and [`docs/scores.md`](../docs/scores.md) for the scan.

# %%
