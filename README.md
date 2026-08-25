# tramdag — Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch

[![Open the demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb)
[![PyPI](https://img.shields.io/pypi/v/tramdag)](https://pypi.org/project/tramdag/)
[![CI](https://github.com/tensorchiefs/tramdag/actions/workflows/ci.yml/badge.svg)](https://github.com/tensorchiefs/tramdag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **Status: beta (0.x), under active development.** The API may change
> between releases until 1.0, so pin a version for reproducibility. Note that
> this README documents the **unreleased 0.4** API: the term constructors
> (`SI`/`CI`/`VC`), `scores`, `varying_coef` and `intercept_contributions` are
> not in `0.3.0` on PyPI. Until 0.4 ships, install from git to follow the docs
> below.

**TRAM-DAGs** model each variable of a structural causal model with a
(transformation-model) flow: one triangular normalizing flow from iid
standard-logistic latents to the observed variables. The triangular adjacency
structure is exactly your causal DAG. Fit it **once** on observational data and answer all
three rungs of Pearl's causal hierarchy — observational (L1), interventional
(L2, the do-operator), and counterfactual (L3, Pearl abduction) — while keeping
**interpretable effects**: every linear-shift coefficient is a log-odds ratio,
exactly as in classical proportional-odds models.

> Beate Sick & Oliver Dürr, *Interpretable Neural Causal Models with TRAM-DAGs*,
> CLeaR 2025 ([arXiv:2503.16206](https://arxiv.org/abs/2503.16206)).
> This repo is the reference implementation (PyTorch, built on
> [zuko](https://zuko.readthedocs.io/stable/)); all of the paper's experiments are
> replicated here with pinned tests.

**5-minute showcase**: the [Colab badge above](https://colab.research.google.com/github/tensorchiefs/tramdag/blob/main/notebooks/demo_tram_dag_colab.ipynb) fits the paper's bimodal benchmark
live and walks L1 → L2 → L3, every answer checked against analytic
ground truth. Further notebooks are available at [`notebooks/`](notebooks/) like the didactic walkthrough of the model:
[`notebooks/intro_tram_dag.py`](notebooks/intro_tram_dag.py).

## Install

```bash
pip install tramdag            # latest release (PyPI)
pip install "git+https://github.com/tensorchiefs/tramdag.git@main"   # dev version (track main)
uv sync                        # or: dev setup from a clone (tests, experiments)
```

Pin the dev install to a commit for reproducibility, e.g. `...tramdag.git@<sha>`.

## 30 seconds of API

```python
from tramdag import CausalFlowDAG, ContinuousNode, OrdinalNode, I, LS, CS

spec = {  # the spec IS the labelled DAG
    "X1": ContinuousNode(),
    "X2": ContinuousNode(I("X1")),
    "T": OrdinalNode(2, LS("X1") + CS("X2")),
    "Y": OrdinalNode(4, I("X1") + CS("X2") + LS("T")),
}
flow = CausalFlowDAG(spec)  # validates acyclicity, builds the flow

# self-stopping training: per-node plateau lr decay + freezing of converged
# nodes (exact, since the per-node NLLs have independent gradients);
# see docs/training-speed.md for benchmarks and the classic two-phase recipe
flow.fit(
    train_df,
    val_df,
    epochs=4000,
    learning_rate=1e-2,
    batch_size=512,
    schedule="plateau",
    plateau_patience=30,
    freeze_patience=120,
)

# all-`ls` model? fit it classically instead: deterministic float64 L-BFGS,
# exact MLE matching statsmodels/R (see docs/fitting.md)
flow.fit_classical(train_df)  # raises on cs/ci/vc specs

flow.log_prob(df)  # L1: joint log-likelihood per row
flow.sample(1000)  # L1: observational sampling
flow.sample(1000, do={"T": 1})  # L2: interventional (graph mutilation)
flow.pmf(df, node="Y", do={"T": 1})  # L2: analytic interventional PMF
flow.density(df, node="X2", grid=grid, do={"X1": 0.5})  # ... and density, continuous nodes

u = flow.abduct(df)  # L3 step 1: latents from observations
cf = flow.sample(do={"T": 1}, u=u)  # L3 steps 2+3: counterfactuals

flow.ls_coefficients()  # interpret: per-edge log-odds-ratios (LS terms)
# per-parent partial effects of an additive complex intercept (centered):
# flow.intercept_contributions("Y", df) on a CI("A", "B", allow_interaction=False)

# heterogeneous treatment effects: a small, penalized effect head beta(x)*T
# (VC term) with a first-class read-out — see docs/varying-coefficients.md
# e.g. CS("X1", "X2") + VC("X1", t="T") ->
# flow.varying_coef("Y", df)               # beta(x): deterministic, y-free

flow.scores(df, node="Y")  # per-observation scores dl_i/dtheta
flow.effect_modifier_scan(df, "Y", t="T")  # which VC modifiers? (CUSUM
# scan from a cheap all-ls fit) — docs/scores.md

flow.save("flow.pt")
flow = CausalFlowDAG.load("flow.pt")
```

## The model in detail: spec → math → networks

Per node, the transformation is additive on the latent (log-odds) scale — one
intercept term `I` plus any number of shifts (notation:
[`docs/notation.md`](docs/notation.md)):

`u = h_ϑ(x) + Σ β·x_pa + Σ g(x_pa) + (β₀ + b_Θ(x_mod))·x_t`

| term | math | what gets built | interpretability |
|---|---|---|---|
| `I()` / bare `I` / omitted | `h_ϑ(x)` — constant ϑ | `SimpleIntercept`: one free parameter vector, no network | the baseline transform |
| `I("A")` | `h_ϑ(a)(x)` — ϑ bends with the parent | `ComplexIntercept`: MLP `[8, 8] → n_params` | the parent reshapes the whole distribution; no single coefficient |
| `I("A","B")` (default `allow_interaction=True`) | `h_ϑ(a,b)(x)` | **one joint** MLP over both parents — they interact in ϑ | maximal flexibility |
| `I("A","B", allow_interaction=False)` | `h_ϑ(a)+ϑ(b)(x)` | one MLP **per parent**, parameter vectors summed in coefficient space | per-parent partial effects via `flow.intercept_contributions` |
| `LS("A")` | `β·a` | `Linear(width, 1)`, no bias — **one parameter per feature column** (one for a continuous parent, `levels` for a one-hot ordinal) | `exp(β)` is an odds ratio |
| `CS("A")` | `g(a)`, additive | `ComplexShift`: MLP `[64, 128, 64] → 1` | plot `g` |
| `CS("A","B")` | `g(a,b)` — joint | one MLP over the concatenated features | interaction *in the shift* |
| `CS("A") + CS("B")` | `g₁(a) + g₂(b)` | two MLPs, scalars added | GAM-style, each effect plottable |
| `VC("A","B", t="T")` | `(β₀ + b_Θ(a,b))·x_t` | scalar `β₀` + zero-initialised penalized MLP `[16] → 1`; the treatment value multiplies, it never enters the net | `β₀` ≈ constant effect, `flow.varying_coef` reads `β(x)` |

A node takes **at most one `I` term with parents** — a term list is therefore
always purely additive on the latent scale, and interactions exist only
*inside* a term. Lists and `+` sums are interchangeable:
`[LS("A"), CS("B")]` ≡ `LS("A") + CS("B")`.

Two knobs on the terms:

- **`transform=` on `I`** picks the basis of `h_ϑ` for a continuous node —
  `"bernstein"` (default, 20 coefficients, tails extrapolate with the boundary
  slope), `"spline"` (monotone RQ spline, 23 params at `bins=8`, fixed tail
  slope) or `"affine"` (2 params: the latent is exactly logistic). Ordinal
  nodes have no basis: their intercept is the cutpoint vector,
  `P(x ≤ k) = σ(ϑ_k − shift)`.
- **`units=` on `I`/`CS`/`VC`** sizes the term's network, e.g. `units=[16]`
  for one hidden layer of 16 neurons. The defaults match the PyTorch
  reference this package grew out of ([buehlpa/TramDag](https://github.com/buehlpa/TramDag),
  `tram_models.py`), **not** the TRAM-DAG paper's own R nets — so a
  replication sets `units=` and `activation=` explicitly, as the configs in
  `experiments/paper/` do.

Feature widths: a continuous parent enters raw (1 column), an ordinal parent
one-hot (`levels` columns). Abduction is exact for continuous nodes and
truncated-logistic for ordinal ones, so `flow.sample(u=flow.abduct(df))`
reproduces `df` exactly / level-exactly.

There are two ways to fit the model: a stochastic deep learning optimizer (`fit`) and a 2nd order optimization like in the classical statistical models (`fit_classical`). The latter is more efficient for all-`ls` models (each node-conditional is then a classical transformation model). For more details, see the [`docs/fitting.md`](docs/fitting.md) file.

## Validation

Two layers, deliberately separate.

**The framework's own test suite** ([`tests/`](tests/)) measures the library
against three inline data-generating processes it carries itself, so it needs no
research code to run. Its strongest claim is an equality, not a similarity: an
all-`ls` outcome node *is* an ordered-logit model, so the flow's MLE must match
`statsmodels` `OrderedModel` on the same design matrix — it does, and the
varying-coefficient acceptance bars (effect recovery, propensity centering) are
pinned the same way. What the tests guarantee, and how each ground truth is
obtained, is documented in [`tests/README.md`](tests/README.md).

**The paper replications** ([`experiments/`](experiments/)) are separate: one
self-contained script per dataset, with its hyperparameters in a sibling YAML
file and its expected results committed under each area's `ground_truth/`. A
dedicated workflow runs them and compares.

| experiment | paper | demonstrates |
|---|---|---|
| `triangle.py` (`linear-ls`, `linear-cs`, `atan-cs`, `sin-cs`) | §6.1, C.3 | LS coefficient recovery (β = 2, −0.2, +0.3), CS curve ≡ −f(x₂) for non-monotone f |
| `triangle_mixed.py` (`linear-ls`, `exp-cs`) | §6.2 | mixed data L1/L2 + the C.4 odds-ratio check (OR ≈ 7.4) |
| `vaca.py` | §5.1–5.2 | the bimodal L1 case a default CNF misses; L2 `p(x₃ \| do(x₂))` against analytic means |
| `carefl.py` | §5.3 | L3 counterfactual curves vs **analytic** truth |
| `validate_ls.py` | — | flow ≡ `statsmodels` ≡ R `MASS::polr` to ~4 decimals, on a frozen synthetic cohort with a known true effect |

Sign note: ordinal shifts are *subtracted* here but *added* in the paper, so
fitted ordinal weights are the paper's with flipped sign (each `truth.json`
records both conventions). A figure-by-figure account of what is reproduced,
and what is not (the competing CNF/NSF baselines), is in
[`experiments/paper/PAPER_COVERAGE.md`](experiments/paper/PAPER_COVERAGE.md).

**Training speed** — schedules, per-node freezing, L-BFGS and device benchmarks:
[`docs/training-speed.md`](docs/training-speed.md).

## Testing policy
See the [`tests/README.md`](tests/README.md) file for more details.

## Layout

```
src/tramdag/            spec.py transforms.py conditioners.py flow.py
                        scores.py utils.py        <- the framework, and nothing else
tests/                  unit tests, identities, acceptance bars, three inline DGPs
experiments/            research code, one directory per area, each self-contained:
                          paper/       the replications + generators + frozen data
                          benchmarks/  training-speed and machine measurements
                          misc/        the classical-MLE validation
                        paper/misc: <name>.py + <name>.yaml, data/,
                        ground_truth/, tests/, results/ (benchmarks reads the
                        other areas' data and pins no ground truth)
notebooks/              four executed examples: didactic intro, Colab demo,
                        additive-vs-joint intercepts, varying coefficients
docs/                   code-map.md (every class/function + all knobs),
                        fitting.md, notation.md, training-speed.md,
                        varying-coefficients.md, scores.md
```

Implementation conventions (latent-scale signs, raw/one-hot parent encoding,
log-space ordinal likelihood, seeding) are documented in
[`CLAUDE.md`](CLAUDE.md) and pinned by tests.

## Citation

If you use `tramdag`, please cite the method paper:

```bibtex
@inproceedings{sick2025tramdag,
  title     = {Interpretable Neural Causal Models with TRAM-DAGs},
  author    = {Sick, Beate and D{\"u}rr, Oliver},
  booktitle = {Proceedings of the 4th Conference on Causal Learning and Reasoning (CLeaR)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {275},
  year      = {2025},
}
```

<!-- For the stroke application (and the `magic-mrclean` cohort design) additionally:

```bibtex
@article{duerr2026stroke,
  title  = {Estimating Individualized Treatment Effects in Acute Ischemic Stroke
            with Causal Transformation Models (TRAM-DAG): A Multi-Centre
            Observational Study with External RCT Validation},
  author = {D{\"u}rr, Oliver and Herzog, Lisa and B{\"u}hler, Pascal and
            Wegener, Susanne and Sick, Beate},
  journal = {arXiv preprint arXiv:2606.12623},
  year   = {2026},
}
``` -->
