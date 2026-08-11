# Case study: the stroke ITE analysis

This document gives the background and the full detail for the `magic-mrclean`
experiments. These experiments are the public, synthetic counterpart of the
clinical analysis in Dürr, Herzog, Bühler, Wegener & Sick, *Estimating
Individualized Treatment Effects in Acute Ischemic Stroke with Causal
Transformation Models (TRAM-DAG)*
([arXiv:2606.12623](https://arxiv.org/abs/2606.12623)). The clinical MAGIC /
MR CLEAN data is private. It is never part of this repo.

## The DAG and the pipeline

The DAG has five nodes with all forward edges: `Age → mRS_pre → NIHSSa → T → mRS_3m`.
In the observational cohort, age and stroke severity confound the treatment
`T` (thrombectomy). The estimand is the ATE of `T` on a good outcome,
`P(mRS_3m ≤ 2 | do(T))`, averaged over the (younger) trial population. The
pipeline computes this ATE analytically from the interventional PMF of the
outcome node. It does not use Monte Carlo.

```bash
cd experiments
uv run python sim_flow.py nl                  # headline storyline (synthetic, default)
uv run python sim_flow.py nl                  # all-ls vs flexible, known truth
# the single-config runners (all_ls_flow, nihss6_flow) are parked in stale/
uv run python validate_ls.py                  # all-ls flow vs statsmodels MLE
# counterfactual_demo.py is parked in stale/
uv run python all_ls_long.py                  # 4x-longer convergence check
```

Experiments default to the public synthetic source `magic-mrclean/nl`. The
`magic` source (clinical cohort, NIHSSa ≥ 6, N = 1275) works only inside the
original paper monorepo. Every run uses the same 80/10/10 split
(`random_state=42`). Each run writes plots, per-patient interventional PMFs
(`rct_predicted_proba.csv`), and a checkpoint to `results/<name>/`.

## The synthetic cohort (`magic-mrclean`)

This cohort is a hand-specified SCM in the model family of the flow itself
(logistic latents). It has the same schema as the stroke study and realistic
marginals. It also has **known ground truth** that the real data cannot
provide: the true ATE, true individual counterfactuals (shared-latent pairs),
and a provably misspecified baseline. The cohort has two variants:

- **`ls`** — every parent effect is a linear shift. Each node-conditional is
  exactly a classical proportional-odds model. Therefore the flow, the R
  reference, and the truth must coincide. This variant is the clean
  equivalence baseline.
- **`nl`** — this variant adds an accelerating age effect on disability and a
  heterogeneous treatment effect `tau(Age)` that fades in the elderly. It also
  reduces the treatment probability for the very old. An all-`ls` model must
  collapse `tau(Age)` to a constant and is therefore biased. A flexible
  (`ci`/`cs`) flow can recover the truth. The trial cohort (`rct.csv`) enrolls
  younger patients than the observational cohort, as real trials do. This
  extrapolation is exactly what breaks the misspecified model.

The table shows the known-truth recovery (seed 7), the ATE on P(good) over the
trial population:

| nl variant | ATE | vs true **+0.104** |
|---|---|---|
| naive observational contrast | +0.303 | confounded (overstates 2.9×) |
| all-`ls` flow | +0.076 | undershoots (misses the age-varying effect) |
| flexible (`ci`/`cs`) flow | +0.101 | **recovers the truth** |

This result reproduces the clinical-data finding in miniature (flexible +0.108
vs all-`ls` +0.054), but against a *known* answer.

**R cross-check.** [`data/magic-mrclean/fit_ls.R`](../data/magic-mrclean/fit_ls.R)
refits the all-`ls` DAG node-by-node in classical R (`MASS::polr` / `tram::Colr`
/ `glm`). Its committed `ref_ls/` outputs let `tests/test_simulations.py` assert
flow ≡ R fit (outcome-node coefficients and ATE) without an R installation.

## Clinical-data results (context — not reproducible from this repo)

| model | ATE on P(good) | source |
|---|---|---|
| MR CLEAN RCT (ground truth) | **+0.135** [+0.057, +0.213] | Berkhemer et al. 2015 |
| this flow, nihss6 config | **+0.063** | `experiments/stale/nihss6_flow.py` (parked) |
| this flow, all-ls | **+0.057** | `experiments/stale/all_ls_flow.py` (parked) |
| classical proportional-odds MLE, same 80% split | +0.055 | `experiments/validate_ls.py` |
| original TRAM-DAG `md_dag_ls` (all-ls) | +0.054 | paper monorepo |
| original TRAM-DAG `nihss6` | +0.108 (seed 2: +0.092) | paper monorepo |
| classical MLE, full data | +0.082 | paper monorepo |

Reading notes:

- When the all-ls flow trains to convergence without early stopping (the
  default, `restore_best=False`), **it IS the classical MLE**. On the
  synthetic `ls` cohort, its outcome-node coefficients match `statsmodels`
  *and* R `polr` to 4 decimals (Age 0.0526, NIHSSa 0.1630, T −0.9424, and ATE
  +0.1429 vs +0.1428). See `experiments/validate_ls.py` and
  `test_simulations.py::test_all_ls_flow_is_exact_mle`. The earlier residual
  of +0.057 vs +0.055 on the clinical data was exactly the early-stopping
  effect (see CHANGELOG).
- **Flexible (`ci`/`cs`) models are different**: their MLE *overfits the
  observational confounding*. Therefore they need `restore_best=True`
  (early-stopping regularization) to recover the causal effect. The synthetic
  `nl` cohort confirms this: the flexible MLE undershoots (+0.076), but the
  early-stopped fit recovers the true ATE (+0.10), with a lower validation
  NLL. `run_experiment` defaults `restore_best` per style accordingly (off for
  all-`ls`, on for flexible).
- The treatment effect is **weakly identified** in this observational cohort:
  the likelihood around the T-coefficient is nearly flat. Refits drift to the
  80%-split optimum of ≈ +0.054. The original +0.108 is a
  near-likelihood-equivalent solution, not a sharper optimum. Both flow
  results sit inside the established acceptance band [+0.03, +0.14]. The
  meaningful check is the match to the band, not to the point estimate.

## Counterfactual demo

`experiments/counterfactual_demo.py` is a capability beyond the original
scripts. It abducts the latents of the 128 held-out test patients and confirms
exact factual reconstruction. It then predicts the outcome of each patient
under the opposite treatment (`results/<name>/counterfactuals_test.csv`,
`plots/counterfactual_mrs3m.png`).
