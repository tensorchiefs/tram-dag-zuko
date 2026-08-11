# Case study: the stroke ITE analysis

Background and full detail for the `magic-mrclean` experiments — the public,
synthetic counterpart of the clinical analysis in Dürr, Herzog, Bühler, Wegener
& Sick, *Estimating Individualized Treatment Effects in Acute Ischemic Stroke
with Causal Transformation Models (TRAM-DAG)*
([arXiv:2606.12623](https://arxiv.org/abs/2606.12623)). The clinical MAGIC /
MR CLEAN data is private and never part of this repo.

## The DAG and the pipeline

Five nodes with all forward edges: `Age → mRS_pre → NIHSSa → T → mRS_3m`.
Treatment `T` (thrombectomy) is confounded by age and stroke severity in the
observational cohort; the estimand is the ATE of `T` on a good outcome,
`P(mRS_3m ≤ 2 | do(T))`, averaged over the (younger) trial population — computed
analytically from the outcome node's interventional PMF, no Monte Carlo.

```bash
cd experiments
uv run python sim_flow.py nl                  # headline storyline (synthetic, default)
uv run python sim_flow.py nl                  # all-ls vs flexible, known truth
# the single-config runners (all_ls_flow, nihss6_flow) are parked in stale/
uv run python validate_ls.py                  # all-ls flow vs statsmodels MLE
# counterfactual_demo.py is parked in stale/
uv run python all_ls_long.py                  # 4x-longer convergence check
```

Experiments default to the public synthetic source `magic-mrclean/nl`; passing
`magic` (clinical cohort, NIHSSa ≥ 6, N = 1275) only works inside the original
paper monorepo. Every run uses the same 80/10/10 split (`random_state=42`) and
writes plots, per-patient interventional PMFs (`rct_predicted_proba.csv`) and a
checkpoint to `results/<name>/`.

## The synthetic cohort (`magic-mrclean`)

A hand-specified SCM in the flow's own model family (logistic latents) shaped
like the stroke study — same schema, realistic marginals, and **known ground
truth** the real data cannot provide: the true ATE, true individual
counterfactuals (shared-latent pairs), and a provably misspecified baseline.
Two variants:

- **`ls`** — every parent effect is a linear shift: each node-conditional is
  exactly a classical proportional-odds model, so flow, R reference and truth
  must coincide. The clean equivalence baseline.
- **`nl`** — adds an accelerating age effect on disability, a heterogeneous
  treatment effect `tau(Age)` that fades in the elderly, and reduced treatment
  probability for the very old. An all-`ls` model must collapse `tau(Age)` to a
  constant and is therefore biased; a flexible (`ci`/`cs`) flow can recover the
  truth. The trial cohort (`rct.csv`) enrolls younger than the observational
  cohort, as real trials do — exactly the extrapolation that breaks the
  misspecified model.

Known-truth recovery (seed 7), ATE on P(good) over the trial population:

| nl variant | ATE | vs true **+0.104** |
|---|---|---|
| naive observational contrast | +0.303 | confounded (overstates 2.9×) |
| all-`ls` flow | +0.076 | undershoots (misses the age-varying effect) |
| flexible (`ci`/`cs`) flow | +0.101 | **recovers the truth** |

This reproduces the clinical-data finding in miniature (flexible +0.108 vs
all-`ls` +0.054) but against a *known* answer.

**R cross-check.** [`data/magic-mrclean/fit_ls.R`](../data/magic-mrclean/fit_ls.R)
refits the all-`ls` DAG node-by-node in classical R (`MASS::polr` / `tram::Colr`
/ `glm`); its committed `ref_ls/` outputs let `tests/test_simulations.py` assert
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

- **The all-ls flow IS the classical MLE** when trained to convergence without
  early stopping (the default, `restore_best=False`): on the synthetic `ls`
  cohort its outcome-node coefficients match `statsmodels` *and* R `polr` to 4
  decimals (Age 0.0526, NIHSSa 0.1630, T −0.9424; ATE +0.1429 vs +0.1428) — see
  `experiments/validate_ls.py` and
  `test_simulations.py::test_all_ls_flow_is_exact_mle`. The earlier
  +0.057-vs-+0.055 residual on the clinical data was exactly the early-stopping
  effect (see CHANGELOG).
- **Flexible (`ci`/`cs`) models are different**: their MLE *overfits the
  observational confounding*, so they need `restore_best=True` (early-stopping
  regularization) to recover the causal effect — confirmed on the synthetic `nl`
  cohort, where the flexible MLE undershoots (+0.076) but the early-stopped fit
  recovers the true ATE (+0.10), with a lower validation NLL. `run_experiment`
  defaults `restore_best` per style accordingly (off for all-`ls`, on for
  flexible).
- The treatment effect is **weakly identified** in this observational cohort:
  the likelihood around the T-coefficient is nearly flat, and refits drift to
  the 80%-split optimum of ≈ +0.054 — the original +0.108 is a
  near-likelihood-equivalent solution, not a sharper optimum. Both flow results
  sit inside the established acceptance band [+0.03, +0.14]; matching the band,
  not the point estimate, is the meaningful check.

## Counterfactual demo

`experiments/counterfactual_demo.py` (a capability beyond the original scripts)
abducts the latents of the 128 held-out test patients, verifies exact factual
reconstruction, and predicts each patient's outcome under the opposite
treatment (`results/<name>/counterfactuals_test.csv`,
`plots/counterfactual_mrs3m.png`).
