# `magic-mrclean` — synthetic stroke cohort with known ground truth

A fully synthetic dataset shaped like the MAGIC / MR CLEAN stroke study, used as the
**public, reproducible** default for the tramdag experiments. It contains **no patient
data**: every row is generated from a hand-specified structural causal model (SCM), so
the true interventional effects are known exactly. It is a drop-in substitute for the
private clinical data — identical column schema, dtypes and ranges — so the same
experiment and analysis scripts run on either by changing one string.

Generator: [`src/tramdag/simulations/magic_mrclean.py`](../../src/tramdag/simulations/magic_mrclean.py).

## DAG and schema

```
Age ──▶ mRS_pre ──▶ NIHSSa ──▶ T ──▶ mRS_3m      (every forward edge present)
```

| column   | type  | meaning                          | range  |
|----------|-------|----------------------------------|--------|
| `Age`    | float | age at stroke (years, 1 decimal) | 20–103 |
| `mRS_pre`| int   | pre-stroke disability (mRS)      | 0–5    |
| `NIHSSa` | float | admission stroke severity (NIHSS)| 6–42   |
| `T`      | int   | thrombectomy (0/1)               | 0/1    |
| `mRS_3m` | int   | 3-month outcome (mRS)            | 0–6    |

"Good outcome" = `mRS_3m <= 2`. Latents are iid **standard logistic** (the TRAM base
distribution), so the data lives natively in the family the flow fits.

## Two variants

| variant | each parent effect | role |
|---------|--------------------|------|
| `ls/`   | **linear shift** only | every node-conditional is a classical proportional-odds / linear-transformation model, so an all-`ls` flow, the R reference, and the true SCM all coincide — the clean equivalence baseline |
| `nl/`   | linear **plus three mild non-linearities (★)** | the all-`ls` model is misspecified and **biased low** for the true ATE; a flexible (`ci`/`cs`) flow recovers it |

The three ★ terms in `nl` (and only those) differ from `ls`:

1. **accelerating disability** — `mRS_pre` gains a `+0.18·relu((Age−73)/10)²` term;
2. **age² severity** — `NIHSSa` gains a `+0.20·relu((Age−73)/10)²` shift;
3. **heterogeneous, age-fading treatment effect** — the outcome shift uses
   `τ(Age) = −0.85 + 0.85·(1 − σ((78−Age)/6))`, i.e. full benefit (≈ −0.85) in the
   young that fades to ≈ 0 in the very old; `nl` also smoothly withholds treatment
   from the very old. (`ls` uses a constant `τ = −0.85`.)

**Trial inclusion.** The `obs.csv` (observational) and `rct.csv` (randomized) draws
use *different* covariate populations: the trial enrols a **younger** cohort (age
location shifted down ~9 years), as real stroke trials do. Combined with the
heterogeneous `τ(Age)` in `nl`, this is what makes the all-`ls` model biased: fit on
the older observational cohort it learns an old-weighted average effect, then
**undershoots** the larger true effect in the younger trial population. The flexible
model learns `τ(Age)` and extrapolates correctly. In the `ls` variant `τ` is constant,
so the population shift causes no bias and every model still agrees with the truth.

## Files per variant

| file | contents |
|------|----------|
| `obs.csv` | N=1275 **observational** draw — `T` follows the confounded assignment mechanism (the MAGIC analog); full-age cohort |
| `rct.csv` | N=500 **randomized** draw — `T ~ Bernoulli(0.5)` independent of covariates (the MR CLEAN analog); younger trial-inclusion cohort |
| `truth.json` | Monte-Carlo (n=5·10⁵) ground truth: `true_ate` (do-effect of `T` on P(good), averaged over the **trial** population — `ate_population: "rct"`), `p_good_do_T0/1`, and `naive_obs_diff` (the confounded observational contrast, for comparison) |

## Ground truth (seed 7)

True ATE is the do-effect on P(good) averaged over the trial (younger) population, to
match what the experiments score on `rct.csv`.

| variant | true ATE | naive obs. diff | all-`ls` flow | flexible flow |
|---------|----------|-----------------|---------------|----------------|
| `ls`    | **+0.132** | +0.255 | +0.13 ✓ | +0.13 ✓ |
| `nl`    | **+0.104** | +0.303 | **+0.076** (biased low) | **+0.101** (recovers) |

Two things to read off:

1. The huge gap between the **naive observational contrast** (+0.26 / +0.30) and the
   true ATE makes the **do-operator demonstrably necessary** — both flows correct it.
2. On `nl`, the all-`ls` flow **undershoots** the true ATE while the flexible (`ci`/`cs`)
   flow recovers it — echoing the real-data finding (TRAM-DAG `nihss6` +0.108 vs
   `md_dag_ls` +0.054). On `ls` (constant effect) both models and the truth coincide.

## Regenerating

The CSVs are **frozen** (committed) so Python and R see byte-identical data. To
regenerate (e.g. a new seed → bump to a new folder rather than overwriting):

```bash
uv run python -m tramdag.simulations.magic_mrclean --out data/magic-mrclean --seed 7
```

## R reference

[`fit_ls.R`](fit_ls.R) fits the all-`ls` DAG node-by-node with classical R
(`MASS::polr` / `tram` / `glm`) and writes `<variant>/ref_ls/{coefficients,ate}.csv`.
Run it per variant (requires `tram`, `MASS`):

```bash
Rscript fit_ls.R ls
Rscript fit_ls.R nl
```

The committed `ref_ls/` outputs let `tests/test_simulations.py` check the Python flow
against the R fit without R installed.
