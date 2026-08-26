# Paper replication — protocol, hyperparameters and results, per experiment

The eight variants under `experiments/paper/` replicate the TRAM-DAG paper
(Sick & Dürr, CLeaR 2025, arXiv:2503.16206) against its own R code
(`tensorchiefs/tram-dag`). This page lists, for each experiment, the DGP, the
model, every hyperparameter with its source, what deviates from the paper and
why, and the numbers: what the paper reports, what the previous protocol of
this repository measured, and what the current 1:1 protocol measures.

Measured 2026-08-25 on `feat/followups` (seeds: DGP 42, init 7, shuffle 0);
CI run 32878435001 reproduced every number within the committed tolerances.
"Previous" is the protocol before that branch: a 90/10 split of one draw,
batch 512, the fit restarted in chunks (`fit_in_chunks`), and for VACA/CAREFL
a second "polish" fit at a lower rate — a repository recipe, not the paper's.
Its numbers come from the ground-truth files of that revision (`73f9f39`;
`{max}` bounds were 2.5× the measurement).

The paper states four training numbers — n = 40000, 500 epochs, Adam,
Bernstein order 20 — and shows results as figures. Where it gives no number,
the "paper" column names the figure and what it shows.

## What is common to all eight

| item | reference | here |
|---|---|---|
| latent | standard logistic, shifts added on the continuous scale, subtracted for ordinal `P(Y<=k) = sigmoid(theta_k - shift)` | same (tests pin both signs) |
| Bernstein basis | `len_theta` unconstrained coefficients, `to_theta` softplus-cumsum, domain = train min/max → [0, 1], tangent-linear extrapolation outside | zuko Bernstein, `n_coeffs` unconstrained (zuko ties two control points on, so the free-parameter count matches), domain = train 5 %/95 % quantiles → [−5, 5], linear extrapolation — **deviation D1** |
| init | Keras glorot | torch default — **D2** (glorot measured worse on VACA: 0.19 vs 0.026) |
| optimizer | Keras Adam, eps 1e-7 | torch Adam, eps 1e-8 |
| seeds | R scripts run unseeded | every seed is a repo choice |
| calibrated start | none | `marginal_init=False` hard-coded in `helpers.fit_paper` (framework default is True) |
| intercept output layer | Keras dense with bias | bias-free — **D3** (a bias is redundant with the first Bernstein coefficient) |

D1 is the only deviation with a measured effect (see CAREFL below).

## Triangle, continuous (`triangle.py`) — paper Sec. 6.1, App. C.3

**DGP** (`summerof24/triangle_structured_continous.R`): x1 ~ 0.5 N(0.25, 0.1²) + 0.5 N(0.73, 0.05²);
h(x2 | x1) = 5 x2 + 2 x1 = u2; h(x3 | x1, x2) = 0.63 x3 − 0.2 x1 − f(x2) = u3, u ~ logistic.
f: `linear` −0.3 x, `atan` 0.75 atan(5 (x + 0.12)), `sin` 2 sin(3x) + x.
True weights in the flow's convention: β12 = +2, β13 = −0.2, β23 = +0.3 (linear only).

**Model**: `x1: SI`, `x2: SI + LS(x1)`, `x3: SI + LS(x1) + {LS(x2) | CS(x2)}`, Bernstein.
CS net = reference `hidden_features_CS = c(2, 25, 25, 2)`, sigmoid (the ReLU line
in `create_param_net` is commented out).

| hyperparameter | paper / R code | previous | now |
|---|---|---|---|
| train / validation | `dgp(40000)` / `dgp(40000)`, two draws | 36000 / 4000, one draw split | 40000 / 40000, two draws |
| epochs | 500 | 500 (in chunks of 10, Adam restarted each chunk) | 500, one continuous run |
| lr | 0.001 (`optimizer_adam()` default) | 0.001 | 0.001 |
| batch | 32 (Keras `fit()` default) | 512 | 32 |
| `len_theta` / `n_coeffs` | 20 | 20 | 20 |
| schedule / early stop | none / none, final weights | none / none | none / none |
| coefficient read-out | after every epoch (Keras loop) | at chunk boundaries | `fit(epoch_callback=)`, every epoch |

**Results**

| variant | metric | paper | previous | now |
|---|---|---|---|---|
| linear-ls | β12 / β13 / β23 | Fig. 14: trajectories at 2 / −0.2 / 0.3 | 1.983 / −0.145 / 0.285 | 1.987 / −0.170 / 0.282 |
| linear-cs | β13; max \|ĝ − (−f)\| on [−1, 1] | Fig. 17: fitted CS is a straight line | −0.151; 0.507 | −0.175; 0.116 |
| atan-cs | β13; cs max err | Fig. 7 right / 15 / 16: CS on −f, coefficients at 2 / −0.2 | −0.149; 0.229 | −0.169; 0.085 |
| sin-cs | β13; cs max err | Fig. 18: CS follows the non-monotone f, saturating at the grid ends | −0.185; 2.29 | −0.199; 1.016 |
| all | val NLL x3 | — | 2.4603–2.4731 | 2.4607–2.4764 |

β13 at −0.17 instead of −0.2: x1 has sd 0.254, so SE(β13) ≈ 0.036 at n = 40000
— inside one SE. The `sin-cs` error of 1.0 is the reference architecture's
capacity: the two-unit bottleneck saturates at both ends of the grid exactly
as in the paper's Fig. 18.

## Triangle, mixed (`triangle_mixed.py`) — paper Sec. 6.2, App. C.4, App. B

**DGP** (`triangle_structured_mixed.R`): x1, x2 as above; x3 ordinal with four
levels, cutpoints θ = (−2, 0.42, 1.02) (from the R code — the paper does not
state them), level = #{k : u3 > θ_k + 0.2 x1 + f(x2)}; f: `linear` −0.3 x,
`exp` 0.5 exp(x). The paper adds the ordinal shift, the flow subtracts it, so
the fitted weights are β13 = −0.2, β23 = +0.3.

**Model**: `x1: SI`, `x2: SI + LS(x1)`, `x3: OrdinalNode(4, LS(x1) + {LS(x2) | CS(x2)})`.
CS net = reference `c(2, 2, 2, 2)`, sigmoid.

| hyperparameter | paper / R code | previous | now |
|---|---|---|---|
| train / validation | `dgp(40000)` / `dgp(10000)` | 36000 / 4000 split | 40000 / 10000, two draws |
| epochs, lr, batch, schedule | 500, 0.001, 32, none | 500 (chunks of 10), 0.001, 512 | 500 continuous, 0.001, 32, none |
| `n_coeffs` (x1, x2) | 20 | 20 | 20 |
| odds-ratio check (App. C.4) | odds(x2 ≤ −1) under do(x1 += 1), theory e² ≈ 7.39 | 40000 rows, seed 99 | same |
| counterfactual PMF (App. B) | Fig. 10 explains why point counterfactuals fail for the ordinal node | 2000 rows × 200 draws | same |

**Results**

| variant | metric | paper | previous | now |
|---|---|---|---|---|
| linear-ls | β12 / β13 / β23 | Fig. 19: 2 / −0.2 / 0.3 | 1.983 / −0.268 / 0.322 | 1.987 / −0.246 / 0.319 |
| linear-ls | odds ratio predicted / DGP | e² ≈ 7.39 | 7.26 / 7.19 | 7.29 / 7.19 |
| linear-ls | CF PMF TV vs analytic; P(true level) flow / analytic / mode bound | — (App. B qualitative) | 0.084; 0.714 / 0.728 / 0.806 | 0.044; 0.718 / 0.728 / 0.806 |
| exp-cs | β13; cs max err | Fig. 20: distributions match | −0.227; 0.885 | −0.207; 0.126 |
| exp-cs | CF PMF TV; P(true level) | — | 0.075; 0.924 / 0.921 | 0.022; 0.919 / 0.921 |

## VACA / CNF benchmark (`vaca.py`) — paper Sec. 5.1–5.2, App. C.1

**DGP** (Sanchez-Martin et al. 2022, App. E.1): x1 ~ 0.5 N(−2, 1.5) + 0.5 N(1.5, 1);
x2 = −x1 + N(0, 1); x3 = x1 + 0.25 x2 + N(0, 1). Gaussian noise — outside the
logistic-latent family, so the all-`CI` flow has to fit it. Analytic target:
E[x3 | do(x2 = a)] = −0.25 + 0.25 a; the paper's Fig. 5 shows a ∈ {−3, −2, 0}.

**Model**: `x1: SI`, `x2: CI(x1)`, `x3: CI(x1, x2)`, Bernstein `n_coeffs = 31`
(reference M = 30, `len_theta = 31`), nets `dense(10, tanh) → dense(100, tanh) → dense(31)`
(`comparison/utils.R::make_model`).

| hyperparameter | R code | previous | now |
|---|---|---|---|
| train / validation | nTrain 2500 / `dgp(5000)` | 18000 / 2000 split | 2500 / 5000, two draws |
| epochs | 10000 (`Figure_Triangle_Linear_Bimodal.R`) | 400 in chunks of 50, then 120 "polish" | 10000, one run |
| lr | 0.001 | 0.01, polish 0.001 | 0.001 |
| batch | full batch (one `apply_gradients` per epoch) | 512 | 2500 = n_train |
| schedule | `update_learning_rate`: ReduceLROnPlateau on the summed val NLL, factor 0.1, patience 50, min_lr 1e-7, strict `<` | none | `plateau`, same numbers, **per node** — D4 |
| input scaling | `scale_df`: everything min-max to [0, 1] | raw | `net_input_scaling: minmax` (network inputs; targets D1) |
| n_compare | — | 50000 | 50000 |

**Results** — the check is the flow's error against the analytic mean, not a pinned flow value.

| metric | paper | previous | now |
|---|---|---|---|
| \|E[x3 \| do(x2 = −3)] − (−1.0)\| | Fig. 5: densities overlap | 0.037 | 0.026 |
| \|E[x3 \| do(x2 = −2)] − (−0.75)\| | Fig. 5 | — (do(−1) was scored: 0.012) | 0.034 |
| \|E[x3 \| do(x2 = 0)] − (−0.25)\| | Fig. 5 | 0.010 | 0.077 |
| sd(x1) flow vs analytic 2.0767 | Fig. 4: bimodal x1 fitted (the default CNF fails) | 2.074 | error 0.0077 |
| val NLL x3 | — | 1.4356 | 1.4351 |

Under the exact R protocol with **raw** inputs the errors were 0.731 / 0.426 /
0.017 — the tanh nets saturate (40 % of the rows have |x1| > 2, 43 % |x2| > 2).
Min-max scaled inputs, as in `scale_df`, give the row above. The other
deviations were each measured under the R protocol: Bernstein on min/max
(D1 off) 0.134 at do(−3), glorot init 0.19, no plateau rule 0.56 / 0.35 / 0.14
with val NLL 1.459 — the plateau rule is what stops the full-batch overfit, so
per-node vs global (D4) is not where a gap comes from.

## CAREFL benchmark (`carefl.py`) — paper Sec. 5.3, App. C.2

**DGP** (Khemakhem et al. 2021): x1, x2 ~ Laplace(0, 1/√2); x3 = x1 + 0.5 x2³ + ε;
x4 = −x2 + 0.5 x1² + ε, ε ~ Laplace(0, 1/√2). Counterfactuals are analytic by
noise abduction. Observation `x_obs`: noise (2, 1.5, 1.4, −1) → (2, 1.5, 5.0875, −0.5)
in the SCM's units — the paper prints (2, 1.5, 0.81, −0.28) because CAREFL's
runner divides x3, x4 by their sample sds (6.01, 1.91). **Before this branch
the repository used the printed values as raw coordinates**, a 4σ-off point.
Queries: x3^cf | do(x2 = α) and x4^cf | do(x1 = α), α ∈ [−3, 3].

**Model**: `x1, x2: SI`, `x3, x4: CI(x1, x2)`, Bernstein `n_coeffs = 31`, the
same `make_model` nets as VACA.

| hyperparameter | R code (`carefl_fig5.r`) | previous | now |
|---|---|---|---|
| train / validation | 2500 / `dgp(5000)` | 18000 / 2000 split | 2500 / 5000 |
| epochs | 7000 | 300 in chunks of 50 + 100 polish | 7000 |
| lr, batch, schedule, input scaling | 0.001, full batch, plateau 0.1/50/1e-7, `scale_df` | 0.01→0.001, 512, none, raw | 0.001, 2500, plateau (per node), `minmax` |
| scoring | the single `x_obs`, curves over α (Fig. 6) | + 300 held-out rows at α ∈ {−1.5, 0, 1.5} | same |

**Results**

| metric | paper | previous (wrong x_obs) | now |
|---|---|---|---|
| Fig. 6 max \|x3^cf error\| over α at `x_obs` | Fig. 6: flow tracks the DGP curve | 1.64 | 1.52 (at α = 3, x3 ≈ 17) |
| Fig. 6 max \|x4^cf error\| | Fig. 6 | 0.38 | 0.47 |
| held-out CF MAE x3, α = −1.5 / 0 / 1.5 | — | 0.109 / 0.053 / 0.095 | 0.177 / 0.065 / 0.123 |
| held-out CF MAE x4, α = −1.5 / 0 / 1.5 | — | 0.078 / 0.059 / 0.085 | 0.117 / 0.059 / 0.115 |
| val NLL x3 / x4 | — | 1.376 / 1.345 | 1.395 / 1.373 |

The previous protocol trained on 7× more rows (18000), which is why its
held-out MAEs are lower; they are not comparable to the paper's nTrain = 2500.
With D1 switched off (Bernstein domain on train min/max, `RANGE_Q = 0`) under
the R protocol: x3 MAE 0.083 / 0.038 / 0.050 (better), x4 0.143 / 0.083 / 0.154
(worse), val NLL 1.444 / 1.476 (worse). The quantile domain stays the default;
a `range_quantile` knob would make R's choice available.

## Runtime and the epochs

Triangle: 500 epochs × 1250 steps of batch 32 — 3.2 s per epoch single-process
at 2–4 torch threads (5.5 s at 32 threads, the step is overhead-bound), 42–69 min
per variant on the 2-core CI runners; VACA 4.7 min, CAREFL 6.6 min; the
workflow's ten jobs run in parallel, 70 min wall against a 300-min cap.

A shorter budget was measured as the candidate deviation for CI runtime: at
100 epochs the LS coefficients are unchanged (they settle after ~40 epochs,
Fig. 14) but the complex shift is not — cs max err 0.235 (linear-cs, 0.116 at
500) and 0.395 (mixed exp-cs, 0.126); at 250 epochs 0.112 and 0.203. The
paper's 500 stay.

## Repository choices the paper does not state

Seeds; the validation draw sizes for the triangle scripts (the R code's
`test`); `n_compare`; `n_heldout = 300` and the α set {−1.5, 0, 1.5} scored
next to the single `x_obs`; the do(x2) set {−3, −2, 0} read off Fig. 5; the
mixed cutpoints from the R code; the odds-ratio sample (40000 rows) and the
counterfactual-PMF sample (2000 rows × 200 draws).
