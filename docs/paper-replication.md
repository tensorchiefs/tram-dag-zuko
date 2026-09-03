# Paper replication — protocol, hyperparameters and results, per experiment

The eight variants under `experiments/paper/` replicate the TRAM-DAG paper
(Sick & Dürr, CLeaR 2025, arXiv:2503.16206) against its own R code
(`tensorchiefs/tram-dag`). This page lists, for each experiment, the DGP, the
model, every hyperparameter with its source, what deviates from the paper and
why, and the numbers: what the paper reports, what the previous protocol of
this repository measured, and what the current 1:1 protocol measures.

Measured 2026-08-26 on `refactor/lean-fit` (seeds: DGP 42, init 7, shuffle 0)
with the exact reference protocol — global plateau rule and Keras init. On
2026-09-01 the *optimization* was re-tuned for CI runtime (fewer epochs, lr
scaled to match; data, model and init untouched); on **2026-09-02 the
VACA and CAREFL cuts were reverted to the reference protocols 1:1** after a
visual pass against paper Figs. 5/6 (the triangle cuts stay): the reference
runs cut CAREFL's Fig. 6 curve errors ~30% and improved VACA's do(x2=0),
with every bound holding — ground truths re-pinned from those runs. The
tuning campaign, with what was tried and what failed, is in
[the 2026-09-01 tuning round](#the-2026-09-01-tuning-round) below. The
2026-08-25 numbers of `feat/followups` (per-node plateau, torch init; CI run
32878435001) are kept in the "R 1:1, torch init" column where they differ.
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
| Bernstein basis | `len_theta` unconstrained coefficients, `to_theta` softplus-cumsum, domain → [0, 1] with tangent-linear extrapolation outside; the domain comes from the train **5 %/95 % quantiles in the triangle scripts** (`quantile(..., c(0.05, 0.95))`, the min/max lines commented out) and from **min/max** in the comparison scripts (`scale_df`) | zuko Bernstein, `n_coeffs` unconstrained (zuko ties two control points on, so the free-parameter count matches), domain = train 5 %/95 % quantiles → [−5, 5], linear extrapolation. Triangle: a match up to the reparametrization [0,1] vs [−5,5], the tail rule and order 21 vs 19. VACA/CAREFL: quantiles instead of min/max — **deviation D1** |
| init | triangle scripts: `LinearMasked` layers with Keras `random_normal` (N(0, 0.05²)) on weights and biases, the LS `beta` layer included; comparison scripts: `layer_dense` default, glorot-uniform weights and zero biases | `init: normal` (triangle) and `init: glorot` (VACA/CAREFL), `CausalFlowDAG(init=)`; torch's default init remains the framework default and was the decisive deviation under the full-batch protocol, see VACA |
| optimizer | Keras Adam, eps 1e-7 | torch Adam, eps 1e-8 — measured: no effect (VACA identical to four digits) |
| seeds | triangle scripts unseeded (`SEED = -1`); comparison scripts `dgp(..., seed=42)` on R's RNG | not replayable in torch, so every seed is a repo choice (42 for the DGP is kept as a nod to the reference) |
| calibrated start | none | `flow.calibrate(train, marginal_init=False)` in `helpers.fit_paper` (framework default is True) |
| intercept output layer | Keras dense with bias | bias-free — **D3** (the same function class; the bias adds a constant to all unconstrained coefficients) |
| plateau rule (VACA/CAREFL) | `update_learning_rate`: one optimizer, reduce when the summed validation NLL has not improved for 50 epochs (strict `<`), factor 0.1, min 1e-7 | torch `ReduceLROnPlateau(patience=49, threshold=0, threshold_mode="abs", factor=0.1, min_lr=1e-7)` on the summed `history["val"]` entry (fit computes it) — the same rule, verified against torch's source; both keep it since 2026-09-02 (VACA's 2026-09-01 drop went with the reverted epoch cut) |

D1 (VACA/CAREFL only) and D3 remain; both were measured (below). The per-node
plateau approximation of 2026-08-25 (D4) is gone with the lean `fit`. Two
further repo choices are CAREFL's: a fresh 2500-row draw with a separate
validation draw where the R run used CAREFL's file with `val = train`, and raw
units where the R run sd-standardized x3/x4.

## Triangle, continuous (`triangle.py`) — paper Sec. 6.1, App. C.3

**DGP** (`summerof24/triangle_structured_continous.R`): x1 ~ 0.5 N(0.25, 0.1²) + 0.5 N(0.73, 0.05²);
h(x2 | x1) = 5 x2 + 2 x1 = u2; h(x3 | x1, x2) = 0.63 x3 − 0.2 x1 − f(x2) = u3, u ~ logistic.
f: `linear` −0.3 x, `atan` 0.75 atan(5 (x + 0.12)), `sin` 2 sin(3x) + x.
True weights in the flow's convention: β12 = +2, β13 = −0.2, β23 = +0.3 (linear only).

**Model**: `x1: SI`, `x2: SI + LS(x1)`, `x3: SI + LS(x1) + {LS(x2) | CS(x2)}`, Bernstein.
CS net = reference `hidden_features_CS = c(2, 25, 25, 2)`, sigmoid (the ReLU line
in `create_param_net` is commented out). **Corrected 2026-09-02**: the vector
reads as in/out dims around the hidden stack **(25, 25)** — the earlier
literal `[2, 25, 25, 2]` reading put a 2-sigmoid bottleneck on the input and
could not reproduce paper Fig. 18 (sin) at any protocol; (25, 25) lands on
Figs. 17/18 (linear cs err 0.13 → 0.025, sin 1.22 → 0.24 on the paper's
[−1, 0] window). Mixed likewise: c(2, 2, 2, 2) → hidden (2, 2).

| hyperparameter | paper / R code | previous | now |
|---|---|---|---|
| train / validation | `dgp(40000)` / `dgp(40000)`, two draws | 36000 / 4000, one draw split | 40000 / 40000, two draws |
| epochs | 500 | 500 (in chunks of 10, Adam restarted each chunk) | **300** (`linear-cs`: 500), one continuous run — the CI deviation, floors below |
| lr | 0.001 (`optimizer_adam()` default) | 0.001 | **0.004**, the CI deviation |
| batch | 32 (Keras `fit()` default) | 512 | **256**, the CI deviation |
| `len_theta` / `n_coeffs` | 20 | 20 | 20 |
| schedule / early stop / init | none / none, final weights / random_normal | none / none / torch | none / none / `init: normal` |
| coefficient read-out | after every epoch (Keras loop) | at chunk boundaries | `fit(callbacks=)`, every epoch |

**Results**

| variant | metric | paper | previous | paper protocol, `init: normal` (batch 32 / lr 0.001 / 500 epochs) | CI config (batch 256 / lr 0.004, 300 epochs — `linear-cs` 500), the pinned ground truth |
|---|---|---|---|---|---|
| linear-ls | β12 / β13 / β23 | Fig. 14 trajectories; App. C.3 text: 1.98 / −0.21 / 0.26 | 1.983 / −0.145 / 0.285 | 1.987 / −0.170 / 0.282 | 1.981 / −0.161 / 0.281 |
| linear-cs | β13; max \|ĝ − (−f)\| on [−1, 1] | Fig. 17: fitted CS is a straight line | −0.151; 0.507 | −0.173; 0.122 (glorot init: 0.110, torch: 0.116) | −0.178; 0.088 |
| atan-cs | β13; cs max err | Fig. 7 right / 15 / 16: CS on −f; text: β12 = 2.07, β13 = −0.203 | −0.149; 0.229 | −0.168; 0.059 (glorot: 0.080, torch: 0.085) | −0.168; 0.108 (hidden (25,25), 2026-09-02) |
| sin-cs | β13; cs max err | Fig. 18: CS follows the non-monotone f on x2 ∈ [−1, 0] | −0.185; 2.29 | −0.195; 1.10 (glorot: 0.997, torch: 1.016) | −0.168; 0.240 on the paper's [−1, 0] window (hidden (25,25), 2026-09-02) |
| all | \|E[x3 \| do(x1 = −1)] flow − DGP\| | Figs. 16/17: histograms overlap | — | 0.09–0.14 | 0.08–0.11 |
| all | val NLL x3 | — | 2.4603–2.4731 | 2.4607–2.4764 | 2.4606–2.4764 |

β13 at −0.16 to −0.18 instead of −0.2: x1 has sd 0.254, so SE(β13) ≈ 0.036 at
n = 40000 — within ~1 SE. The old `sin-cs` error of ~1.0 was the misread
2-unit-bottleneck net, not "reference capacity" (retracted 2026-09-02): with
hidden (25, 25) the fit lands on Fig. 18 including its small −1-endpoint
deviation, and the paper's figure never plots beyond [−1, 0].

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
| epochs, lr, batch, schedule, init | 500, 0.001, 32, none, random_normal | 500 (chunks of 10), 0.001, 512, torch | **350 (`exp-cs`) / 200 (`linear-ls`)** continuous, none, `init: normal`; **lr 0.004 (`linear-ls`: 0.002) / batch 256** — the CI deviations, floors below |
| `n_coeffs` (x1, x2) | 20 | 20 | 20 |
| odds-ratio check (App. C.4) | odds(x2 ≤ −1) under do(x1 += 1), theory e² ≈ 7.39 | 40000 rows, seed 99 | same |
| counterfactual PMF (App. B) | Fig. 10 explains why point counterfactuals fail for the ordinal node | 2000 rows × 200 draws | same |

**Results**

| variant | metric | paper | previous | paper protocol, `init: normal` (500 epochs) | CI config (batch 256; `exp-cs` 350 epochs / lr 0.004, `linear-ls` 200 / 0.002), pinned |
|---|---|---|---|---|---|
| linear-ls | β12 / β13 / β23 | Fig. 19: 2 / −0.2 / 0.3 | 1.983 / −0.268 / 0.322 | 1.987 / −0.246 / 0.319 | 1.978 / −0.258 / 0.333 |
| linear-ls | odds ratio predicted / DGP | e² ≈ 7.39; C.4 text: 7.74, CI [7.16, 8.38] | 7.26 / 7.19 | 7.29 / 7.19 | 7.23 / 7.19 |
| linear-ls | CF PMF TV vs analytic; P(true level) flow / analytic / mode bound | — (App. B qualitative) | 0.084; 0.714 / 0.728 / 0.806 | 0.044; 0.718 / 0.728 / 0.806 | 0.050; 0.716 / 0.728 / 0.806 |
| exp-cs | β13; cs max err | Fig. 20: distributions match | −0.227; 0.885 | −0.207; 0.142 (glorot: 0.071, torch: 0.126) | −0.200; 0.156 |
| exp-cs | CF PMF TV; P(true level) | — | 0.075; 0.924 / 0.921 | 0.023; 0.917 / 0.921 | 0.024; 0.920 / 0.921 |

## VACA / CNF benchmark (`vaca.py`) — paper Sec. 5.1–5.2, App. C.1

**DGP** (Sanchez-Martin et al. 2022, App. E.1): x1 ~ 0.5 N(−2, 1.5) + 0.5 N(1.5, 1);
x2 = −x1 + N(0, 1); x3 = x1 + 0.25 x2 + N(0, 1). Gaussian noise — outside the
logistic-latent family, so the all-`CI` flow has to fit it. Analytic target:
E[x3 | do(x2 = a)] = −0.25 + 0.25 a. The paper's Sec. 5.2 text says a ∈ {−3, −2, 0}; its Fig. 5 panels and the R code (`vaca_triangle.r`) intervene at a ∈ {−3, −1, 0} — the code is followed (the frozen `data/vaca/truth.json` holds the same three).

**Model**: `x1: SI`, `x2: CI(x1)`, `x3: CI(x1, x2)`, Bernstein `n_coeffs = 31`
(reference M = 30, `len_theta = 31`), nets `dense(10, tanh) → dense(100, tanh) → dense(31)`
(`comparison/utils.R::make_model`).

| hyperparameter | R code | previous | now |
|---|---|---|---|
| train / validation | nTrain 2500 / `dgp(5000)` | 18000 / 2000 split | 2500 / 5000, two draws |
| epochs | 10000 (`Figure_Triangle_Linear_Bimodal.R`, the sourcing script — not in our copy of the R code, so EPOCHS/M/nTrain for VACA rest on that reading) | 400 in chunks of 50, then 120 "polish" | **10000**, one run — the reference, 1:1 (the 2026-09-01 4800-epoch cut was reverted 2026-09-02) |
| lr | 0.001 | 0.01, polish 0.001 | **0.001** — the reference |
| batch | full batch (one `apply_gradients` per epoch) | 512 | 2500 = n_train |
| schedule | `update_learning_rate`: ReduceLROnPlateau on the summed val NLL, factor 0.1, patience 50, min_lr 1e-7, strict `<` | none | **the same rule** (restored 2026-09-02 with the reference epochs; it fires ~epoch 9050 and freezes an all-bounds point) |
| input scaling | `scale_df`: everything min-max to [0, 1] | raw | `input_transform: minmax` on the CI terms (network inputs; targets D1) |
| n_compare | — | 50000 | 50000 |

**Results** — the check is the flow's error against the analytic mean, not a pinned flow value.

| metric | paper | previous | R 1:1, torch init, per-node plateau (08-25) | R 1:1, glorot, global plateau (08-26) | now: the reference 10000 @ 0.001 + plateau — the pinned ground truth |
|---|---|---|---|---|---|
| \|E[x3 \| do(x2 = −3)] − (−1.0)\| | Fig. 5: densities overlap | 0.037 | 0.026 | 0.097 | 0.096 |
| \|E[x3 \| do(x2 = −1)] − (−0.5)\| | Fig. 5 | 0.012 | — (do(−2) was scored: 0.034) | 0.088 | 0.080 |
| \|E[x3 \| do(x2 = 0)] − (−0.25)\| | Fig. 5 | 0.010 | 0.077 | 0.019 | 0.022 |
| sd(x1) flow vs analytic 2.0767 | Fig. 4: bimodal x1 fitted (the default CNF fails) | 2.074 | error 0.0077 | error 0.032 | 2.036, error 0.040 |
| val NLL x3 | — | 1.4356 | 1.4351 | 1.4496 | 1.4427 |

The 08-25 and 08-26 columns are not the same three interventions: the do(x2)
set moved from the paper text's {−3, −2, 0} to the R code's {−3, −1, 0} (see
the DGP paragraph above), so only do(x2 = −3) and do(x2 = 0) compare directly.
At the earlier grid and with glorot the errors were 0.098 / 0.159 / 0.026 at
this seed and 0.268 / 0.119 / 0.005 and 0.217 / 0.031 / 0.021 at init seeds 8
and 9 — the spread quoted below.

How the gap was traced, one change at a time under the exact protocol —
10000 epochs at lr 0.001 with the plateau rule, before the 2026-09-01 epoch
cut (inputs min-max scaled unless stated; do(x2) errors at the then-current
grid −3 / −2 / 0):

| variant | error | reading |
|---|---|---|
| raw parents into the nets | 0.731 / 0.426 / 0.017 | the tanh nets saturate: 40 % of the rows have \|x1\| > 2, 43 % \|x2\| > 2 |
| per-node plateau, torch init | 0.026 / 0.034 / 0.077 | the 2026-08-25 approximation |
| global plateau (R's rule), torch init | 0.523 / 0.334 / 0.129 | the summed NLL keeps improving through x1 while x3 overfits |
| … without any plateau | 0.562 / 0.354 / 0.142 | the rule is what stops the full-batch overfit |
| … Adam eps 1e-7 | 0.523 / 0.334 / 0.129 | no effect |
| … Bernstein on train min/max (D1 off) | 0.154 / 0.012 / 0.062 | helps, not the cause |
| … glorot init | 0.035 / 0.006 / 0.007 | the cause (a different random draw than the config's seed 7) |
| … glorot + min/max | 0.035 / 0.056 / 0.057 | |

Under the exact protocol the result is **seed-sensitive at the off-manifold
point** do(x2 = −3): 0.03–0.27 over four init draws, 0.005–0.026 at do(x2 = 0)
(measured under the 10000-epoch protocol; the paper shows one run's
densities). The committed bound is 2.5× the seed-7 measurement of the
reference 10000-epoch run (the 2026-09-01 4800-epoch cut was reverted
2026-09-02), and this table is the honest picture.

## CAREFL benchmark (`carefl.py`) — paper Sec. 5.3, App. C.2

**The Fig. 6 x4 dip, root-caused 2026-09-03**: the flow's parabola bottom
sits ~0.55 (raw) below the analytic truth near α = 0. It is finite-sample
tail-quantile variance at the reference's own n_train = 2500 — invariant to
the training protocol (3000 vs 7000 epochs), to val=train vs held-out, and
to init seeds (0.586/0.601/0.610 at seeds 7/8/9); the observed noise sits at
the Laplace 12% quantile, and the fitted conditional's 12% quantile errs
+0.27 at the sparse observation region (x1 = 2, ~190 nearby rows) against
−0.19 in the dense middle — the obs-anchored counterfactual inherits the
difference. Fresh 2500-row draws move it (err@α0 0.54/0.31/0.43 at dgp
seeds 42/43/44), so the paper's single run sits inside the band.

**DGP** (Khemakhem et al. 2021): x1, x2 ~ Laplace(0, 1/√2); x3 = x1 + 0.5 x2³ + ε;
x4 = −x2 + 0.5 x1² + ε, ε ~ Laplace(0, 1/√2). Counterfactuals are analytic by
noise abduction. Observation `x_obs`: noise (2, 1.5, 1.4, −1) → (2, 1.5, 5.0875, −0.5) in the
SCM's units; the R code's `xObs.csv` is that point divided by CAREFL's sample
sds (6.0104, 1.9114): (2, 1.5, 0.8465, −0.2616). The paper prints (2, 1.5, 0.81,
−0.28), slightly off it (CAREFL's own text). **Before `feat/followups` the
repository used the printed values as raw coordinates**, a 4σ-off point. The R
run also scores in sd-standardized units, so its Fig. 6 y-axis is 6× ours for
x3 — the `fig6_max_abs_err` entries are in raw units.
Queries: x3^cf | do(x2 = α) and x4^cf | do(x1 = α), α ∈ [−3, 3].

**Model**: `x1, x2: SI`, `x3, x4: CI(x1, x2)`, Bernstein `n_coeffs = 31`, the
same `make_model` nets as VACA.

| hyperparameter | R code (`carefl_fig5.r`) | previous | now |
|---|---|---|---|
| train / validation | CAREFL's own `data/CAREFL_CF/X.csv` (`USE_EXTERNAL_DATA = TRUE`), x3/x4 sd-standardized, **`val = train`** — the plateau rule watched the training NLL | 18000 / 2000 split | 2500 rows drawn from the same SCM / a separate `dgp(5000)` draw, raw units — **repo choice** |
| epochs | 7000 | 300 in chunks of 50 + 100 polish | **7000** — the reference, 1:1 (the 2026-09-01 3000-epoch cut was reverted 2026-09-02) |
| lr, batch, schedule, input scaling, init | 0.001, full batch, plateau 0.1/50/1e-7, `scale_df`, glorot | 0.01→0.001, 512, none, raw, torch | **0.002** (the CI deviation), 2500, the same global rule, `minmax`, glorot |
| scoring | the single `x_obs`, curves over α (Fig. 6) | + 300 held-out rows at α ∈ {−1.5, 0, 1.5} | same |

**Results**

| metric | paper | previous (wrong x_obs) | R 1:1, torch init, per-node plateau (08-25) | R 1:1, glorot, global plateau, 7000 @ 0.001 (08-26) | now: 3000 epochs @ lr 0.002 — the pinned ground truth |
|---|---|---|---|---|---|
| Fig. 6 max \|x3^cf error\| over α at `x_obs` | Fig. 6: flow tracks the DGP curve | 1.64 | 1.52 (at α = 3, x3 ≈ 17) | 1.87 (at α = 3) | 2.68 (at α = 3) |
| Fig. 6 max \|x4^cf error\| | Fig. 6 | 0.38 | 0.47 | 0.59 | 0.77 |
| held-out CF MAE x3, α = −1.5 / 0 / 1.5 | — | 0.109 / 0.053 / 0.095 | 0.177 / 0.065 / 0.123 | 0.181 / 0.065 / 0.142 (seeds 8, 9: 0.158–0.171 / 0.069–0.084 / 0.136–0.154) | 0.208 / 0.060 / 0.128 |
| held-out CF MAE x4, α = −1.5 / 0 / 1.5 | — | 0.078 / 0.059 / 0.085 | 0.117 / 0.059 / 0.115 | 0.204 / 0.134 / 0.162 (0.172–0.178 / 0.113–0.129 / 0.145–0.160) | 0.124 / 0.097 / 0.122 |
| val NLL x3 / x4 | — | 1.376 / 1.345 | 1.395 / 1.373 | 1.403 / 1.419 | 1.405 / 1.393 |

(Historical, reverted 2026-09-02.) Five of the six held-out MAEs are better at 3000 @ 0.002 than at the
reference's 7000 @ 0.001 (only x3 at α = −1.5 is worse, 0.208 vs 0.181); the
Fig. 6 single-point errors are larger — one noisy yardstick, which is why the
held-out rows are scored next to it.

The previous protocol trained on 7× more rows (18000), which is why its
held-out MAEs are lower; they are not comparable to the paper's nTrain = 2500.
Under the exact protocol the CAREFL numbers are stable across init seeds
(spread ≈ 0.03) and the x4 query is 1.5–2× worse than under the per-node
approximation — the reference's rule, faithfully applied, is not the best
optimizer for this DGP, and the paper's Fig. 6 is a visual match either way.
With D1 switched off (Bernstein domain on train min/max, `RANGE_Q = 0`, under
the per-node protocol): x3 MAE 0.083 / 0.038 / 0.050 (better), x4 0.143 / 0.083
/ 0.154 (worse), val NLL 1.444 / 1.476 (worse). The quantile domain stays the
default; a `range_quantile` knob would make R's choice available.

## Runtime and the epochs

The reference protocol's cost, the motivation for every deviation here —
triangle: 500 epochs × 1250 steps of batch 32, 3.2 s per epoch single-process
at 2–4 torch threads (5.5 s at 32 threads, the step is overhead-bound), 42–69 min
per variant on the 2-core CI runners; VACA 355 s, CAREFL 338 s (2026-08-31); the
workflow's ten jobs run in parallel, 70 min wall against a 300-min cap.

The first deviation taken for CI runtime (2026-08-26) was the triangle
**batch size and learning rate**: batch 256 at lr 0.004 instead of the paper's
batch 32 at lr 0.001, 8× fewer optimizer steps per epoch. This is the earlier
selection grid, kept as the record of that choice — it ran at the paper's 500
epochs with glorot init, against the then-committed ground-truth bands (cs max
err unless noted):

| batch / lr / epochs | linear-ls β13 | linear-cs | atan-cs | mixed exp-cs (TV) | steps vs paper |
|---|---|---|---|---|---|
| **32 / 0.001 / 500 (paper)** | −0.170 | 0.110 | 0.080 | 0.071 (0.019) | 1× |
| 128 / 0.001 / 500 | −0.170 | 0.170 | 0.041 | 0.126 (0.023) | 1/4 |
| 128 / 0.004 / 500 | −0.178 | 0.160 | 0.094 | 0.070 (0.014) | 1/4 |
| 128 / 0.004 / 250 | −0.170 | 0.147 | 0.070 | 0.131 (0.035) | 1/8 |
| **256 / 0.004 / 500 (CI)** | −0.171 | 0.132 | 0.073 | 0.072 (0.017) | 1/8 |
| 256 / 0.008 / 500 | −0.181 | 0.163 | 0.113 | 0.073 (0.017) | 1/8 |
| 512 / 0.010 / 500 | −0.169 | 0.183 | 0.104 | 0.119 (0.015) | 1/16 |

Batch 256 / lr 0.004 is the only row that keeps every cs error within ~0.02 (linear-cs: 0.022) of the
paper protocol; larger batches or rates bend the misspecified linear-cs curve
(0.16–0.18) and lr 0.008 hurts atan-cs. At 500 epochs and this batch/lr, CI
run 32974872751 took 7–11 min per triangle job instead of 42–69, the workflow
about 12 min wall instead of 70. The grid ran with glorot init; the committed
configs use `init: normal` (the triangle scripts' initializer). The
paper-protocol numbers stay in this document as the reference; the ground
truth is pinned from the CI config, which since 2026-09-01 also cuts the
epochs — the round below.

## The 2026-09-01 tuning round

The frame (repo decision 2026-09-01): the *data, model and init* stay the reference's;
allowed were dropping the network-input scaling, trying sigmoid/relu activations,
and tuning lr / batch size / epochs for CI runtime. Outcome: the triangle
scripts already ran raw parents + sigmoid (their reference does), so they
comply by construction and only their epochs moved; VACA and CAREFL **keep
tanh + min-max network inputs (`input_transform: minmax`)** because every alternative was tried and
measurably fails. Every tuned variant's ground truth was re-pinned from its
final run (centers to the run, `{max}` at 2.5x; `fit_seconds` waits for the
next CI measurement) — **the bounds and pins quoted in this section are the
pre-repin 2026-08-26 values the round was measured against**, and re-pinning
loosened several. The full decision trail is in
each config's YAML header.

| experiment | was | now |
|---|---|---|
| triangle `linear-ls` / `atan-cs` / `sin-cs` | 500 epochs | 300 epochs |
| triangle `linear-cs` | 500 epochs | 500 epochs (kept) |
| triangle-mixed `exp-cs` | 500 epochs | 350 epochs |
| triangle-mixed `linear-ls` | 500 epochs @ lr 0.004 | 200 epochs @ lr 0.002 |
| vaca | 10000 full-batch epochs @ lr 0.001, plateau | reverted 2026-09-02: back to the reference 10000 @ 0.001 + plateau |
| carefl | 7000 full-batch epochs @ lr 0.001, plateau | reverted 2026-09-02: back to the reference 7000 @ 0.001 |
| validate_ls `adam` (misc) | phases 4000/2000/1000 @ 1e-2/1e-3/1e-4 | 800/700/500, batch 256 kept |

**Triangle** (batch 256 / lr 0.004 / sigmoid / `init: normal` / raw parents
all unchanged): at 300 epochs every metric holds its bound; the floor is
real — 150 epochs fails atan's do(x1) bound (0.235 > 0.206). `linear-cs`
keeps 500: at 300 its fitted cs curve visibly flattens past |x2| > 0.5, max
err 0.140 vs 0.088 — the DGP line spans only ±0.3 and the sigmoid net's slow
tail convergence is half of that. relu is ruled out: the reference's 2-unit
input bottleneck dies (atan: β13 +0.53, cs err 1.42).

**Triangle-mixed**: the cs variant's slow parts are the 2-unit sigmoid CS net
and the do(x1) mean — cs max err 0.344 / 0.177 / 0.156 and do err 0.033 /
0.060 / 0.022 at 100 / 250 / 350 epochs (0.107 / 0.016 at 500), so 100 failed
the then-current cs bound (0.344 > 0.2686) and 250 the do bound
(0.060 > 0.0398); `exp-cs` trains 350.
`linear-ls` has no CS net to wait for and trains 200 at lr 0.002, which reads
the weakly identified β13/β23 out slightly closer to the 500-epoch pins than
lr 0.004 does (β23 0.333 vs 0.337, pin 0.306); at 100 epochs β13 sits 57%
into its tolerance (−0.271 vs the pinned −0.243). relu dies at the sd-0.05
normal init (cs err 0.859, β13 −0.055).

**VACA** — the instrumented finding: probing the reference trajectory (10000
@ 0.001, plateau) shows the do(x2) errors descend until epoch ~6000–8500
(best 0.005–0.10), then creep back up; the plateau anneal happens to fire at
~9050 and freezes 0.101 / 0.099 / 0.030 (an instrumented re-run; the small
offset from the 08-26 pins 0.097 / 0.088 / 0.019 is run-to-run sampling of
the do-means) — the reference protocol's quality IS its anneal landing
inside that window. At lr 0.002 the same smooth
full-batch descent runs 2× compressed with an all-bounds window at epochs
~4350–5000 (binding edges: x1's marginal std enters its 2.0766 ± 0.08 band at
~4320, the do(x2=+0) error exits after ~5000); 4800 sits inside it, every
margin ≥ 2×, at under half the reference's wall clock. The plateau rule is
dropped because on the compressed trajectory the summed validation NLL keeps
strictly improving past the window, so it cannot fire inside it; lr 0.004 is
out because x1's std clock does not compress with lr (1.978 at its do-window,
epochs ~2050–2350). Rejected, both measured: relu + raw parents wanders
0.17–0.45 on do(x2=−3), holding the bound only at isolated epochs (tanh on
raw parents saturates outright, 0.731); minibatch stepping (batch 256, even
with minmax + tanh) converges with a systematic −0.11 common-mode offset of
all three do-means — 2.4× the do(x2=+0) bound — while the observational fit
stays perfect.

**CAREFL** (full batch + plateau kept): 3000 @ 0.002 replaced 7000 @ 0.001 (reverted 2026-09-02 — the reference protocol cuts the Fig. 6 curve errors ~30%);
five of the six held-out counterfactual MAEs come out better than the
7000-epoch pins. Rejected: raw parents — sigmoid saturates on the Laplace
parents just like tanh, and relu underfits x3 (val NLL 1.46–1.47 vs the
1.403 ± 0.05 band) while growing a fragile Bernstein tail (one held-out row
inverted to x4 ≈ 580, cf MAE 4.36 vs bound 0.33); minibatch — the fig6 grid
ends live in the sparse x2 tail that minibatch gradients underweight
(cf MAE x3 do(x2=+1.5) 0.40 — over the old 0.355 and the re-pinned 0.319
bound alike — and fig6 x3 err 5.9 against the full-batch run's 2.7).

**validate_ls `adam`** (misc, not a paper experiment but tuned in the same
round): phases 800/700/500 instead of 4000/2000/1000 — 2000 epochs instead of
7000 — still lands on the classical MLE, named-coef gap to statsmodels
1.6e-5 vs the old 1.8e-5, ~3.5× less fit wall clock.

Local wall times under the tuned protocol (32-core box, 4 torch threads):
triangle 162–281 s per variant, mixed 88–165 s, vaca 46.5 s, carefl 72–82 s.
The CI timings quoted above and the `fit_seconds` pins in `ground_truth/`
are still the 2026-08-31 pre-cut measurements; both will be re-measured on
the next push and the pins updated then.

## Repository choices the paper does not state

Seeds; the validation draw sizes for the triangle scripts (the R code's
`test`); `n_compare`; `n_heldout = 300` and the α set {−1.5, 0, 1.5} scored
next to the single `x_obs`; the
mixed cutpoints from the R code; the odds-ratio sample (40000 rows) and the
counterfactual-PMF sample (2000 rows × 200 draws).
