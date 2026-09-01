# Fitting a TRAM-DAG: how training works

The technical reference for training a [`CausalFlowDAG`](../src/tramdag/flow.py):
the likelihood, the two fitting paths (`fit`, `fit_classical`) and their hooks.

## How the flow is built: one module, one sub-model per node

A `CausalFlowDAG` is a single `torch.nn.Module` holding **one independent
sub-model per variable** — an intercept producing the transform parameters `θ`,
the monotone 1-D transform `h` (no learnable weights of its own, only range
buffers), and one shift module per shift term. The nodes share **no
parameters**: one module bundles them and one optimizer trains them, but the
DAG structure lives entirely in *which parents each node reads* — there is no
edge weight matrix or shared trunk. Which class implements which term is the
[code map](code-map.md); parent features enter continuous-raw / ordinal-one-hot.

## How the likelihood is computed

A TRAM-DAG maps iid standard-logistic latents `U` to the observed `X` in causal
order. Node `i` reads only its parents (earlier variables, as *data*). Therefore
the Jacobian of `U → X` is **triangular**, and its log-determinant is the sum of
the per-node 1-D terms. The joint log-likelihood therefore **decomposes per node**:

```
log p(x) = Σ_i log p(x_i | pa(x_i))
```

[`CausalFlowDAG.node_log_prob`](../src/tramdag/flow.py) computes one term per node
and `log_prob` sums them. For a node, given `θ, shift` from `theta_shift`:

- **continuous** — change of variables through the monotone transform:

  ```
  u = h(x; θ) + shift
  log p(x | pa) = log f_logistic(z) + log |dz/dx|
  ```

  That is, the term is the standard-logistic density at the latent `u`
  ([`StandardLogistic.log_prob`](../src/tramdag/transforms.py)) plus the
  transform's log-derivative. `ut.forward` returns this log-derivative as `ladj`.
  This is the 1-D Jacobian term that makes the result a proper density, not only
  a score.
- **ordinal** — an ordered-logit / proportional-odds head,
  `P(x ≤ k) = σ(θ_k − shift)`. [`ordinal_log_prob`](../src/tramdag/transforms.py)
  evaluates it as the log of the cutpoint-interval probability. The computation
  runs in log-space via `logsigmoid`/`log1mexp`, because the naive sigmoid
  difference underflows to exactly-zero gradients in float32.

The training loss is the summed per-node **mean** NLL over the batch
(`Σ_i mean_rows(−log p(x_i | pa))`). In contrast, `log_prob` returns the per-row
joint, which scores whole observations.

**Consequence used by both optimizers:** because parents enter as data, the
per-node gradients are independent. Therefore a joint fit of the summed loss is
identical to a separate fit of each node. This independence licenses per-node
learning rates and freezing (a callback, below) and the all-`ls` classical fit.

## Path A — stochastic optimization (`fit`)

[`CausalFlowDAG.fit`](../src/tramdag/flow.py) is the general-purpose trainer. Any
`cs`/`ci` edge requires it. Mechanics:

- **One optimizer over all parameters** — `Adam(lr=learning_rate)` by
  default, or any `torch.optim.Optimizer` you pass as `optimizer=` (exactly
  per-node training, see the consequence above; `per_node_adam` builds the
  per-node parameter groups when you want per-node rates).
- **Minibatches**: a fresh `torch.randperm` shuffle each epoch (`seed=` seeds
  it). The loss is the summed per-node mean NLL on the batch, plus the `VC`
  penalty.
- **`calibrate(train_df, marginal_init=True)`**, called by the first `fit`:
  the transform ranges from the train 5%/95% quantiles (each Bernstein/spline
  domain), the statistics of every term-level `input_transform=`, and
  the calibrated start — Bernstein nodes at the linear map onto the latent
  5%/95% quantiles, ordinal cutpoints at the empirical class log-odds, a pure
  init that leaves the MLE unchanged. Call it yourself to switch the start
  off. A checkpoint carries the flag, so a loaded model is never recalibrated.
  The start itself is also a public step: `flow.init_marginals(train_df)`
  resets every Bernstein/ordinal simple intercept to its column's marginal
  (spline and affine have no calibrated start), any time — e.g. to
  restart a trained or loaded flow (`calibrate` won't, it is once-only).
- **Validation, Keras-shaped** — `validation_data=` (a DataFrame) or
  `validation_split=` (a float: the LAST fraction of `train_df`, no shuffle,
  and only the head calibrates — no leakage) makes `fit` compute the
  per-node validation NLL after every epoch, once, into
  `flow.history["val"]` (`validation_batch_size=` chunks the pass). The
  shipped callbacks read it there. `verbose=N` prints every Nth epoch plus
  the final one (0, the default, is silent).
- **`callbacks=`** — one entry or a list. A
  [`tramdag.callbacks.Callback`](../src/tramdag/callbacks.py) hooks all
  three points of the loop: `on_fit_begin(flow, optimizer)` once after
  calibration (the shipped callbacks reset their state here, so instances
  are reusable), `on_epoch_end(flow, epoch, optimizer)` after every epoch,
  once the epoch's train NLLs are in `flow.history["train"]` — every
  callback runs each epoch, and the fit stops after an epoch in which any
  returned `True` — and `on_fit_end(flow, optimizer)`, which runs *before*
  the `VC` re-centering, so a hook that swaps the weights hands them to the
  re-centering. A bare callable in the list is an `on_epoch_end` hook —
  this is where schedules, snapshots and coefficient trajectories live. The
  common recipes ship in `tramdag.callbacks`: `RestoreBest`
  (best-validation weights, restored automatically), `EarlyStopping`
  (patience on the validation NLL) and `PerNodePlateau` + `per_node_adam`
  (per-node decay and freezing), all reading `history["val"]`.
- **`vc_ehat=`**: the out-of-fold propensities a centered `VC` term needs,
  `{node: {t: array}}` with one value per training row (see
  [varying-coefficients.md](varying-coefficients.md)).

### Training strategies

Every strategy below is `fit` plus a callback or a few lines of your own —
pick by model class, each with a copy-paste example (they assume a built
`flow = CausalFlowDAG(spec)` and pandas `train_df`/`val_df`). The empirical
rule of thumb: **all-`ls` models train to the MLE and keep the final
weights; flexible (CI/CS/VC) models validate and keep the best weights**
(they overfit observational confounding at the MLE, see the finding below).

| Strategy | When |
|---|---|
| exact MLE — [`fit_classical`](#path-b--classical-optimization-fit_classical) | all-`ls` spec; deterministic, seconds |
| plain Adam | all-`ls` with a shift `fit_classical` refuses, quick looks |
| multi-phase Adam | a tighter MLE without a scheduler |
| best-validation weights — `RestoreBest` | any CI/CS/VC model; the recommended recipe (register it — fit has no default) |
| … + early stopping — `EarlyStopping` | cap the wasted epochs after the best |
| global plateau schedule | decaying one shared rate beats picking one |
| per-node plateau — `PerNodePlateau` | nodes converge at different speeds; self-stopping |

**Plain Adam** — one loop, constant rate, final weights:

```python
flow = CausalFlowDAG(spec, seed=0)
flow.fit(train_df, epochs=500, learning_rate=1e-3, batch_size=256, verbose=100)
```

**Multi-phase Adam** — re-calling `fit` continues training, so decreasing
rates are a loop. This is the `validate_ls` protocol, whose three phases land
within ~1e-5 of statsmodels — a single converged constant-rate run gets
~1e-3:

```python
for epochs, lr in [(800, 1e-2), (700, 1e-3), (500, 1e-4)]:
    flow.fit(train_df, epochs=epochs, learning_rate=lr)
```

**Best-validation weights** — the flexible-model default, one import, one
registration (the restore happens automatically at fit end):

```python
from tramdag.callbacks import RestoreBest

flow.fit(
    train_df,
    epochs=4000,
    validation_data=val_df,   # or validation_split=0.1
    verbose=50,
    callbacks=RestoreBest(),
)
```

`validation_split` takes the LAST rows unshuffled — shuffle the DataFrame
first if its row order means anything.

**… plus early stopping** — `EarlyStopping` tracks its own best epoch, so
the pair composes in any order:

```python
from tramdag.callbacks import EarlyStopping, RestoreBest

flow.fit(train_df, epochs=4000, validation_split=0.1,
         callbacks=[RestoreBest(), EarlyStopping(patience=200)])
```

**Per-node plateau** — one rate per node (`per_node_adam` tags one parameter
group per node; valid because the per-node gradients are independent), each
decaying and finally freezing on its own validation NLL; the fit stops when
every node froze. The demo notebook runs it end to end; before 0.4 it was
`fit(schedule="plateau", freeze_patience=)`, measured in
`experiments/benchmarks/bench_training.py`:

```python
from tramdag.callbacks import PerNodePlateau, per_node_adam

flow.fit(train_df, epochs=4000, validation_split=0.1,
         optimizer=per_node_adam(flow, lr=1e-2),
         callbacks=PerNodePlateau())   # patience=15, freeze=50
```

**Global plateau schedule** — anything else is a few lines of your own: a
learning-rate schedule is torch's, stepped from the hook on the validation
NLL `fit` already computed (so this too needs `validation_data=` or
`validation_split=`); the snapshot half is what `RestoreBest` does inside,
written out:

```python
import copy

import torch

opt = torch.optim.Adam(flow.parameters(), lr=1e-2)
plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=30)
best = {"nll": float("inf"), "state": None}

def on_epoch(flow, epoch, opt):
    nll = sum(flow.history["val"][-1].values())   # fit computed it
    plateau.step(nll)
    if nll < best["nll"]:
        best.update(nll=nll, state=copy.deepcopy(flow.state_dict()))
    return opt.param_groups[0]["lr"] < 1e-5      # stop once the rate bottomed out

flow.fit(train_df, epochs=4000, batch_size=512, validation_data=val_df,
         optimizer=opt, callbacks=on_epoch)
flow.load_state_dict(best["state"])
```

(One difference to `RestoreBest`: a post-fit `load_state_dict` skips the VC
re-centering, so on a spec with a `VC` term put the restore in a `Callback`
subclass's `on_fit_end` instead.)

The exact-MLE and warm-start strategies are Path B, below.


Freezing helps and parallelizing the node loop does not: freezing deletes
whole epochs, while node-level overlap only time-slices the cores that each
node's batched BLAS ops already saturate — measured as contention, not speedup.
Only when per-node kernels under-utilize the hardware (tiny nodes on a big GPU)
could overlap pay, and there the tool is fusing same-shaped nodes, not threads.

Benchmarks and schedule trade-offs are in
[training-speed.md](training-speed.md). The worked walkthrough is
[`notebooks/intro_tram_dag.py`](../notebooks/intro_tram_dag.py).

## Path B — classical optimization (`fit_classical`)

[`CausalFlowDAG.fit_classical`](../src/tramdag/flow.py) is the dedicated optimizer
for **all-`ls`** models, where every node-conditional is a classical
transformation model (ordered-logit / Colr). It raises on any `cs`/`ci`/`vc` term.

- **Full-batch, float64, L-BFGS** (strong-Wolfe line search). There are no
  minibatches, no schedule, and no early stopping. Therefore the fit is
  **deterministic** (same init → bit-identical) and lands on the **exact MLE**
  — `fit_classical` matches `statsmodels`/R to ~4 decimals; a converged Adam
  `fit` gets within ~1e-3.
- **Solver budget** (`max_iter=400`, `tol=1e-9`, `history_size=50`): one
  L-BFGS run with torch's own stopping rule — it ends when the NLL or the
  parameters move by less than `tol`, or at `max_iter`. The report's
  `n_iter` is torch's count and `converged` says whether a tolerance, not
  the cap, ended the run. `history_size` is the L-BFGS memory.
- **float64 is a transient compute mode**: the fit runs in double and
  restores float32 afterwards; checkpoints stay float32.
- **Convergence**: the flag is true when torch's `tolerance_change` ended the
  run before `max_iter` did, and it is *advisory*. A
  Bernstein intercept and weakly-identified directions (rare one-hot levels, a
  flat treatment-effect ridge) continue to drift along zero-curvature valleys
  after the likelihood is at the optimum. Correctness comes from a comparison
  with classical software (`python -m misc.validate_ls classical`), not from
  the flag.
- Read the fitted coefficients with `ls_coefficients()`.

### Warm-start handoff: classical fit, then keep training

`fit_classical` leaves the model at the MLE in float32, ready for any normal
operation — and continuing with `fit()` from there **stays put**, which is both
a check that the classical solution really is the optimum and a way to use it as
a fast, principled initialization:

```python
flow.fit_classical(train_df)                       # exact MLE, seconds
before = flow.ls_coefficients()["y"]
flow.fit(train_df, epochs=300, learning_rate=1e-3)  # a gentle Adam phase ...
after = flow.ls_coefficients()["y"]                 # ... barely moves
```

A small drift means the classical fit was already at the optimum. The same
handoff warm-starts a `VC` term's `beta0` — the measured recipe is in
[varying-coefficients.md](varying-coefficients.md).

## Memory and disk during fitting

Neither fitting path writes to disk: parameters, optimizer state and the
`history` dict live in RAM, and whatever a callback records is yours. The only
disk I/O in the module is the explicit `save()`/`load()`; the `results/`
artifacts in this repo come from the experiment scripts, not the library.

## Optimizer choice

Today: **Adam** for flexible models, **L-BFGS** (float64) for all-`ls`. The
per-node decomposition makes optimizer swaps cheap through `optimizer=`;
candidates (IRLS for the `ls` path, per-node mixing, modern Adam variants) are
benchmarked with `experiments/benchmarks/bench_training.py` before adoption.
