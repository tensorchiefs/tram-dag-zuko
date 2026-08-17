# Fitting a TRAM-DAG: how training works

This page is the technical reference for [`CausalFlowDAG`](../src/tramdag/flow.py).
It explains how the module builds the flow and how it computes the likelihood. It
describes the two fitting paths: the stochastic optimizer
([`fit`](../src/tramdag/flow.py)) and the classical optimizer
([`fit_classical`](../src/tramdag/flow.py)). It also covers per-node freezing, the
memory model, and future directions for optimizer choice.

## How the flow is built: one module, one sub-model per node

A `CausalFlowDAG` is a single `torch.nn.Module`, but it is **not one monolithic
network**. At construction ([`CausalFlowDAG.__init__`](../src/tramdag/flow.py)), it
topologically sorts the DAG. It then builds an `nn.ModuleDict` with **one
[`_Node`](../src/tramdag/flow.py) sub-module per variable**. The nodes share *no
parameters*. Each node owns the pieces for its own conditional `p(x_i | pa(x_i))`:

- an **intercept** that produces the transform parameters `θ`. This is
  [`SimpleIntercept`](../src/tramdag/conditioners.py), a free parameter vector that
  is data-independent. If the node has `ci` parents, it is
  [`ComplexIntercept`](../src/tramdag/conditioners.py) instead, a small MLP whose
  output `θ` depends on those parents.
- a **monotone 1-D transform** `h`
  ([`BernsteinUT` / `SplineUT` / `AffineUT`](../src/tramdag/transforms.py)).
  The transform itself carries **no learnable weights**, only the fitted
  range buffers `xmin`/`xmax`. The `θ` from the intercept sets its shape
  entirely.
- a `ModuleDict` of **shift modules**, one per parent edge:
  [`LinearShift`](../src/tramdag/conditioners.py) (`ls`, a single weight) or
  [`ComplexShift`](../src/tramdag/conditioners.py) (`cs`, an MLP).

So is this one network or several? It is **one module, several independent per-node
sub-models** (each itself a small intercept + shifts assembly). One module bundles
them, and one optimizer trains them, but their parameters are disjoint. The DAG
structure lives entirely in *which parents each node reads*. There is no edge
weight matrix or shared trunk.

For a batch, `_Node.theta_shift` assembles a node's `θ` (from the intercept) and
total `shift` (sum of its shift modules over the parent features). `_encode_parent`
encodes the parent features: continuous parents raw, ordinal parents one-hot.

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
  z = h(x; θ) + shift
  log p(x | pa) = log f_logistic(z) + log |dz/dx|
  ```

  That is, the term is the standard-logistic density at the latent `z`
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
learning rates, per-node freezing (below), and the all-`ls` classical fit.

## Path A — stochastic optimization (`fit`)

[`CausalFlowDAG.fit`](../src/tramdag/flow.py) is the general-purpose trainer. Any
`cs`/`ci` edge requires it. Mechanics:

- **Adam** with **one parameter group per node**. `fit` builds each group from
  `self.nodes[name].parameters()`, so each node can carry its own learning rate.
- **Minibatches**: a fresh `torch.randperm` shuffle occurs each epoch. The loss is
  the summed per-node mean NLL on the batch.
- **Transform ranges**: `_set_ranges` sets them once at the start from the train
  5%/95% quantiles. These ranges define each Bernstein/spline domain.
- **Learning-rate schedules** (`schedule=`): `None` (constant), `"onecycle"`,
  `"cosine"`, or `"plateau"`. The `"plateau"` schedule decays each node's lr off
  *its own* validation NLL.
- **Per-node freezing** (`freeze_patience=`, see below).
- **`restore_best`**: when enabled, `fit` snapshots each node's best-validation
  weights and restores them at the end (early-stopping regularization). The
  default is `False`, so the fit sits at the training-data MLE.
- **`marginal_init=`**: opt-in calibrated initialization of the unconditional
  intercepts — Bernstein nodes start at the linear map onto the latent
  5%/95% quantiles, ordinal cutpoints at the empirical class log-odds. A pure
  init; the converged MLE is unchanged.
- **`vc_warm_start=`** (default on): each `VC` term's `beta0` starts at the
  classical all-`ls` solution of its node's conditional, so the penalized
  head only learns deviations.

### How per-node freezing works

Set `freeze_patience=N` to remove converged nodes from training. Each epoch, `fit`
tracks every node's own validation NLL and counts epochs since its last
improvement (by more than `min_delta`). When that counter reaches `N`, `fit`
**freezes** the node. Under `schedule="plateau"`, the freeze occurs only after the
schedule decays the node's learning rate. This order prevents a freeze while a
smaller step can still help.

A freeze does more than stop updates. `fit` **excludes the frozen node from the
computation entirely**: it calls `node_log_prob(values, nodes=active)` with only
the still-active nodes, so a frozen node runs no forward and no backward pass.
That is a real FLOP saving. It is also *correct*, precisely because the per-node
gradients are independent (above): removal of a converged node's term from the
loss does not change any other node's gradient. The slowest node sets the total
training time. Once the other nodes converge, they cost nothing.

When **every** node is frozen, the fit returns early (the self-stopping behavior).
`fit` records the freeze epochs in `flow.history["frozen"]`. The freeze state is
local to a single `fit` call (a later `fit` call trains all nodes again). This
composes with `restore_best`, which still restores each node's best-validation
snapshot at the end.

#### Freezing vs. parallelism (why one helps and the other usually does not)

A tempting argument says: "freezing a node saves time, so node computation costs
time, so parallel computation of nodes will also save time." The premise is
correct, but the conclusion does not follow. The two act through different
mechanisms:

- **Freezing removes work.** The frozen node's forward/backward never runs again.
  Removal of work is an *unconditional* win, and most of its payoff is
  structural. Once *all* nodes freeze, the fit stops and deletes whole epochs.
  Parallelism can never do that (it can shrink time-per-epoch, not the number of
  epochs).
- **Parallelism only overlaps work.** It runs the same computations concurrently,
  and this helps *only if the hardware sits idle* during the sequential node
  loop. The hardware usually is not idle. Parallelism already exists one level
  down: each node's batched tensor ops run across all rows, and the BLAS/GPU
  backend already uses all cores. While node A's matmul runs, the cores are busy,
  so node B "at the same time" only time-slices the same cores. The result is
  contention, often slightly slower.

So the speed gain from freezing does **not** imply spare capacity for parallel
work across nodes. The exception is the regime where per-node ops *under-utilize*
the hardware: small batches, tiny models, or many small nodes on a big GPU where
each kernel leaves it mostly idle. In that regime, node overlap can help. But the
right tool is **fusion** (batch same-shaped nodes into one larger tensor op, or
CUDA streams), not a threaded Python loop (which the GIL fights). That is the
"vectorize/fuse the per-node loop" direction in
[Optimizer choice](#optimizer-choice--current-and-future) below. It is a
conditional win in that regime, distinct from the unconditional win of freezing.

Benchmarks, schedule trade-offs, and the recommended self-stopping recipe are in
[training-speed.md](training-speed.md). The worked walkthrough is
[`notebooks/intro_tram_dag.py`](../notebooks/intro_tram_dag.py).

## Path B — classical optimization (`fit_classical`)

[`CausalFlowDAG.fit_classical`](../src/tramdag/flow.py) is the dedicated optimizer
for **all-`ls`** models, where every node-conditional is a classical
transformation model (ordered-logit / Colr). It raises on any `cs`/`ci` edge.

- **Full-batch, float64, L-BFGS** (strong-Wolfe line search). There are no
  minibatches, no schedule, and no early stopping. Therefore the fit is
  **deterministic** (same init → bit-identical) and lands on the **exact MLE**
  (matches `statsmodels`/R to ~1e-3 on well-identified coefficients).
- **float64 is a transient compute mode**: `self.double()` upcasts parameters
  *and* the transforms' range buffers in one call. The fit runs in double. A
  `finally: self.float()` restores float32. The stored model stays float32, and
  so do `save`/`load`. The data path (`_tensorize`, `sample`, `pmf`) reads the
  model dtype, so it follows along.
- **Convergence**: the flag comes from NLL flatness and is *advisory*. A
  Bernstein intercept and weakly-identified directions (rare one-hot levels, a
  flat treatment-effect ridge) continue to drift along zero-curvature valleys
  after the likelihood is at the optimum. Correctness comes from a comparison
  with classical software (`experiments/validate_ls.py --classical`), not from
  the flag.
- Read the fitted coefficients with `ls_coefficients()`.

Walkthrough: [`notebooks/stale/classical_fit_tram_dag.py`](../notebooks/stale/classical_fit_tram_dag.py) (parked).

## Memory and disk during fitting

**Neither fitting path writes to disk.** Everything during `fit`/`fit_classical`
lives in RAM:

- model parameters and (for `fit`) the per-node Adam optimizer state
- the `history` dict (per-epoch train/val NLL, lr, wall-clock), which is an
  in-memory attribute and not a file
- `restore_best` snapshots: `copy.deepcopy` of node `state_dict`s held in memory
  (`flow.py`), never serialized

The **only** disk I/O in the module is the explicit, user-called
[`save`](../src/tramdag/flow.py) (`torch.save` of spec + state_dict + history) and
its counterpart `load`. So a fit produces no temp files, no checkpoints, and no
scratch directory. Persistence is opt-in via `flow.save(path)`. (The *experiment
scripts* write the `results/` and `docs/perf/` artifacts in this repo, not the
library's fitting code.)

## Optimizer choice — current and future

Today: **Adam** for flexible models, **L-BFGS** (float64) for all-`ls`. The
per-node-param-group structure already in `fit` makes several extensions cheap.
These extensions are worth consideration as the package matures:

- **IRLS / Fisher scoring for the all-`ls` path.** The classical fitting
  algorithm for proportional-odds / GLM-type models is iteratively reweighted
  least squares (Newton with the expected information). For the `ls` case, it
  will likely reach the MLE in *fewer, more stable* steps than L-BFGS. Also, the
  Fisher information that it computes is exactly the ingredient for the planned
  **standard-error table** (currently flagged as next in the CHANGELOG). It is a
  strong candidate to back `fit_classical` in a future version.
- **Second-order / curvature-aware methods for flexible nodes.** K-FAC or a
  Gauss-Newton approximation can help the `ci`/`cs` MLPs. However, full-batch
  quasi-Newton is a poor fit for them (minibatch noise also regularizes, which is
  why `fit_classical` refuses them).
- **Modern first-order variants.** AdamW (decoupled weight decay), RAdam (warmup-
  free), or Lion/Sophia are drop-in alternatives to Adam. The benchmark harness
  ([`experiments/stale/bench_training.py`](../experiments/stale/bench_training.py),
  parked — needs an API migration) exists to evaluate exactly such swaps on
  time-to-target.
- **Per-node optimizer selection.** The loss decomposes, and the optimizer
  already holds one group per node. Therefore different nodes can in principle
  use different optimizers (for example, L-BFGS for the `ls` nodes and Adam for
  an MLP node in a mixed model). This is a direction for mixed-flexibility DAGs.
- **Warm-start handoffs.** `fit_classical → fit` already works (the classical MLE
  as a fast, principled initialization). The reverse (Adam to escape, L-BFGS to
  polish) is a natural pattern to formalize.

These are *directions*, not commitments. The autoresearch loop
([`docs/research/MISSION_autoresearch.md`](research/MISSION_autoresearch.md)) is a
good venue to test the speed-oriented ones empirically before any adoption.
