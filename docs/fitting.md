# Fitting a TRAM-DAG: how training works

Technical reference for [`CausalFlowDAG`](../src/tramdag/flow.py): how the flow is
built, how the likelihood is computed, and the two fitting paths — the stochastic
optimizer ([`fit`](../src/tramdag/flow.py)) and the classical one
([`fit_classical`](../src/tramdag/flow.py)) — plus per-node freezing, the memory
model, and where optimizer choice could go next.

## How the flow is built: one module, one sub-model per node

A `CausalFlowDAG` is a single `torch.nn.Module`, but it is **not one monolithic
network**. At construction ([`CausalFlowDAG.__init__`](../src/tramdag/flow.py)) it
topologically sorts the DAG and builds an `nn.ModuleDict` with **one
[`_Node`](../src/tramdag/flow.py) sub-module per variable**. The nodes share *no
parameters*; each owns the pieces for its own conditional `p(x_i | pa(x_i))`:

- an **intercept** producing the transform parameters `θ`:
  [`SimpleIntercept`](../src/tramdag/conditioners.py) (a free parameter vector,
  data-independent) — or, if the node has `ci` parents,
  [`ComplexIntercept`](../src/tramdag/conditioners.py) (a small MLP whose output
  `θ` depends on those parents);
- a **monotone 1-D transform** `h`
  ([`BernsteinUT` / `SplineUT` / `AffineUT`](../src/tramdag/transforms.py)) —
  note the transform itself carries **no learnable weights**, only the fitted
  range buffers `xmin`/`xmax`; its shape is set entirely by `θ` from the
  intercept;
- a `ModuleDict` of **shift modules**, one per parent edge:
  [`LinearShift`](../src/tramdag/conditioners.py) (`ls`, a single weight) or
  [`ComplexShift`](../src/tramdag/conditioners.py) (`cs`, an MLP).

So "one network or several?" — **one module, several independent per-node
sub-models** (each itself a small intercept + shifts assembly). They are bundled
into one module and trained by one optimizer, but their parameters are disjoint.
The DAG structure lives entirely in *which parents each node reads*; there is no
edge weight matrix or shared trunk.

`_Node.theta_shift` assembles a node's `θ` (from the intercept) and total `shift`
(sum of its shift modules over the parent features) for a batch; parent features
are encoded by `_encode_parent` — continuous parents raw, ordinal parents one-hot.

## How the likelihood is computed

A TRAM-DAG maps iid standard-logistic latents `U` to the observed `X` in causal
order; because node `i` reads only its parents (earlier variables, as *data*), the
Jacobian of `U → X` is **triangular**, so its log-determinant is the sum of the
per-node 1-D terms. The joint log-likelihood therefore **decomposes per node**:

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

  i.e. the standard-logistic density at the latent `z`
  ([`StandardLogistic.log_prob`](../src/tramdag/transforms.py)) plus the
  transform's log-derivative (`ladj`, returned by `ut.forward` — the 1-D Jacobian
  term that makes it a proper density, not just a score).
- **ordinal** — an ordered-logit / proportional-odds head,
  `P(x ≤ k) = σ(θ_k − shift)`, evaluated as the log of the cutpoint-interval
  probability by [`ordinal_log_prob`](../src/tramdag/transforms.py) (done in
  log-space via `logsigmoid`/`log1mexp`, because the naive sigmoid difference
  underflows to exactly-zero gradients in float32).

The training loss is the summed per-node **mean** NLL over the batch
(`Σ_i mean_rows(−log p(x_i | pa))`); `log_prob` instead returns the per-row joint
for scoring whole observations.

**Consequence used by both optimizers:** because parents enter as data, the
per-node gradients are independent — optimizing the summed loss jointly is
identical to optimizing each node separately. This is what licenses per-node
learning rates, per-node freezing (below), and the all-`ls` classical fit.

## Path A — stochastic optimization (`fit`)

[`CausalFlowDAG.fit`](../src/tramdag/flow.py) is the general-purpose trainer
(required for any `cs`/`ci` edge). Mechanics:

- **Adam** with **one parameter group per node** — built from
  `self.nodes[name].parameters()`, so each node can carry its own learning rate.
- **Minibatches**: a fresh `torch.randperm` shuffle each epoch; the loss is the
  summed per-node mean NLL on the batch.
- **Transform ranges** are set once at the start from the train 5%/95% quantiles
  (`_set_ranges`), defining each Bernstein/spline domain.
- **Learning-rate schedules** (`schedule=`): `None` (constant), `"onecycle"`,
  `"cosine"`, or `"plateau"` — the last decays each node's lr off *its own*
  validation NLL.
- **Per-node freezing** (`freeze_patience=`, see below).
- **`restore_best`**: optionally snapshots each node's best-validation weights and
  restores them at the end (early-stopping regularization). Default `False` so the
  fit sits at the training-data MLE.

### How per-node freezing works

Set `freeze_patience=N` to let converged nodes drop out of training. Each epoch,
`fit` tracks every node's own validation NLL and counts epochs since its last
improvement (by more than `min_delta`). A node is **frozen** once that counter
reaches `N` (under `schedule="plateau"`, only after its learning rate has already
been decayed, so a node isn't frozen while a smaller step would still help).

Freezing is not just "stop updating" — the frozen node is **excluded from the
computation entirely**: `node_log_prob(values, nodes=active)` is called with only
the still-active nodes, so a frozen node runs no forward and no backward pass.
That is a real FLOP saving, and it is *correct* precisely because the per-node
gradients are independent (above): removing a converged node's term from the loss
does not change any other node's gradient. The slowest node sets the total
training time; the rest stop costing anything once they have converged.

When **every** node is frozen the fit returns early (the self-stopping behavior).
Freeze epochs are recorded in `flow.history["frozen"]`; the freeze state is local
to a single `fit` call (a later `fit` call trains all nodes again). This composes
with `restore_best`, which still restores each node's best-validation snapshot at
the end.

#### Freezing vs. parallelism (why one helps and the other usually doesn't)

It is tempting to reason "freezing a node speeds things up, so node computation
costs time, so computing nodes *in parallel* should also speed things up." The
premise is right but the conclusion doesn't follow, because the two act through
different mechanisms:

- **Freezing removes work** — the frozen node's forward/backward simply never
  runs again. Removing work is an *unconditional* win, and most of its payoff is
  structural: once *all* nodes freeze the fit stops, deleting whole epochs —
  something parallelism can never do (it can shrink time-per-epoch, not the number
  of epochs).
- **Parallelism only overlaps work** — it runs the same computations
  concurrently, which helps *only if the hardware is sitting idle* during the
  sequential node loop. It usually isn't: parallelism is already extracted one
  level down, where each node's batched tensor ops run across all rows and the
  BLAS/GPU backend already uses all cores. While node A's matmul runs the cores
  are busy, so doing node B "at the same time" just time-slices the same cores —
  contention, often slightly slower.

So freezing speeding things up does **not** imply spare capacity to parallelize
across nodes. The exception is the regime where per-node ops *under-utilize* the
hardware — small batches, tiny models, or many small nodes on a big GPU where each
kernel leaves it mostly idle. There, overlapping nodes can help, but the right
tool is **fusion** (batch same-shaped nodes into one larger tensor op, or CUDA
streams), not threading the Python loop (which the GIL fights). That is the
"vectorize/fuse the per-node loop" direction in
[Optimizer choice](#optimizer-choice--current-and-future) below — a conditional
win in that regime, distinct from freezing's unconditional one.

Benchmarks, schedule trade-offs, and the recommended self-stopping recipe are in
[training-speed.md](training-speed.md); the worked walkthrough is
[`notebooks/intro_tram_dag.py`](../notebooks/intro_tram_dag.py).

## Path B — classical optimization (`fit_classical`)

[`CausalFlowDAG.fit_classical`](../src/tramdag/flow.py) is the dedicated optimizer
for **all-`ls`** models, where every node-conditional is a classical
transformation model (ordered-logit / Colr). It raises on any `cs`/`ci` edge.

- **Full-batch, float64, L-BFGS** (strong-Wolfe line search) — no minibatches, no
  schedule, no early stopping ⇒ **deterministic** (same init → bit-identical) and
  on the **exact MLE** (matches `statsmodels`/R to ~1e-3 on well-identified
  coefficients).
- **float64 is a transient compute mode**: `self.double()` upcasts parameters
  *and* the transforms' range buffers in one call; the fit runs in double; a
  `finally: self.float()` restores float32. The stored model — and
  `save`/`load` — stay float32. The data path (`_tensorize`, `sample`, `pmf`)
  reads the model dtype so it follows along.
- **Convergence** is judged by NLL flatness and is *advisory*: a Bernstein
  intercept and weakly-identified directions (rare one-hot levels, a flat
  treatment-effect ridge) keep drifting along zero-curvature valleys after the
  likelihood is at the optimum. Correctness is verified against classical software
  (`experiments/validate_ls.py --classical`), not the flag.
- Read the fitted coefficients with `ls_coefficients()`.

Walkthrough: [`notebooks/stale/classical_fit_tram_dag.py`](../notebooks/stale/classical_fit_tram_dag.py) (parked).

## Memory and disk during fitting

**Neither fitting path writes to disk.** Everything during `fit`/`fit_classical`
lives in RAM:

- model parameters and (for `fit`) the per-node Adam optimizer state;
- the `history` dict (per-epoch train/val NLL, lr, wall-clock) — an in-memory
  attribute, not a file;
- `restore_best` snapshots — `copy.deepcopy` of node `state_dict`s held in memory
  (`flow.py`), never serialized.

The **only** disk I/O in the module is the explicit, user-called
[`save`](../src/tramdag/flow.py) (`torch.save` of spec + state_dict + history) and
its counterpart `load`. So a fit produces no temp files, no checkpoints, no
scratch directory; persistence is opt-in via `flow.save(path)`. (The `results/`
and `docs/perf/` artifacts in this repo are written by the *experiment scripts*,
not by the library's fitting code.)

## Optimizer choice — current and future

Today: **Adam** for flexible models, **L-BFGS** (float64) for all-`ls`. The
per-node-param-group structure already in `fit` makes several extensions cheap,
and worth considering as the package matures:

- **IRLS / Fisher scoring for the all-`ls` path.** The classical fitting algorithm
  for proportional-odds / GLM-type models is iteratively reweighted least squares
  (Newton with the expected information). For the `ls` case it would likely reach
  the MLE in *fewer, more stable* steps than L-BFGS — and the Fisher information it
  computes is exactly the ingredient for the planned **standard-error table**
  (currently flagged as next in the CHANGELOG). Strong candidate to back
  `fit_classical` in a future version.
- **Second-order / curvature-aware methods for flexible nodes.** K-FAC or a
  Gauss-Newton approximation could help the `ci`/`cs` MLPs, though full-batch
  quasi-Newton is a poor fit for them (minibatch noise also regularizes — see why
  `fit_classical` refuses them).
- **Modern first-order variants.** AdamW (decoupled weight decay), RAdam (warmup-
  free), or Lion/Sophia are drop-in alternatives to Adam; the benchmark harness
  ([`experiments/bench_training.py`](../experiments/bench_training.py)) is built to
  evaluate exactly such swaps on time-to-target.
- **Per-node optimizer selection.** Because the loss decomposes and the optimizer
  already holds one group per node, different nodes could in principle use
  different optimizers (e.g. L-BFGS for the `ls` nodes, Adam for an MLP node in a
  mixed model) — an interesting direction for mixed-flexibility DAGs.
- **Warm-start handoffs.** `fit_classical → fit` already works (the classical MLE
  as a fast, principled initialization); the reverse (Adam to escape, L-BFGS to
  polish) is a natural pattern to formalize.

These are *directions*, not commitments. The autoresearch loop
([`docs/research/MISSION_autoresearch.md`](research/MISSION_autoresearch.md)) is a
good venue to test the speed-oriented ones empirically before adopting any.
