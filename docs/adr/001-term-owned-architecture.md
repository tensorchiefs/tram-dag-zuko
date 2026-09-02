# ADR 001 — Term-owned architecture for 1.0-RC

Date: 2026-09-02 · Status: accepted · Branch: `rc/1.0-architecture`

## Context

At 0.4 the package was seven modules with flow.py a 2020-line monolith; per-effect
behavior (validation, construction, evaluation, penalty, post-fit steps, score
columns, adjacency cells, classical-fit eligibility) lived in five parallel string
switches across spec.py/flow.py/scores.py — VC alone had ~27 sites. Three
architecture proposals (term-owned, one-dispatch-table, pure seam-split) were
drafted against a seven-subsystem survey and judged from three lenses
(maintainability, migration risk, extensibility without over-engineering).

## Decision

**Term-owned flow, executed in the seam-split's order.**

1. Verbatim code motion first: `nodes.py` (node model), `fitting.py`
   (fit/fit_classical as free functions), `readouts.py` (stateless read-outs +
   the new public `shift_curve`) — state-dict paths and the seeded RNG stream
   untouched, one-line delegates keep the public API on `CausalFlowDAG`.
2. The **registry** (`terms.py`): one definition per effect. Built-in shift terms
   subclass their conditioners (`LSTerm(ShiftTerm, LinearShift)` …), so
   checkpoints and RNG draws stay bit-stable; each term owns validation
   (`check_arity`/`edge_parents`), construction (`build`), evaluation
   (`shift_value`/`theta_value`), `post_init`, `regularizer`, post-fit
   `finalize`, `score_columns`, the side-input contract
   (`side_keys`/`check_side`/`live_side`/`extra_columns`), adjacency `cells`,
   `term_is_classical` and its `option_defaults`.
3. The intercept slot is a term too (`SITerm`/`CITerm`/`AdditiveCITerm`);
   `_Node._theta` is one line and the marginal init is a hook
   (`has_marginal_start`/`marginal_start`, `transform.marginal_init_theta`).
4. **Node kinds stay an if/else in ONE place**: the four adjacent functions
   `kind_log_prob`/`kind_sample`/`kind_abduct`/`kind_marginal_theta` in
   nodes.py. The judges cut the drafted per-kind Head protocol as an n=2
   abstraction — a third node kind earns the protocol.
5. Extension points: `register_term` (a `ShiftTerm` subclass under its own
   effect name) and `fn_shift` (a callable / `nn.Module` in the additive
   shifts); `I(transform=<_ScaledUT subclass>)` for a custom basis.

## Refused (deliberately, so a later proposal can find the reasoning)

- No per-kind node protocol (n=2), no new node kinds at 1.0.
- No per-effect `Term` subclasses in the data layer — `Term` stays ONE frozen
  dataclass serialized by effect name; the polymorphism lives once, in terms.py.
- No transform/activation registry beyond `make_univariate_transform` accepting
  a class; no plugin entry points; no config system; no new dependencies.
- No custom latent distribution — the standard logistic IS the TRAM semantics;
  every pinned number depends on it.
- No collapse of ComplexIntercept/ComplexShift into factories: their names
  anchor checkpoint paths and the seeded RNG stream that ten CI ground truths pin.
- No new `fit` arguments: `vc_ehat=` stays the public spelling of the (now
  generic) side-input channel.
- Error messages reworded only where a responsibility physically moved.

## Consequences

- flow.py ≈ 900 lines of framework only; no effect string-switch survives
  outside terms.py; a new effect touches one class.
- One deliberate checkpoint break (0.4 unreleased): additive-CI keys
  `nodes.*.intercept_nets.*` → `nodes.*.intercept.nets.*` (parameters proven
  bit-identical under the rename).
- Every migration step landed with the full quick suite, a seeded
  state-dict-diff smoke (`tests/tools/statedict_smoke.py`), and experiment
  reproductions against the committed ground truths; guards were only
  tightened, never re-based silently.
