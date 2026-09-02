# Architecture

Nine modules, one rule: **term-specific behavior lives on the term's registry
entry; node-kind behavior lives in four adjacent functions; everything else is
framework.** The decisions and their refused alternatives are recorded in
[ADR 001](adr/001-term-owned-architecture.md).

## Module map

```mermaid
graph TD
    subgraph data["pure data"]
        spec["spec.py<br/>DSL: Term, nodes, constructors,<br/>normalization, Kahn sort, (de)serialization"]
    end
    subgraph torch["torch modules"]
        terms["terms.py<br/>registry: one TermDef per effect;<br/>ShiftTerm/InterceptTerm hooks;<br/>LSTerm CSTerm VCTerm FnTerm,<br/>SITerm CITerm AdditiveCITerm"]
        conditioners["conditioners.py<br/>raw nn heads (frozen:<br/>anchors checkpoints + RNG)"]
        transforms["transforms.py<br/>Bernstein/Spline/Affine,<br/>ordinal_* likelihood,<br/>StandardLogistic"]
        nodes["nodes.py<br/>_Node (intercept + shifts),<br/>_InputTransform,<br/>kind_log_prob/sample/abduct/<br/>marginal_theta"]
        flow["flow.py<br/>CausalFlowDAG: build, calibrate,<br/>log_prob, sample/abduct/pmf/density,<br/>save/load, delegates"]
    end
    subgraph functions["free functions over a flow"]
        fitting["fitting.py<br/>fit (Adam loop, callbacks),<br/>fit_classical (L-BFGS)"]
        readouts["readouts.py<br/>ls_coefficients, varying_coef,<br/>to_matrix, contributions,<br/>design_matrix, shift_curve"]
        scores["scores.py<br/>node_scores,<br/>effect_modifier_scan"]
    end
    callbacks["callbacks.py<br/>Callback, EarlyStopping,<br/>PerNodePlateau, per_node_adam"]

    spec --> terms
    terms --> conditioners
    nodes --> terms
    nodes --> spec
    nodes --> transforms
    flow --> nodes
    flow --> fitting
    flow --> readouts
    flow --> scores
    fitting --> callbacks
```

(`spec.py` consults the registry lazily, so the data layer stays importable
without torch executing any model code; `fitting.py`/`readouts.py` import
`flow` under `TYPE_CHECKING` only — the graph is acyclic.)

## The term contract

```mermaid
classDiagram
    class TermDef {
        <<registry entry>>
        effect: str
        slot: "intercept"|"shift"
        option_defaults: dict
        check_arity(name, term)
        edge_parents(name, term, spec)
        cells(term)
        term_is_classical(term)
    }
    class ShiftTerm {
        key / parents / net_parents
        build(term, spec)
        shift_value(node, feats, vc_ehat)
        post_init()
        has_regularizer / regularizer()
        finalizes / finalize(node, feats)
        score_columns(node, flow, feats, dlds, ehat)
        side_keys() / check_side() / live_side() / extra_columns()
    }
    class InterceptTerm {
        groups / ci_parents
        build(term, spec, n_params)
        theta_value(node, feats, n)
        has_marginal_start / marginal_start(theta)
    }
    TermDef <|-- ShiftTerm
    TermDef <|-- InterceptTerm
    ShiftTerm <|-- LSTerm
    ShiftTerm <|-- CSTerm
    ShiftTerm <|-- VCTerm
    ShiftTerm <|-- FnTerm
    InterceptTerm <|-- SITerm
    InterceptTerm <|-- CITerm
    InterceptTerm <|-- AdditiveCITerm
    LSTerm --|> LinearShift : nn
    CSTerm --|> ComplexShift : nn
    VCTerm --|> VaryingCoef : nn
    SITerm --|> SimpleIntercept : nn
    CITerm --|> ComplexIntercept : nn
```

Built-in terms subclass their conditioners, so state-dict paths
(`nodes.<n>.shifts.<key>.…`) and the seeded RNG stream are those of 0.4.
A custom effect subclasses `ShiftTerm`, sets `effect`/`slot`/
`option_defaults`, implements `build` + `shift_value`, and registers with
`register_term`; the cheap path for a one-off is `fn_shift`.

## Node kinds

Two kinds (continuous, ordinal) stay an if/else — in ONE place:
`kind_log_prob` / `kind_sample` / `kind_abduct` / `kind_marginal_theta`,
adjacent in nodes.py. A third kind is the trigger for a protocol, not before.

## Guards that pin all of this

- `tests/tools/statedict_smoke.py` — seeded per-DGP state dicts, bit-compared.
- The inline DGP truths (`tests/conftest.py`) and 45+ regex-pinned refusals.
- `experiments/*/ground_truth/*.json` — ten CI-checked replications with
  wall-time tripwires; centers move only with a documented reason.
