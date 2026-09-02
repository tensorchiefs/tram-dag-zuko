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
        flow["flow.py<br/>CausalFlowDAG: build, calibrate,<br/>log_prob, sample/abduct/pmf/density,<br/>save/load; composes the mixins"]
    end
    subgraph functions["flow behavior by concern"]
        fitting["fitting.py<br/>_FitMixin: fit (Adam loop, callbacks),<br/>fit_classical (L-BFGS)"]
        readouts["readouts.py<br/>_ReadoutsMixin: ls_coefficients,<br/>varying_coef, to_matrix, contributions,<br/>design_matrix, shift_curve"]
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
without torch executing any model code; `fitting.py`/`readouts.py` are
mixins `CausalFlowDAG` composes and import `flow` under `TYPE_CHECKING`
only — the graph is acyclic.)

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
        shift_value(node, feats)
        post_init()
        has_regularizer / regularizer()
        finalizes / finalize(node, feats)
        score_columns(node, flow, feats, dlds, ehat)
        side_columns() / check_column() / live_side() / extra_columns()
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
`option_defaults`, implements `build` (which must set `key`/`parents`/`net_parents`) +
`shift_value`, and registers with
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

<!-- AUTOGEN:diagrams (tools/gen_diagrams.py) — do not edit by hand -->
## Generated views

Regenerate with ``uv run python tools/gen_diagrams.py`` — the package UML
comes from pyreverse, the call graphs from a profile trace of one flow
construction and one three-epoch ``fit`` on a 3-node SI/LS/CS/VC spec
(tramdag-internal edges only; ``3x`` = once per node).

### Package UML (pyreverse)

```mermaid
classDiagram
  class tramdag {
  }
  class callbacks {
  }
  class conditioners {
  }
  class fitting {
  }
  class flow {
  }
  class nodes {
  }
  class readouts {
  }
  class scores {
  }
  class spec {
  }
  class terms {
  }
  class transforms {
  }
  tramdag --> flow
  tramdag --> spec
  tramdag --> terms
  fitting --> callbacks
  flow --> fitting
  flow --> nodes
  flow --> readouts
  flow --> scores
  flow --> spec
  flow --> terms
  flow --> transforms
  nodes --> spec
  nodes --> terms
  nodes --> transforms
  readouts --> conditioners
  readouts --> terms
  scores --> transforms
  spec --> terms
  terms --> conditioners
  terms --> spec
  fitting ..> flow
  terms ..> nodes
```

### Call graph — flow construction (traced)

```mermaid
flowchart LR
  subgraph conditioners
    n0["ComplexShift.__init__"]
    n28["LinearShift.__init__"]
    n27["SimpleIntercept.__init__"]
    n2["VaryingCoef.__init__"]
    n1["_nn"]
  end
  subgraph flow
    n3["CausalFlowDAG.__init__"]
    n4["CausalFlowDAG._apply_init"]
  end
  subgraph nodes
    n5["_Node.__init__"]
    n14["_Node._add_input_transform"]
    n7["_Node._build_intercept"]
    n8["_Node._build_shifts"]
  end
  subgraph spec
    n20["_check_input_transform"]
    n18["_check_node"]
    n19["_check_term"]
    n25["_kahn_sort"]
    n26["feat_width"]
    n9["node_parents"]
    n6["validate_and_sort"]
  end
  subgraph terms
    n15["CSTerm.build"]
    n12["InterceptTerm.build"]
    n16["LSTerm.build"]
    n21["LSTerm.check_arity"]
    n22["TermDef.check_arity"]
    n23["TermDef.edge_parents"]
    n17["VCTerm.build"]
    n24["VCTerm.edge_parents"]
    n13["get_term"]
  end
  subgraph transforms
    n29["BernsteinUT.__init__"]
    n10["BernsteinUT.n_params"]
    n30["_ScaledUT.__init__"]
    n11["make_univariate_transform"]
  end
    n0 --> n1
    n2 --> n1
    n3 --> n4
    n3 -- "3x" --> n5
    n3 --> n6
    n5 -- "3x" --> n7
    n5 -- "3x" --> n8
    n5 -- "3x" --> n9
    n5 -- "2x" --> n10
    n5 -- "2x" --> n11
    n7 -- "3x" --> n12
    n7 -- "3x" --> n13
    n8 -- "3x" --> n14
    n8 --> n15
    n8 --> n16
    n8 --> n17
    n8 -- "6x" --> n13
    n18 -- "6x" --> n19
    n19 -- "6x" --> n20
    n19 --> n21
    n19 -- "5x" --> n22
    n19 -- "5x" --> n23
    n19 --> n24
    n19 -- "6x" --> n13
    n25 -- "3x" --> n9
    n6 -- "3x" --> n18
    n6 --> n25
    n15 --> n0
    n15 --> n26
    n12 -- "3x" --> n27
    n16 --> n28
    n16 --> n26
    n17 --> n2
    n17 --> n26
    n29 -- "2x" --> n30
    n11 -- "2x" --> n29
```

### Call graph — one fit (traced)

```mermaid
flowchart LR
  subgraph callbacks
    n0["EarlyStopping.__init__"]
    n1["EarlyStopping._reset"]
    n2["EarlyStopping.on_epoch_end"]
    n4["EarlyStopping.on_fit_begin"]
    n8["EarlyStopping.on_fit_end"]
    n3["_last_val"]
  end
  subgraph conditioners
    n51["ComplexShift.forward"]
    n53["LinearShift.forward"]
    n54["SimpleIntercept.forward"]
    n6["VaryingCoef.beta"]
    n5["VaryingCoef.forward"]
    n56["VaryingCoef.l2"]
    n55["VaryingCoef.recenter"]
  end
  subgraph fitting
    n7["_FitMixin.fit"]
    n9["_check_fit_sizes"]
    n10["_epoch_pass"]
    n20["_fit_epoch"]
    n11["_log_epoch"]
    n12["_normalize_callbacks"]
    n13["_split_validation"]
    n21["_val_nll"]
  end
  subgraph flow
    n37["CausalFlowDAG._check_levels"]
    n14["CausalFlowDAG._check_side_columns"]
    n32["CausalFlowDAG._dtype"]
    n27["CausalFlowDAG._encode_parent"]
    n26["CausalFlowDAG._features"]
    n28["CausalFlowDAG._marginal_start"]
    n31["CausalFlowDAG._np_dtype"]
    n15["CausalFlowDAG._recenter_vc"]
    n34["CausalFlowDAG._set_range"]
    n36["CausalFlowDAG._side_feats"]
    n16["CausalFlowDAG._tensorize"]
    n17["CausalFlowDAG.calibrate"]
    n38["CausalFlowDAG.init_marginals"]
    n22["CausalFlowDAG.node_log_prob"]
  end
  subgraph nodes
    n52["_Node.net_input"]
    n39["_Node.set_input_stats"]
    n40["_Node.theta_shift"]
    n41["kind_log_prob"]
    n29["kind_marginal_theta"]
  end
  subgraph terms
    n42["CSTerm.shift_value"]
    n43["LSTerm.shift_value"]
    n30["SITerm.marginal_start"]
    n44["SITerm.theta_value"]
    n18["ShiftTerm.has_regularizer"]
    n24["ShiftTerm.side_columns"]
    n33["VCTerm.finalize"]
    n19["VCTerm.has_regularizer"]
    n57["VCTerm.regressor"]
    n23["VCTerm.regularizer"]
    n45["VCTerm.shift_value"]
    n25["VCTerm.side_columns"]
  end
  subgraph transforms
    n58["BernsteinUT._build"]
    n49["BernsteinUT.marginal_init_theta"]
    n46["StandardLogistic.log_prob"]
    n59["_ScaledUT._log_dt_dx"]
    n60["_ScaledUT._scale"]
    n47["_ScaledUT.forward"]
    n35["_ScaledUT.set_range"]
    n63["_log1mexp"]
    n61["ordinal_bounds"]
    n62["ordinal_cutpoints"]
    n48["ordinal_log_prob"]
    n50["ordinal_marginal_init_theta"]
  end
    n0 --> n1
    n2 -- "3x" --> n3
    n4 --> n1
    n5 -- "9x" --> n6
    n7 -- "3x" --> n2
    n7 --> n4
    n7 --> n8
    n7 --> n9
    n7 -- "3x" --> n10
    n7 -- "3x" --> n11
    n7 --> n12
    n7 --> n13
    n7 --> n14
    n7 --> n15
    n7 -- "2x" --> n16
    n7 --> n17
    n7 -- "2x" --> n18
    n7 --> n19
    n10 -- "3x" --> n20
    n10 -- "3x" --> n21
    n20 -- "6x" --> n22
    n20 -- "6x" --> n23
    n21 -- "3x" --> n22
    n14 -- "2x" --> n24
    n14 --> n25
    n26 -- "30x" --> n27
    n28 -- "3x" --> n29
    n28 -- "3x" --> n30
    n31 -- "2x" --> n32
    n15 --> n26
    n15 --> n33
    n34 -- "2x" --> n35
    n36 -- "18x" --> n24
    n36 -- "9x" --> n25
    n16 -- "2x" --> n31
    n17 --> n37
    n17 -- "2x" --> n34
    n17 --> n38
    n17 -- "3x" --> n39
    n38 --> n37
    n38 -- "3x" --> n28
    n22 -- "9x" --> n26
    n22 -- "27x" --> n36
    n22 -- "27x" --> n40
    n22 -- "27x" --> n41
    n40 -- "9x" --> n42
    n40 -- "9x" --> n43
    n40 -- "27x" --> n44
    n40 -- "9x" --> n45
    n41 -- "18x" --> n46
    n41 -- "18x" --> n47
    n41 -- "9x" --> n48
    n29 -- "2x" --> n49
    n29 --> n50
    n42 -- "9x" --> n51
    n42 -- "9x" --> n52
    n43 -- "9x" --> n53
    n44 -- "27x" --> n54
    n33 --> n55
    n33 --> n52
    n23 -- "6x" --> n56
    n45 -- "9x" --> n5
    n45 -- "9x" --> n52
    n45 -- "9x" --> n57
    n47 -- "18x" --> n58
    n47 -- "18x" --> n59
    n47 -- "18x" --> n60
    n61 -- "9x" --> n62
    n48 -- "9x" --> n63
    n48 -- "9x" --> n61
```
<!-- AUTOGEN:end -->
