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
    plots["plots.py<br/>plot_dag, plot_marginals,<br/>plot_training (matplotlib optional)"]

    spec --> terms
    plots --> spec
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

Regenerate with ``uv run python tools/gen_diagrams.py`` — the package and
class UML come from pyreverse (classes: names and inheritance only), the
call graphs from a profile trace of one flow construction and one
three-epoch ``fit`` on a 3-node SI/LS/CS/VC spec (tramdag-internal edges
only; ``3x`` = once per node).

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
  class plots {
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
  tramdag --> plots
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
  plots --> spec
  readouts --> conditioners
  readouts --> terms
  scores --> transforms
  spec --> terms
  terms --> conditioners
  terms --> spec
  fitting ..> flow
  terms ..> nodes
```

### Class UML (pyreverse)

The built-in terms are conditioners with a term-hook mixin: each concrete
term class inherits its network from ``conditioners`` and its contract
from ``ShiftTerm``/``InterceptTerm``.

```mermaid
classDiagram
  class AdditiveCITerm {
  }
  class AffineUT {
  }
  class BernsteinUT {
  }
  class CITerm {
  }
  class CSTerm {
  }
  class Callback {
  }
  class CausalFlowDAG {
  }
  class ComplexIntercept {
  }
  class ComplexShift {
  }
  class ContinuousNode {
  }
  class EarlyStopping {
  }
  class FnTerm {
  }
  class InterceptDef {
  }
  class InterceptTerm {
  }
  class LSTerm {
  }
  class LinearShift {
  }
  class OrdinalNode {
  }
  class PerNodePlateau {
  }
  class SITerm {
  }
  class ShiftTerm {
  }
  class SimpleIntercept {
  }
  class SplineUT {
  }
  class StandardLogistic {
  }
  class Term {
  }
  class TermDef {
  }
  class VCTerm {
  }
  class VaryingCoef {
  }
  class _FitMixin {
  }
  class _FnCallback {
  }
  class _InputTransform {
  }
  class _Node {
  }
  class _ReadoutsMixin {
  }
  class _ScaledUT {
  }
  EarlyStopping --|> Callback
  PerNodePlateau --|> Callback
  _FnCallback --|> Callback
  CausalFlowDAG --|> _FitMixin
  CausalFlowDAG --|> _ReadoutsMixin
  AdditiveCITerm --|> InterceptTerm
  CITerm --|> ComplexIntercept
  CITerm --|> InterceptTerm
  CSTerm --|> ComplexShift
  CSTerm --|> ShiftTerm
  FnTerm --|> ShiftTerm
  InterceptDef --|> InterceptTerm
  InterceptTerm --|> TermDef
  LSTerm --|> LinearShift
  LSTerm --|> ShiftTerm
  SITerm --|> SimpleIntercept
  SITerm --|> InterceptTerm
  ShiftTerm --|> TermDef
  VCTerm --|> VaryingCoef
  VCTerm --|> ShiftTerm
  AffineUT --|> _ScaledUT
  BernsteinUT --|> _ScaledUT
  SplineUT --|> _ScaledUT
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
    n7["_Node._build_intercept"]
    n8["_Node._build_shifts"]
  end
  subgraph spec
    n19["_check_input_transform"]
    n17["_check_node"]
    n18["_check_term"]
    n24["_kahn_sort"]
    n25["feat_width"]
    n9["node_parents"]
    n6["validate_and_sort"]
  end
  subgraph terms
    n14["CSTerm.build"]
    n12["InterceptTerm.build"]
    n15["LSTerm.build"]
    n20["LSTerm.check_arity"]
    n21["TermDef.check_arity"]
    n22["TermDef.edge_parents"]
    n16["VCTerm.build"]
    n23["VCTerm.edge_parents"]
    n26["_attach_input_transform"]
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
    n8 --> n14
    n8 --> n15
    n8 --> n16
    n8 -- "6x" --> n13
    n17 -- "6x" --> n18
    n18 -- "6x" --> n19
    n18 --> n20
    n18 -- "5x" --> n21
    n18 -- "5x" --> n22
    n18 --> n23
    n18 -- "6x" --> n13
    n24 -- "3x" --> n9
    n6 -- "3x" --> n17
    n6 --> n24
    n14 --> n0
    n14 --> n25
    n14 --> n26
    n12 -- "3x" --> n27
    n15 --> n28
    n15 --> n25
    n16 --> n2
    n16 --> n25
    n16 --> n26
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
    n47["ComplexShift.forward"]
    n49["LinearShift.forward"]
    n50["SimpleIntercept.forward"]
    n6["VaryingCoef.beta"]
    n5["VaryingCoef.forward"]
    n52["VaryingCoef.l2"]
    n51["VaryingCoef.recenter"]
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
    n32["CausalFlowDAG._check_columns"]
    n33["CausalFlowDAG._check_levels"]
    n14["CausalFlowDAG._check_side_columns"]
    n29["CausalFlowDAG._dtype"]
    n27["CausalFlowDAG._encode_parent"]
    n26["CausalFlowDAG._features"]
    n28["CausalFlowDAG._np_dtype"]
    n15["CausalFlowDAG._recenter_vc"]
    n31["CausalFlowDAG._side_feats"]
    n16["CausalFlowDAG._tensorize"]
    n17["CausalFlowDAG.calibrate"]
    n22["CausalFlowDAG.node_log_prob"]
  end
  subgraph nodes
    n38["_Node.net_input"]
    n36["_Node.theta_shift"]
    n37["kind_log_prob"]
  end
  subgraph terms
    n40["CSTerm.shift_value"]
    n34["InterceptTerm.calibrate"]
    n41["LSTerm.shift_value"]
    n42["SITerm.theta_value"]
    n18["ShiftTerm.has_regularizer"]
    n24["ShiftTerm.side_columns"]
    n35["TermDef.calibrate"]
    n39["TermDef.input_transform"]
    n30["VCTerm.finalize"]
    n19["VCTerm.has_regularizer"]
    n53["VCTerm.regressor"]
    n23["VCTerm.regularizer"]
    n43["VCTerm.shift_value"]
    n25["VCTerm.side_columns"]
  end
  subgraph transforms
    n54["BernsteinUT._build"]
    n44["StandardLogistic.log_prob"]
    n55["_ScaledUT._log_dt_dx"]
    n56["_ScaledUT._scale"]
    n45["_ScaledUT.forward"]
    n48["_ScaledUT.set_range"]
    n59["_log1mexp"]
    n57["ordinal_bounds"]
    n58["ordinal_cutpoints"]
    n46["ordinal_log_prob"]
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
    n28 -- "2x" --> n29
    n15 --> n26
    n15 --> n30
    n31 -- "18x" --> n24
    n31 -- "9x" --> n25
    n16 -- "2x" --> n32
    n16 -- "2x" --> n28
    n17 --> n32
    n17 --> n33
    n17 -- "3x" --> n34
    n17 -- "3x" --> n35
    n22 -- "9x" --> n26
    n22 -- "27x" --> n31
    n22 -- "27x" --> n36
    n22 -- "27x" --> n37
    n38 -- "19x" --> n39
    n36 -- "9x" --> n40
    n36 -- "9x" --> n41
    n36 -- "27x" --> n42
    n36 -- "9x" --> n43
    n37 -- "18x" --> n44
    n37 -- "18x" --> n45
    n37 -- "9x" --> n46
    n40 -- "9x" --> n47
    n40 -- "9x" --> n38
    n34 -- "3x" --> n35
    n34 -- "2x" --> n48
    n41 -- "9x" --> n49
    n42 -- "27x" --> n50
    n35 -- "6x" --> n39
    n30 --> n51
    n30 --> n38
    n23 -- "6x" --> n52
    n43 -- "9x" --> n5
    n43 -- "9x" --> n38
    n43 -- "9x" --> n53
    n45 -- "18x" --> n54
    n45 -- "18x" --> n55
    n45 -- "18x" --> n56
    n57 -- "9x" --> n58
    n46 -- "9x" --> n59
    n46 -- "9x" --> n57
```
<!-- AUTOGEN:end -->
