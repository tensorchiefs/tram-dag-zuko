# Architecture

Ten modules, one rule: **term-specific behavior lives on the effect's two
classes — its `Term` subclass (spec) and its module (`terms.py`); node-kind
behavior lives in four adjacent functions; everything else is framework.** The decisions and their refused alternatives are recorded in
[ADR 001](adr/001-term-owned-architecture.md).

## Module map

```mermaid
graph TD
    subgraph data["pure data"]
        spec["spec.py<br/>DSL: Term + one subclass per effect<br/>(I LS CS VC Fn), nodes, normalization,<br/>Kahn sort, (de)serialization"]
    end
    subgraph torch["torch modules"]
        terms["terms.py<br/>one module per effect, data = its Term class;<br/>ShiftTerm/InterceptTerm hooks; module_for;<br/>LSTerm CSTerm VCTerm FnTerm,<br/>SITerm CITerm AdditiveCITerm"]
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

(`spec.py` imports nothing from `terms.py`, so the data layer stays importable
without torch executing any model code; `fitting.py`/`readouts.py` are
mixins `CausalFlowDAG` composes and import `flow` under `TYPE_CHECKING`
only — the graph is acyclic.)

## The term contract

```mermaid
classDiagram
    class Term {
        <<spec.py, frozen data>>
        parents
        effect: the class name
        option fields with defaults
        __post_init__(): arity, option values
        edge_parents(name, spec)
        cells()
        classical
        options() / from_serialized()
    }
    Term <|-- I
    Term <|-- LS
    Term <|-- CS
    Term <|-- VC
    Term <|-- Fn
    class TermDef {
        <<terms.py, module>>
        data: the Term subclass it builds
        input_transform / calibrate(train_df)
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
A custom effect is two classes: a `tramdag.Term` subclass (its annotated
attributes are the options; `__post_init__`, `edge_parents`, `cells` its
rules) and a `ShiftTerm` subclass declaring `data =` that class and
implementing `build` (which must set `key`/`parents`/`net_parents`) +
`shift_value`. Subclassing is the registration; the cheap path for a
one-off is `fn_shift`.

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
  class CS {
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
  class Fn {
  }
  class FnTerm {
  }
  class I {
  }
  class InterceptTerm {
  }
  class LS {
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
  class VC {
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
  CS --|> Term
  Fn --|> Term
  I --|> Term
  LS --|> Term
  VC --|> Term
  AdditiveCITerm --|> InterceptTerm
  CITerm --|> ComplexIntercept
  CITerm --|> InterceptTerm
  CSTerm --|> ComplexShift
  CSTerm --|> ShiftTerm
  FnTerm --|> ShiftTerm
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
  CS --o CSTerm : data
  Fn --o FnTerm : data
  I --o InterceptTerm : data
  LS --o LSTerm : data
  VC --o VCTerm : data
```

### Call graph — flow construction (traced)

```mermaid
flowchart LR
  subgraph conditioners
    n0["ComplexShift.__init__"]
    n25["LinearShift.__init__"]
    n24["SimpleIntercept.__init__"]
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
    n19["Term.edge_parents"]
    n20["VC.edge_parents"]
    n17["_check_node"]
    n18["_check_term"]
    n21["_kahn_sort"]
    n26["_subclasses"]
    n22["feat_width"]
    n9["node_parents"]
    n6["validate_and_sort"]
  end
  subgraph terms
    n14["CSTerm.build"]
    n12["InterceptTerm.build"]
    n15["LSTerm.build"]
    n16["VCTerm.build"]
    n23["_attach_input_transform"]
    n13["module_for"]
  end
  subgraph transforms
    n27["BernsteinUT.__init__"]
    n10["BernsteinUT.n_params"]
    n28["_ScaledUT.__init__"]
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
    n8 -- "3x" --> n13
    n17 -- "6x" --> n18
    n18 -- "5x" --> n19
    n18 --> n20
    n21 -- "3x" --> n9
    n6 -- "3x" --> n17
    n6 --> n21
    n14 --> n0
    n14 --> n22
    n14 --> n23
    n12 -- "3x" --> n24
    n15 --> n25
    n15 --> n22
    n16 --> n2
    n16 --> n22
    n16 --> n23
    n13 -- "6x" --> n26
    n27 -- "2x" --> n28
    n11 -- "2x" --> n27
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
    n48["ComplexShift.forward"]
    n50["LinearShift.forward"]
    n51["SimpleIntercept.forward"]
    n6["VaryingCoef.beta"]
    n5["VaryingCoef.forward"]
    n53["VaryingCoef.l2"]
    n52["VaryingCoef.recenter"]
  end
  subgraph fitting
    n7["_FitMixin.fit"]
    n9["_check_fit_sizes"]
    n10["_epoch_pass"]
    n21["_fit_epoch"]
    n11["_learning_rates"]
    n12["_log_epoch"]
    n13["_normalize_callbacks"]
    n14["_split_validation"]
    n22["_val_nll"]
  end
  subgraph flow
    n33["CausalFlowDAG._check_columns"]
    n34["CausalFlowDAG._check_levels"]
    n15["CausalFlowDAG._check_side_columns"]
    n30["CausalFlowDAG._dtype"]
    n28["CausalFlowDAG._encode_parent"]
    n27["CausalFlowDAG._features"]
    n29["CausalFlowDAG._np_dtype"]
    n16["CausalFlowDAG._recenter_vc"]
    n32["CausalFlowDAG._side_feats"]
    n17["CausalFlowDAG._tensorize"]
    n18["CausalFlowDAG.calibrate"]
    n23["CausalFlowDAG.node_log_prob"]
  end
  subgraph nodes
    n39["_Node.net_input"]
    n37["_Node.theta_shift"]
    n38["kind_log_prob"]
  end
  subgraph terms
    n41["CSTerm.shift_value"]
    n35["InterceptTerm.calibrate"]
    n42["LSTerm.shift_value"]
    n43["SITerm.theta_value"]
    n19["ShiftTerm.has_regularizer"]
    n25["ShiftTerm.side_columns"]
    n36["TermDef.calibrate"]
    n40["TermDef.input_transform"]
    n31["VCTerm.finalize"]
    n20["VCTerm.has_regularizer"]
    n54["VCTerm.regressor"]
    n24["VCTerm.regularizer"]
    n44["VCTerm.shift_value"]
    n26["VCTerm.side_columns"]
  end
  subgraph transforms
    n55["BernsteinUT._build"]
    n45["StandardLogistic.log_prob"]
    n56["_ScaledUT._log_dt_dx"]
    n57["_ScaledUT._scale"]
    n46["_ScaledUT.forward"]
    n49["_ScaledUT.set_range"]
    n60["_log1mexp"]
    n58["ordinal_bounds"]
    n59["ordinal_cutpoints"]
    n47["ordinal_log_prob"]
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
    n7 -- "3x" --> n12
    n7 --> n13
    n7 --> n14
    n7 --> n15
    n7 --> n16
    n7 -- "2x" --> n17
    n7 --> n18
    n7 -- "2x" --> n19
    n7 --> n20
    n10 -- "3x" --> n21
    n10 -- "3x" --> n22
    n21 -- "6x" --> n23
    n21 -- "6x" --> n24
    n22 -- "3x" --> n23
    n15 -- "2x" --> n25
    n15 --> n26
    n27 -- "30x" --> n28
    n29 -- "2x" --> n30
    n16 --> n27
    n16 --> n31
    n32 -- "18x" --> n25
    n32 -- "9x" --> n26
    n17 -- "2x" --> n33
    n17 -- "2x" --> n29
    n18 --> n33
    n18 --> n34
    n18 -- "3x" --> n35
    n18 -- "3x" --> n36
    n23 -- "9x" --> n27
    n23 -- "27x" --> n32
    n23 -- "27x" --> n37
    n23 -- "27x" --> n38
    n39 -- "19x" --> n40
    n37 -- "9x" --> n41
    n37 -- "9x" --> n42
    n37 -- "27x" --> n43
    n37 -- "9x" --> n44
    n38 -- "18x" --> n45
    n38 -- "18x" --> n46
    n38 -- "9x" --> n47
    n41 -- "9x" --> n48
    n41 -- "9x" --> n39
    n35 -- "3x" --> n36
    n35 -- "2x" --> n49
    n42 -- "9x" --> n50
    n43 -- "27x" --> n51
    n36 -- "6x" --> n40
    n31 --> n52
    n31 --> n39
    n24 -- "6x" --> n53
    n44 -- "9x" --> n5
    n44 -- "9x" --> n39
    n44 -- "9x" --> n54
    n46 -- "18x" --> n55
    n46 -- "18x" --> n56
    n46 -- "18x" --> n57
    n58 -- "9x" --> n59
    n47 -- "9x" --> n60
    n47 -- "9x" --> n58
```
<!-- AUTOGEN:end -->
