"""Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch."""

# %% imports ---------------------------------------------------------------------------
from importlib.metadata import version

from .flow import CausalFlowDAG
from .plots import plot_dag
from .spec import (
    CI,
    CS,
    LS,
    SI,
    VC,
    ComplexShift,
    ContinuousNode,
    Fn,
    FnShift,
    I,
    Intercept,
    LinearShift,
    OrdinalNode,
    Term,
    VaryingCoefficient,
    complex_intercept,
    complex_shift,
    fn_shift,
    intercept,
    linear_shift,
    node_parents,
    simple_intercept,
    spec_from_dict,
    spec_to_dict,
    validate_and_sort,
    varying_coefficient,
)

# %% global variables ------------------------------------------------------------------
__all__ = [
    "CI",
    "CS",
    "LS",
    "SI",
    "VC",
    "CausalFlowDAG",
    "ComplexShift",
    "ContinuousNode",
    "Fn",
    "FnShift",
    "I",
    "Intercept",
    "LinearShift",
    "OrdinalNode",
    "Term",
    "VaryingCoefficient",
    "complex_intercept",
    "complex_shift",
    "fn_shift",
    "intercept",
    "linear_shift",
    "node_parents",
    "plot_dag",
    "simple_intercept",
    "spec_from_dict",
    "spec_to_dict",
    "validate_and_sort",
    "varying_coefficient",
]
__version__ = version("tramdag")
