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
    node_parents,
    spec_from_dict,
    spec_to_dict,
    validate_and_sort,
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
    "node_parents",
    "plot_dag",
    "spec_from_dict",
    "spec_to_dict",
    "validate_and_sort",
]
__version__ = version("tramdag")
