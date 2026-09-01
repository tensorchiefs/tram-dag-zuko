"""Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch."""

# %% imports ---------------------------------------------------------------------------
from importlib.metadata import version

from .flow import CausalFlowDAG
from .readouts import shift_curve
from .spec import (
    CI,
    CS,
    LS,
    SI,
    VC,
    ContinuousNode,
    I,
    OrdinalNode,
    Term,
    complex_intercept,
    complex_shift,
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
    "ContinuousNode",
    "I",
    "OrdinalNode",
    "Term",
    "complex_intercept",
    "complex_shift",
    "intercept",
    "linear_shift",
    "node_parents",
    "shift_curve",
    "simple_intercept",
    "spec_from_dict",
    "spec_to_dict",
    "validate_and_sort",
    "varying_coefficient",
]
__version__ = version("tramdag")
