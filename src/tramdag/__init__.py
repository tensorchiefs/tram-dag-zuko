"""tramdag — Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch.

.. include:: ../../README.md
"""

# %% imports ---------------------------------------------------------------------------
from importlib.metadata import version

from .flow import CausalFlowDAG
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
    simple_intercept,
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
    "simple_intercept",
    "varying_coefficient",
]
__version__ = version("tramdag")
