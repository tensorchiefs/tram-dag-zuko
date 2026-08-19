"""tramdag — Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch.

.. include:: ../../README.md
"""

from importlib.metadata import version

from .env import machine_info
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
from .utils import load_config

__all__ = [
    "CausalFlowDAG",
    "ContinuousNode",
    "OrdinalNode",
    "machine_info",
    "load_config",
    # term-formula notation: short aliases and their definitions
    "Term",
    "I",
    "intercept",
    "SI",
    "simple_intercept",
    "CI",
    "complex_intercept",
    "LS",
    "linear_shift",
    "CS",
    "complex_shift",
    "VC",
    "varying_coefficient",
]
__version__ = version("tramdag")
