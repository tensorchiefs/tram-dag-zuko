"""tramdag — Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch.

.. include:: ../../README.md
"""

from importlib.metadata import version

from . import simulations
from .env import machine_info
from .flow import CausalFlowDAG
from .spec import (
    CS,
    LS,
    VC,
    ContinuousNode,
    I,
    OrdinalNode,
    Term,
    complex_shift,
    intercept,
    linear_shift,
    varying_coefficient,
)

__all__ = [
    "CausalFlowDAG",
    "ContinuousNode",
    "OrdinalNode",
    "machine_info",
    "simulations",
    # term-formula notation: short aliases and their definitions
    "Term",
    "I",
    "intercept",
    "LS",
    "linear_shift",
    "CS",
    "complex_shift",
    "VC",
    "varying_coefficient",
]
__version__ = version("tramdag")
