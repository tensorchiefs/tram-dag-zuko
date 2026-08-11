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
    Transformation,
    term,
)

__all__ = [
    "CausalFlowDAG",
    "ContinuousNode",
    "OrdinalNode",
    "machine_info",
    "simulations",
    # term-formula notation
    "Term",
    "Transformation",
    "I",
    "LS",
    "CS",
    "VC",
    "term",
]
__version__ = version("tramdag")
