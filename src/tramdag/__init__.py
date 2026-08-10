"""tramdag — Interpretable Neural Causal Models (TRAM-DAGs) in PyTorch.

One triangular normalizing flow (built on `zuko <https://zuko.readthedocs.io>`_)
whose Jacobian sparsity is the causal DAG: fit once on observational data,
then answer observational (L1), interventional (L2) and counterfactual (L3)
queries. Reference: Sick & Dürr, *Interpretable Neural Causal Models with
TRAM-DAGs*, CLeaR 2025 (arXiv:2503.16206).

Conventional import alias::

    import tramdag as td

    flow = td.CausalFlowDAG(spec)
    td.simulations.REGISTRY          # synthetic DGPs with known ground truth
"""

from importlib.metadata import version

from . import simulations
from .env import machine_info
from .flow import CausalFlowDAG
from .spec import (CS, LS, VC, CShift, ContinuousNode, I, Intercept, LinShift,
                   OrdinalNode, Term, term)

__all__ = ["CausalFlowDAG", "ContinuousNode", "OrdinalNode", "machine_info",
           "simulations",
           # term-formula notation
           "Term", "I", "LS", "CS", "VC", "term", "Intercept", "LinShift",
           "CShift"]
__version__ = version("tramdag")
