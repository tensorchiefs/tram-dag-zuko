"""Synthetic-cohort generators for tramdag.

Research code, not framework code: it lives with the experiments that
consume it. The stroke cohort (``magic_mrclean``) and the VC validation
DGP (``vc_shift``) moved out with the stroke storyline — recover them
with ``git checkout pre-experiments-cut -- <path>`` if needed.

Each scenario is one module exposing a numpy-only SCM generator class with known
causal ground truth. New scenarios register here so experiments/tests can look
them up by name. Frozen CSVs live under ``data/<name>/`` and are a contract —
regenerate only deliberately via each module's CLI.
"""

from .carefl import Carefl4
from .triangle import TriangleContinuous, TriangleMixed
from .vaca import VacaTriangle

__all__ = [
    "TriangleContinuous",
    "TriangleMixed",
    "VacaTriangle",
    "Carefl4",
]
