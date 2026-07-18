"""Synthetic-cohort generators for tramdag.

Each scenario is one module exposing a numpy-only SCM generator class with known
causal ground truth. New scenarios register here so experiments/tests can look
them up by name. Frozen CSVs live under ``data/<name>/`` and are a contract —
regenerate only deliberately via each module's CLI.
"""

from .carefl import Carefl4
from .magic_mrclean import MagicMrClean
from .triangle import TriangleContinuous, TriangleMixed
from .vaca import VacaTriangle
from .vc_shift import VCLogisticShift

REGISTRY = {
    "magic-mrclean": MagicMrClean,
    "triangle": TriangleContinuous,
    "triangle-mixed": TriangleMixed,
    "vaca": VacaTriangle,
    "carefl": Carefl4,
    "vc-shift": VCLogisticShift,
}

__all__ = ["MagicMrClean", "TriangleContinuous", "TriangleMixed",
           "VacaTriangle", "Carefl4", "VCLogisticShift", "REGISTRY"]
