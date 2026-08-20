"""Synthetic-cohort generators for tramdag.

Research code, not framework code: it lives with the experiments that
consume it. The stroke cohort (``magic_mrclean``) and the VC validation
DGP (``vc_shift``) moved out with the stroke storyline — recover them
with ``git checkout pre-experiments-cut -- <path>`` if needed.

Each scenario is one module exposing a numpy-only SCM generator class with
known causal ground truth, imported from that module by name
(``from paper.simulations.triangle import TriangleContinuous``). Frozen CSVs
live under ``data/<name>/`` and are a contract — regenerate only deliberately
via each module's CLI.
"""
