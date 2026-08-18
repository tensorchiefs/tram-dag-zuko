"""Helpers shared by the SCM generators.

Each generator owns its structural equations; what they share is the
boilerplate around them — drawing the standard-logistic latent, resolving
the ``(n, rng, latents)`` triple, and the seed offsets of the named
datasets.
"""

from __future__ import annotations

import numpy as np


def logistic(rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw the standard logistic latent, the TRAM base distribution."""
    return rng.logistic(loc=0.0, size=size)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Give the logistic function of ``x``."""
    return 1.0 / (1.0 + np.exp(-x))


def resolve_latents(gen, n, rng, latents) -> tuple[dict[str, np.ndarray], int]:
    """Give the latents to simulate from, and their row count.

    Draws a fresh set from ``gen.draw_latents`` unless ``latents`` is
    given, in which case ``n`` and ``rng`` are ignored.

    Raises
    ------
    ValueError
        If neither ``n`` nor ``latents`` is given.
    """
    if latents is not None:
        return latents, len(next(iter(latents.values())))
    if n is None:
        raise ValueError("provide either n or latents")
    return gen.draw_latents(n, rng or np.random.default_rng(gen.seed)), n
