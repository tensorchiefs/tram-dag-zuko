"""Helpers shared by the SCM generators.

Each generator owns its structural equations; what they share is the
boilerplate around them — drawing the standard-logistic latent, resolving
the ``(n, rng, latents)`` triple, and the seed offsets of the named
datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


class DatasetDraws:
    """The named dataset draws every generator shares.

    Each generator owns ``simulate`` and ``draw_latents``; the draws below
    only pick the random stream. The seed offsets are part of the frozen
    data contract in ``data/``, so they live here once: observational
    ``+1``, interventional ``+501``, counterfactual ``+2``.
    """

    def observational(self, n: int, seed_offset: int = 0) -> pd.DataFrame:
        """Draw an observational sample of ``n`` rows."""
        rng = np.random.default_rng(self.seed + 1 + seed_offset)
        return self.simulate(n, rng=rng)

    def interventional(self, n: int, do: dict[str, float]) -> pd.DataFrame:
        """Draw ``n`` rows from the mutilated SCM — the L2 ground truth."""
        rng = np.random.default_rng(self.seed + 501)
        return self.simulate(n, rng=rng, do=do)

    def counterfactual_pair(
        self, n: int, do: dict[str, float]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Draw a factual sample and its counterfactual under ``do``.

        Both arms share one latent draw, so the pair gives true individual
        counterfactuals — which real data can never supply. Use it to score
        a flow's abduction.
        """
        rng = np.random.default_rng(self.seed + 2)
        latents = self.draw_latents(n, rng)
        return self.simulate(latents=latents), self.simulate(latents=latents, do=do)
