"""The CAREFL counterfactual benchmark DGP (TRAM-DAG paper App. C.2).

Originally from Khemakhem et al. (2021, CAREFL Fig. 5), used in the paper to
benchmark TRAM-DAG's L3 (counterfactual) queries (Fig. 6)::

    x1, x2 ~ Laplace(0, 1/sqrt(2))
    x3 = x1 + 0.5 x2^3 + Laplace(0, 1/sqrt(2))
    x4 = -x2 + 0.5 x1^2 + Laplace(0, 1/sqrt(2))

All-continuous additive-noise SCM, so individual counterfactuals are **analytic**
via noise abduction (no Monte Carlo): eps3 = x3 - x1 - 0.5 x2^3 and
eps4 = x4 + x2 - 0.5 x1^2 are recovered exactly, then the mutilated SCM is
re-evaluated. The paper picks the observation ``X_OBS = (2.00, 1.50, 0.81, -0.28)``
and sweeps two queries for alpha in [-3, 3]:

    (i)  x3^cf  given do(x2 = alpha):  x1 + 0.5 alpha^3 + eps3
    (ii) x4^cf  given do(x1 = alpha): -x2 + 0.5 alpha^2 + eps4

CLI::

    uv run python -m tramdag.simulations.carefl --out data/carefl --seed 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import DatasetDraws, resolve_latents

X_OBS = {"x1": 2.00, "x2": 1.50, "x3": 0.81, "x4": -0.28}  # the paper's observation
ALPHA_GRID = np.round(np.linspace(-3.0, 3.0, 61), 4)
_SCALE = 1.0 / np.sqrt(2.0)


@dataclass
class Carefl4(DatasetDraws):
    """SCM generator for the 4-variable CAREFL benchmark."""

    seed: int = 42

    def draw_latents(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw the latent noise of every variable, ``n`` rows each."""
        return {
            k: rng.laplace(loc=0.0, scale=_SCALE, size=n)
            for k in ["x1", "x2", "x3", "x4"]
        }

    def simulate(
        self,
        n: int | None = None,
        *,
        rng: np.random.Generator | None = None,
        do: dict[str, float] | None = None,
        latents: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """Simulate the SCM, with optional interventions and reused latents.

        Parameters
        ----------
        n : int | None, optional
            Number of rows, by default ``None``. Then ``latents`` sets the count.
        rng : np.random.Generator | None, optional
            Random source, by default ``None``.
        do : dict[str, float] | None, optional
            Variables to hold at a fixed value, by default ``None``.
        latents : dict[str, np.ndarray] | None, optional
            Latent values to reuse, by default ``None``. Then they are drawn fresh.

        Returns
        -------
        pd.DataFrame
            One column per variable.
        """
        do = do or {}
        latents, n = resolve_latents(self, n, rng, latents)

        def clamp_or(name, value):
            return np.full(n, float(do[name])) if name in do else value

        x1 = clamp_or("x1", latents["x1"])
        x2 = clamp_or("x2", latents["x2"])
        x3 = clamp_or("x3", x1 + 0.5 * x2**3 + latents["x3"])
        x4 = clamp_or("x4", -x2 + 0.5 * x1**2 + latents["x4"])
        return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})

    # -------------------------------------------------------------- ground truth
    @staticmethod
    def abduct_noise(obs: dict[str, float] | pd.DataFrame) -> dict[str, np.ndarray]:
        """Give the exact noise values consistent with an observation.

        The computation is vectorized, so ``obs`` can hold arrays.

        Parameters
        ----------
        obs : dict[str, float] | pd.DataFrame
            Observed values of ``x1`` to ``x4``.

        Returns
        -------
        dict[str, np.ndarray]
            One noise array per variable.
        """
        x1, x2 = np.asarray(obs["x1"], float), np.asarray(obs["x2"], float)
        x3, x4 = np.asarray(obs["x3"], float), np.asarray(obs["x4"], float)
        return {
            "x1": x1,
            "x2": x2,
            "x3": x3 - x1 - 0.5 * x2**3,
            "x4": x4 + x2 - 0.5 * x1**2,
        }

    def true_counterfactual(
        self, obs: dict[str, float], do: dict[str, float]
    ) -> dict[str, float]:
        """Give the analytic counterfactual of one observation under ``do``.

        Parameters
        ----------
        obs : dict[str, float]
            The factual observation, values of ``x1`` to ``x4``.
        do : dict[str, float]
            Hard interventions ``{node: value}``.

        Returns
        -------
        dict[str, float]
            The counterfactual values of every variable.
        """
        eps = self.abduct_noise({k: np.atleast_1d(v) for k, v in obs.items()})
        cf = self.simulate(do=do, latents=eps)
        return {k: float(cf[k].iloc[0]) for k in cf}

    def true_cf_curves(
        self, obs: dict[str, float] = X_OBS, alphas: np.ndarray = ALPHA_GRID
    ) -> dict:
        """Compute the two analytic counterfactual curves of paper Fig. 6.

        Parameters
        ----------
        obs : dict[str, float], optional
            The factual observation, by default the paper's ``X_OBS``.
        alphas : np.ndarray, optional
            Intervention grid, by default ``ALPHA_GRID``.

        Returns
        -------
        dict
            The observation, the intervention grid, and the counterfactual
            values of x3 under do(x2) and of x4 under do(x1).
        """
        x3_cf = [self.true_counterfactual(obs, {"x2": a})["x3"] for a in alphas]
        x4_cf = [self.true_counterfactual(obs, {"x1": a})["x4"] for a in alphas]
        return {
            "x_obs": dict(obs),
            "alphas": [float(a) for a in alphas],
            "x3_cf_do_x2": x3_cf,
            "x4_cf_do_x1": x4_cf,
        }


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    """Regenerate the frozen CSV files of this data-generating process."""
    p = argparse.ArgumentParser(description="Generate the CAREFL benchmark data.")
    p.add_argument("--out", type=Path, default=Path("data/carefl"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-obs", type=int, default=5000)
    args = p.parse_args(argv)

    gen = Carefl4(seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    obs = gen.observational(args.n_obs)
    obs.to_csv(args.out / "obs.csv", index=False)

    truth = {
        "source": "arXiv:2503.16206 App. C.2 (orig. Khemakhem 2021 Fig. 5)",
        "seed": args.seed,
        "n_obs": args.n_obs,
        "scm": {
            "x1": "Laplace(0, 1/sqrt(2))",
            "x2": "Laplace(0, 1/sqrt(2))",
            "x3": "x1 + 0.5*x2^3 + Laplace",
            "x4": "-x2 + 0.5*x1^2 + Laplace",
        },
        **gen.true_cf_curves(),
    }
    (args.out / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(
        f"[carefl] n={len(obs)}  cf sanity: "
        f"x3_cf(do x2=1.5)={gen.true_counterfactual(X_OBS, {'x2': 1.5})['x3']:.3f} "
        f"(factual {X_OBS['x3']})"
    )


if __name__ == "__main__":
    main()
