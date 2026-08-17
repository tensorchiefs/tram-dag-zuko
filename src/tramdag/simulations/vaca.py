"""The VACA / CNF benchmark DGP (TRAM-DAG paper App. C.1, arXiv:2503.16206).

Originally from Sanchez-Martin et al. (2022, VACA App. E.1), used in the paper to
benchmark TRAM-DAG against Causal Normalizing Flows (Javaloy et al. 2024) on L1
(observational fit, Fig. 4) and L2 (interventional distributions, Fig. 5)::

    x1 ~ 0.5 N(-2, 1.5) + 0.5 N(1.5, 1)        (bimodal; sd of the 1st comp. is
                                                 sqrt(1.5) — the paper's headline
                                                 L1 case that the default CNF
                                                 fails to fit)
    x2 = -x1 + N(0, 1)
    x3 =  x1 + 0.25 x2 + N(0, 1)

Gaussian noise, so this DGP is deliberately *outside* the flow's logistic-latent
family — a flexible (all-``ci``) TRAM-DAG still has to fit it. Interventional
queries in the paper: p(x3 | do(x2 = a)) for a in {-3, -1, 0}.

CLI::

    uv run python -m tramdag.simulations.vaca --out data/vaca --seed 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DO_X2_VALUES = (-3.0, -1.0, 0.0)  # the paper's Fig. 5 interventions


@dataclass
class VacaTriangle:
    """SCM generator for the VACA bimodal triangle."""

    seed: int = 42

    def draw_latents(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw the latent noise of every variable.

        Parameters
        ----------
        n : int
            Number of rows to draw.
        rng : np.random.Generator
            Random source.

        Returns
        -------
        dict[str, np.ndarray]
            One array of length ``n`` per variable.
        """
        return {
            "x1_mix": rng.uniform(size=n),
            "x1_a": rng.normal(size=n),  # N(-2, sqrt(1.5)) branch
            "x1_b": rng.normal(size=n),  # N(1.5, 1) branch
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
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
        if latents is None:
            if n is None:
                raise ValueError("provide either n or latents")
            rng = rng or np.random.default_rng(self.seed)
            latents = self.draw_latents(n, rng)
        n = len(latents["x2"])

        if "x1" in do:
            x1 = np.full(n, float(do["x1"]))
        else:
            x1 = np.where(
                latents["x1_mix"] < 0.5,
                -2.0 + np.sqrt(1.5) * latents["x1_a"],
                1.5 + 1.0 * latents["x1_b"],
            )
        x2 = np.full(n, float(do["x2"])) if "x2" in do else -x1 + latents["x2"]
        x3 = (
            np.full(n, float(do["x3"]))
            if "x3" in do
            else x1 + 0.25 * x2 + latents["x3"]
        )
        return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    # ----------------------------------------------------------------- datasets
    def observational(self, n: int, seed_offset: int = 0) -> pd.DataFrame:
        """Draw an observational sample.

        Parameters
        ----------
        n : int
            Number of rows.
        seed_offset : int, optional
            Added to the generator seed, by default ``0``.

        Returns
        -------
        pd.DataFrame
            The sample.
        """
        rng = np.random.default_rng(self.seed + 1 + seed_offset)
        return self.simulate(n, rng=rng)

    def interventional(
        self, n: int, do: dict[str, float], seed_offset: int = 0
    ) -> pd.DataFrame:
        """Draw a sample under an intervention.

        Parameters
        ----------
        n : int
            Number of rows.
        do : dict[str, float]
            Variables to hold at a fixed value.
        seed_offset : int, optional
            Added to the generator seed, by default ``0``.

        Returns
        -------
        pd.DataFrame
            The sample.
        """
        rng = np.random.default_rng(self.seed + 501 + seed_offset)
        return self.simulate(n, rng=rng, do=do)

    # -------------------------------------------------------------- ground truth
    def true_moments(self, mc_n: int = 1_000_000) -> dict:
        """Give the observational and interventional ground-truth moments.

        Under ``do(x2=a)``: ``x3 = x1 + 0.25 a + N(0,1)``, so
        ``E = E[x1] + 0.25 a`` and ``Var = Var[x1] + 1``. These are exact.
        Monte Carlo values for the observational moments are stored too,
        with the same estimator a test uses.

        Parameters
        ----------
        mc_n : int, optional
            Monte Carlo sample size, by default 1_000_000.

        Returns
        -------
        dict
            ``mc_n``, the observational ``obs_mean`` and ``obs_std`` per
            column, and the analytic x3 moments per ``do(x2)`` value.
        """
        mu1 = 0.5 * (-2.0) + 0.5 * 1.5
        var1 = 0.5 * (1.5 + (-2.0 - mu1) ** 2) + 0.5 * (1.0 + (1.5 - mu1) ** 2)
        obs = self.observational(mc_n, seed_offset=777)
        out = {
            "mc_n": mc_n,
            "obs_mean": {c: float(obs[c].mean()) for c in obs},
            "obs_std": {c: float(obs[c].std()) for c in obs},
            "do_x2": {},
        }
        for a in DO_X2_VALUES:
            out["do_x2"][str(a)] = {
                "mean_x3_analytic": mu1 + 0.25 * a,
                "std_x3_analytic": float(np.sqrt(var1 + 1.0)),
            }
        return out


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    """Regenerate the frozen CSV files of this data-generating process.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments, by default ``None`` (``sys.argv``).

    Returns
    -------
    None
    """
    p = argparse.ArgumentParser(description="Generate the VACA benchmark data.")
    p.add_argument("--out", type=Path, default=Path("data/vaca"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-obs", type=int, default=5000)
    p.add_argument("--mc-n", type=int, default=1_000_000)
    args = p.parse_args(argv)

    gen = VacaTriangle(seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    obs = gen.observational(args.n_obs)
    obs.to_csv(args.out / "obs.csv", index=False)

    truth = {
        "source": "arXiv:2503.16206 App. C.1 (orig. Sanchez-Martin 2022 E.1)",
        "seed": args.seed,
        "n_obs": args.n_obs,
        "scm": {
            "x1": "0.5 N(-2,1.5) + 0.5 N(1.5,1)",
            "x2": "-x1 + N(0,1)",
            "x3": "x1 + 0.25*x2 + N(0,1)",
        },
        **gen.true_moments(args.mc_n),
    }
    (args.out / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(
        f"[vaca] n={len(obs)}  x1 bimodal: mean={obs['x1'].mean():+.3f} "
        f"std={obs['x1'].std():.3f}"
    )


if __name__ == "__main__":
    main()
