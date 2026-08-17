"""Validation DGP for the varying-coefficient (VC) shift term (issue #28).

A logistic-shift SCM whose outcome conditional is *exactly* a transformation
model with a heterogeneous treatment effect on the latent scale::

    X1, X2, X3 ~ iid N(0, 1)
    T | x      ~ Bernoulli(sigmoid(0.4 X1 + 0.4 X2))       (confounded assignment)
    h(Y) + g(x) + beta(x) * T = U_Y,   U_Y ~ standard logistic
      h(y)    = 2 y                                        (Colr-type affine)
      g(x)    = 0.5 X1^2 + X2 - 0.5 X3                     (nonlinear prognostic)
      beta(x) = b0 + b2 X2 + b3 X3 = -1.0 + 0.8 X2 - 0.6 X3

so ``Y = (U_Y - g(x) - beta(x) T) / 2``. The DGP is exactly in-class for::

    "Y": ContinuousNode([CS("X1", "X2", "X3"), VC("X2", "X3", t="T")])

``X2`` is deliberately both a confounder (enters the propensity) *and* an
effect modifier — the configuration where the unregularized ``CS(on, x...)``
reduced form fails hardest (corr ~ 0.5 against the true effect function even
in-class; tramdag-simu PR #21). ``true_beta`` gives the pointwise ground truth
the acceptance test scores against (corr >= 0.9 at n = 5000, issue #28).

Everything is numpy-only: this module is the ground truth, deliberately
independent of the flow implementation.

CLI (regenerate the frozen CSV)::

    uv run python -m tramdag.simulations.vc_shift --out data/vc-shift --seed 42
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

B0, B2, B3 = -1.0, 0.8, -0.6  # beta(x) = B0 + B2*X2 + B3*X3
H_SCALE = 2.0  # h(y) = H_SCALE * y


def _logistic(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.logistic(loc=0.0, size=size)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class VCLogisticShift:
    """SCM generator for the VC validation cohort (issue #28).

    Parameters
    ----------
    seed : int, optional
        Master seed, by default 42. Each dataset draw uses an independent
        child stream.
    """

    seed: int = 42

    # ------------------------------------------------------------------ latents
    def draw_latents(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw all noise of the SCM.

        The sources get Gaussian primitives. T gets a logistic assignment
        latent and Y gets the logistic TRAM latent.

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
            "X1": rng.normal(size=n),
            "X2": rng.normal(size=n),
            "X3": rng.normal(size=n),
            "T": _logistic(rng, n),
            "Y": _logistic(rng, n),
        }

    # --------------------------------------------------------------------- SCM
    def simulate(
        self,
        n: int | None = None,
        *,
        rng: np.random.Generator | None = None,
        do: dict[str, float] | None = None,
        latents: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """Forward-sample the SCM.

        Parameters
        ----------
        n : int | None, optional
            Number of rows. Required unless ``latents`` is given.
        rng : np.random.Generator | None, optional
            Random source. Defaults to a generator seeded with
            ``self.seed``.
        do : dict[str, float] | None, optional
            Hard interventions ``{node: value}``. The node is clamped and
            its structural equation skipped.
        latents : dict[str, np.ndarray] | None, optional
            Reuse a fixed latent draw. If given, ``n`` and ``rng`` are
            ignored.

        Returns
        -------
        pd.DataFrame
            The sample, one column per variable.

        Raises
        ------
        ValueError
            If both ``n`` and ``latents`` are omitted.
        """
        do = do or {}
        if latents is None:
            if n is None:
                raise ValueError("provide either n or latents")
            rng = rng or np.random.default_rng(self.seed)
            latents = self.draw_latents(n, rng)
        n = len(next(iter(latents.values())))

        x = {}
        for name in ("X1", "X2", "X3"):
            x[name] = np.full(n, float(do[name])) if name in do else latents[name]
        if "T" in do:
            T = np.full(n, float(do["T"]))
        else:
            logit_T = 0.4 * x["X1"] + 0.4 * x["X2"]
            T = (latents["T"] > -logit_T).astype(float)  # P(T=1) = sigmoid(logit_T)
        if "Y" in do:
            Y = np.full(n, float(do["Y"]))
        else:
            g = 0.5 * x["X1"] ** 2 + x["X2"] - 0.5 * x["X3"]
            beta = B0 + B2 * x["X2"] + B3 * x["X3"]
            Y = (latents["Y"] - g - beta * T) / H_SCALE
        return pd.DataFrame(
            {"X1": x["X1"], "X2": x["X2"], "X3": x["X3"], "T": T, "Y": Y}
        )

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

    # -------------------------------------------------------------- ground truth
    def true_beta(self, x) -> np.ndarray:
        """Give the true effect function ``beta(x)`` on the latent scale.

        The scale is log-odds. A fitted VC term must recover this function
        through :meth:`~tramdag.CausalFlowDAG.varying_coef`.

        Parameters
        ----------
        x : pd.DataFrame
            Rows with ``X2`` and ``X3`` columns. Other columns are
            ignored.

        Returns
        -------
        np.ndarray
            The effect values, shape ``(n,)``.
        """
        return (
            B0
            + B2 * np.asarray(x["X2"], dtype=float)
            + B3 * np.asarray(x["X3"], dtype=float)
        )

    def counterfactual_pair(
        self, n: int, do: dict[str, float], seed_offset: int = 0
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Draw a factual sample and its counterfactual under ``do``.

        Both share the same latents, so the pair gives true individual
        counterfactuals.

        Parameters
        ----------
        n : int
            Number of rows.
        do : dict[str, float]
            Hard interventions for the counterfactual arm.
        seed_offset : int, optional
            Added to the generator seed, by default 0.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            The factual and the counterfactual sample.
        """
        rng = np.random.default_rng(self.seed + 2 + seed_offset)
        latents = self.draw_latents(n, rng)
        return self.simulate(latents=latents), self.simulate(latents=latents, do=do)


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
    p = argparse.ArgumentParser(description="Generate the VC validation cohort.")
    p.add_argument("--out", type=Path, default=Path("data/vc-shift"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-obs", type=int, default=5000)
    args = p.parse_args(argv)

    gen = VCLogisticShift(seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    obs = gen.observational(args.n_obs)
    obs["T"] = obs["T"].astype(int)
    obs.to_csv(args.out / "obs.csv", index=False)

    truth = {
        "source": "tramdag issue #28 (VC acceptance DGP)",
        "seed": args.seed,
        "n_obs": args.n_obs,
        "beta": {"b0": B0, "b2": B2, "b3": B3, "formula": f"{B0} + {B2}*X2 + {B3}*X3"},
        "g": "0.5*X1^2 + X2 - 0.5*X3",
        "h": f"{H_SCALE}*y",
        "propensity": "sigmoid(0.4*X1 + 0.4*X2)",
    }
    (args.out / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(
        f"[vc-shift] n={len(obs)}  T-rate={obs['T'].mean():.3f}  "
        f"sd(beta_true)={gen.true_beta(obs).std():.3f}  -> {args.out}"
    )


if __name__ == "__main__":
    main()
