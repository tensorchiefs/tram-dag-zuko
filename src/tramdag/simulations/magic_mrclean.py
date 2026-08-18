"""Synthetic stroke cohort in the shape of the MAGIC / MR CLEAN study.

A hand-specified structural causal model (SCM) over the same five-node DAG as the
real analysis::

    Age -> mRS_pre -> NIHSSa -> T -> mRS_3m   (plus all forward edges)

It exists so the package has a *public, reproducible* dataset with **known ground
truth** (true ATE, individual counterfactuals) that can later be swapped for the
private clinical data — same column schema, same dtypes, same ranges.

Two variants, identical except for three deliberately mild non-linearities (★):

- ``"ls"`` — every parent effect is a **linear shift**. Each node-conditional is
  then exactly a classical (proportional-odds / linear transformation) model, so
  an all-``ls`` flow, the R reference (``polr``/``tram``/``glm``) and the true SCM
  all coincide. This is the clean equivalence baseline.
- ``"nl"`` — adds an accelerating age effect on disability, a heterogeneous
  treatment effect ``tau(Age)`` that fades in the elderly, and a smooth reduction
  of treatment probability for the very old. The all-``ls`` model is now provably
  misspecified (it must collapse ``tau(Age)`` to one constant), so it is biased
  for the true ATE, while a flexible (``ci``/``cs``) flow can recover it.

Latents are iid standard logistic (the TRAM base distribution), so the data lives
natively in the family the flow fits. Everything is numpy-only: this module is the
*ground truth*, deliberately independent of the flow implementation.

CLI (regenerate the frozen CSVs)::

    uv run python -m tramdag.simulations.magic_mrclean --out data/magic-mrclean --seed 7
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import DatasetDraws, logistic, resolve_latents, sigmoid

COLUMNS = ["Age", "mRS_pre", "NIHSSa", "T", "mRS_3m"]

# Ordered-logit cutpoints, P(Y <= k) = sigmoid(theta_k - shift); chosen so the
# observational marginals roughly match the real cohort (mRS_pre skewed to 0,
# mRS_3m spread across all 7 levels). See data/magic-mrclean/README.md.
CUTS_PRE = np.array([0.20, 1.15, 2.05, 3.20, 5.20])  # 6 levels (0..5)
CUTS_Y = np.array([-1.85, -0.75, 0.05, 0.80, 1.55, 2.35])  # 7 levels (0..6)


def _ordinal(shift: np.ndarray, cuts: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Sample an ordered-logit level from the latent ``u``.

    The rule is ``Y = #{ j : cuts_j - shift < u }``, the same rule that the
    flow's ``ordinal_sample`` applies.
    """
    return (u[:, None] > (cuts[None, :] - shift[:, None])).sum(axis=1)


@dataclass
class MagicMrClean(DatasetDraws):
    """SCM generator for the synthetic stroke cohort.

    Parameters
    ----------
    variant : str, optional
        ``"ls"`` (all linear shifts) or ``"nl"`` (mild non-linearities),
        by default ``"nl"``.
    seed : int, optional
        Master seed, by default 7. Each draw uses an independent child
        stream.
    """

    variant: str = "nl"
    seed: int = 7

    def __post_init__(self):
        """Validate the variant and set the switch for the non-linear terms.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``variant`` is neither ``"ls"`` nor ``"nl"``.
        """
        if self.variant not in ("ls", "nl"):
            raise ValueError(f"variant must be 'ls' or 'nl', got {self.variant!r}")
        self.nl = float(self.variant == "nl")  # 0.0 disables the ★ terms

    # ------------------------------------------------------------------ latents
    def draw_latents(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw the latent noise of every variable, ``n`` rows each."""
        return {k: logistic(rng, n) for k in COLUMNS}

    # --------------------------------------------------------------------- SCM
    def simulate(
        self,
        n: int | None = None,
        *,
        rng: np.random.Generator | None = None,
        randomize_T: bool = False,
        population: str = "obs",
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
        randomize_T : bool, optional
            If True, assign ``T ~ Bernoulli(0.5)`` independently of its
            parents (the RCT design). Otherwise ``T`` follows the
            confounded observational mechanism. Default False.
        population : str, optional
            Covariate population, by default ``"obs"``. ``"obs"`` is the
            full observational cohort. ``"rct"`` mimics trial inclusion —
            a **younger** enrolled population (age location shifted down).
            Only the ``Age`` source marginal differs; all structural
            equations are unchanged. With the heterogeneous ``nl``
            treatment effect this fit-vs-eval shift is what biases an
            all-``ls`` model: it cannot extrapolate ``tau(Age)`` from the
            older observational cohort to the younger trial. The ``ls``
            DGP has constant ``tau`` and is therefore unaffected.
        do : dict[str, float] | None, optional
            Hard interventions ``{node: value}`` (graph mutilation). The
            node is clamped and its structural equation skipped.
        latents : dict[str, np.ndarray] | None, optional
            Reuse a fixed latent draw, for counterfactuals and paired
            interventions. If given, ``n`` and ``rng`` are ignored.

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
        latents, n = resolve_latents(self, n, rng, latents)
        nl = self.nl
        age_loc = 73.0 - (9.0 if population == "rct" else 0.0)  # trial enrolls younger

        # --- Age: location-scale logistic (sd ~ 13.8), clipped to a plausible range
        if "Age" in do:
            Age = np.full(n, float(do["Age"]))
        else:
            Age = np.clip(age_loc + 7.6 * latents["Age"], 20.0, 103.0)
        a = (Age - 73.0) / 10.0  # standardized age
        relu_a = np.maximum(a, 0.0)

        # --- mRS_pre: pre-stroke disability, worse (and ★ accelerating) with age
        if "mRS_pre" in do:
            mRS_pre = np.full(n, float(do["mRS_pre"]))
        else:
            eta_pre = 0.55 * a + nl * (0.18 * relu_a**2)
            mRS_pre = _ordinal(eta_pre, CUTS_PRE, latents["mRS_pre"]).astype(float)

        # --- NIHSSa: stroke severity; linear shift on parents, ★ age^2 in nl.
        #     Free monotone marginal map keeps it in the realistic [6, 42] range.
        if "NIHSSa" in do:
            NIHSSa = np.full(n, float(do["NIHSSa"]))
        else:
            shift_nih = 0.45 * a + 0.25 * mRS_pre + nl * (0.20 * relu_a**2)
            lat_nih = shift_nih + 0.85 * latents["NIHSSa"]
            NIHSSa = np.clip(6.0 + 36.0 * sigmoid((lat_nih - 1.75) / 1.5), 6.0, 42.0)
        s = (NIHSSa - 15.0) / 6.0  # standardized severity

        # --- T: thrombectomy assignment. Observational mechanism is confounded by
        #     age and severity; ★ in nl it is smoothly withheld from the very old.
        if "T" in do:
            T = np.full(n, float(do["T"]))
        elif randomize_T:
            T = (latents["T"] > 0.0).astype(float)  # Bernoulli(0.5), latent-driven
        else:
            logit_T = (
                1.9
                - 0.45 * np.maximum(a + 0.5, 0.0)
                - 0.30 * np.maximum(s - 1.0, 0.0)
                + nl * (-1.3 * sigmoid((Age - 82.0) / 4.0))
            )
            T = (latents["T"] > -logit_T).astype(float)  # P(T=1) = sigmoid(logit_T)

        # --- mRS_3m: 3-month outcome. Treatment lowers the latent (better outcome);
        #     ★ in nl the benefit tau(Age) fades to ~0 in the elderly.
        if "mRS_3m" in do:
            mRS_3m = np.full(n, float(do["mRS_3m"]))
        else:
            tau = -0.85 + nl * (0.85 - 0.85 * sigmoid((78.0 - Age) / 6.0))
            zeta = 0.85 * s + 0.55 * a + 0.45 * mRS_pre + tau * T
            mRS_3m = _ordinal(zeta, CUTS_Y, latents["mRS_3m"]).astype(float)

        return pd.DataFrame(
            {"Age": Age, "mRS_pre": mRS_pre, "NIHSSa": NIHSSa, "T": T, "mRS_3m": mRS_3m}
        )

    def rct(self, n: int, seed_offset: int = 0) -> pd.DataFrame:
        """Draw a randomized-trial sample.

        Treatment is randomized and the covariates follow the trial population, so
        the contrast this yields is unconfounded.

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
        rng = np.random.default_rng(self.seed + 1001 + seed_offset)
        return self.simulate(n, rng=rng, randomize_T=True, population="rct")

    # -------------------------------------------------------------- ground truth
    def true_ate(self, n: int = 500_000, on: str = "rct") -> dict:
        """Estimate the true ATE of T on ``P(mRS_3m <= 2)`` by Monte Carlo.

        Both arms use the same latent draw, so the result is the do-effect
        and carries none of the confounding in T.

        Parameters
        ----------
        n : int, optional
            Monte Carlo sample size, by default 500_000.
        on : str, optional
            Covariate population the ATE is averaged over, by default
            ``"rct"``. ``"rct"`` mirrors the **younger trial** population
            that :meth:`rct` enrolls. ``"obs"`` is the observational
            cohort.

        Returns
        -------
        dict
            ``p_good_do_T0``, ``p_good_do_T1``, ``ate_population``,
            ``true_ate``, the confounded ``naive_obs_diff``, and ``mc_n``.
        """
        rng = np.random.default_rng(self.seed + 9001)
        latents = self.draw_latents(n, rng)
        d0 = self.simulate(latents=latents, population=on, do={"T": 0})
        d1 = self.simulate(latents=latents, population=on, do={"T": 1})
        good0 = float((d0["mRS_3m"] <= 2).mean())
        good1 = float((d1["mRS_3m"] <= 2).mean())
        # naive (observational, confounded) contrast for contrast with the truth
        obs = self.observational(n, seed_offset=777)
        naive = float((obs.loc[obs["T"] == 1, "mRS_3m"] <= 2).mean()) - float(
            (obs.loc[obs["T"] == 0, "mRS_3m"] <= 2).mean()
        )
        return {
            "p_good_do_T0": good0,
            "p_good_do_T1": good1,
            "ate_population": on,
            "true_ate": good1 - good0,
            "naive_obs_diff": naive,
            "mc_n": n,
        }


# --------------------------------------------------------------------------- CLI
def _write_variant(
    out_dir: Path, variant: str, seed: int, n_obs: int, n_rct: int, mc_n: int
) -> dict:
    gen = MagicMrClean(variant=variant, seed=seed)
    vdir = out_dir / variant
    vdir.mkdir(parents=True, exist_ok=True)

    obs = gen.observational(n_obs)
    rct = gen.rct(n_rct)
    # integer-valued columns stored as ints for a clean CSV / R read
    for df in (obs, rct):
        for c in ["mRS_pre", "T", "mRS_3m"]:
            df[c] = df[c].astype(int)
        df["Age"] = df["Age"].round(1)
        df["NIHSSa"] = df["NIHSSa"].round(2)
    obs.to_csv(vdir / "obs.csv", index=False)
    rct.to_csv(vdir / "rct.csv", index=False)

    truth = {
        "variant": variant,
        "seed": seed,
        "n_obs": n_obs,
        "n_rct": n_rct,
        **gen.true_ate(mc_n),
    }
    (vdir / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")

    print(
        f"[{variant}] obs={len(obs)} rct={len(rct)}  "
        f"true ATE={truth['true_ate']:+.4f}  naive={truth['naive_obs_diff']:+.4f}  "
        f"P(good)={(obs['mRS_3m'] <= 2).mean():.3f}  T-rate={obs['T'].mean():.3f}"
    )
    return truth


def main(argv: list[str] | None = None) -> None:
    """Regenerate the frozen CSV files of this data-generating process."""
    p = argparse.ArgumentParser(
        description="Generate the magic-mrclean synthetic cohort."
    )
    p.add_argument("--out", type=Path, default=Path("data/magic-mrclean"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-obs", type=int, default=1275)
    p.add_argument("--n-rct", type=int, default=500)
    p.add_argument("--mc-n", type=int, default=500_000)
    p.add_argument("--variants", nargs="+", default=["ls", "nl"])
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        _write_variant(args.out, variant, args.seed, args.n_obs, args.n_rct, args.mc_n)
    print(f"\nWrote {', '.join(args.variants)} -> {args.out}")


if __name__ == "__main__":
    main()
