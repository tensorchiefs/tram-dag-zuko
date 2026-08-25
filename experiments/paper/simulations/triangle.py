"""The TRAM-DAG paper's triangle DGPs (Sick & Duerr, CLeaR 2025, arXiv:2503.16206).

Two families over the same DAG ``x1 -> x2 -> x3 <- x1``, built *as TRAMs* with
standard-logistic latents (paper Section 6; original code
``summerof24/triangle_structured_{continous,mixed}.R`` in tensorchiefs/tram-dag):

- source   x1 ~ 0.5 N(0.25, 0.1^2) + 0.5 N(0.73, 0.05^2)          (bimodal GMM)
- ``TriangleContinuous`` (Sec. 6.1):   h(x2|x1) = 5 x2 + 2 x1 = u2,
  h(x3|x1,x2) = 0.63 x3 - 0.2 x1 - f(x2) = u3   (u ~ standard logistic)
- ``TriangleMixed`` (Sec. 6.2): same x1, x2; x3 ordinal with 4 levels,
  cutpoints theta = (-2, 0.42, 1.02), level = #{k : u3 > theta_k + 0.2 x1 + f(x2)}
  (stored 0..3; the paper counts levels 1..4).

``f`` selects the x2 -> x3 effect (paper / R-script variants)::

    linear  -0.3 x                  (=> +0.3 coefficient on x2 in h3)
    exp     0.5 exp(x)
    atan    0.75 atan(5 (x + 0.12))  (complex-shift experiment, Fig. 7)
    sin     2 sin(3 x) + x           (non-monotone, App. C.3.4)

Convention mapping to ``CausalFlowDAG`` fits (see ``zuko_expectations``):
continuous nodes share the paper's sign (``z = h(x) + shift``), so the fitted
``ls`` weights converge to +2 (x1->x2) and -0.2 (x1->x3) and a ``cs`` module
learns ``-f(x2)`` up to an additive constant. The ordinal node flips the sign
(flow: ``P(Y<=k) = sigmoid(theta_k - shift)``, paper *adds* the shift), so the
fitted weights converge to -0.2 (x1->x3) and, for ``linear``, +0.3 (x2->x3) —
and the ``cs`` module again learns ``-f(x2)``.

CLI (regenerate the frozen CSVs for both families)::

    uv run python -m paper.simulations.triangle --out paper/data --seed 42
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import DatasetDraws, resolve_latents, sigmoid

# %% global variables ------------------------------------------------------------------
F_VARIANTS = {
    "linear": (lambda x: -0.3 * x, "-0.3*x"),
    "exp": (lambda x: 0.5 * np.exp(x), "0.5*exp(x)"),
    "atan": (lambda x: 0.75 * np.arctan(5.0 * (x + 0.12)), "0.75*atan(5*(x+0.12))"),
    "sin": (lambda x: 2.0 * np.sin(3.0 * x) + x, "2*sin(3*x) + x"),
}

THETA_MIXED = np.array([-2.0, 0.42, 1.02])  # ordinal cutpoints (4 levels)

# the variants actually used in the paper (frozen by the CLI)
PAPER_VARIANTS = {"continuous": ("linear", "atan", "sin"), "mixed": ("linear", "exp")}


# %% private functions -----------------------------------------------------------------
def _clamp(value, n: int) -> np.ndarray:
    """Broadcast a ``do`` value to shape ``(n,)``.

    The value is a scalar or one value per row. A per-row array covers the soft
    intervention ``x1 -> x1 + 1`` of paper appendix C.4.
    """
    return np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()


def _write_variant(cls, out_dir: Path, f: str, seed: int, n_obs: int) -> None:
    gen = cls(f=f, seed=seed)
    vdir = out_dir / f
    vdir.mkdir(parents=True, exist_ok=True)

    obs = gen.observational(n_obs)
    if gen.family == "mixed":
        obs["x3"] = obs["x3"].astype(int)
    obs.to_csv(vdir / "obs.csv", index=False)

    truth = {
        "source": "arXiv:2503.16206 Sec. 6 (Sick & Duerr, CLeaR 2025)",
        "family": gen.family,
        "f": f,
        "f_formula": F_VARIANTS[f][1],
        "seed": seed,
        "n_obs": n_obs,
        "paper": gen.paper_truth(),
        "zuko": gen.zuko_expectations(),
    }
    (vdir / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    if gen.family == "mixed":
        x3_range = f"levels {sorted(obs['x3'].unique())}"
    else:
        x3_range = f"in [{obs['x3'].min():.2f}, {obs['x3'].max():.2f}]"
    print(
        f"[{gen.family}/{f}] n={len(obs)}  "
        f"x2 in [{obs['x2'].min():.2f}, {obs['x2'].max():.2f}]  "
        f"x3 {x3_range}"
    )


# %% public functions ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """Regenerate the frozen CSV files of this data-generating process."""
    p = argparse.ArgumentParser(
        description="Generate the TRAM-DAG paper triangle data."
    )
    p.add_argument("--out", type=Path, default=Path("paper/data"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-obs", type=int, default=5000)
    p.add_argument("--force", action="store_true", help="overwrite an existing folder")
    args = p.parse_args(argv)
    if args.out.exists() and not args.force:
        raise SystemExit(
            f"{args.out} exists: the frozen data is a contract. A new seed or new "
            "equations belong in a NEW folder (--out); pass --force to overwrite."
        )

    for f in PAPER_VARIANTS["continuous"]:
        _write_variant(
            TriangleContinuous, args.out / "triangle", f, args.seed, args.n_obs
        )
    for f in PAPER_VARIANTS["mixed"]:
        _write_variant(
            TriangleMixed, args.out / "triangle-mixed", f, args.seed, args.n_obs
        )
    print(f"\nWrote triangle + triangle-mixed -> {args.out}")


# %% private classes -------------------------------------------------------------------
@dataclass
class _TriangleBase(DatasetDraws):
    """Shared x1 (GMM source) and x2 (Colr-type TRAM) mechanisms.

    Parameters
    ----------
    f : str, optional
        Name of the x2 -> x3 effect function, one of ``F_VARIANTS``, by
        default ``"linear"``.
    seed : int, optional
        Master seed, by default 42.
    """

    f: str = "linear"
    seed: int = 42

    def __post_init__(self):
        if self.f not in F_VARIANTS:
            raise ValueError(f"f must be one of {sorted(F_VARIANTS)}, got {self.f!r}")
        self.f_callable = F_VARIANTS[self.f][0]

    # latents
    def draw_latents(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw all noise of the SCM, ``n`` rows each.

        The GMM source gets its primitives (component indicator plus two
        normal branches). x2 and x3 get their TRAM latents.
        """
        return {
            "x1_mix": rng.uniform(size=n),  # component indicator
            "x1_a": rng.normal(size=n),  # N(0.25, 0.1) branch
            "x1_b": rng.normal(size=n),  # N(0.73, 0.05) branch
            "x2": rng.logistic(size=n),
            "x3": rng.logistic(size=n),
        }

    # SCM
    def _x1_x2(self, do: dict, latents: dict) -> tuple[np.ndarray, np.ndarray]:
        n = len(latents["x2"])
        if "x1" in do:
            x1 = _clamp(do["x1"], n)
        else:
            x1 = np.where(
                latents["x1_mix"] < 0.5,
                0.25 + 0.10 * latents["x1_a"],
                0.73 + 0.05 * latents["x1_b"],
            )
        if "x2" in do:
            x2 = _clamp(do["x2"], n)
        else:
            x2 = (latents["x2"] - 2.0 * x1) / 5.0  # h(x2|x1) = 5 x2 + 2 x1 = u2
        return x1, x2

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
            Hard interventions ``{node: value}``. A per-row array is also
            accepted, for the soft intervention of paper appendix C.4.
        latents : dict[str, np.ndarray] | None, optional
            Reuse a fixed latent draw. If given, ``n`` and ``rng`` are
            ignored.

        Returns
        -------
        pd.DataFrame
            The sample, columns ``x1``, ``x2``, ``x3``.

        Raises
        ------
        ValueError
            If both ``n`` and ``latents`` are omitted.
        """
        do = do or {}
        latents, n = resolve_latents(self, n, rng, latents)
        x1, x2 = self._x1_x2(do, latents)
        x3 = self._x3(x1, x2, do, latents)
        return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    def _x3(self, x1, x2, do, latents):  # pragma: no cover - abstract
        raise NotImplementedError

    # ground truth
    def true_shift_curve(self, x2_grid: np.ndarray) -> np.ndarray:
        """Give the limit of a fitted ``cs`` module on the x2 to x3 edge.

        The module converges to ``-f(x2)`` up to an additive constant, in
        both families. See the module docstring.

        Parameters
        ----------
        x2_grid : np.ndarray
            Evaluation points.

        Returns
        -------
        np.ndarray
            ``-f(x2)`` at the grid points.
        """
        return -self.f_callable(np.asarray(x2_grid, dtype=float))

    def zuko_expectations(self) -> dict:
        """Give the expected parameter values in the conventions of the flow.

        Returns
        -------
        dict
            Expected fitted values, keyed by parameter name.
        """
        exp = {"w_x2_from_x1": 2.0, "w_x3_from_x1": -0.2, "cs_curve": "-f(x2) + const"}
        if self.f == "linear":
            exp["w_x3_from_x2"] = 0.3
        return exp


# %% public classes --------------------------------------------------------------------
class TriangleContinuous(_TriangleBase):
    """Paper Sec. 6.1: the all-continuous triangle.

    ``h(x3|x1,x2) = 0.63 x3 - 0.2 x1 - f(x2)``.
    """

    family = "continuous"

    def _x3(self, x1, x2, do, latents):
        if "x3" in do:
            return _clamp(do["x3"], len(x1))
        return (latents["x3"] + 0.2 * x1 + self.f_callable(x2)) / 0.63

    def paper_truth(self) -> dict:
        """State the true parameters of this data-generating process.

        Returns
        -------
        dict
            The coefficients and transformation functions used to generate the
            data, in the notation of the paper.
        """
        t = {
            "beta12": 2.0,
            "beta13": -0.2,
            "h2": "5*x2 + 2*x1",
            "h3": f"0.63*x3 - 0.2*x1 - ({F_VARIANTS[self.f][1]})",
        }
        if self.f == "linear":
            t["beta23"] = 0.3
        return t


class TriangleMixed(_TriangleBase):
    """Paper Sec. 6.2: the triangle with an ordinal x3.

    x3 has 4 levels, stored as 0 to 3, with
    ``level = #{k : u3 > theta_k + 0.2 x1 + f(x2)}`` and
    ``theta = (-2, 0.42, 1.02)``.
    """

    family = "mixed"
    theta = THETA_MIXED

    def _x3(self, x1, x2, do, latents):
        if "x3" in do:
            return _clamp(do["x3"], len(x1))
        cuts = self.theta[None, :] + (0.2 * x1 + self.f_callable(x2))[:, None]
        return (latents["x3"][:, None] > cuts).sum(axis=1).astype(float)

    def true_pmf(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Give the analytic class probabilities of x3 given its parents.

        Parameters
        ----------
        x1, x2 : np.ndarray
            Parent values, each shape ``(n,)``.

        Returns
        -------
        np.ndarray
            The class probabilities, shape ``(n, 4)``.
        """
        shift = 0.2 * np.asarray(x1, float) + self.f_callable(np.asarray(x2, float))
        cuts = self.theta[None, :] + shift[:, None]
        cdf = 1.0 / (1.0 + np.exp(-cuts))
        cdf = np.concatenate(
            [np.zeros((len(cdf), 1)), cdf, np.ones((len(cdf), 1))], axis=1
        )
        return np.diff(cdf, axis=1)

    def true_counterfactual_pmf(
        self, observed: pd.DataFrame, do: dict[str, float]
    ) -> np.ndarray:
        """Give the exact counterfactual distribution of x3, row by row.

        For a **discretized** variable the individual counterfactual is not
        identified (paper App. B): observing ``x3 = k`` pins the latent only
        to the interval between two cutpoints, not to a value. What *is*
        identified is a distribution — the truncated-logistic mass of that
        interval, redistributed over the cutpoints of the intervened world.
        That distribution is the object a flow's averaged abduction draws
        should be compared against. It is not an upper bound on every score:
        ``P(realised level)`` is maximized by naming the modal level, not by
        reporting this law (see ``triangle_mixed.score_counterfactuals``).

        Parameters
        ----------
        observed : pd.DataFrame
            Factual rows, with ``x1``, ``x2`` and the observed ``x3`` level.
        do : dict[str, float]
            The intervention, for example ``{"x1": -1.0}``.

        Returns
        -------
        np.ndarray
            Class probabilities, shape ``(n, 4)``, one row per observation.
        """
        x1 = observed["x1"].to_numpy(dtype=float)
        x2 = observed["x2"].to_numpy(dtype=float)
        level = observed["x3"].to_numpy(dtype=int)

        factual_cuts = self.theta[None, :] + (0.2 * x1 + self.f_callable(x2))[:, None]
        # x2 is a *descendant* of x1, so intervening on x1 moves it too. Its
        # latent is recoverable from the structural equation h(x2|x1) =
        # 5 x2 + 2 x1, which gives x2_cf = x2 + 0.4 (x1 - a) under do(x1 = a).
        if "x1" in do:
            x1_cf = np.full_like(x1, float(do["x1"]))
            x2_cf = x2 + 0.4 * (x1 - x1_cf)
        else:
            x1_cf = x1
            x2_cf = x2
        if "x2" in do:  # a direct intervention overrides the propagated value
            x2_cf = np.full_like(x2, float(do["x2"]))
        cf_cuts = self.theta[None, :] + (0.2 * x1_cf + self.f_callable(x2_cf))[:, None]

        # the observed level says the latent lies in [lo, hi)
        padded = np.concatenate(
            [
                np.full((len(x1), 1), -np.inf),
                factual_cuts,
                np.full((len(x1), 1), np.inf),
            ],
            axis=1,
        )
        lo = padded[np.arange(len(x1)), level]
        hi = padded[np.arange(len(x1)), level + 1]
        mass = sigmoid(hi) - sigmoid(lo)  # logistic mass of the known interval

        # redistribute that interval over the intervened cutpoints
        cf_padded = np.concatenate(
            [
                np.full((len(x1), 1), -np.inf),
                cf_cuts,
                np.full((len(x1), 1), np.inf),
            ],
            axis=1,
        )
        lo_c = np.maximum(cf_padded[:, :-1], lo[:, None])
        hi_c = np.minimum(cf_padded[:, 1:], hi[:, None])
        overlap = np.clip(sigmoid(hi_c) - sigmoid(lo_c), 0.0, None)
        return overlap / mass[:, None]

    def paper_truth(self) -> dict:
        """State the true parameters of this data-generating process.

        Returns
        -------
        dict
            The coefficients, cutpoints and shift functions used to generate the
            data, in the notation of the paper.
        """
        t = {
            "beta13": 0.2,
            "theta": self.theta.tolist(),
            "h3": f"theta_k + 0.2*x1 + ({F_VARIANTS[self.f][1]})",
            "levels": 4,
            "level_offset": "paper counts 1..4, stored 0..3",
            "zuko_sign": -1,
        }
        if self.f == "linear":
            t["beta23"] = -0.3
        return t

    def zuko_expectations(self) -> dict:
        """Give the same truth in the conventions of ``CausalFlowDAG``.

        The ordinal shift is subtracted here and added in the paper, so the
        expected weights carry the opposite sign.

        Returns
        -------
        dict
            Expected fitted values, keyed by parameter name.
        """
        exp = {
            "w_x2_from_x1": 2.0,
            "w_x3_from_x1": -0.2,
            "theta": self.theta.tolist(),
            "cs_curve": "-f(x2) + const",
        }
        if self.f == "linear":
            exp["w_x3_from_x2"] = 0.3
        return exp


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
