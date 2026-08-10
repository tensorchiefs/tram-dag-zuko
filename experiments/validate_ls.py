"""Spot-on validation of the all-`ls` flow against the classical MLE.

The outcome node of an all-`ls` flow is a proportional-odds model. Trained to
convergence without early stopping it must therefore reproduce the classical
MLE exactly, both the statsmodels OrderedModel and the R polr reference.

Fits both on the same full dataset (no held-out split), with the flow using
``restore_best=False`` so it sits at the training-data maximum likelihood — then
compares outcome-node coefficients and the analytic ATE on the RCT covariates.

This exactness is only possible because fit() no longer early-stops by default;
with best-validation restoration the flow would sit off the train optimum.

Usage: uv run python validate_ls.py [source] [--classical]
       source: magic-mrclean/ls (default) | magic-mrclean/nl | magic
       --classical: fit via flow.fit_classical (deterministic float64 L-BFGS)
                    instead of the multi-phase Adam schedule
"""

import sys

import pandas as pd
import torch
from common import DATA_R_REF, load_data
from statsmodels.miscmodels.ordinal_model import OrderedModel

from tramdag import LS, CausalFlowDAG, ContinuousNode, OrdinalNode

PHASES = [(4000, 1e-2), (2000, 1e-3), (1000, 1e-4)]  # to tight convergence


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the parents exactly as the flow does.

    A continuous parent stays raw. An ordinal parent is one-hot encoded with
    level 0 dropped: with cutpoints the full one-hot is unidentified, so only
    the differences against level 0 are comparable.
    """
    X = pd.DataFrame(index=df.index)
    X["Age"] = df["Age"].values
    for k in range(6):
        X[f"mRS_pre_{k}"] = (df["mRS_pre"].values == k).astype(float)
    X["NIHSSa"] = df["NIHSSa"].values
    X["T"] = df["T"].values
    return X.drop(columns=["mRS_pre_0"])


def all_ls_spec() -> dict:
    return {
        "Age": ContinuousNode(transform="bernstein"),
        "mRS_pre": OrdinalNode(levels=6, terms=[LS("Age")]),
        "NIHSSa": ContinuousNode(
            transform="bernstein", terms=[LS("Age"), LS("mRS_pre")]
        ),
        "T": OrdinalNode(levels=2, terms=[LS("Age"), LS("mRS_pre"), LS("NIHSSa")]),
        "mRS_3m": OrdinalNode(
            levels=7, terms=[LS("Age"), LS("mRS_pre"), LS("NIHSSa"), LS("T")]
        ),
    }


def main(source: str = "magic-mrclean/ls", classical: bool = False):
    obs, rct, truth = load_data(source)
    fitter = "fit_classical (float64 L-BFGS)" if classical else "Adam (multi-phase)"
    print(f"=== spot-on all-ls comparison on '{source}' (N={len(obs)}) — {fitter} ===")

    # --- classical MLE (statsmodels), full data ---
    res = OrderedModel(obs["mRS_3m"].astype(int), design(obs), distr="logit").fit(
        method="bfgs", disp=False
    )

    # --- flow: full data, no early stopping, train to the MLE ---
    torch.manual_seed(0)
    flow = CausalFlowDAG(all_ls_spec())
    if classical:
        flow.fit_classical(obs)
    else:
        for ep, lr in PHASES:
            flow.fit(
                obs,
                epochs=ep,
                learning_rate=lr,
                batch_size=256,
                verbose=0,
                seed=0 if lr == 1e-2 else None,
                restore_best=False,
            )

    node = flow.nodes["mRS_3m"]
    w_age = float(node.shifts["Age"].weight.detach())
    w_nih = float(node.shifts["NIHSSa"].weight.detach())
    w_pre = node.shifts["mRS_pre"].weight.detach().numpy().ravel()
    w_t = node.shifts["T"].weight.detach().numpy().ravel()

    rows = [
        ("Age", w_age, res.params["Age"]),
        ("NIHSSa", w_nih, res.params["NIHSSa"]),
        ("T (=1 vs 0)", w_t[1] - w_t[0], res.params["T"]),
    ]
    for k in range(1, 6):
        rows.append(
            (f"mRS_pre_{k} (vs 0)", w_pre[k] - w_pre[0], res.params[f"mRS_pre_{k}"])
        )

    print(f"\n{'coefficient':<20}{'flow':>10}{'statsmodels':>13}{'|diff|':>9}")
    maxdiff = 0.0
    for name, a, b in rows:
        maxdiff = max(maxdiff, abs(a - b))
        print(f"{name:<20}{a:>10.4f}{b:>13.4f}{abs(a - b):>9.4f}")
    print(f"{'max |diff|':<20}{'':>10}{'':>13}{maxdiff:>9.4f}")

    # --- optional R polr reference (committed ref_ls/) ---
    rref = DATA_R_REF(source)
    if rref is not None and (rref / "coefficients.csv").exists():
        ry = pd.read_csv(rref / "coefficients.csv")
        ry = ry[ry["node"] == "mRS_3m"].set_index("term")["estimate"]
        print(
            f"\nR polr reference (mRS_3m): Age={ry['Age']:+.4f}  "
            f"NIHSSa={ry['NIHSSa']:+.4f}  T={ry['T']:+.4f}"
        )

    # --- ATE on the RCT covariates ---
    p0 = res.model.predict(
        res.params, exog=design(rct.assign(T=0)).values, which="prob"
    )
    p1 = res.model.predict(
        res.params, exog=design(rct.assign(T=1)).values, which="prob"
    )
    ate_mle = float((p1[:, :3].sum(axis=1) - p0[:, :3].sum(axis=1)).mean())
    pf0 = flow.pmf(rct, node="mRS_3m", do={"T": 0})
    pf1 = flow.pmf(rct, node="mRS_3m", do={"T": 1})
    ate_flow = float((pf1[:, :3].sum(axis=1) - pf0[:, :3].sum(axis=1)).mean())
    print(
        f"\nATE on RCT covariates:  flow {ate_flow:+.4f}   statsmodels {ate_mle:+.4f}"
        f"   |diff| {abs(ate_flow - ate_mle):.4f}"
    )
    if truth is not None:
        print(f"  (known true ATE for this DGP: {truth['true_ate']:+.4f})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--classical"]
    main(args[0] if args else "magic-mrclean/ls", classical="--classical" in sys.argv)
