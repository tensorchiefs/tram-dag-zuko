"""Shared helpers for the tramdag stroke experiments.

Data conventions match the TRAM-DAG runs in the parent repository:
MAGIC cohort filtered to NIHSSa >= 6 (N=1275), IAT -> T, 80/10/10 split with
random_state=42; evaluation on the MR CLEAN RCT cohort (data/exp_data.csv).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportion_confint

from tramdag import CausalFlowDAG, ContinuousNode, OrdinalNode, term

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
SIM_DATA = Path(__file__).resolve().parents[1] / "data"

NODES = ["Age", "mRS_pre", "NIHSSa", "T", "mRS_3m"]
MRS_LEVELS = list(range(7))

# RCT benchmark (Berkhemer et al., 2015)
RCT_ATE = 0.135
RCT_CI = (0.057, 0.213)

# Default data source. "magic" is the private clinical cohort; the synthetic
# "magic-mrclean/<variant>" folders are the public, reproducible substitute.
DEFAULT_SOURCE = "magic-mrclean/nl"


# ------------------------------------------------------------------- data
def load_magic(nihss_min: float = 6.0) -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "data" / "obs_data.csv")
    df = df.rename(columns={"IAT": "T"}).drop(columns=["IVT", "hospital"])
    df = df[df["NIHSSa"] >= nihss_min].reset_index(drop=True)
    return df[NODES]


def load_magic_rct() -> pd.DataFrame:
    rct = pd.read_csv(REPO_ROOT / "data" / "exp_data.csv").rename(columns={
        "Age_rct": "Age", "NIHSSa_rct": "NIHSSa", "mRS_pre_rct": "mRS_pre",
        "T_rct": "T", "mRS_3m_rct": "mRS_3m"})
    rct = rct[NODES].dropna().reset_index(drop=True)
    return rct.astype({"NIHSSa": int, "mRS_pre": int, "T": int, "mRS_3m": int})


def load_data(source: str = DEFAULT_SOURCE):
    """Return ``(obs_df, rct_df, truth)`` for a data source.

    ``source`` is either ``"magic"`` (the private clinical cohort) or a synthetic
    folder ``"magic-mrclean/<variant>"`` under ``tramdag/data/``. ``truth`` is the
    parsed ``truth.json`` for a synthetic source (with the known true ATE), or
    ``None`` for the clinical data.
    """
    if source == "magic":
        return load_magic(), load_magic_rct(), None
    base = SIM_DATA / source
    if not (base / "obs.csv").exists():
        raise FileNotFoundError(
            f"unknown data source '{source}' (looked in {base}). "
            "Use 'magic' or 'magic-mrclean/{ls,nl}'.")
    obs = pd.read_csv(base / "obs.csv")[NODES]
    rct = pd.read_csv(base / "rct.csv")[NODES].astype(
        {"NIHSSa": float, "mRS_pre": int, "T": int, "mRS_3m": int})
    truth = json.loads((base / "truth.json").read_text())
    return obs, rct, truth


def DATA_R_REF(source: str) -> Path | None:
    """Path to the committed R reference (ref_ls/) for a synthetic source, or None."""
    if source == "magic":
        return None
    ref = SIM_DATA / source / "ref_ls"
    return ref if ref.exists() else None


def split(df: pd.DataFrame):
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df


# ------------------------------------------------------------------- specs
def build_spec(style: str) -> dict:
    """DAG spec for the fully-connected stroke DAG.

    style="flexible": the nihss6 configuration (Age 'ci', mRS_pre 'ls',
    NIHSSa 'cs', T 'ls' — per-edge terms as in nihss6/configuration.json).
    style="ls": all edges linear shift (classical proportional-odds analog).
    """
    if style == "flexible":
        t = {"Age": "ci", "mRS_pre": "ls", "NIHSSa": "cs", "T": "ls"}
    elif style == "ls":
        t = {"Age": "ls", "mRS_pre": "ls", "NIHSSa": "ls", "T": "ls"}
    else:
        raise ValueError(f"unknown style '{style}'")
    return {
        "Age":     ContinuousNode(transform="bernstein"),
        "mRS_pre": OrdinalNode(levels=6, terms=[term(t["Age"], "Age")]),
        "NIHSSa":  ContinuousNode(transform="bernstein",
                                  terms=[term(t["Age"], "Age"), term(t["mRS_pre"], "mRS_pre")]),
        "T":       OrdinalNode(levels=2,
                               terms=[term(t["Age"], "Age"), term(t["mRS_pre"], "mRS_pre"), term(t["NIHSSa"], "NIHSSa")]),
        "mRS_3m":  OrdinalNode(levels=7,
                               terms=[term(t["Age"], "Age"), term(t["mRS_pre"], "mRS_pre"), term(t["NIHSSa"], "NIHSSa"), term(t["T"], "T")]),
    }


# ------------------------------------------------------------------- plots
def saver(plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str):
        plt.savefig(plots_dir / name, dpi=150, bbox_inches="tight")
        plt.close()

    return save


def plot_loss_history(flow: CausalFlowDAG, save):
    fig, ax = plt.subplots(figsize=(8, 5))
    for which, style in [("train", "-"), ("val", "--")]:
        hist = flow.history[which]
        if not hist:
            continue
        total = [sum(e.values()) for e in hist]
        ax.plot(total, style, label=f"{which} (total)")
    ax.set_xlabel("epoch"); ax.set_ylabel("NLL"); ax.legend()
    ax.set_title("Joint flow training (sum of per-node NLLs)")
    save("loss_history.png")


def plot_training_speed(flow: CausalFlowDAG, save):
    """NLL vs wall-clock time, with the learning-rate schedule on a twin axis."""
    t = flow.history.get("time", [])
    lr = flow.history.get("lr", [])
    if not t:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for which, style in [("train", "-"), ("val", "--")]:
        total = [sum(e.values()) for e in flow.history[which]]
        ax.plot(t, total, style, label=f"{which} NLL")
    ax.set_xlabel("wall-clock time [s]"); ax.set_ylabel("total NLL")
    ax2 = ax.twinx()
    ax2.step(t, lr, where="post", color="grey", alpha=0.6, lw=1.2, label="lr")
    ax2.set_yscale("log"); ax2.set_ylabel("learning rate", color="grey")
    ax2.tick_params(axis="y", colors="grey")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    n_ep, total_s = len(t), t[-1]
    ax.set_title(f"Training speed: {n_ep} epochs in {total_s:.0f} s "
                 f"({n_ep / total_s:.1f} epochs/s, CPU)")
    save("training_speed.png")
    print(f"Training wall-clock: {total_s:.1f} s for {n_ep} epochs "
          f"({n_ep / total_s:.1f} epochs/s)")


def plot_samples_vs_true(flow: CausalFlowDAG, train_df: pd.DataFrame, save, n=10_000):
    sampled = flow.sample(n, seed=1)
    fig, axes = plt.subplots(2, len(NODES), figsize=(4 * len(NODES), 7))
    for j, node in enumerate(NODES):
        for row, (label, data) in enumerate([("observed (train)", train_df[node]),
                                             ("sampled", sampled[node])]):
            ax = axes[row][j]
            if node in ("Age", "NIHSSa"):
                ax.hist(data, bins=30, color="steelblue", edgecolor="white")
            else:
                lv = range(int(train_df[node].max()) + 1)
                counts = pd.Series(data).round().astype(int).value_counts(
                    normalize=True).reindex(lv, fill_value=0)
                ax.bar(counts.index, counts.values, color="steelblue", edgecolor="white")
            ax.set_title(f"{label} — {node}", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save("samples_vs_true.png")
    return sampled


def plot_interventional(flow: CausalFlowDAG, save, n=10_000):
    s_obs = flow.sample(n, seed=2)
    s_t0 = flow.sample(n, do={"T": 0}, seed=2)
    s_t1 = flow.sample(n, do={"T": 1}, seed=2)
    sets = {"Observational": s_obs, "do(T=0)": s_t0, "do(T=1)": s_t1}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    x = np.arange(7)
    for ax, (name, s) in zip(axes, sets.items()):
        counts = s["mRS_3m"].round().astype(int).value_counts(
            normalize=True).reindex(MRS_LEVELS, fill_value=0)
        ax.bar(x, counts.values, color="steelblue", edgecolor="white")
        good = (s["mRS_3m"] <= 2).mean()
        ax.set_title(f"{name}\nP(mRS<=2) = {good:.3f}")
        ax.set_xlabel("mRS_3m"); ax.set_xticks(x)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("probability")
    plt.tight_layout()
    save("interventional_mrs3m.png")
    ate_sample = (s_t1["mRS_3m"] <= 2).mean() - (s_t0["mRS_3m"] <= 2).mean()
    print(f"Population (sampling) ATE on P(good): {ate_sample:+.4f}")
    return sets


# --------------------------------------------------------------- evaluation
def evaluate_rct(flow: CausalFlowDAG, results_dir: Path, save,
                 rct: pd.DataFrame | None = None, truth: dict | None = None) -> pd.DataFrame:
    """Per-patient interventional PMFs on the RCT covariates -> ATE/ITE,
    calibration and RCT-vs-predicted distribution plots.

    Mirrors the analytic evaluation in the original implementation's scripts
    (predict_ordinal_pmf in stroke_fully_connected.py / nihss6.py). ``truth``,
    if given (synthetic data), supplies the known true ATE for comparison.
    """
    if rct is None:
        rct = load_magic_rct()
    pmf_t0 = flow.pmf(rct, node="mRS_3m", do={"T": 0})
    pmf_t1 = flow.pmf(rct, node="mRS_3m", do={"T": 1})
    assert np.allclose(pmf_t0.sum(axis=1), 1, atol=1e-5)
    assert np.allclose(pmf_t1.sum(axis=1), 1, atol=1e-5)

    result = rct.rename(columns={"T": "T_actual", "mRS_3m": "mRS_3m_observed"}).copy()
    for k in MRS_LEVELS:
        result[f"p_mRS{k}_T0"] = pmf_t0[:, k]
        result[f"p_mRS{k}_T1"] = pmf_t1[:, k]
    result["p_good_T0"] = pmf_t0[:, :3].sum(axis=1)
    result["p_good_T1"] = pmf_t1[:, :3].sum(axis=1)
    result["p_good_diff"] = result["p_good_T1"] - result["p_good_T0"]
    out_csv = results_dir / "rct_predicted_proba.csv"
    result.to_csv(out_csv, index=False)

    ate = result["p_good_diff"].mean()
    print(f"\n=== RCT evaluation (N={len(result)}) ===")
    print(f"  P(good | do(T=0)) = {result['p_good_T0'].mean():.4f}")
    print(f"  P(good | do(T=1)) = {result['p_good_T1'].mean():.4f}")
    print(f"  ATE  = {ate:+.4f}   median ITE = {result['p_good_diff'].median():+.4f}")
    if truth is not None:
        err = ate - truth["true_ate"]
        print(f"  TRUE ATE (known) = {truth['true_ate']:+.4f}   "
              f"flow error = {err:+.4f}   (naive obs. diff {truth['naive_obs_diff']:+.4f})")
    else:
        print(f"  MR CLEAN reference: +{RCT_ATE:.3f} "
              f"(95% CI [{RCT_CI[0]:+.3f}, {RCT_CI[1]:+.3f}])")
    print(f"  Saved per-patient PMFs -> {out_csv}")

    # --- observed RCT arms vs predicted distribution ---
    width = 0.38
    x = np.arange(7)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for arm, color in [(0, "#e07b54"), (1, "steelblue")]:
        obs = rct.loc[rct["T"] == arm, "mRS_3m"].value_counts(
            normalize=True).reindex(MRS_LEVELS, fill_value=0)
        axes[0].bar(x + (arm - 0.5) * width, obs.values * 100, width=width,
                    label=f"RCT (T={arm})", color=color)
        pred = (pmf_t1 if arm else pmf_t0).mean(axis=0)
        axes[1].bar(x + (arm - 0.5) * width, pred * 100, width=width,
                    label=f"predicted do(T={arm})", color=color)
    axes[0].set_title("MR CLEAN observed"); axes[1].set_title("Flow predicted (RCT covariates)")
    for ax in axes:
        ax.set_xlabel("mRS_3m"); ax.set_xticks(x); ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("percent")
    plt.tight_layout()
    save("rct_vs_predicted.png")

    # --- calibration, T=0 arm ---
    sub0 = result[result["T_actual"] == 0]
    n0 = len(sub0)
    pred_cal = sub0[[f"p_mRS{k}_T0" for k in MRS_LEVELS]].mean().values
    counts = sub0["mRS_3m_observed"].value_counts().reindex(MRS_LEVELS, fill_value=0).values
    obs_cal = counts / n0
    lo, hi = proportion_confint(counts, n0, alpha=0.05, method="wilson")
    yerr = np.vstack([obs_cal - lo, hi - obs_cal])
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.4
    ax.bar(x - w / 2, pred_cal, width=w, label="predicted (do T=0)", color="steelblue")
    ax.bar(x + w / 2, obs_cal, width=w, label="observed", color="#e07b54")
    ax.errorbar(x + w / 2, obs_cal, yerr=yerr, fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_xticks(x); ax.set_xlabel("mRS at 3 months"); ax.set_ylabel("probability")
    ax.set_title(f"T=0 arm calibration (N={n0}); "
                 f"P(good): pred {pred_cal[:3].sum():.3f} vs obs {obs_cal[:3].sum():.3f}")
    ax.legend()
    plt.tight_layout()
    save("calibration_T0.png")

    # --- ITE histogram ---
    ite = result["p_good_diff"].values
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ite, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="grey", ls=":", lw=1)
    ax.axvline(ate, color="red", ls="--", lw=2, label=f"ATE = {ate:+.3f}")
    ax.set_xlabel("ITE = P(good | do(T=1)) - P(good | do(T=0))")
    ax.set_ylabel("patients"); ax.legend()
    ax.set_title(f"Per-patient thrombectomy effect (N={len(ite)}, benefit>0: "
                 f"{100 * (ite > 0).mean():.0f}%)")
    plt.tight_layout()
    save("ite_histogram.png")

    return result


def run_experiment(name: str, style: str, source: str = DEFAULT_SOURCE,
                   phases=((3000, 1e-2), (1000, 1e-3)), batch_size=256, seed=123,
                   restore_best: bool | None = None):
    """Full pipeline: fit -> diagnostics -> RCT evaluation -> save model.

    ``source`` selects the data ("magic" private clinical, or a synthetic
    "magic-mrclean/{ls,nl}"). ``phases`` is a sequence of (epochs, lr) phases.
    ``restore_best`` toggles early-stopping (best-validation) weight restoration;
    if None it defaults per style — off for the constrained all-`ls` model (its
    MLE is the proportional-odds estimate, no overfitting), on for the flexible
    `ci`/`cs` model, whose MLE overfits the observational confounding so it needs
    the regularization to recover the causal effect.
    """
    if restore_best is None:
        restore_best = style != "ls"
    results_dir = RESULTS_ROOT / name
    results_dir.mkdir(parents=True, exist_ok=True)
    save = saver(results_dir / "plots")

    df, rct, truth = load_data(source)
    train_df, val_df, test_df = split(df)
    print(f"data '{source}': N={len(df)}  train/val/test = "
          f"{len(train_df)}/{len(val_df)}/{len(test_df)}"
          + (f"  | true ATE {truth['true_ate']:+.4f}" if truth else ""))

    torch.manual_seed(seed)  # weight init happens at construction -> must be seeded
    flow = CausalFlowDAG(build_spec(style))
    n_par = sum(p.numel() for p in flow.parameters())
    print(f"Model '{name}' (style={style}): {n_par} parameters")

    print(f"restore_best (early stopping) = {restore_best}")
    for i, (epochs, lr) in enumerate(phases):
        print(f"--- phase {i + 1}/{len(phases)}: {epochs} epochs at lr {lr:g} ---")
        flow.fit(train_df, val_df, epochs=epochs, learning_rate=lr,
                 batch_size=batch_size, verbose=200, seed=seed if i == 0 else None,
                 restore_best=restore_best)

    print("\nPer-node val NLL:", {k: round(v, 4) for k, v in flow.nll(val_df).items()})
    print("Per-node test NLL:", {k: round(v, 4) for k, v in flow.nll(test_df).items()})

    plot_loss_history(flow, save)
    plot_training_speed(flow, save)
    plot_samples_vs_true(flow, train_df, save)
    plot_interventional(flow, save)
    result = evaluate_rct(flow, results_dir, save, rct=rct, truth=truth)

    flow.save(results_dir / "flow.pt")
    print(f"\nModel saved -> {results_dir / 'flow.pt'}")
    return flow, result
