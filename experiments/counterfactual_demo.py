"""Counterfactual showcase: Pearl's abduction -> action -> prediction.

Loads a fitted flow (default: the all-ls run) and, for the held-out test
patients, answers the per-patient question "what would this patient's
3-month mRS have been under the opposite treatment?":

    u  = flow.abduct(df)                 # step 1: latents from observations
    cf = flow.sample(do={"T": 1 - t}, u=u)  # steps 2+3: mutilate + push forward

Continuous latents are exact, ordinal latents are interval-identified and
sampled from the truncated logistic, so the counterfactual outcome is itself
a draw; we average over draws to get per-patient counterfactual PMFs.

Usage: uv run python counterfactual_demo.py [all_ls|nihss6]
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS_ROOT, load_magic, saver, split

from tramdag import CausalFlowDAG

N_DRAWS = 200  # truncated-logistic abduction draws per patient


def main(name: str = "all_ls"):
    results_dir = RESULTS_ROOT / name
    flow = CausalFlowDAG.load(results_dir / "flow.pt")
    save = saver(results_dir / "plots")

    _, _, test_df = split(load_magic())
    test_df = test_df.reset_index(drop=True)
    n = len(test_df)
    print(f"Counterfactuals for {n} test patients (model '{name}', {N_DRAWS} draws)")

    # --- sanity: factual reconstruction (deterministic for continuous nodes,
    #     level-exact for ordinal nodes since the latent lies in the observed bin)
    u = flow.abduct(test_df, seed=0)
    rec = flow.sample(u=u)
    for col in ["Age", "NIHSSa"]:
        err = np.abs(rec[col].values - test_df[col].values).max()
        print(f"  factual reconstruction {col}: max abs error {err:.2e}")
    for col in ["mRS_pre", "T", "mRS_3m"]:
        match = (rec[col].values == test_df[col].values).mean()
        print(f"  factual reconstruction {col}: {100 * match:.1f}% level-exact")

    # --- counterfactual mRS_3m under the flipped treatment, averaged over
    #     abduction draws (ordinal latents are only interval-identified)
    t_factual = test_df["T"].values
    cf_counts = np.zeros((n, 7))
    for d in range(N_DRAWS):
        u = flow.abduct(test_df, seed=1000 + d)
        cf = flow.sample(do={"T": 1}, u=u)
        cf0 = flow.sample(do={"T": 0}, u=u)
        # pick the arm opposite to the factual treatment, per patient
        mrs_cf = np.where(t_factual == 0, cf["mRS_3m"].values, cf0["mRS_3m"].values)
        for k in range(7):
            cf_counts[:, k] += mrs_cf == k
    cf_pmf = cf_counts / N_DRAWS

    out = test_df.rename(columns={"T": "T_factual", "mRS_3m": "mRS_3m_factual"}).copy()
    for k in range(7):
        out[f"p_cf_mRS{k}"] = cf_pmf[:, k]
    out["p_cf_good"] = cf_pmf[:, :3].sum(axis=1)
    out["good_factual"] = (out["mRS_3m_factual"] <= 2).astype(int)
    out_csv = results_dir / "counterfactuals_test.csv"
    out.to_csv(out_csv, index=False)
    print(f"  saved per-patient counterfactual PMFs -> {out_csv}")

    # untreated patients: what if they had been treated (and vice versa)?
    for arm, label in [(0, "untreated -> do(T=1)"), (1, "treated -> do(T=0)")]:
        sub = out[out["T_factual"] == arm]
        print(f"  {label}: factual P(good) {sub['good_factual'].mean():.3f}  "
              f"counterfactual P(good) {sub['p_cf_good'].mean():.3f}  (N={len(sub)})")

    # --- plot: factual vs counterfactual outcome distribution per arm
    x = np.arange(7)
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, arm in zip(axes, [0, 1]):
        sub = out[out["T_factual"] == arm]
        fact = sub["mRS_3m_factual"].value_counts(normalize=True).reindex(
            range(7), fill_value=0).values
        cf_mean = sub[[f"p_cf_mRS{k}" for k in range(7)]].mean().values
        ax.bar(x - width / 2, fact, width=width, label="factual", color="#e07b54")
        ax.bar(x + width / 2, cf_mean, width=width,
               label=f"counterfactual do(T={1 - arm})", color="steelblue")
        ax.set_title(f"factually T={arm} (N={len(sub)})")
        ax.set_xlabel("mRS_3m")
        ax.set_xticks(x)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("probability")
    plt.tight_layout()
    save("counterfactual_mrs3m.png")
    print(f"  plot -> {results_dir / 'plots' / 'counterfactual_mrs3m.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all_ls")
