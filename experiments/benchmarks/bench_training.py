"""Training-speed benchmark: lr schedules x batch size x device (+ LBFGS).

How fast can a CausalFlowDAG reach a *known-good* fit? Two workloads with
exact targets:

- **W1 stroke-ls** — all-`ls` 5-node DAG on the frozen synthetic cohort
  (experiments/misc/data/magic-mrclean/ls/obs.csv, n=1275, full-data fit). The
  optimum is the classical proportional-odds MLE (pinned against
  statsmodels in tests);
  target = train NLL within 1e-3 of a cached long-run reference.
- **W2 vaca-ci** — all-`ci` flow on the frozen VACA cohort
  (experiments/paper/data/vaca/obs.csv, n=5000, 90/10 split); target = val NLL
  within 2e-3 of a cached reference.

Because ``fit`` records per-epoch val NLL *and* wall-clock time in
``flow.history``, every config runs once and time-to-target is read off the
history post hoc — no instrumentation overhead.

Usage (from experiments/)::

    uv run python -m benchmarks.bench_training            # full grid (~30-45 min)
    uv run python -m benchmarks.bench_training --quick    # 1 seed, cpu only

Outputs -> results/bench-training/{results.csv, ranking.csv, nll_vs_time_*.png,
reference.json}.
"""

# %% imports ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tramdag import LS, CausalFlowDAG, ContinuousNode, I, OrdinalNode

# %% global variables ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
OUT = HERE / "results" / "bench-training"
# two target tiers (NLL gap vs the long-run reference):
# tight = exact-MLE equivalence; practical = coefficient-/density-equivalent
# (a stroke-ls fit with gap ~3e-3 already matches the R coefficients within
# the test tolerances, see tests/test_fit_schedules.py)
TOL_TIGHT = {"stroke-ls": 1e-3, "vaca-ci": 2e-3}
TOL_PRACT = {"stroke-ls": 5e-3, "vaca-ci": 1e-2}

# (schedule label, list of fit-phases, extra fit kwargs); budgets per workload
# 'onecycle' and 'cosine' were measured in the June 2026 grid and are still in
# the table of docs/training-speed.md, but fit() dropped both schedules in 0.4
# (they lost to plateau on every workload), so they cannot be run again here.
CONFIGS = {
    "stroke-ls": [
        ("baseline-2phase", [(3000, 1e-2), (1000, 1e-3)], {}),
        ("constant", [(4000, 1e-2)], {}),
        # min_delta 1e-5: the stroke metric is train NLL (full-data MLE fit),
        # evaluated deterministically -> a fine threshold is noise-safe and
        # lets the schedule keep recognizing slow progress near the optimum
        (
            "plateau+freeze",
            [(4000, 1e-2)],
            {
                "schedule": "plateau",
                "plateau_patience": 30,
                "freeze_patience": 120,
                "min_delta": 1e-5,
            },
        ),
    ],
    "vaca-ci": [
        ("baseline-2phase", [(400, 1e-2), (120, 1e-3)], {}),
        ("constant", [(520, 1e-2)], {}),
        (
            "plateau+freeze",
            [(1500, 1e-2)],
            {"schedule": "plateau", "plateau_patience": 15, "freeze_patience": 50},
        ),
    ],
}


# %% private functions -----------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, cpu only")
    ap.add_argument(
        "--figures-only",
        action="store_true",
        help="refit seed 0 / cpu just to (re)draw the curves; "
        "does not overwrite results.csv or ranking.csv",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return ap.parse_args()


def _devices_for(label: str, mps_ok: bool) -> list[str]:
    """Give the devices to run: cpu, plus the mps spotlight.

    The spotlight measures mps on two representative schedules only.
    """
    devices = ["cpu"]
    if mps_ok and label in ("constant", "plateau+freeze"):
        devices.append("mps")
    return devices


def _grid(workload: str, seeds: list[int], mps_ok: bool):
    """Yield one (label, phases, extra, batch, device, seed) run per cell."""
    cells = itertools.product(CONFIGS[workload], (512, "full"))
    for (label, phases, extra), batch in cells:
        for device in _devices_for(label, mps_ok):
            # the mps spotlight runs the first seed only
            for seed in seeds if device == "cpu" else seeds[:1]:
                yield label, phases, extra, batch, device, seed


def _run_one(workload, ref, label, phases, extra, batch, device, seed):
    """Run one recipe, read time-to-target off its history, print one line.

    Gives the result row plus the (times, nll) history the curves need.
    """
    t0 = time.perf_counter()
    flow = run_config(workload, phases, extra, batch, device, seed)
    wall = time.perf_counter() - t0
    nll = total_monitored_nll(flow.history)
    times = np.array(flow.history["time"])
    hit_t = np.nonzero(nll <= ref + TOL_TIGHT[workload])[0]
    hit_p = np.nonzero(nll <= ref + TOL_PRACT[workload])[0]
    t_tight = float(times[hit_t[0]]) if len(hit_t) else None
    t_pract = float(times[hit_p[0]]) if len(hit_p) else None
    row = {
        "workload": workload,
        "schedule": label,
        "batch": batch,
        "device": device,
        "seed": seed,
        "time_to_target_s": t_tight,
        "time_to_practical_s": t_pract,
        "epochs_to_target": int(hit_t[0]) + 1 if len(hit_t) else None,
        "total_time_s": wall,
        "epochs_run": len(nll),
        "final_nll": float(nll[-1]),
    }
    tt = f"{t_tight:6.1f}s" if t_tight else "  MISS "
    tp = f"{t_pract:6.1f}s" if t_pract else "  MISS "
    print(
        f"  {label:16s} b={batch!s:5s} {device:3s} "
        f"seed {seed}: practical @ {tp}  tight @ {tt}  "
        f"(ran {len(nll)} ep, {wall:.1f}s)"
    )
    return row, times, nll


def _run_workload(workload, seeds, mps_ok, rows, curves) -> None:
    """Run the schedule x batch x device grid of one workload."""
    ref = reference_nll(workload)
    print(
        f"[{workload}] reference NLL {ref:.4f} "
        f"(tight +{TOL_TIGHT[workload]}, practical +{TOL_PRACT[workload]})"
    )
    for label, phases, extra, batch, device, seed in _grid(workload, seeds, mps_ok):
        row, times, nll = _run_one(
            workload, ref, label, phases, extra, batch, device, seed
        )
        rows.append(row)
        if seed == seeds[0] and device == "cpu":
            curves.setdefault(workload, {})[f"{label}/b{batch}"] = (times, nll - ref)


def _run_lbfgs_grid(seeds, rows) -> None:
    """Run the LBFGS recipes, cold and warm-started, one line each."""
    for seed in seeds:
        for warm in (0, 150):
            r = run_lbfgs(seed, warm_epochs=warm)
            rows.append(r)
            tt = (
                f"{r['time_to_target_s']:6.2f}s" if r["time_to_target_s"] else "  MISS "
            )
            tp = (
                f"{r['time_to_practical_s']:6.2f}s"
                if r["time_to_practical_s"]
                else "  MISS "
            )
            print(
                f"  {r['schedule']:16s} b=full  cpu seed {seed}: "
                f"practical @ {tp}  tight @ {tt}"
            )


def _write_results(rows) -> None:
    """Write results.csv and the cpu median ranking."""
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    # ranking: median time-to-target over seeds (cpu only)
    med = (
        df[df["device"] == "cpu"]
        .groupby(["workload", "schedule", "batch"])[
            ["time_to_practical_s", "time_to_target_s"]
        ]
        .median()
        .sort_values("time_to_practical_s")
    )
    ranking = med.reset_index()
    ranking.to_csv(OUT / "ranking.csv", index=False)
    print("\n=== ranking (median seconds to practical / tight target, cpu) ===")
    print(ranking.to_string(index=False))


def _plot_workload(workload, cs, sched_color, seed) -> None:
    """Save the curve archive and the NLL-vs-time figure of one workload."""
    np.savez(
        OUT / f"curves_{workload}.npz",
        **{label: np.vstack([t, gap]) for label, (t, gap) in cs.items()},
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, (t, gap) in cs.items():
        sched, b = label.rsplit("/", 1)
        ax.plot(
            t,
            np.maximum(gap, 1e-5),
            lw=1.4,
            color=sched_color[sched],
            ls="-" if b == "b512" else ":",
            label=label,
        )
    ax.axhline(TOL_TIGHT[workload], color="k", ls="--", lw=1, label="tight tol")
    ax.axhline(TOL_PRACT[workload], color="k", ls=":", lw=1, label="practical tol")
    ax.set_yscale("log"), ax.set_xlabel("wall-clock seconds")
    ax.set_ylabel("monitored NLL − reference\n(train for stroke-ls)")
    ax.set_title(
        f"{workload}: convergence vs wall-clock (seed {seed}, cpu; "
        "solid = batch 512, dotted = full batch)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"nll_vs_time_{workload}.png", dpi=150)
    plt.close(fig)


def _plot_curves(curves, seed) -> None:
    """Draw one convergence figure per workload, schedule colors shared."""
    schedules = list(dict.fromkeys(c[0] for w in CONFIGS for c in CONFIGS[w]))
    sched_color = {s: f"C{i}" for i, s in enumerate(schedules)}
    for workload, cs in curves.items():
        _plot_workload(workload, cs, sched_color, seed)


# %% public functions ------------------------------------------------------------------
# ----------------------------------------------------------------- workloads
def all_ls_spec():
    """Give the 5-node all-``ls`` spec of the frozen cohort.

    Every edge is a linear shift, so each node-conditional is a classical
    transformation model and the optimum is the classical MLE. Written out
    here rather than imported: this benchmark is the only caller left.
    """
    return {
        "Age": ContinuousNode(),
        "mRS_pre": OrdinalNode(6, [LS("Age")]),
        "NIHSSa": ContinuousNode([LS("Age"), LS("mRS_pre")]),
        "T": OrdinalNode(2, [LS("Age"), LS("mRS_pre"), LS("NIHSSa")]),
        "mRS_3m": OrdinalNode(7, [LS("Age"), LS("mRS_pre"), LS("NIHSSa"), LS("T")]),
    }


def stroke_data():
    # the two workloads keep the frozen CSVs they were measured on, read from
    # the areas that own them (docs/training-speed.md reports those numbers)
    obs = pd.read_csv(
        EXPERIMENTS / "misc" / "data" / "magic-mrclean" / "ls" / "obs.csv"
    )
    return obs, None  # full-data MLE fit: val = train


def vaca_data():
    obs = pd.read_csv(EXPERIMENTS / "paper" / "data" / "vaca" / "obs.csv")
    cut = int(len(obs) * 0.9)
    return obs.iloc[:cut], obs.iloc[cut:]


def vaca_spec():
    return {
        "x1": ContinuousNode(),
        "x2": ContinuousNode([I("x1")]),
        "x3": ContinuousNode([I("x1", "x2")]),
    }


# not in "global variables": the dict references the workload functions above
# at definition time
WORKLOADS = {
    "stroke-ls": dict(data=stroke_data, spec=all_ls_spec),
    "vaca-ci": dict(data=vaca_data, spec=vaca_spec),
}


def total_monitored_nll(history) -> np.ndarray:
    return np.array([sum(ep.values()) for ep in history["val"]])


def run_config(workload, phases, extra, batch, device, seed):
    train, val = WORKLOADS[workload]["data"]()
    torch.manual_seed(seed)
    flow = CausalFlowDAG(WORKLOADS[workload]["spec"](), device=device)
    bs = len(train) if batch == "full" else int(batch)
    for epochs, lr in phases:
        flow.fit(
            train,
            val,
            epochs=epochs,
            learning_rate=lr,
            batch_size=bs,
            verbose=0,
            **extra,
        )
    return flow


# ----------------------------------------------------------------- reference
def reference_nll(workload: str) -> float:
    """Long-run reference NLL per workload, cached to reference.json."""
    cache = OUT / "reference.json"
    refs = json.loads(cache.read_text()) if cache.exists() else {}
    if workload in refs:
        return refs[workload]
    print(f"[ref] computing long-run reference for {workload} ...")
    train, val = WORKLOADS[workload]["data"]()
    torch.manual_seed(123)
    flow = CausalFlowDAG(WORKLOADS[workload]["spec"]())
    if workload == "stroke-ls":  # the tests' full-MLE recipe
        for epochs, lr in [(4000, 1e-2), (2000, 1e-3), (1000, 1e-4)]:
            flow.fit(
                train, val, epochs=epochs, learning_rate=lr, batch_size=512, verbose=0
            )
    else:
        flow.fit(
            train,
            val,
            epochs=3000,
            learning_rate=1e-2,
            batch_size=512,
            verbose=0,
            schedule="plateau",
            plateau_patience=25,
            freeze_patience=120,
        )
        flow.fit(train, val, epochs=300, learning_rate=1e-3, batch_size=512, verbose=0)
    refs[workload] = float(total_monitored_nll(flow.history).min())
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(refs, indent=2) + "\n")
    return refs[workload]


# --------------------------------------------------------------------- LBFGS
def run_lbfgs(seed: int, warm_epochs: int = 0) -> dict:
    """Run full-batch LBFGS on the stroke all-ls workload.

    A few Adam epochs can warm-start the fit. Adam escapes the early plateau
    and LBFGS then polishes the result.
    """
    train, _ = stroke_data()
    torch.manual_seed(seed)
    flow = CausalFlowDAG(all_ls_spec())
    ref = reference_nll("stroke-ls")
    t0 = time.perf_counter()
    if warm_epochs:
        flow.fit(
            train, epochs=warm_epochs, learning_rate=1e-2, batch_size=512, verbose=0
        )
    flow._set_ranges(train)
    vals = flow._tensorize(train)
    flow.train()
    opt = torch.optim.LBFGS(
        flow.parameters(),
        lr=1.0,
        max_iter=40,
        history_size=30,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = torch.stack(
            [-lp.mean() for lp in flow.node_log_prob(vals).values()]
        ).sum()
        loss.backward()
        return loss

    t_tight = t_pract = None
    loss = float("inf")
    iters = 0
    for _ in range(15):  # up to 15 x 40 inner iterations
        iters += 1
        loss = float(opt.step(closure))
        t = time.perf_counter() - t0
        if t_pract is None and loss <= ref + TOL_PRACT["stroke-ls"]:
            t_pract = t
        if t_tight is None and loss <= ref + TOL_TIGHT["stroke-ls"]:
            t_tight = t
            break
    label = f"adam{warm_epochs}+lbfgs" if warm_epochs else "lbfgs"
    return {
        "workload": "stroke-ls",
        "schedule": label,
        "batch": "full",
        "device": "cpu",
        "seed": seed,
        "time_to_target_s": t_tight,
        "time_to_practical_s": t_pract,
        "total_time_s": time.perf_counter() - t0,
        "epochs_run": warm_epochs + iters * 40,
        "final_nll": loss,
        "epochs_to_target": None,
    }


# ----------------------------------------------------------------------- main
def main():
    args = _parse_args()
    seeds = args.seeds[:1] if (args.quick or args.figures_only) else args.seeds
    OUT.mkdir(parents=True, exist_ok=True)

    mps_ok = torch.backends.mps.is_available() and not (args.quick or args.figures_only)
    rows, curves = [], {}
    for workload in WORKLOADS:
        _run_workload(workload, seeds, mps_ok, rows, curves)
    if not args.figures_only:
        _run_lbfgs_grid(seeds, rows)
        _write_results(rows)
    _plot_curves(curves, seeds[0])
    print(f"\n-> {OUT}")


# %% main ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
