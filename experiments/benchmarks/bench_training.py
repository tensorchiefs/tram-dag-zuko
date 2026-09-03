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

The benchmark's per-epoch callback records validation NLL *and* wall-clock
time (``fit`` itself records only the per-node train NLL), so every config
runs once and time-to-target is read off that record post hoc — no
instrumentation overhead.

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
import yaml

from tramdag import CausalFlowDAG, spec_from_dict
from tramdag.callbacks import PerNodePlateau, per_node_adam

# %% global variables ------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "bench-training"
# workload specs, data, target tiers and the recipe grid — see the YAML
CFG = yaml.safe_load((HERE / "bench_training.yaml").read_text())
WORKLOADS = CFG["workloads"]
TOL_TIGHT = {w: cfg["tol_tight"] for w, cfg in WORKLOADS.items()}
TOL_PRACT = {w: cfg["tol_pract"] for w, cfg in WORKLOADS.items()}
CONFIGS = {
    w: [(c["label"], c["phases"], c["extra"]) for c in cfg["configs"]]
    for w, cfg in WORKLOADS.items()
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
    _, hist = run_config(workload, phases, extra, batch, device, seed)
    wall = time.perf_counter() - t0
    nll = total_monitored_nll(hist)
    times = np.array(hist["time"])
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
        for warm in CFG["lbfgs"]["warm_epochs"]:
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
def workload_spec(workload: str) -> dict:
    """Rebuild one workload's spec from the YAML (options-free: the defaults)."""
    return spec_from_dict(WORKLOADS[workload]["spec"])


def workload_data(workload: str):
    """Read one workload's frozen CSV and split it as the YAML says.

    The two workloads keep the frozen CSVs they were measured on, read from
    the areas that own them (docs/training-speed.md reports those numbers).
    A missing ``val_split`` means the full-data MLE fit: val = train.
    """
    cfg = WORKLOADS[workload]
    obs = pd.read_csv(HERE / cfg["data"])
    if cfg["val_split"] is None:
        return obs, None
    cut = int(len(obs) * (1.0 - cfg["val_split"]))
    return obs.iloc[:cut], obs.iloc[cut:]


def all_ls_spec():
    """Give the stroke workload's spec (the tests' anchor for `is_classical`)."""
    return workload_spec("stroke-ls")


def stroke_data():
    return workload_data("stroke-ls")


def vaca_data():
    return workload_data("vaca-ci")


def total_monitored_nll(history) -> np.ndarray:
    return np.array([sum(ep.values()) for ep in history["val"]])


def run_config(workload, phases, extra, batch, device, seed):
    """Run the phases of one recipe; give the flow and its (val, time) record."""
    train, val = workload_data(workload)
    val = train if val is None else val  # stroke-ls: the full-data MLE fit
    torch.manual_seed(seed)
    flow = CausalFlowDAG(workload_spec(workload), device=device)
    bs = len(train) if batch == "full" else int(batch)
    hist = {"val": [], "time": []}
    t0 = time.perf_counter()
    extras = extra if isinstance(extra, list) else [extra] * len(phases)
    for (epochs, lr), phase_extra in zip(phases, extras, strict=True):
        opt = per_node_adam(flow, lr)
        sched = PerNodePlateau(**phase_extra) if phase_extra else None

        def monitor(f, epoch, opt, sched=sched):
            hist["val"].append(f.history["val"][-1])  # fit computed it
            hist["time"].append(time.perf_counter() - t0)
            return sched is not None and sched.on_epoch_end(f, epoch, opt)

        flow.fit(
            train,
            epochs=epochs,
            batch_size=bs,
            validation_data=val,
            optimizer=opt,
            callbacks=monitor,
        )
    return flow, hist


def reference_nll(workload: str) -> float:
    """Long-run reference NLL per workload, cached to reference.json."""
    cache = OUT / "reference.json"
    refs = json.loads(cache.read_text()) if cache.exists() else {}
    if workload in refs:
        return refs[workload]
    print(f"[ref] computing long-run reference for {workload} ...")
    recipe = WORKLOADS[workload]["reference"]
    _, hist = run_config(
        workload,
        recipe["phases"],
        recipe["extra"],
        recipe["batch"],
        "cpu",
        recipe["seed"],
    )
    refs[workload] = float(total_monitored_nll(hist).min())
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(refs, indent=2) + "\n")
    return refs[workload]


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
    lb = CFG["lbfgs"]
    if warm_epochs:
        flow.fit(
            train,
            epochs=warm_epochs,
            learning_rate=lb["warm_lr"],
            batch_size=lb["warm_batch"],
        )
    # cold start: the same calibrated start a fit would take (no-op when warm).
    # Not fit_classical: this loop needs per-chunk time-to-target readings and
    # float32, and node_log_prob (not the no-grad log_prob) keeps the graph.
    flow.calibrate(train)
    vals = flow._tensorize(train)
    flow.train()
    opt = torch.optim.LBFGS(
        flow.parameters(),
        lr=lb["lr"],
        max_iter=lb["max_iter"],
        history_size=lb["history_size"],
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
    for _ in range(lb["chunks"]):  # up to chunks x max_iter inner iterations
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
        "epochs_run": warm_epochs + iters * lb["max_iter"],
        "final_nll": loss,
        "epochs_to_target": None,
    }


# %% main ------------------------------------------------------------------------------
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


if __name__ == "__main__":
    main()
