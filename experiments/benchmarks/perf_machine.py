"""Cross-machine performance benchmark for tramdag — single self-contained file.

Runs two fixed workloads for exactly 200 epochs each (constant lr, no early
stopping — identical work on every machine), on every available device
(cpu, cuda, mps), and writes one JSON with timings + a machine fingerprint.

- **intro**  (n=5,000):  the 4-node SCM from ``notebooks/intro_tram_dag.py``
  (mixed continuous + ordinal, ls/cs terms) — measures per-step overhead,
  the regime of typical tabular fits.
- **large**  (n=50,000): all-``ci`` flow on the bimodal VACA benchmark, whose
  generator is copied into this file (the wheel ships ``src/tramdag`` only, and
  this script has to run from a bare ``curl``) — measures throughput, the
  regime where GPUs help.

Usage on any machine (no repo clone needed)::

    pip install tramdag                    # torch comes as a dependency
    curl -O https://raw.githubusercontent.com/tensorchiefs/tramdag/main/experiments/benchmarks/perf_machine.py
    python perf_machine.py                 # -> <YYYY-MM-DD-HHMM>_<host>.json + summary
    python perf_machine.py --devices cpu   # restrict devices

Collecting results: copy each machine's JSON into the repo's ``docs/perf/``
(when run from a repo clone the JSON is written there directly), then::

    python -m benchmarks.perf_machine --report docs/perf   # table + docs/perf/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import tramdag as td
from tramdag import CS, LS, CausalFlowDAG, ContinuousNode, I, OrdinalNode

EPOCHS = 200  # fixed: every machine does identical work


def vaca_dgp(n: int, seed: int = 42) -> pd.DataFrame:
    """Draw the bimodal benchmark SCM, written out so this file stands alone.

    ``x1`` is a Gaussian mixture, ``x2 = -x1 + N(0,1)`` and
    ``x3 = x1 + 0.25 x2 + N(0,1)`` — the TRAM-DAG paper's App. C.1 DGP.
    """
    rng = np.random.default_rng(seed + 1)
    # the draw ORDER is part of the sample: mixture indicator, both branch
    # normals, then the two noises. Any other order gives different rows and
    # makes final_val_nll incomparable with the committed machine results.
    mixture = rng.uniform(size=n)
    first = rng.normal(size=n)
    second = rng.normal(size=n)
    noise_x2 = rng.normal(size=n)
    noise_x3 = rng.normal(size=n)
    x1 = np.where(mixture < 0.5, -2.0 + np.sqrt(1.5) * first, 1.5 + second)
    x2 = -x1 + noise_x2
    x3 = x1 + 0.25 * x2 + noise_x3
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


# ----------------------------------------------------------- intro-NB workload
def intro_dgp(n: int, seed: int = 1) -> pd.DataFrame:
    """Sample the SCM of notebooks/intro_tram_dag.py, with known truth."""
    rng = np.random.default_rng(seed)
    z = {k: rng.logistic(size=n) for k in "1234"}
    x1 = (z["1"] + 0.4) / 1.2
    x2 = (z["2"] - 1.5 * x1 - 1.0) / 2.0
    x3 = np.arcsinh(z["3"] - 0.8 * x1 - 0.5 * x2**2)
    cut = np.array([-2.0, 0.0, 1.5])[None, :] - 1.0 * x3[:, None]
    y = (z["4"][:, None] > cut).sum(axis=1)
    return pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "Y": y.astype(float)})


WORKLOADS = {
    "intro": dict(
        n=5_000,
        batch=512,
        data=lambda: intro_dgp(5_000),
        spec=lambda: {
            "X1": ContinuousNode(),
            "X2": ContinuousNode([LS("X1")]),
            "X3": ContinuousNode([LS("X1"), CS("X2")]),
            "Y": OrdinalNode(4, [LS("X3")]),
        },
    ),
    "large": dict(
        n=50_000,
        batch=4096,
        data=lambda: vaca_dgp(50_000),
        spec=lambda: {
            "x1": ContinuousNode(),
            "x2": ContinuousNode([I("x1")]),
            "x3": ContinuousNode([I("x1", "x2")]),
        },
    ),
}


# ------------------------------------------------------------------- machine
def code_version() -> dict:
    """Give the tramdag version, plus the git commit inside a repo clone."""
    v = {"tramdag": td.__version__, "git_commit": None, "git_dirty": None}
    try:
        import subprocess

        here = str(Path(__file__).resolve().parent)
        v["git_commit"] = subprocess.check_output(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        v["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", here, "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        pass  # pip-installed / curl'd outside a repo: package version only
    return v


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


# ----------------------------------------------------------------- benchmark
def run_workload(name: str, device: str) -> dict:
    w = WORKLOADS[name]
    df = w["data"]()
    cut = int(w["n"] * 0.9)
    train, val = df.iloc[:cut], df.iloc[cut:]

    torch.manual_seed(0)
    flow = CausalFlowDAG(w["spec"](), device=device)
    # warm-up: kernel compilation / first-touch allocations, excluded from timing
    flow.fit(train, val, epochs=3, learning_rate=1e-2, batch_size=w["batch"], verbose=0)
    t0 = time.perf_counter()
    flow.fit(
        train, val, epochs=EPOCHS, learning_rate=1e-2, batch_size=w["batch"], verbose=0
    )
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    samp = flow.sample(100_000, seed=0)
    sample_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    flow.abduct(df.iloc[:10_000], seed=0)
    abduct_s = time.perf_counter() - t0

    return {
        "workload": name,
        "device": device,
        "n": w["n"],
        "batch": w["batch"],
        "epochs": EPOCHS,
        "fit_s": round(fit_s, 2),
        "epochs_per_s": round(EPOCHS / fit_s, 2),
        "sample_100k_s": round(sample_s, 2),
        "abduct_10k_s": round(abduct_s, 2),
        # cross-machine sanity check: same seed + data -> NLL must be ~equal
        "final_val_nll": round(sum(flow.history["val"][-1].values()), 4),
        "_sample_mean_x_last": round(float(samp.iloc[:, -1].mean()), 4),
    }


# -------------------------------------------------------------------- report
def report(directory: str) -> None:
    rows = []
    for f in sorted(Path(directory).glob("*.json")):
        d = json.loads(f.read_text())
        if "results" not in d:
            continue
        commit = (d.get("code") or {}).get("git_commit")
        for r in d["results"]:
            if "error" in r:
                continue
            rows.append(
                {
                    "host": d["machine"]["hostname"],
                    "chip": d["machine"]["processor"],
                    "gpu": d["machine"]["cuda"]
                    or ("mps" if d["machine"]["mps"] else "-"),
                    "code": commit or d.get("code", {}).get("tramdag", "?"),
                    **{
                        k: r[k]
                        for k in [
                            "workload",
                            "device",
                            "fit_s",
                            "epochs_per_s",
                            "sample_100k_s",
                            "final_val_nll",
                        ]
                    },
                }
            )
    if not rows:
        print(f"no benchmark JSONs found in {directory}")
        return
    t = pd.DataFrame(rows).sort_values(["workload", "fit_s"])
    print(t.to_string(index=False))
    # markdown table without extra dependencies (tabulate not required)
    cols = list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in t.itertuples(index=False)
    ]
    md = Path(directory) / "REPORT.md"
    md.write_text(
        "# tramdag cross-machine benchmark\n\n"
        f"Fixed {EPOCHS}-epoch workloads (see "
        "`experiments/benchmarks/perf_machine.py`); "
        "`final_val_nll` must agree across machines (same seed & data).\n\n"
        + "\n".join(lines)
        + "\n\n"
        "## Add your machine\n\n"
        "```bash\n"
        "# on the new machine (no repo clone needed; GPU optional)\n"
        "pip install tramdag\n"
        "curl -O https://raw.githubusercontent.com/tensorchiefs/tramdag/main/"
        "experiments/benchmarks/perf_machine.py\n"
        "python perf_machine.py            "
        "# ~2-5 min -> <YYYY-MM-DD-HHMM>_<host>.json\n"
        "```\n\n"
        "Then copy the JSON into `docs/perf/` in the repo, commit it, and\n"
        "regenerate this file (or ask Claude to do it):\n\n"
        "```bash\n"
        "python experiments/benchmarks/perf_machine.py --report docs/perf\n"
        "```\n\n"
        "Sanity check before committing: the new row's `final_val_nll` should\n"
        "match the existing rows to ~1e-3 — same seed and data everywhere.\n"
        "*(This file is auto-generated by `--report`; don't edit it by hand.)*\n"
    )
    print(f"-> {md}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="subset of {cpu, cuda, mps}; default: all available",
    )
    ap.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="output directory (default: docs/perf inside a clone, "
        "else current directory)",
    )
    ap.add_argument(
        "--report",
        metavar="DIR",
        help="merge perf_*.json from DIR into a table and exit",
    )
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return

    info = td.machine_info()
    devices = args.devices or available_devices()
    print(
        f"tramdag {info['tramdag']} | torch {info['torch']} | "
        f"{info['hostname']} ({info['processor']}, {info['cpu_count']} cores, "
        f"{info['ram_gb']} GB) | devices: {devices}\n"
    )

    results = []
    for wl in WORKLOADS:
        for dev in devices:
            print(f"[{wl} / {dev}] {EPOCHS} epochs ...", end=" ", flush=True)
            try:
                r = run_workload(wl, dev)
                results.append(r)
                print(
                    f"{r['fit_s']:7.1f}s  ({r['epochs_per_s']:.1f} ep/s)  "
                    f"sample {r['sample_100k_s']:.1f}s  "
                    f"NLL {r['final_val_nll']}"
                )
            except Exception as e:  # e.g. an op unsupported on mps
                results.append(
                    {"workload": wl, "device": dev, "error": f"{type(e).__name__}: {e}"}
                )
                print(f"FAILED — {type(e).__name__}: {e}")

    # default output: docs/perf/ when running inside a repo clone, else cwd
    checkout = next(
        (d for d in Path(__file__).resolve().parents if (d / ".git").exists()), None
    )
    default_out = checkout / "docs" / "perf" if checkout else Path.cwd()
    out_dir = Path(args.out) if args.out else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d-%H%M}"
    out = out_dir / f"{stamp}_{info['hostname']}.json"
    out.write_text(
        json.dumps(
            {
                "machine": info,
                "code": code_version(),
                "epochs": EPOCHS,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n-> {out.resolve()}")
    print(
        "collect the JSONs from all machines in docs/perf/, then:"
        "  python -m benchmarks.perf_machine --report docs/perf"
    )


if __name__ == "__main__":
    main()
