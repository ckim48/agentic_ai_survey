#!/usr/bin/env python3
import argparse
import copy
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.simulator import (build_tasks, load_trace, make_infrastructure,
                           run_scheme, sha256_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/simulation.yaml")
    args = ap.parse_args()
    cp = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(cp, "r") as f:
        cfg = yaml.safe_load(f)
    path = cfg["experiment"]["dataset_path"]
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    outdir = os.path.join(ROOT, "results", "sensitivity")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    start = time.time()
    seed0 = int(cfg["experiment"]["seed_start"])
    for n in cfg["experiment"]["vehicle_counts"]:
        for k in range(int(cfg["experiment"]["independent_runs"])):
            seed = seed0 + k
            tr = load_trace(path, int(n), seed, int(cfg["experiment"]["frames_per_run"]),
                            float(cfg["experiment"]["minimum_vehicle_coverage"]),
                            cfg["experiment"].get("region_bbox_lonlat"))
            infra = make_infrastructure(tr, cfg)
            tasks = build_tasks(tr, cfg, seed)
            for delay in (0.00, 0.04, 0.08, 0.12):
                c = copy.deepcopy(cfg)
                c["orchestration"]["independent_mismatch_delay_s"] = delay
                m, _ = run_scheme("Independent Agents", tr, tasks, c, infra)
                rows.append({"study": "mismatch_delay", "value": delay,
                             "vehicles": n, "seed": seed, **m})
            for rounds in (1, 2, 3):
                c = copy.deepcopy(cfg)
                c["experiment"]["max_negotiation_rounds"] = rounds
                m, _ = run_scheme("AAI-CDOS", tr, tasks, c, infra)
                rows.append({"study": "negotiation_rounds", "value": rounds,
                             "vehicles": n, "seed": seed, **m})
            print("sensitivity vehicles=%d seed=%d elapsed=%.1fs" %
                  (n, seed, time.time() - start), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(outdir, "raw.csv"), index=False)

    raw = pd.DataFrame(rows)
    metrics = ["deadline_success", "p95_latency_s", "mean_dt_age_s"]
    summary = []
    for keys, g in raw.groupby(["study", "value", "vehicles"]):
        row = {"study": keys[0], "value": keys[1], "vehicles": int(keys[2]),
               "runs": len(g)}
        for metric in metrics:
            x = g[metric].to_numpy(float)
            sd = x.std(ddof=1)
            row[metric + "_mean"] = x.mean()
            row[metric + "_ci95"] = 1.96 * sd / np.sqrt(len(x))
        summary.append(row)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump({"dataset_sha256": sha256_file(path), "elapsed_s": time.time() - start,
                   "runs": int(cfg["experiment"]["independent_runs"]),
                   "frames": int(cfg["experiment"]["frames_per_run"])}, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    specs = [("mismatch_delay", axes[0], "Reconciliation delay (ms)"),
             ("negotiation_rounds", axes[1], "Maximum negotiation rounds")]
    for study, ax, xlabel in specs:
        q = s[s.study == study]
        for n in sorted(q.vehicles.unique()):
            g = q[q.vehicles == n].sort_values("value")
            x = g.value.to_numpy() * (1000 if study == "mismatch_delay" else 1)
            ax.errorbar(x, 100 * g.deadline_success_mean,
                        yerr=100 * g.deadline_success_ci95, marker="o", capsize=3,
                        label="N=%d" % n)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Deadline satisfaction (%)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    figdir = os.path.join(ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, "sensitivity." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
