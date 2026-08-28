#!/usr/bin/env python3
"""Run controlled network-environment robustness experiments."""

import argparse
import copy
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.simulator import run_one_configuration, runtime_manifest
from scripts.summarize_results import METRICS


def deep_update(target, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict):
            target.setdefault(key, {})
            deep_update(target[key], value)
        else:
            target[key] = value


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw):
    rows = []
    for (condition_id, condition, scheme, vehicles), group in raw.groupby(
            ["condition_id", "condition", "scheme", "vehicles"], sort=False):
        row = {
            "condition_id": condition_id,
            "condition": condition,
            "scheme": scheme,
            "vehicles": int(vehicles),
            "runs": int(len(group)),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_sd"] = sd
            row[metric + "_ci95"] = 1.96 * sd / np.sqrt(max(len(values), 1))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/simulation.yaml")
    parser.add_argument("--sweep", default="config/environment_sweep.yaml")
    parser.add_argument("--output", default="results/environment_sweep")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--frames", type=int)
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    sweep_path = args.sweep if os.path.isabs(args.sweep) else os.path.join(ROOT, args.sweep)
    output = args.output if os.path.isabs(args.output) else os.path.join(ROOT, args.output)
    with open(config_path, "r") as stream:
        base_cfg = yaml.safe_load(stream)
    with open(sweep_path, "r") as stream:
        sweep = yaml.safe_load(stream)

    vehicle_count = int(sweep["experiment"]["fixed_vehicle_count"])
    runs = int(args.runs or sweep["experiment"]["independent_runs"])
    frames = int(args.frames or sweep["experiment"]["frames_per_run"])
    seed0 = int(base_cfg["experiment"]["seed_start"])
    data_path = base_cfg["experiment"]["dataset_path"]
    data_path = data_path if os.path.isabs(data_path) else os.path.join(ROOT, data_path)
    os.makedirs(output, exist_ok=True)

    raw_rows, route_rows = [], []
    started = time.time()
    for condition_index, condition in enumerate(sweep["conditions"]):
        cfg = copy.deepcopy(base_cfg)
        deep_update(cfg, condition.get("overrides", {}))
        for run_index in range(runs):
            seed = seed0 + run_index
            results, routes, trace = run_one_configuration(
                cfg, data_path, vehicle_count, seed, frames)
            for row in results:
                row.update({
                    "condition_id": condition["id"],
                    "condition": condition["label"],
                    "condition_index": condition_index,
                })
                raw_rows.append(row)
            for row in routes:
                row.update({
                    "condition_id": condition["id"],
                    "condition": condition["label"],
                    "condition_index": condition_index,
                })
                route_rows.append(row)
            print("condition=%s seed=%d frames=%d elapsed=%.1fs" %
                  (condition["id"], seed, len(trace["times"]), time.time() - started),
                  flush=True)
            write_csv(os.path.join(output, "raw_runs.csv"), raw_rows)
            write_csv(os.path.join(output, "route_shares.csv"), route_rows)

    raw = pd.DataFrame(raw_rows)
    summary = summarize(raw)
    condition_order = {item["id"]: index for index, item in enumerate(sweep["conditions"])}
    summary["condition_index"] = summary["condition_id"].map(condition_order)
    summary = summary.sort_values(["condition_index", "scheme"]).reset_index(drop=True)
    summary.to_csv(os.path.join(output, "summary.csv"), index=False)

    manifest = runtime_manifest(base_cfg, data_path)
    manifest.update({
        "experiment_type": "controlled environment robustness sweep",
        "fixed_vehicle_count": vehicle_count,
        "runs_per_condition_scheme": runs,
        "frames_per_run": frames,
        "paired_seed_start": seed0,
        "elapsed_s": time.time() - started,
        "conditions": sweep["conditions"],
        "guardrail": "Seoul data supplies mobility only; radio, routing, queues, compute, and DT tasks are simulated.",
    })
    with open(os.path.join(output, "run_manifest.json"), "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)

    lines = [
        "# Environment robustness sweep", "",
        "This controlled experiment fixes the load at %d vehicles and replays the same "
        "%d seeds and %d Seoul mobility snapshots per condition." %
        (vehicle_count, runs, frames), "",
        "The condition labels correspond to explicit parameter overrides in "
        "`config/environment_sweep.yaml`; they are not post-hoc relabeling.", "",
        "The Seoul trace supplies mobility only. Radio, routing, queueing, computing, "
        "DT tasks, and policy outcomes are simulated.", "", "## Conditions", "",
    ]
    for item in sweep["conditions"]:
        lines.append("- **%s:** %s" % (item["label"], item["description"]))
    with open(os.path.join(output, "README.md"), "w") as stream:
        stream.write("\n".join(lines) + "\n")
    print(summary[["condition", "scheme", "deadline_success_mean",
                   "p95_position_error_m_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
