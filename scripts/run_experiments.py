#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.simulator import run_one_configuration, runtime_manifest


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/simulation.yaml")
    ap.add_argument("--runs", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--output")
    args = ap.parse_args()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    runs = int(args.runs or cfg["experiment"]["independent_runs"])
    frames = int(args.frames or cfg["experiment"]["frames_per_run"])
    output = args.output or cfg["experiment"]["output_dir"]
    output = output if os.path.isabs(output) else os.path.join(ROOT, output)
    data_path = cfg["experiment"]["dataset_path"]
    data_path = data_path if os.path.isabs(data_path) else os.path.join(ROOT, data_path)
    os.makedirs(output, exist_ok=True)

    raw, routes = [], []
    seed0 = int(cfg["experiment"]["seed_start"])
    started = time.time()
    for n in cfg["experiment"]["vehicle_counts"]:
        for k in range(runs):
            seed = seed0 + k
            r, s, tr = run_one_configuration(cfg, data_path, int(n), seed, frames)
            raw.extend(r)
            routes.extend(s)
            print("vehicles=%d seed=%d frames=%d elapsed=%.1fs" %
                  (n, seed, len(tr["times"]), time.time() - started), flush=True)
            write_csv(os.path.join(output, "raw_runs.csv"), raw)
            write_csv(os.path.join(output, "route_shares.csv"), routes)

    manifest = runtime_manifest(cfg, data_path)
    manifest.update({"runs_completed": runs, "frames_per_run": frames,
                     "elapsed_s": time.time() - started})
    with open(os.path.join(output, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print("done:", output)


if __name__ == "__main__":
    main()
