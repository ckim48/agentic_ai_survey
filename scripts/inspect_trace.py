#!/usr/bin/env python3
import hashlib
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    with open(os.path.join(ROOT, "config", "simulation.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    path = os.path.join(ROOT, cfg["experiment"]["dataset_path"])
    d = np.load(path, allow_pickle=False)
    pos, times = d["pos"], d["times"]
    valid = np.isfinite(pos).all(axis=2)
    nonempty = valid.any(axis=1)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    print("path:", os.path.realpath(path))
    print("sha256:", h.hexdigest())
    print("ids:", len(d["ids"]))
    print("raw snapshots:", len(times), "nonempty:", int(nonempty.sum()))
    print("duration_s:", float(times[-1] - times[0]))
    print("median_dt_s:", float(np.median(np.diff(times))))
    print("active vehicles/frame min/median/max:", int(valid.sum(1).min()),
          float(np.median(valid.sum(1))), int(valid.sum(1).max()))
    print("vehicles coverage>=0.70:", int((valid.mean(0) >= 0.70).sum()))


if __name__ == "__main__":
    main()
