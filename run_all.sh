#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/inspect_trace.py
python3 scripts/run_experiments.py --config config/simulation.yaml
python3 scripts/summarize_results.py --config config/simulation.yaml
