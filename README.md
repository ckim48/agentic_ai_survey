# AAI-CDOS trace-driven simulation

This directory contains a reproducible first-pass evaluation of **AAI-CDOS**
for vehicular digital-twin (DT) synchronization in an urban space-air-ground
integrated network (SAGIN). Mobility is replayed from a real Seoul V2X trace;
radio, routing, queueing, and compute states are simulated.

## What is real and what is simulated

- **Real:** anonymized terminal IDs, WGS84 vehicle positions, and capture times
  from the Seoul T-data V2X vehicle-status feed (dataset 15102656). The trace is
  not redistributed in this public repository; obtain it from an authorized
  source and place it at `data/seoul_v2x_trace_evening45.npz`.
- **Simulated:** channel fading, bandwidth sharing, terrestrial/UAV/LEO links,
  E2E routes, compute queues, task sizes, workloads, deadlines, and agents.
- The source trace is never modified. Its expected SHA-256 is documented in
  `data/README.md`, and every new run records the digest it actually used.

The trace contains 1,094 unique terminal IDs and 444 raw snapshots over about
89.5 minutes. The experiment selects vehicles whose median location is inside
the declared Gangnam bounding box and whose coverage is at least 70%. Empty
API-outage snapshots are discarded; missing positions of a selected vehicle
are linearly interpolated inside each replay window.

## Experiment

We evaluate 10, 25, and 50 connected vehicles. Every configuration uses the
same vehicle trajectories, task requests, channel samples, resource states,
and hard application deadlines across all schemes. There are 30 independent
runs (seeds 2026--2055), and summaries report the mean, sample standard
deviation, and normal-approximation 95% confidence interval.

Schemes:

1. `AAI-CDOS`: iterative observe-plan-tool-verify-coordinate-execute-feedback,
   joint radio/routing/compute evaluation, hard-deadline verifier, memory, and
   at most three negotiation rounds.
2. `Single Domain`: radio-centric access choice with co-located processing.
3. `Independent Agents`: radio, routing, and compute choices are optimized
   independently without joint-plan verification. When independently selected
   access and processing intents conflict, the simulator adds the configured
   80-ms reconciliation/redispatch delay; this baseline assumption is exposed
   in YAML and should be included in sensitivity analysis.
4. `One-Shot LLM`: one noisy holistic proposal without memory, tool feedback,
   negotiation, or verification.

The 30-seed network-performance experiment uses a deterministic **planner
surrogate** so that it is exactly reproducible and cannot incur unbounded API
cost. Columns ending in `_proxy` remain local estimates and must not be called
GPT-4o measurements. A separate bounded audit in `results/llm_audit/` invokes
the real GPT-4o Responses API for O-RAN, core-network, compute, and E2E agents,
plus a One-Shot LLM baseline. That audit records returned token usage and
wall-clock latency. It samples replay states and therefore measures planner
telemetry and selected-plan feasibility, not 30-seed network performance.
The observed cloud-API wall time is longer than the application deadlines, so
the audit does not validate synchronous per-event cloud control; such a design
needs asynchronous/precomputed decisions or a lower-latency local model path.

## Literature-grounded settings

The parameter mapping is documented in `REFERENCES.md` and
`config/simulation.yaml`. Core task and MEC parameters come from IEEE
Transactions papers published in 2023 or later:

- B. Li *et al.*, "Delay-Aware Digital Twin Synchronization in Mobile Edge
  Networks With Semantic Communications," *IEEE Transactions on Vehicular
  Technology*, 2025, DOI: 10.1109/TVT.2025.3548844. We use 0.6--0.8 Mbit DT
  updates, 300 cycles/bit, 0.01--0.1 W UE power, and 10 GHz edge CPU.
- L. Zhang *et al.*, "Digital Twin-Assisted Edge Computation Offloading in
  Industrial Internet of Things With NOMA," *IEEE Transactions on Vehicular
  Technology*, 2023, DOI: 10.1109/TVT.2023.3270859. We use its urban/industrial
  NLoS path-loss form, 9 dB receiver noise figure, shadowing, and 24 dBm upper
  power sensitivity setting.
- Y. Liu *et al.*, "Online Computation Offloading for Collaborative
  Space/Aerial-Aided Edge Computing Toward 6G System," *IEEE Transactions on
  Vehicular Technology*, vol. 73, no. 2, 2024, DOI:
  10.1109/TVT.2023.3312676, motivates collaborative space/aerial edge
  offloading and time-varying queues.

SAGIN geometry values not specified by those Transactions papers are taken
from recent IEEE journal studies and are explicitly marked as such: 780 km LEO
altitude, five UAVs at 100 m, 4 GHz air links, and 400 MHz air bandwidth from
Gao *et al.*, *IEEE JSAC*, 2024 (DOI: 10.1109/JSAC.2024.3459073); 150 MHz and
200 Gcycles/s satellite capacity from Zhang *et al.*, *IEEE Access*, 2024
(DOI: 10.1109/ACCESS.2024.3486564).

## Reproduce

```bash
git clone https://github.com/ckim48/agentic_ai_survey.git
cd agentic_ai_survey
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Copy the authorized Seoul V2X NPZ to data/ first.
python3 scripts/inspect_trace.py
python3 scripts/run_experiments.py --config config/simulation.yaml
python3 scripts/summarize_results.py --config config/simulation.yaml
```

Sensitivity study:

```bash
python3 scripts/run_sensitivity.py --config config/simulation.yaml
```

Bounded real-GPT-4o telemetry audit (the key is read only from the process
environment and API requests set `store=false`):

```bash
export OPENAI_API_KEY="..."
python3 scripts/run_llm_audit.py --smoke
python3 scripts/run_llm_audit.py
unset OPENAI_API_KEY
```

The default audit is capped at 220 HTTP request attempts. It sends task and
simulated resource metrics only; terminal IDs, vehicle coordinates, trace
timestamps, and the API key are never included in prompts or output files.

Quick smoke test:

```bash
python3 scripts/run_experiments.py --config config/simulation.yaml \
  --runs 2 --frames 30 --output results/smoke
python3 scripts/summarize_results.py --config config/simulation.yaml \
  --input results/smoke/raw_runs.csv --output results/smoke
```

## Outputs

- `results/full/raw_runs.csv`: one row per scheme/configuration/seed.
- `results/full/summary.csv`: means, SDs, and 95% CIs.
- `results/full/route_shares.csv`: access/compute selection shares.
- A local run creates `results/full/run_manifest.json` with its config, trace
  digest, host, runtime, and planner-mode disclosure. Host-specific manifests
  are intentionally not committed.
- `figures/main_performance.{png,pdf}`: deadline success, p95 latency, DT age,
  and position-error comparisons.
- `figures/agent_overhead.{png,pdf}`: negotiation and surrogate overhead.
- `results/sensitivity/` and `figures/sensitivity.{png,pdf}`: negotiation-round
  and independent-agent reconciliation-delay sensitivity.
- `paper/simulation_setup.tex`: drop-in LaTeX setup text and parameter table.
- `paper/results_autofill.tex`: automatically generated result macros.
- `results/llm_audit/`: public, sanitized GPT-4o decision summaries, agentic
  metrics, examples, and validation reports. Raw API correlation logs are not
  committed.
- `paper/llm_audit_table.tex`: auto-generated bounded-audit overhead table.
- `results/llm_audit/four_scheme_{events,summary}.csv`: AAI-CDOS, Single
  Domain, Independent Agents, and One-Shot LLM decisions evaluated on the same
  sampled pre-decision states; no extra API calls are used.
- `figures/llm_audit_four_scheme.{png,pdf}` and
  `paper/llm_audit_four_scheme_{table,results}.tex`: complete four-scheme audit
  view and an explicit disclosure when AAI-CDOS and One-Shot select identically.
- `results/llm_audit/conversation_trace_public.jsonl`: the public reconstructed
  structured input, domain-agent proposal, E2E negotiation, verifier feedback,
  execution, and memory trace with API correlation IDs removed.
- `figures/agentic_interaction_metrics.{png,pdf}` and
  `paper/agentic_{metrics_table,results}.tex`: agent agreement, negotiation,
  token, and per-role API-latency evidence.
- `results/dt_analysis/`, `figures/dt_outcomes_four_scheme.{png,pdf}`, and
  `figures/dt_deadline_by_class.{png,pdf}`: dedicated policy-level DT age,
  position error, synchronization latency, and deadline results over 30 seeds.
- `paper/aai_cdos_results_bundle.tex` and `paper/RESULTS_GUIDE.md`: drop-in
  result includes plus explicit guidance for keeping real-agent telemetry and
  surrogate policy-level DT outcomes scientifically separate.

## Interpretation guardrails

This is a trace-driven network simulation, not a packet-level 3GPP/NS-3 study.
The real Seoul feed supplies mobility only, and satellite visibility is not
derived from orbital ephemerides. Do not describe radio samples as measured
Seoul V2X channel data. Do not describe `_proxy` columns as GPT-4o telemetry.
Do not use the bounded LLM audit as a replacement for the full 30-seed
network-performance comparison.
