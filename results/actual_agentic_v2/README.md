# Actual-model agentic experiment

Every compared planner in this directory used the returned model shown
in `run_manifest.json`. All schemes were evaluated on the same sampled
pre-decision simulator state. Seoul data supplies mobility only; radio,
routing, queue, compute, and task states are simulated.

| Vehicles | Scheme | Events | Deadline success | DT latency | Planner wall | Calls | Cost |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | AAI-CDOS | 24 | 100.0% | 121.5 ms | 3074.7 ms | 120 | $0.1867 |
| 10 | Independent Agents | 24 | 50.0% | 2231.1 ms | 1652.5 ms | 72 | $0.0813 |
| 10 | One-Shot LLM | 24 | 8.3% | 3666.8 ms | 1043.0 ms | 24 | $0.0395 |
| 10 | Single Domain | 24 | 66.7% | 392.7 ms | 1076.2 ms | 24 | $0.0207 |
| 25 | AAI-CDOS | 24 | 91.7% | 229.2 ms | 2907.4 ms | 116 | $0.1854 |
| 25 | Independent Agents | 24 | 66.7% | 3584.7 ms | 1181.5 ms | 72 | $0.0813 |
| 25 | One-Shot LLM | 24 | 12.5% | 7763.3 ms | 1152.7 ms | 24 | $0.0395 |
| 25 | Single Domain | 24 | 70.8% | 446.7 ms | 931.0 ms | 24 | $0.0207 |
| 50 | AAI-CDOS | 24 | 75.0% | 285.1 ms | 3697.0 ms | 148 | $0.2488 |
| 50 | Independent Agents | 24 | 45.8% | 8080.7 ms | 1304.2 ms | 72 | $0.0812 |
| 50 | One-Shot LLM | 24 | 12.5% | 14876.9 ms | 1220.8 ms | 24 | $0.0395 |
| 50 | Single Domain | 24 | 50.0% | 515.0 ms | 1147.9 ms | 24 | $0.0206 |

Important evidence boundary:

- `dt_update_latency_ms` excludes cloud planner wall time.
- Planner wall time is measured separately and must be disclosed.
- The experiment is a same-state counterfactual decision benchmark, not
  a packet-level physical testbed deployment.
- Natural-language figure copy may paraphrase structured decisions, but
  must not be presented as verbatim model output.

Completed events: 72; completed model calls: 744; request attempts: 744.
