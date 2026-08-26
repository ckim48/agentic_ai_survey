# Real GPT-4o planner audit

This is a bounded, class-stratified telemetry audit over states from the
Seoul V2X replay. It does not replace the 30-seed deterministic network
performance experiment. No coordinates or terminal identifiers were sent.

Model requested: `gpt-4o`; Responses API `store=false`.

| Vehicles | Events | API calls | AAI accept | One-shot feasible | AAI wall/event | Est. cost |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 50 | 100.0% | 100.0% | 3596 ms | $0.0725 |
| 25 | 10 | 50 | 100.0% | 100.0% | 3522 ms | $0.0724 |
| 50 | 10 | 56 | 70.0% | 70.0% | 3350 ms | $0.0859 |

## Four-scheme comparison on identical sampled states

| Vehicles | AAI-CDOS | Single Domain | Independent Agents | One-Shot LLM |
|---:|---:|---:|---:|---:|
| 10 | 100% | 50% | 90% | 100% |
| 25 | 100% | 60% | 100% | 100% |
| 50 | 70% | 40% | 60% | 70% |

These are selected-plan deadline-feasibility rates on a common AAI
pre-decision state, not separate policy-level trajectories. AAI-CDOS and
One-Shot LLM selected the same access/compute plan in all 30 sampled events;
this audit therefore does not establish a decision-quality advantage of
multi-agent negotiation over One-Shot GPT-4o. The deterministic Single Domain
and Independent Agents baselines made no API calls.

Completed calls: 156; HTTP request attempts (including retries): 156.

Cost is estimated from returned token usage and the prices recorded in
`config/llm_audit.yaml`; billing records remain authoritative.

The measured cloud-API planner wall time is longer than the 0.15--1.0 s
application deadlines. These measurements therefore do not support a
synchronous per-event cloud-control claim; deployment would require
asynchronous/precomputed decisions or a substantially lower-latency serving
path/model.
