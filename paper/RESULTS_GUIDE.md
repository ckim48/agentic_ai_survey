# Results integration guide

The evidence has two layers and should not be merged into one latency claim.

## Policy-level DT outcomes

Use `dt_results.tex` and `dt_outcomes_table.tex` for the 30-seed Seoul V2X
trace-driven comparison of AAI-CDOS, Single Domain, Independent Agents, and
One-Shot LLM. The associated figures are `dt_outcomes_four_scheme` and
`dt_deadline_by_class`. These results contain DT update success, network E2E
latency, DT information age, and position error. The planners in this layer are
deterministic surrogates, and cloud GPT-4o wall time is not part of network E2E
latency.

## Real-GPT-4o agent-process evidence

Use `agentic_results.tex`, `agentic_metrics_table.tex`,
`llm_audit_results.tex`, and `llm_audit_table.tex` for actual API tokens,
per-role latency, domain proposal agreement, negotiation rounds, verifier
rejections, and examples of intermediate agent exchanges. The associated
figure is `agentic_interaction_metrics`.

`results/llm_audit/conversation_trace.jsonl` contains one sanitized structured
trace per audited event: observation, O-RAN/CN/compute inputs and proposals,
each E2E coordination round, verifier feedback, execution, and memory update.
`conversation_examples.md` contains one one-round and one three-round example.

The 30-event audit found identical AAI-CDOS and One-Shot access/compute plans in
30/30 events. Do not claim a measured decision-quality gain over One-Shot from
this audit. It does show the internal multi-agent process and its overhead.

## Convenience include

`aai_cdos_results_bundle.tex` includes all generated result paragraphs and
tables. Remove any table already included elsewhere in the manuscript to avoid
duplicate labels.
