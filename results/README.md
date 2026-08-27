# Result artifacts

- `full/`: 30-seed, four-scheme raw run summaries and route shares.
- `sensitivity/`: negotiation-round and independent-agent mismatch sensitivity.
- `dt_analysis/`: dedicated DT synchronization outcomes and AAI improvement
  calculations.
- `llm_audit/`: sanitized real-GPT-4o audit summaries, four-scheme same-state
  comparison, agent-process metrics, examples, and public interaction traces.
- `actual_agentic_v2/`: final fair same-state experiment in which AAI-CDOS, Independent
  Agents, One-Shot LLM, and Single Domain all execute their configured real
  model planners over the same valid joint-plan catalog. AAI-CDOS alone uses
  iterative domain-agent negotiation, deterministic verifier feedback, and
  bounded memory.

Host-specific manifests, raw API correlation logs, connection-test records,
and private mobility data are excluded from this public repository.
