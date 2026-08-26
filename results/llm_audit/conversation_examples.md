# Sanitized agent-interaction examples

These are deterministic reconstructions of the structured inputs sent
during the completed GPT-4o audit plus the saved structured outputs.
They are not free-form chat transcripts and contain no coordinates or
terminal identifiers.

## Accepted in one round: `n10_s2026_e1`

Task: `routine`, deadline 1000.0 ms, 0.661 Mbit; feasible candidates 8/8.

Domain proposals:

- `oran`: `{"access": "ground", "confidence": 0.95, "rationale_code": "lowest_latency"}`
- `cn`: `{"confidence": 0.95, "plan_id": "p0", "rationale_code": "lowest_latency"}`
- `compute`: `{"compute": "ground1", "confidence": 0.95, "rationale_code": "queue_avoidance"}`

Coordination and verifier:

- `e2e_round_1` chose `p0`; verifier `accepted` (139.2 ms vs 1000.0 ms).

Final execution: `ground -> ground1`, 139.2 ms, deadline met: `True`, fallback: `False`.

## Verifier-driven multi-round case: `n50_s2026_e1`

Task: `cooperative`, deadline 350.0 ms, 0.686 Mbit; feasible candidates 0/8.

Domain proposals:

- `oran`: `{"access": "leo", "confidence": 0.95, "rationale_code": "deadline_margin"}`
- `cn`: `{"confidence": 0.9, "plan_id": "p7", "rationale_code": "lowest_latency"}`
- `compute`: `{"compute": "ground0", "confidence": 0.95, "rationale_code": "queue_avoidance"}`

Coordination and verifier:

- `e2e_round_1` chose `p7`; verifier `rejected` (367.8 ms vs 350.0 ms).
- `e2e_round_2` chose `p3`; verifier `rejected` (860.1 ms vs 350.0 ms).
- `e2e_round_3` chose `p0`; verifier `rejected` (21980.9 ms vs 350.0 ms).

Final execution: `leo -> sat0`, 367.8 ms, deadline met: `False`, fallback: `True`.
