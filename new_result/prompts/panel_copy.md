# Ready-to-paste DT panel copy

All numeric values below come from the sanitized audited event
`n10_s2026_e1` in `results/llm_audit/conversation_trace_public.jsonl`.

## Case Study

**Real-Time Vehicular Digital-Twin Synchronization**

Connected vehicles periodically transmit mobility and operational states to
their digital twins through terrestrial, UAV-assisted, and LEO-supported
communication and computing resources.

## User Service Intent

> Maintain a routine vehicular DT update within a hard 1000-ms synchronization
> deadline. Coordinate radio access, E2E routing, and DT computing placement
> using the current domain observations. Reject any joint plan that violates
> the deadline.

Structured intent:

```text
task_class: routine
update_size: 0.661 Mbit
hard_deadline: 1000 ms
objective: timely and consistent vehicle-to-twin synchronization
```

## Partial Output

```text
## E2E Service Domain
[Perception]
Task: routine DT update, 0.661 Mbit
Hard deadline: 1000 ms

[Reasoning]
Compare joint radio-route-compute candidates.
Reject a plan if its predicted E2E latency exceeds the deadline.

## O-RAN Domain
[Perception]
Ground: 23.57 Mbit/s, SNR 10.55 dB
UAV:    15.93 Mbit/s, SNR -10.52 dB
LEO:     3.84 Mbit/s, SNR  -5.04 dB

[Proposal]
access: ground
rationale: lowest_latency

## Core-Network Domain
[Perception]
Candidate p0 path delay: 3.10 ms
Candidate p0 predicted E2E latency: 139.2 ms

[Proposal]
plan_id: p0
rationale: lowest_latency

## Computing Domain
[Perception]
ground1 queue: 88.2 ms
ground1 service: 19.8 ms

[Proposal]
compute: ground1
rationale: queue_avoidance

## E2E Verifier
selected_plan: p0
predicted_latency: 139.2 ms
hard_deadline: 1000 ms
verdict: accepted
```

## Final Decision

```text
access: ground
route: p0
DT compute: ground1
predicted E2E synchronization latency: 139.2 ms
deadline met: true
coordination rounds: 1
verifier rejections: 0
```

## Bottom Takeaway

At 50 vehicles, AAI-CDOS attains **89.2% deadline satisfaction** with a
**27.2-m P95 DT position error** in the 30-seed policy-level simulation.

The single decision above is an actual bounded GPT-4o audit example. The
aggregate curves are a separate deterministic planner-surrogate evaluation.
