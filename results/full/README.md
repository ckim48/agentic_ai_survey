# Full experiment summary

Thirty independent seeds, 180 nonempty Seoul-Gangnam V2X snapshots per run,
and 95% confidence intervals. Planner overhead columns are local proxies, not GPT-4o telemetry.

## 10 vehicles

- AAI-CDOS deadline satisfaction: 99.17% +/- 0.11 percentage points (95% CI).
- Gain over the strongest baseline: 12.75 percentage points.
- P95 latency: 192.8 ms (40.2% below the lowest-latency baseline).
- Mean DT age: 0.193 s (89.6% below the best baseline).

## 25 vehicles

- AAI-CDOS deadline satisfaction: 97.28% +/- 0.37 percentage points (95% CI).
- Gain over the strongest baseline: 23.56 percentage points.
- P95 latency: 306.4 ms (28.1% below the lowest-latency baseline).
- Mean DT age: 0.468 s (89.1% below the best baseline).

## 50 vehicles

- AAI-CDOS deadline satisfaction: 89.22% +/- 1.11 percentage points (95% CI).
- Gain over the strongest baseline: 32.00 percentage points.
- P95 latency: 470.9 ms (18.1% below the lowest-latency baseline).
- Mean DT age: 1.692 s (82.5% below the best baseline).

## Guardrail

The Seoul source provides mobility only. Radio/network/compute states are simulated,
and the 80-ms independent-agent reconciliation delay is an explicit baseline assumption.
Run sensitivity analysis before using causal or deployment claims.
