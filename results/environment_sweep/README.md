# Environment robustness sweep

This controlled experiment fixes the load at 25 vehicles and replays the same 30 seeds and 180 Seoul mobility snapshots per condition.

The condition labels correspond to explicit parameter overrides in `config/environment_sweep.yaml`; they are not post-hoc relabeling.

The Seoul trace supplies mobility only. Radio, routing, queueing, computing, DT tasks, and policy outcomes are simulated.

## Conditions

- **Nominal:** Baseline radio, routing, and computing configuration.
- **Radio degraded:** Lower access bandwidth, higher receiver noise, and excess path loss.
- **Edge congested:** Reduced processing rates model background edge-compute load.
- **UAV blockage:** The aerial radio-access link is unavailable; ground and LEO remain usable.
