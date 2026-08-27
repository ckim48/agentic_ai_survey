# Actual-model figure evidence

- Source experiment: `results/actual_agentic_v2/`
- Returned model: `gpt-4o-2024-08-06`
- Comparison: 4 independent seeds, 6 class-balanced events per seed and load,
  72 common simulator states, 288 scheme evaluations, and 744 model calls.
- Mobility: authorized Seoul V2X trace.
- Simulated: radio, route, queue, compute, and DT task states.
- DT-update latency excludes cloud planner wall time; planner overhead remains
  available in the experiment summary.
- Error bars: two-sided 95% Student-t confidence intervals over seed-level
  statistics.
- The representative panel uses event `n25_s2028_e4`. Agent decisions are actual
  structured model outputs; prose labels are compact paraphrases.
