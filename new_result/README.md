# Actual-agentic DT case-study figure package

This folder contains the publication assets for the vehicular digital-twin
(DT) case study. The recommended assets are generated from the final fair
actual-model experiment in `results/actual_agentic_v2/`.

## Recommended current assets

- `charts/actual_agentic_dt_performance_4panel.{png,pdf,svg}`: four-panel
  publication figure.
- `charts/chart_a_deadline_success_pct.*`: deadline satisfaction.
- `charts/chart_b_mean_latency_ms.*`: mean DT-update latency.
- `charts/chart_c_mean_staleness_ms.*`: mean DT staleness.
- `charts/chart_d_p95_position_error_m.*`: P95 position error.
- `prompts/actual_partial_output_panel.txt`: compact actual two-round agent
  trace for the center panel.
- `prompts/actual_figure_caption.tex`: evidence-complete LaTeX caption.
- `prompts/actual_results_paragraph.tex`: ready-to-edit result paragraph.
- `data/representative_actual_agent_trace.json`: structured evidence behind
  the center-panel copy.
- `data/actual_agentic_chart_values.csv`: plotted means and confidence
  intervals.
- `scripts/generate_actual_agentic_assets.py`: reproducible generator.

The final experiment contains 72 common simulator states, 288 scheme
evaluations, and 744 actual model calls. Error bars are two-sided 95% Student-t
confidence intervals over four independent seeds. Mobility comes from the
authorized Seoul V2X trace; radio, routing, queue, compute, and task states are
simulated. Cloud-planner wall time is measured separately and excluded from
DT-update latency.

Regenerate the current assets from the repository root:

```bash
python new_result/scripts/generate_actual_agentic_assets.py \
  --results results/actual_agentic_v2 --output new_result
```

## Legacy surrogate package

The files named `dt_infographic_*`, together with `prompts/panel_copy.md`,
`prompts/figure_caption.tex`, and `prompts/evidence_boundary.md`, belong to an
earlier mixed-evidence package: a bounded GPT audit plus separate 30-seed
deterministic planner-surrogate curves. They are retained only for comparison
and must not replace the actual-model assets above.

### Legacy figure structure

1. **Case study:** dense urban vehicular DT synchronization over terrestrial,
   UAV, and LEO resources.
2. **AAI-CDOS:** a user service intent and the closed-loop
   observe--plan--tool--verify--coordinate--execute--feedback workflow.
3. **Partial output:** one sanitized, actually audited GPT-4o decision example
   (`n10_s2026_e1`).
4. **Final decision:** the accepted joint plan from that audited event.
5. **Aggregate results:** two compact plots from the separate 30-seed
   planner-surrogate policy evaluation.

The audited GPT-4o example and the aggregate simulation charts must remain
visually separated. The single audited decision did not generate the 30-seed
network-performance curves.

### Legacy contents

- `reference/fl_case_study_reference.png`: user-supplied FL layout reference.
- `reference/dt_chart_style_reference.png`: user-supplied IEEE line-chart style reference.
- `reference/dt_bar_chart_style_reference.png`: user-supplied grouped-bar style reference.
- `charts/dt_infographic_results.{png,pdf}`: line-and-marker result variant.
- `charts/dt_infographic_results_bar.{png,pdf}`: recommended compact grouped-bar variant.
- `data/dt_infographic_metrics.csv`: exact plotted means and 95% confidence intervals.
- `prompts/dt_figure_edit_prompt.md`: image-editing/illustration prompt.
- `prompts/panel_copy.md`: ready-to-paste English panel text grounded in the audit.
- `prompts/figure_caption.tex`: proposed VTM-style LaTeX caption.
- `prompts/evidence_boundary.md`: claims that are and are not supported.
- `scripts/generate_dt_infographic_charts.py`: reproducible chart generator.
- `requirements.txt`: Python packages used by the chart generator.

### Regenerate the legacy charts

From the repository root:

```bash
pip install -r new_result/requirements.txt
python new_result/scripts/generate_dt_infographic_charts.py
```

The chart source is `results/dt_analysis/dt_outcomes_summary.csv`.
PNG export also requires `pdftocairo` (Poppler) on `PATH`; the PDF output is
generated directly by ReportLab.

### Suggested legacy use

Use this composite as the case-study overview figure. Retain a separate,
larger result figure in the simulation subsection if the two compact charts
become too small at final print size. IEEE VTM counts each visible subfigure
and each table toward the manuscript's combined figure/table limit.
