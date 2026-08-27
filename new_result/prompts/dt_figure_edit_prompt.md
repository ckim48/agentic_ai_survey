# Prompt for converting the FL figure to a vehicular-DT figure

Use the supplied FL/wildfire figure only as a layout reference. Preserve its
left-to-right explanatory flow and clean IEEE-magazine infographic style, but
replace every wildfire, federated-learning, IoT-sensor, FL-model, Dirichlet
alpha, accuracy, and UAV-selection element with the vehicular digital-twin
content below. Do not retain any FL terminology or wildfire imagery.

## Overall layout

Create a wide, high-resolution vector-like scientific infographic with five
clearly separated regions:

1. **Case Study -- Real-Time Vehicular DT Synchronization.** Show connected
   vehicles moving through a dense urban road network. Show terrestrial gNB/RSU
   and edge servers as the primary infrastructure, with UAV and LEO satellite
   communication/compute as complementary resources. Pair each physical vehicle
   with a small translucent digital-twin representation. Use arrows to show
   periodic state updates carrying position, velocity, heading, and status.

2. **User Service Intent and AAI-CDOS.** Replace the emergency-FL user request
   with a DT synchronization intent: application class, hard deadline, update
   size, priority, and maximum position error. Show the E2E, O-RAN, core-network,
   and computing agents. Show a closed loop labeled Observe, Plan, Tool,
   Verify, Coordinate, Execute, Feedback. Keep fast scheduling/link adaptation
   inside the domain controllers.

3. **Structured Partial Output.** Use concise monospaced text, not free-form
   chat. Include the audited event `n10_s2026_e1`: routine DT update, 1000-ms
   deadline, 0.661-Mbit update; O-RAN proposes ground access; CN proposes plan
   p0; compute proposes ground1; the E2E verifier accepts p0 at 139.2 ms.
   Use the exact wording from `panel_copy.md` and do not invent coordinates,
   terminal identifiers, or extra measurements.

4. **Audited GPT-4o Final Decision.** Show a small decision box labeled
   `Audited GPT-4o decision example`, with final plan p0, ground access,
   ground1 compute, estimated E2E latency 139.2 ms, hard deadline 1000 ms,
   verifier accepted. Add a small memory/feedback arrow returning the outcome
   to the agents.

5. **30-Seed Policy-Level Simulation Results.** Insert
   `charts/dt_infographic_results_bar.png` without redrawing or changing its
   data.
   Place a visible divider between the audited decision example and the
   aggregate charts. Label the chart block `Planner-surrogate policy evaluation
   (30 seeds)` so it cannot be mistaken for the latency of the GPT-4o call.

## Visual requirements

- Professional IEEE Vehicular Technology Magazine visual style.
- White background, restrained pale-blue and pale-orange section fills.
- Match the compact plot language in `reference/dt_bar_chart_style_reference.png`:
  Times-family serif labels, grouped bars with distinct print-safe hatch
  patterns, a single horizontal legend, thin error bars, and faint dashed grids.
- Colorblind-safe blue/orange/red/green chart colors; do not recolor the
  supplied result chart.
- Minimum final-print text size of 8 pt and no decorative dot-grid background.
- Use simple vector-like vehicles, roads, gNB/RSU, edge server, UAV, satellite,
  and digital-twin icons. Do not use photorealistic imagery.
- Keep the final chart axes, legends, units, error bars, and surrogate disclosure
  legible at two-column width.
- Do not claim that GPT-4o itself produced the aggregate deadline or position-
  error improvement. Do not claim decision-quality superiority over One-Shot
  from the 30-event real-model audit.

## Bottom takeaway

Use this factual sentence:

`At 50 vehicles, AAI-CDOS attains 89.2% deadline satisfaction with a 27.2-m P95
DT position error in the 30-seed policy-level simulation.`

Optionally add, in smaller text:

`The bounded GPT-4o audit demonstrates the multi-agent proposal, verifier, and
feedback process; it is reported separately from network E2E performance.`
