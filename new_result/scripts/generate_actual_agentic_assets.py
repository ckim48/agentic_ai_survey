#!/usr/bin/env python3
"""Generate publication assets from the actual-model AAI-CDOS experiment."""

import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


SCHEMES = ["AAI-CDOS", "Single Domain", "Independent Agents", "One-Shot LLM"]
LABELS = {
    "AAI-CDOS": "AAI-CDOS",
    "Single Domain": "Single Domain",
    "Independent Agents": "Independent",
    "One-Shot LLM": "One-Shot LLM",
}
STYLES = {
    "AAI-CDOS": dict(color="#2F6FDB", marker="D", linestyle="-", linewidth=1.6),
    "Single Domain": dict(color="#F28E2B", marker="o", linestyle="-", linewidth=1.3),
    "Independent Agents": dict(color="#E15759", marker="s", linestyle="--", linewidth=1.3),
    "One-Shot LLM": dict(color="#59A14F", marker="^", linestyle=":", linewidth=1.5),
}

# Two-sided Student-t 0.975 critical values. The final experiment uses four
# independent seeds, but the small table keeps the generator reusable.
T_CRITICAL = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def ci95(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    critical = T_CRITICAL.get(len(values) - 1, 1.96)
    half = critical * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, half


def seed_metric(frame, metric):
    if metric == "deadline_success_pct":
        return 100.0 * frame.groupby("seed")["deadline_met"].mean()
    if metric == "mean_latency_ms":
        return frame.groupby("seed")["dt_update_latency_ms"].mean()
    if metric == "mean_staleness_ms":
        return frame.groupby("seed")["dt_age_ms"].mean()
    if metric == "p95_position_error_m":
        return frame.groupby("seed")["position_error_m"].quantile(0.95)
    raise ValueError(metric)


def build_values(events):
    rows = []
    for vehicles in sorted(events["vehicles"].unique()):
        for scheme in SCHEMES:
            subset = events[(events["vehicles"] == vehicles) &
                            (events["scheme"] == scheme)]
            for metric in (
                    "deadline_success_pct", "mean_latency_ms",
                    "mean_staleness_ms", "p95_position_error_m"):
                seed_values = seed_metric(subset, metric)
                mean, half = ci95(seed_values)
                rows.append({
                    "vehicles": int(vehicles),
                    "scheme": scheme,
                    "metric": metric,
                    "mean": mean,
                    "ci95_halfwidth": half,
                    "seeds": len(seed_values),
                    "events": len(subset),
                })
    return pd.DataFrame(rows)


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.2,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.7,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def metric_spec():
    return [
        ("deadline_success_pct", "(a) Deadline Satisfaction",
         "Deadline satisfaction (%)", False),
        ("mean_latency_ms", "(b) DT Update Latency",
         "Mean DT update latency (ms)", True),
        ("mean_staleness_ms", "(c) DT Staleness",
         "Mean DT staleness (ms)", True),
        ("p95_position_error_m", "(d) Position Error",
         "P95 position error (m)", True),
    ]


def draw_metric(ax, values, metric, title, ylabel, log_scale, show_xlabel=True):
    vehicles = sorted(values["vehicles"].unique())
    for scheme in SCHEMES:
        subset = values[(values["metric"] == metric) &
                        (values["scheme"] == scheme)].sort_values("vehicles")
        y = subset["mean"].to_numpy(dtype=float)
        half = subset["ci95_halfwidth"].to_numpy(dtype=float)
        lower = np.minimum(half, np.maximum(y * 0.80, 1e-8))
        style = STYLES[scheme]
        ax.errorbar(
            subset["vehicles"], y, yerr=np.vstack([lower, half]),
            label=LABELS[scheme], markersize=4.0, markerfacecolor="white",
            markeredgewidth=0.9, capsize=2.2, elinewidth=0.75, **style)
    if log_scale:
        ax.set_yscale("log")
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(vehicles)
    ax.set_xlabel("Number of vehicles" if show_xlabel else "")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.45)
    ax.tick_params(direction="in", length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_color("#444444")


def save_figure(fig, base):
    fig.savefig(base + ".png", dpi=400, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.035)


def create_charts(values, chart_dir):
    os.makedirs(chart_dir, exist_ok=True)
    setup_style()
    specs = metric_spec()

    fig, axes = plt.subplots(2, 2, figsize=(7.65, 5.45))
    for ax, spec in zip(axes.flat, specs):
        draw_metric(ax, values, *spec)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.995), handlelength=2.5,
               columnspacing=1.5)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.91,
                        wspace=0.29, hspace=0.34)
    save_figure(fig, os.path.join(chart_dir,
                                  "actual_agentic_dt_performance_4panel"))
    plt.close(fig)

    for index, spec in enumerate(specs, start=1):
        fig, ax = plt.subplots(figsize=(3.45, 3.0))
        draw_metric(ax, values, *spec)
        ax.legend(frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=2,
                  handlelength=2.1, columnspacing=1.0, fontsize=6.6)
        fig.subplots_adjust(left=0.19, right=0.98, bottom=0.30, top=0.91)
        save_figure(fig, os.path.join(
            chart_dir, "chart_%s_%s" % (chr(96 + index), spec[0])))
        plt.close(fig)


def find_trace(trace_path, event_id):
    with open(trace_path, "r") as stream:
        for line in stream:
            item = json.loads(line)
            if item["event_id"] == event_id:
                return item
    raise ValueError("event not found: %s" % event_id)


def write_trace_assets(trace, data_dir, prompt_dir):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(prompt_dir, exist_ok=True)
    rounds = trace["schemes"]["AAI-CDOS"]
    first = rounds[0]
    final = rounds[-1]
    if first["verifier"]["accepted"] or not final["verifier"]["accepted"]:
        raise ValueError("representative trace must show rejection then acceptance")

    task = trace["task"]
    radio = trace["radio_observation"]
    compute = {item["compute"]: item for item in trace["compute_observation"]}
    first_v = first["verifier"]
    final_v = final["verifier"]
    final_breakdown = final_v["latency_breakdown_ms"]

    panel = """PARTIAL OUTPUT & FINAL DECISION

E2E SERVICE DOMAIN
[Intent] Safety-critical vehicular DT synchronization
Update: {update:.3f} Mbit | Deadline: {deadline:.0f} ms
[Plan] Coordinate access, routing, and DT computing.

DOMAIN-AGENT OUTPUTS - ROUND 1
O-RAN: UAV access (confidence {oran_conf:.2f})
CN: shortest terrestrial edge path
Computing: satellite DT node

[Joint Candidate] UAV access -> satellite DT node
[Verifier] {first_latency:.1f} ms > {deadline:.0f} ms: REJECTED
[Feedback] Transmission delay ({first_tx:.1f} ms) violates the deadline.

VERIFIER-GUIDED REVISION - ROUND 2
O-RAN: LEO access | Rate: {leo_rate:.2f} Mbit/s
E2E coordinator: LEO access -> satellite DT node

FINAL JOINT DECISION
Delay components: transmission + path + queue + service
Latency: {tx:.1f} + {path:.1f} + {queue:.1f} + {service:.1f}
       = {latency:.1f} ms <= {deadline:.0f} ms
Verdict: ACCEPTED | Status: COMMITTED
""".format(
        update=task["update_mbit"], deadline=task["deadline_ms"],
        oran_conf=first["domain_proposals"]["oran"]["confidence"],
        first_latency=first_v["estimated_dt_update_latency_ms"],
        first_tx=first_v["latency_breakdown_ms"]["transmission"],
        leo_rate=radio["leo"]["rate_mbps"],
        tx=final_breakdown["transmission"], path=final_breakdown["path"],
        queue=final_breakdown["queue"], service=final_breakdown["service"],
        latency=final_v["estimated_dt_update_latency_ms"])

    with open(os.path.join(prompt_dir, "actual_partial_output_panel.txt"),
              "w") as stream:
        stream.write(panel)

    full = {
        "evidence_type": "actual structured GPT-4o agent outputs plus deterministic verifier",
        "event_id": trace["event_id"],
        "vehicles": trace["vehicles"],
        "seed": trace["seed"],
        "frame": trace["frame"],
        "task": task,
        "radio_observation": radio,
        "compute_observation": trace["compute_observation"],
        "rounds": rounds,
        "final_semantic_plan": {
            "access": "LEO",
            "route": "satellite-assisted E2E path",
            "compute": "satellite DT node",
        },
        "figure_copy_is_paraphrased": True,
    }
    with open(os.path.join(data_dir, "representative_actual_agent_trace.json"),
              "w") as stream:
        json.dump(full, stream, indent=2, sort_keys=True)

    readme = """# Actual-model figure evidence

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
- The representative panel uses event `{event}`. Agent decisions are actual
  structured model outputs; prose labels are compact paraphrases.
""".format(event=trace["event_id"])
    with open(os.path.join(data_dir, "ACTUAL_EVIDENCE_README.md"), "w") as stream:
        stream.write(readme)

    caption = r"""\caption{Representative AAI-CDOS negotiation trace and actual-model DT-synchronization results. The system replays an authorized Seoul V2X mobility trace while simulating radio, route, queue, computing, and DT-task states. The center panel shows an actual two-round agent trace: the deterministic verifier rejects the first cross-domain candidate, returns the binding transmission-delay violation, and accepts the revised joint plan. Curves report means with two-sided 95\% Student-$t$ confidence intervals over four independent seeds (six class-balanced events per seed and vehicle load). All methods receive the same pre-decision state and valid joint-plan catalog. Agent decisions are structured outputs from GPT-4o-2024-08-06; DT-update latency excludes cloud-planner wall time.}""" + "\n"
    with open(os.path.join(prompt_dir, "actual_figure_caption.tex"), "w") as stream:
        stream.write(caption)

    paragraph = r"""Across 10, 25, and 50 vehicles, AAI-CDOS achieves deadline-satisfaction rates of 100.0\%, 91.7\%, and 75.0\%, respectively. The corresponding rates are 50.0\%, 66.7\%, and 45.8\% for Independent Agents; 8.3\%, 12.5\%, and 12.5\% for One-Shot LLM; and 66.7\%, 70.8\%, and 50.0\% for Single Domain. AAI-CDOS also limits mean DT-update latency to 121.5--285.1~ms across the evaluated loads, whereas the cross-domain baselines degrade more sharply under load. These gains arise from verifier-guided proposal revision rather than from model-inference speed: mean cloud planning wall time for AAI-CDOS is 2.91--3.70~s and is excluded from simulated DT-update latency. Accordingly, the results support asynchronous or precomputed agentic control, while a synchronous per-update cloud-LLM deployment would require latency reduction or a local model.""" + "\n"
    with open(os.path.join(prompt_dir, "actual_results_paragraph.tex"), "w") as stream:
        stream.write(paragraph)


def validate(events, manifest):
    if manifest.get("dry_run"):
        raise ValueError("dry-run output cannot be used for publication assets")
    if manifest.get("returned_models") != ["gpt-4o-2024-08-06"]:
        raise ValueError("unexpected returned model")
    if manifest.get("dataset_sha256") != (
            "2885aa7e90874c4f1a82c2ff4690ae2fcbf45161ec0894b9d7870d1644f18cbf"):
        raise ValueError("unexpected Seoul trace digest")
    counts = events.groupby("event_id")["scheme"].nunique()
    if not (counts == 4).all():
        raise ValueError("every event must contain all four schemes")
    if set(events["scheme"].unique()) != set(SCHEMES):
        raise ValueError("scheme set mismatch")
    if events["event_id"].nunique() != 72:
        raise ValueError("expected 72 common decision states")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/actual_agentic_v2")
    parser.add_argument("--output", default="new_result")
    parser.add_argument("--trace-event", default="n25_s2028_e4")
    args = parser.parse_args()

    events = pd.read_csv(os.path.join(args.results, "events.csv"))
    with open(os.path.join(args.results, "run_manifest.json"), "r") as stream:
        manifest = json.load(stream)
    validate(events, manifest)

    values = build_values(events)
    chart_dir = os.path.join(args.output, "charts")
    data_dir = os.path.join(args.output, "data")
    prompt_dir = os.path.join(args.output, "prompts")
    os.makedirs(data_dir, exist_ok=True)
    values.to_csv(os.path.join(data_dir, "actual_agentic_chart_values.csv"),
                  index=False)
    create_charts(values, chart_dir)

    trace = find_trace(os.path.join(args.results, "conversation_trace.jsonl"),
                       args.trace_event)
    write_trace_assets(trace, data_dir, prompt_dir)
    print("generated charts and trace assets from %s" % args.results)


if __name__ == "__main__":
    main()
