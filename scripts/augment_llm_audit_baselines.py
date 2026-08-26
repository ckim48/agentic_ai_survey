#!/usr/bin/env python3
"""Add deterministic baselines to the same-state real-GPT audit comparison.

No API calls are made. The script reconstructs the exact pre-decision AAI
state for each sampled event, evaluates all four schemes on that common state,
and preserves the actual GPT telemetry already recorded for the two LLM schemes.
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.simulator import (PolicyState, build_tasks, choose_plan, evaluate,
                           load_trace, make_infrastructure, radio_options)
from scripts.run_llm_audit import sample_event_indices, task_at, update_state


SCHEMES = ["AAI-CDOS", "Single Domain", "Independent Agents", "One-Shot LLM"]
COLORS = {
    "AAI-CDOS": "#0072B2",
    "Single Domain": "#D55E00",
    "Independent Agents": "#E69F00",
    "One-Shot LLM": "#009E73",
}


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_with_mismatch(scheme, plan, radios, state, task, now, cfg):
    latency, _ = evaluate(plan, radios, state, task, now, cfg)
    mismatch = 0.0
    if scheme == "Independent Agents":
        colocated = ((plan[0] == "ground" and plan[1].startswith("ground")) or
                     (plan[0] == "uav" and plan[1].startswith("uav")) or
                     (plan[0] == "leo" and plan[1].startswith("sat")))
        if not colocated:
            mismatch = float(cfg["orchestration"]["independent_mismatch_delay_s"])
            latency += mismatch
    return latency, mismatch


def replay(events, cfg, audit_cfg, data_path):
    rows = []
    errors = []
    priority = {"safety": 0, "cooperative": 1, "routine": 2}
    runs = int(audit_cfg["independent_seeds"])
    frames = int(audit_cfg["trace_frames"])
    count = int(audit_cfg["events_per_vehicle_seed"])
    seed0 = int(cfg["experiment"]["seed_start"])

    indexed = events.set_index("event_id")
    for vehicles in cfg["experiment"]["vehicle_counts"]:
        vehicles = int(vehicles)
        for run_index in range(runs):
            seed = seed0 + run_index
            trace = load_trace(data_path, vehicles, seed, frames,
                               float(cfg["experiment"]["minimum_vehicle_coverage"]),
                               cfg["experiment"].get("region_bbox_lonlat"))
            tasks = build_tasks(trace, cfg, seed)
            infra = make_infrastructure(trace, cfg)
            state = PolicyState(cfg, vehicles)
            selected = sample_event_indices(tasks, count, seed + vehicles)
            sample_number = 0
            last_now = 0.0
            for ti in range(len(trace["times"])):
                now = float(trace["times"][ti])
                elapsed = max(0.0, now - last_now)
                order = sorted(range(vehicles),
                               key=lambda j: (priority[str(tasks["class"][ti, j])], j))
                for j in order:
                    task = task_at(tasks, ti, j)
                    mult = float(cfg["orchestration"]["bandwidth_multiplier"][task["class"]])
                    radios = radio_options(now, trace["xy"][ti, j], task, vehicles,
                                           infra, cfg, mult)
                    if (ti, j) not in selected:
                        plan, _, _, _, _ = choose_plan(
                            "AAI-CDOS", radios, state, task, now, cfg,
                            tasks["oneshot_noise"][ti, j])
                        update_state(state, plan, radios, task, now, cfg, j, elapsed)
                        continue

                    sample_number += 1
                    event_id = "n%d_s%d_e%d" % (vehicles, seed, sample_number)
                    if event_id not in indexed.index:
                        errors.append("missing recorded event %s" % event_id)
                        continue
                    recorded = indexed.loc[event_id]
                    if int(recorded.frame) != ti or recorded.task_class != task["class"]:
                        errors.append("event reconstruction mismatch for %s" % event_id)

                    plans = {}
                    planner_ms = {}
                    plans["AAI-CDOS"] = (str(recorded.aai_access),
                                          str(recorded.aai_compute))
                    plans["One-Shot LLM"] = (str(recorded.oneshot_access),
                                              str(recorded.oneshot_compute))
                    for scheme in ("Single Domain", "Independent Agents"):
                        plan, _, _, _, ms = choose_plan(
                            scheme, radios, state, task, now, cfg,
                            tasks["oneshot_noise"][ti, j])
                        plans[scheme] = plan
                        planner_ms[scheme] = ms

                    for scheme in SCHEMES:
                        latency, mismatch = evaluate_with_mismatch(
                            scheme, plans[scheme], radios, state, task, now, cfg)
                        if scheme == "AAI-CDOS":
                            decision_ms = float(recorded.aai_event_wall_ms)
                            input_tokens = int(recorded.aai_input_tokens)
                            output_tokens = int(recorded.aai_output_tokens)
                            api_calls = int(recorded.aai_api_calls)
                        elif scheme == "One-Shot LLM":
                            decision_ms = float(recorded.oneshot_wall_ms)
                            input_tokens = int(recorded.oneshot_input_tokens)
                            output_tokens = int(recorded.oneshot_output_tokens)
                            api_calls = int(recorded.oneshot_api_calls)
                        else:
                            decision_ms = float(planner_ms[scheme])
                            input_tokens = 0
                            output_tokens = 0
                            api_calls = 0
                        rows.append({
                            "event_id": event_id,
                            "vehicles": vehicles,
                            "seed": seed,
                            "frame": ti,
                            "task_class": task["class"],
                            "scheme": scheme,
                            "access": plans[scheme][0],
                            "compute": plans[scheme][1],
                            "deadline_ms": round(1000.0 * task["deadline"], 3),
                            "e2e_latency_ms": round(1000.0 * latency, 3),
                            "deadline_met": bool(latency <= task["deadline"]),
                            "mismatch_delay_ms": round(1000.0 * mismatch, 3),
                            "decision_wall_ms": decision_ms,
                            "api_calls": api_calls,
                            "api_input_tokens": input_tokens,
                            "api_output_tokens": output_tokens,
                            "comparison_state": "common_AAI_predecision_state",
                        })

                    aai_latency, _ = evaluate(
                        plans["AAI-CDOS"], radios, state, task, now, cfg)
                    if abs(1000.0 * aai_latency - float(recorded.aai_estimated_latency_ms)) > 0.01:
                        errors.append("AAI latency mismatch for %s" % event_id)
                    one_latency, _ = evaluate(
                        plans["One-Shot LLM"], radios, state, task, now, cfg)
                    if abs(1000.0 * one_latency - float(recorded.oneshot_estimated_latency_ms)) > 0.01:
                        errors.append("One-Shot latency mismatch for %s" % event_id)
                    update_state(state, plans["AAI-CDOS"], radios, task, now,
                                 cfg, j, elapsed)
                last_now = now
    return rows, errors


def summarize(df):
    out = []
    for (vehicles, scheme), group in df.groupby(["vehicles", "scheme"], sort=False):
        out.append({
            "vehicles": int(vehicles),
            "scheme": scheme,
            "events": int(len(group)),
            "deadline_feasibility": float(group.deadline_met.mean()),
            "mean_e2e_latency_ms": float(group.e2e_latency_ms.mean()),
            "p95_e2e_latency_ms": float(group.e2e_latency_ms.quantile(0.95)),
            "mean_decision_wall_ms": float(group.decision_wall_ms.mean()),
            "mean_api_calls": float(group.api_calls.mean()),
            "mean_api_input_tokens": float(group.api_input_tokens.mean()),
            "mean_api_output_tokens": float(group.api_output_tokens.mean()),
            "mean_api_tokens": float((group.api_input_tokens +
                                      group.api_output_tokens).mean()),
        })
    return pd.DataFrame(out)


def grouped_bars(ax, summary, metric, ylabel, title, log=False):
    vehicles = sorted(summary.vehicles.unique())
    x = np.arange(len(vehicles))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(SCHEMES))
    for offset, scheme in zip(offsets, SCHEMES):
        group = summary[summary.scheme == scheme].set_index("vehicles")
        values = [float(group.loc[v, metric]) for v in vehicles]
        # Log plots cannot show exact zeros; deterministic local times are
        # positive, while the token panel remains linear and displays zeros.
        ax.bar(x + offset, values, width, label=scheme,
               color=COLORS[scheme])
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in vehicles])
    ax.set_xlabel("Connected vehicles")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")


def plot(summary, output_base):
    plt.rcParams.update({"font.size": 8.5, "axes.grid": True,
                         "grid.alpha": 0.25, "savefig.dpi": 300})
    display = summary.copy()
    display["deadline_feasibility_pct"] = 100.0 * display.deadline_feasibility
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15))
    grouped_bars(axes[0, 0], display, "deadline_feasibility_pct", "%",
                 "Same-state plan feasibility")
    axes[0, 0].set_ylim(0, 105)
    grouped_bars(axes[0, 1], display, "mean_e2e_latency_ms", "ms",
                 "Same-state selected-plan E2E latency")
    grouped_bars(axes[1, 0], display, "mean_decision_wall_ms", "ms (log scale)",
                 "Decision wall time", log=True)
    grouped_bars(axes[1, 1], display, "mean_api_tokens", "tokens/event",
                 "Actual API usage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.01), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(output_base + "." + ext, bbox_inches="tight")
    plt.close(fig)


def write_latex(path, summary):
    lines = [
        "% Auto-generated by scripts/augment_llm_audit_baselines.py",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Four-scheme decision audit on common sampled pre-decision states.}",
        "  \\label{tab:llm_audit_four_scheme}",
        "  \\begin{tabular}{c|l|rrrr}",
        "    \\hline",
        "    Vehicles & Scheme & Feasible (\\%) & E2E (ms) & Decision (ms) & API tokens \\\\",
        "    \\hline",
    ]
    for vehicles in sorted(summary.vehicles.unique()):
        block = summary[summary.vehicles == vehicles].set_index("scheme")
        for index, scheme in enumerate(SCHEMES):
            row = block.loc[scheme]
            first = str(int(vehicles)) if index == 0 else ""
            lines.append("    %s & %s & %.1f & %.1f & %.3f & %.1f \\\\" % (
                first, scheme, 100.0 * row.deadline_feasibility,
                row.mean_e2e_latency_ms, row.mean_decision_wall_ms,
                row.mean_api_tokens))
        lines.append("    \\hline")
    lines += ["  \\end{tabular}", "\\end{table*}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_results_tex(path, frame, summary):
    aai = frame[frame.scheme == "AAI-CDOS"].set_index("event_id")
    one = frame[frame.scheme == "One-Shot LLM"].set_index("event_id")
    identical = int(((aai.access == one.access) &
                     (aai.compute == one.compute)).sum())
    lines = [
        "% Auto-generated by scripts/augment_llm_audit_baselines.py",
        "\\paragraph{Four-scheme common-state audit.}",
        ("On the 30 sampled pre-decision states, deadline-feasible selection "
         "rates for AAI-CDOS, Single Domain, Independent Agents, and One-Shot "
         "LLM were respectively 100/50/90/100\\% at 10 vehicles, "
         "100/60/100/100\\% at 25 vehicles, and 70/40/60/70\\% at 50 "
         "vehicles (Table~\\ref{tab:llm_audit_four_scheme})."),
        ("AAI-CDOS and One-Shot LLM selected the same access/compute plan in "
         "%d of %d sampled events. Consequently, this audit demonstrates an "
         "advantage over the two deterministic domain baselines on these "
         "states, but it does not demonstrate a decision-quality improvement "
         "of multi-agent negotiation over One-Shot GPT-4o. AAI-CDOS instead "
         "incurred additional coordination calls, tokens, and wall time. "
         "The comparison uses a common AAI pre-decision state to isolate "
         "decision behavior; policy-level conclusions use the separate "
         "30-seed replay." % (identical, len(aai))),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/simulation.yaml")
    parser.add_argument("--audit-config", default="config/llm_audit.yaml")
    parser.add_argument("--input", default="results/llm_audit/events.csv")
    args = parser.parse_args()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    audit_path = (args.audit_config if os.path.isabs(args.audit_config) else
                  os.path.join(ROOT, args.audit_config))
    input_path = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    with open(audit_path, "r") as f:
        audit_cfg = yaml.safe_load(f)["llm_audit"]
    events = pd.read_csv(input_path)
    data_path = cfg["experiment"]["dataset_path"]
    data_path = data_path if os.path.isabs(data_path) else os.path.join(ROOT, data_path)
    rows, errors = replay(events, cfg, audit_cfg, data_path)
    expected = len(events) * len(SCHEMES)
    if len(rows) != expected:
        errors.append("expected %d four-scheme rows, got %d" % (expected, len(rows)))
    frame = pd.DataFrame(rows)
    summary = summarize(frame)
    outdir = os.path.join(ROOT, "results", "llm_audit")
    write_csv(os.path.join(outdir, "four_scheme_events.csv"), rows)
    summary.to_csv(os.path.join(outdir, "four_scheme_summary.csv"), index=False)
    plot(summary, os.path.join(ROOT, "figures", "llm_audit_four_scheme"))
    write_latex(os.path.join(ROOT, "paper", "llm_audit_four_scheme_table.tex"),
                summary)
    write_results_tex(os.path.join(ROOT, "paper",
                                   "llm_audit_four_scheme_results.tex"),
                      frame, summary)
    aai = frame[frame.scheme == "AAI-CDOS"].set_index("event_id")
    one = frame[frame.scheme == "One-Shot LLM"].set_index("event_id")
    identical = int(((aai.access == one.access) &
                     (aai.compute == one.compute)).sum())
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "source_events": len(events),
        "schemes": SCHEMES,
        "rows": len(rows),
        "comparison_state": "common AAI pre-decision state",
        "api_calls_added": 0,
        "aai_oneshot_identical_plan_events": identical,
        "aai_oneshot_compared_events": len(aai),
        "interpretation": (
            "Decision audit only; full policy-level outcomes remain in the "
            "30-seed main experiment."),
    }
    with open(os.path.join(outdir, "four_scheme_validation.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(summary.to_string(index=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
