#!/usr/bin/env python3
"""Reconstruct sanitized inputs, intermediate agent turns, and verifier traces.

The real API run stored structured outputs and telemetry but intentionally did
not persist prompt inputs. Because the replay, seeds, policies, and final AAI
actions are deterministic, this script reconstructs the exact structured input
objects without making any new API calls. Coordinates, terminal IDs, API keys,
and raw trace timestamps remain excluded.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

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
from scripts.run_llm_audit import (plan_context, sample_event_indices, task_at,
                                   update_state)


DOMAIN_INSTRUCTIONS = {
    "oran": (
        "You are the O-RAN agent in AAI-CDOS. Select one access domain from "
        "the measured radio options. Optimize hard-deadline feasibility, then "
        "link rate. Return only the required structured decision."),
    "compute": (
        "You are the compute agent in AAI-CDOS. Select one exact compute node. "
        "Minimize queue plus service time while respecting the hard deadline. "
        "Return only the required structured decision."),
    "cn": (
        "You are the core-network routing agent in AAI-CDOS. Select one candidate "
        "plan ID using path and end-to-end latency. Treat the deadline as hard. "
        "Return only the required structured decision."),
    "e2e": (
        "You are the E2E coordination agent in AAI-CDOS. Reconcile O-RAN, "
        "core-network, and compute proposals into one joint plan. The deterministic "
        "verifier enforces the hard deadline. Prefer a feasible plan with the "
        "largest deadline margin; never repeat a verifier-rejected plan. Return "
        "only the required structured decision."),
    "one_shot": (
        "You are the One-Shot LLM baseline. Select one joint plan in a single pass "
        "without domain-agent messages, negotiation, memory, or verifier feedback. "
        "Treat the task deadline as hard. Return only the structured decision."),
}


def canonical_sha256(value):
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def public_trace(value):
    """Remove API correlation identifiers from a reconstructed trace."""
    if isinstance(value, dict):
        return {k: public_trace(v) for k, v in value.items()
                if k not in ("request_id", "response_id")}
    if isinstance(value, list):
        return [public_trace(x) for x in value]
    return value


def load_calls(path):
    out = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["event_id"]].append(row)
    return out


def role_call(calls, role):
    matches = [x for x in calls if x["agent_role"] == role]
    if len(matches) != 1:
        raise ValueError("expected exactly one %s call" % role)
    return matches[0]


def compute_view(candidates):
    nodes = sorted(set(x["compute"] for x in candidates))
    out = []
    for node in nodes:
        related = [x for x in candidates if x["compute"] == node]
        out.append({
            "compute": node,
            "minimum_queue_ms": min(x["queue_ms"] for x in related),
            "service_ms": min(x["service_ms"] for x in related),
        })
    return out


def turn(role, instructions, input_object, call):
    return {
        "role": role,
        "instructions": instructions,
        "input": input_object,
        "input_sha256": canonical_sha256(input_object),
        "output": call["decision"],
        "response_id": call["response_id"],
        "request_id": call["request_id"],
        "returned_model": call["returned_model"],
        "input_tokens": call["input_tokens"],
        "output_tokens": call["output_tokens"],
        "api_latency_ms": round(1000.0 * call["latency_s"], 3),
    }


def reconstruct(events, calls_by_event, cfg, audit_cfg, data_path):
    traces, metric_rows, errors = [], [], []
    indexed = events.set_index("event_id")
    priority = {"safety": 0, "cooperative": 1, "routine": 2}
    seed0 = int(cfg["experiment"]["seed_start"])

    for vehicles in cfg["experiment"]["vehicle_counts"]:
        vehicles = int(vehicles)
        for run_index in range(int(audit_cfg["independent_seeds"])):
            seed = seed0 + run_index
            trace_data = load_trace(
                data_path, vehicles, seed, int(audit_cfg["trace_frames"]),
                float(cfg["experiment"]["minimum_vehicle_coverage"]),
                cfg["experiment"].get("region_bbox_lonlat"))
            tasks = build_tasks(trace_data, cfg, seed)
            infra = make_infrastructure(trace_data, cfg)
            state = PolicyState(cfg, vehicles)
            selected = sample_event_indices(
                tasks, int(audit_cfg["events_per_vehicle_seed"]), seed + vehicles)
            sample_number = 0
            last_now = 0.0

            for ti in range(len(trace_data["times"])):
                now = float(trace_data["times"][ti])
                elapsed = max(0.0, now - last_now)
                order = sorted(range(vehicles),
                               key=lambda j: (priority[str(tasks["class"][ti, j])], j))
                for j in order:
                    task = task_at(tasks, ti, j)
                    mult = float(cfg["orchestration"]["bandwidth_multiplier"][task["class"]])
                    radios = radio_options(now, trace_data["xy"][ti, j], task,
                                           vehicles, infra, cfg, mult)
                    if (ti, j) not in selected:
                        plan, _, _, _, _ = choose_plan(
                            "AAI-CDOS", radios, state, task, now, cfg,
                            tasks["oneshot_noise"][ti, j])
                        update_state(state, plan, radios, task, now, cfg, j, elapsed)
                        continue

                    sample_number += 1
                    event_id = "n%d_s%d_e%d" % (vehicles, seed, sample_number)
                    if event_id not in indexed.index or event_id not in calls_by_event:
                        errors.append("missing event/calls for %s" % event_id)
                        continue
                    recorded = indexed.loc[event_id]
                    calls = calls_by_event[event_id]
                    task_summary, radio_summary, candidates, plan_map = plan_context(
                        radios, state, task, now, cfg)
                    plan_ids = [x["plan_id"] for x in candidates]
                    feasible_ids = [x["plan_id"] for x in candidates
                                    if x["deadline_met"]]

                    domain_inputs = {
                        "oran": {"task": task_summary,
                                 "radio_options": radio_summary},
                        "compute": {"task": task_summary,
                                    "compute_nodes": compute_view(candidates)},
                        "cn": {"task": task_summary,
                               "candidate_routes": [
                                   {k: x[k] for k in
                                    ("plan_id", "access", "compute", "path_ms",
                                     "latency_ms", "deadline_met")}
                                   for x in candidates]},
                    }
                    domain_turns = {}
                    proposals = {}
                    for role in ("oran", "compute", "cn"):
                        call = role_call(calls, role)
                        domain_turns[role] = turn(
                            role, DOMAIN_INSTRUCTIONS[role], domain_inputs[role], call)
                        proposals[role] = call["decision"]

                    rejected = []
                    e2e_turns = []
                    round_index = 1
                    while True:
                        role = "e2e_round_%d" % round_index
                        matches = [x for x in calls if x["agent_role"] == role]
                        if not matches:
                            break
                        call = matches[0]
                        e2e_input = {
                            "task": task_summary,
                            "domain_agent_proposals": proposals,
                            "candidate_joint_plans": candidates,
                            "coordination_round": round_index,
                            "verifier_rejected_plan_ids": list(rejected),
                            "verifier_feasible_plan_ids": feasible_ids,
                        }
                        selected_id = call["decision"]["plan_id"]
                        selected_candidate = next(x for x in candidates
                                                  if x["plan_id"] == selected_id)
                        accepted = bool(selected_candidate["deadline_met"])
                        verifier = {
                            "accepted": accepted,
                            "checked_plan_id": selected_id,
                            "candidate_latency_ms": selected_candidate["latency_ms"],
                            "hard_deadline_ms": task_summary["deadline_ms"],
                            "reason": ("deadline_satisfied" if accepted else
                                       "hard_deadline_violation"),
                            "feasible_plan_ids": feasible_ids,
                        }
                        item = turn(role, DOMAIN_INSTRUCTIONS["e2e"], e2e_input, call)
                        item["verifier_feedback"] = verifier
                        e2e_turns.append(item)
                        if accepted:
                            break
                        rejected.append(selected_id)
                        round_index += 1

                    one_call = role_call(calls, "one_shot")
                    one_input = {"task": task_summary,
                                 "candidate_joint_plans": candidates}
                    one_turn = turn("one_shot", DOMAIN_INSTRUCTIONS["one_shot"],
                                    one_input, one_call)
                    final_plan = (str(recorded.aai_access), str(recorded.aai_compute))
                    final_id_matches = [pid for pid, plan in plan_map.items()
                                        if plan == final_plan]
                    final_id = final_id_matches[0] if final_id_matches else "fallback"
                    reconstructed = {
                        "record_type": "reconstructed_sanitized_agentic_trace",
                        "event_id": event_id,
                        "reconstruction_basis": (
                            "deterministic trace/seed/state plus saved structured outputs"),
                        "privacy": {
                            "coordinates_included": False,
                            "terminal_ids_included": False,
                            "raw_trace_timestamps_included": False,
                            "api_key_included": False,
                        },
                        "observe": {
                            "vehicles": vehicles,
                            "seed": seed,
                            "frame_index": ti,
                            "task": task_summary,
                            "radio_options": radio_summary,
                            "candidate_joint_plans": candidates,
                        },
                        "domain_agent_turns": domain_turns,
                        "coordination_turns": e2e_turns,
                        "one_shot_baseline_turn": one_turn,
                        "execute": {
                            "final_plan_id": final_id,
                            "access": final_plan[0],
                            "compute": final_plan[1],
                            "latency_ms": float(recorded.aai_estimated_latency_ms),
                            "deadline_met": bool(recorded.aai_deadline_met),
                            "fallback_after_round_limit": bool(recorded.aai_fallback),
                        },
                        "feedback": {
                            "rounds": int(recorded.aai_rounds),
                            "rejections": int(recorded.aai_rejections),
                            "verifier_accepted": bool(recorded.aai_verifier_accepted),
                            "memory_update": {
                                "task_class": task["class"],
                                "plan": [final_plan[0], final_plan[1]],
                                "success": bool(recorded.aai_deadline_met),
                                "latency_ms": float(recorded.aai_estimated_latency_ms),
                            },
                        },
                    }
                    reconstructed["trace_sha256"] = canonical_sha256(reconstructed)
                    traces.append(reconstructed)

                    oran_agree = proposals["oran"]["access"] == final_plan[0]
                    compute_agree = proposals["compute"]["compute"] == final_plan[1]
                    cn_plan = plan_map[proposals["cn"]["plan_id"]]
                    cn_agree = cn_plan == final_plan
                    one_plan = plan_map[one_call["decision"]["plan_id"]]
                    aai_calls = [x for x in calls if x["agent_role"] != "one_shot"]
                    metric_rows.append({
                        "event_id": event_id,
                        "vehicles": vehicles,
                        "seed": seed,
                        "task_class": task["class"],
                        "feasible_candidates": len(feasible_ids),
                        "candidate_count": len(plan_ids),
                        "oran_final_agreement": bool(oran_agree),
                        "compute_final_agreement": bool(compute_agree),
                        "cn_final_agreement": bool(cn_agree),
                        "all_domain_proposals_align_final": bool(
                            oran_agree and compute_agree and cn_agree),
                        "aai_oneshot_same_plan": bool(one_plan == final_plan),
                        "rounds": int(recorded.aai_rounds),
                        "rejections": int(recorded.aai_rejections),
                        "fallback": bool(recorded.aai_fallback),
                        "verifier_accepted": bool(recorded.aai_verifier_accepted),
                        "aai_api_calls": len(aai_calls),
                        "aai_input_tokens": sum(x["input_tokens"] for x in aai_calls),
                        "aai_output_tokens": sum(x["output_tokens"] for x in aai_calls),
                        "aai_sum_api_latency_ms": 1000.0 * sum(
                            x["latency_s"] for x in aai_calls),
                        "aai_event_wall_ms": float(recorded.aai_event_wall_ms),
                    })
                    update_state(state, final_plan, radios, task, now, cfg, j, elapsed)
                last_now = now
    return traces, metric_rows, errors


def summarize_metrics(frame):
    rows = []
    for vehicles, group in frame.groupby("vehicles"):
        rows.append({
            "vehicles": int(vehicles),
            "events": int(len(group)),
            "mean_feasible_candidates": float(group.feasible_candidates.mean()),
            "oran_final_agreement": float(group.oran_final_agreement.mean()),
            "compute_final_agreement": float(group.compute_final_agreement.mean()),
            "cn_final_agreement": float(group.cn_final_agreement.mean()),
            "all_domain_proposals_align_final": float(
                group.all_domain_proposals_align_final.mean()),
            "aai_oneshot_same_plan": float(group.aai_oneshot_same_plan.mean()),
            "mean_rounds": float(group.rounds.mean()),
            "events_with_rejection": float((group.rejections > 0).mean()),
            "fallback_rate": float(group.fallback.mean()),
            "verifier_acceptance_rate": float(group.verifier_accepted.mean()),
            "mean_aai_api_calls": float(group.aai_api_calls.mean()),
            "mean_aai_tokens": float((group.aai_input_tokens +
                                      group.aai_output_tokens).mean()),
            "mean_aai_event_wall_ms": float(group.aai_event_wall_ms.mean()),
        })
    return pd.DataFrame(rows)


def role_summary(calls_by_event):
    buckets = defaultdict(list)
    for calls in calls_by_event.values():
        for call in calls:
            role = call["agent_role"]
            if role.startswith("e2e_round_"):
                role = "e2e"
            buckets[role].append(call)
    rows = []
    for role in ("oran", "cn", "compute", "e2e", "one_shot"):
        values = buckets[role]
        lat = np.array([1000.0 * x["latency_s"] for x in values])
        rows.append({
            "agent_role": role,
            "calls": len(values),
            "mean_input_tokens": float(np.mean([x["input_tokens"] for x in values])),
            "mean_output_tokens": float(np.mean([x["output_tokens"] for x in values])),
            "mean_api_latency_ms": float(lat.mean()),
            "p95_api_latency_ms": float(np.quantile(lat, 0.95)),
        })
    return pd.DataFrame(rows)


def plot_agentic(summary, roles, output_base):
    plt.rcParams.update({"font.size": 8.5, "axes.grid": True,
                         "grid.alpha": 0.25, "savefig.dpi": 300})
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.2))
    vehicles = summary.vehicles.to_numpy()
    x = np.arange(len(vehicles))
    width = 0.24
    agreements = [
        ("O-RAN", "oran_final_agreement", "#0072B2"),
        ("CN", "cn_final_agreement", "#E69F00"),
        ("Compute", "compute_final_agreement", "#009E73"),
    ]
    for offset, (label, col, color) in zip((-width, 0, width), agreements):
        axes[0, 0].bar(x + offset, 100.0 * summary[col], width,
                       label=label, color=color)
    axes[0, 0].set_title("Domain proposal agreement with final plan")
    axes[0, 0].set_ylabel("%")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].legend(fontsize=8)

    negotiation_ax = axes[0, 1]
    rejection_ax = negotiation_ax.twinx()
    rounds_bar = negotiation_ax.bar(
        x - width / 2, summary.mean_rounds, width,
        label="Mean rounds", color="#0072B2")
    rejection_bar = rejection_ax.bar(
        x + width / 2, 100.0 * summary.events_with_rejection, width,
        label="Events with rejection", color="#D55E00")
    negotiation_ax.set_title("Negotiation and verifier activity")
    negotiation_ax.set_ylabel("rounds/event")
    negotiation_ax.set_ylim(0, 3.2)
    rejection_ax.set_ylabel("rejected events (%)")
    rejection_ax.set_ylim(0, 100)
    negotiation_ax.legend([rounds_bar, rejection_bar],
                          ["Mean rounds", "Events with rejection"],
                          fontsize=8, loc="upper left")

    rx = np.arange(len(roles))
    axes[1, 0].bar(rx, roles.mean_input_tokens, label="Input",
                   color="#56B4E9")
    axes[1, 0].bar(rx, roles.mean_output_tokens,
                   bottom=roles.mean_input_tokens, label="Output",
                   color="#CC79A7")
    axes[1, 0].set_title("Actual tokens per agent call")
    axes[1, 0].set_ylabel("tokens")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(rx - width / 2, roles.mean_api_latency_ms, width,
                   label="Mean", color="#009E73")
    axes[1, 1].bar(rx + width / 2, roles.p95_api_latency_ms, width,
                   label="P95", color="#E69F00")
    axes[1, 1].set_title("Actual API latency per agent call")
    axes[1, 1].set_ylabel("ms")
    axes[1, 1].legend(fontsize=8)

    for ax in axes[0]:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in vehicles])
        ax.set_xlabel("Connected vehicles")
    for ax in axes[1]:
        ax.set_xticks(rx)
        ax.set_xticklabels([x.replace("one_shot", "one-shot").upper()
                            for x in roles.agent_role], rotation=20)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_base + "." + ext, bbox_inches="tight")
    plt.close(fig)


def example_markdown(traces):
    normal = next(x for x in traces if x["feedback"]["rounds"] == 1)
    negotiated = next((x for x in traces if x["feedback"]["rounds"] > 1),
                      traces[-1])
    lines = [
        "# Sanitized agent-interaction examples", "",
        "These are deterministic reconstructions of the structured inputs sent",
        "during the completed GPT-4o audit plus the saved structured outputs.",
        "They are not free-form chat transcripts and contain no coordinates or",
        "terminal identifiers.", "",
    ]
    for title, item in (("Accepted in one round", normal),
                        ("Verifier-driven multi-round case", negotiated)):
        obs = item["observe"]
        lines += ["## %s: `%s`" % (title, item["event_id"]), "",
                  "Task: `%s`, deadline %.1f ms, %.3f Mbit; feasible candidates %d/%d." %
                  (obs["task"]["class"], obs["task"]["deadline_ms"],
                   obs["task"]["update_mbit"],
                   sum(x["deadline_met"] for x in obs["candidate_joint_plans"]),
                   len(obs["candidate_joint_plans"])), "",
                  "Domain proposals:", ""]
        for role in ("oran", "cn", "compute"):
            lines.append("- `%s`: `%s`" % (
                role, json.dumps(item["domain_agent_turns"][role]["output"],
                                 sort_keys=True)))
        lines += ["", "Coordination and verifier:", ""]
        for turn_item in item["coordination_turns"]:
            lines.append("- `%s` chose `%s`; verifier `%s` (%.1f ms vs %.1f ms)." % (
                turn_item["role"], turn_item["output"]["plan_id"],
                "accepted" if turn_item["verifier_feedback"]["accepted"] else "rejected",
                turn_item["verifier_feedback"]["candidate_latency_ms"],
                turn_item["verifier_feedback"]["hard_deadline_ms"]))
        lines += ["", "Final execution: `%s -> %s`, %.1f ms, deadline met: `%s`, fallback: `%s`." %
                  (item["execute"]["access"], item["execute"]["compute"],
                   item["execute"]["latency_ms"], item["execute"]["deadline_met"],
                   item["execute"]["fallback_after_round_limit"]), ""]
    return "\n".join(lines) + "\n"


def write_latex(path, summary):
    lines = [
        "% Auto-generated by scripts/reconstruct_agentic_trace.py",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Observed agentic coordination behavior with real GPT-4o calls.}",
        "  \\label{tab:agentic_coordination}",
        "  \\begin{tabular}{c|rrrrrr}",
        "    \\hline",
        "    Vehicles & O-RAN agree & CN agree & Compute agree & Rounds & Reject events & Tokens/event \\\\",
        "    \\hline",
    ]
    for _, row in summary.iterrows():
        lines.append("    %d & %.1f\\%% & %.1f\\%% & %.1f\\%% & %.2f & %.1f\\%% & %.1f \\\\" % (
            row.vehicles, 100.0 * row.oran_final_agreement,
            100.0 * row.cn_final_agreement,
            100.0 * row.compute_final_agreement, row.mean_rounds,
            100.0 * row.events_with_rejection, row.mean_aai_tokens))
    lines += ["    \\hline", "  \\end{tabular}", "\\end{table*}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_results_tex(path, metrics, summary):
    total = len(metrics)
    rejected = int((metrics.rejections > 0).sum())
    aligned = int(metrics.all_domain_proposals_align_final.sum())
    one_same = int(metrics.aai_oneshot_same_plan.sum())
    lines = [
        "% Auto-generated by scripts/reconstruct_agentic_trace.py",
        "\\paragraph{Agent-interaction analysis.}",
        ("We reconstruct the structured observation, three domain proposals, "
         "E2E coordination input, deterministic verifier feedback, and final "
         "memory update for all %d audited events. Each turn is linked to its "
         "saved OpenAI request/response identifiers and a SHA-256 digest of "
         "the reconstructed input. No coordinates or terminal identifiers are "
         "included." % total),
        ("All three domain proposals aligned with the final joint plan in %d "
         "of %d events. Verifier rejection triggered multi-round negotiation "
         "in %d events, all at the 50-vehicle load, where no candidate met the "
         "hard deadline. AAI-CDOS and One-Shot selected the same plan in %d of "
         "%d events; thus the present audit quantifies agentic process and "
         "overhead but does not establish a decision-quality gain over "
         "One-Shot GPT-4o." % (aligned, total, rejected, one_same, total)),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/simulation.yaml")
    parser.add_argument("--audit-config", default="config/llm_audit.yaml")
    parser.add_argument("--events", default="results/llm_audit/events.csv")
    parser.add_argument("--calls", default="results/llm_audit/calls.jsonl")
    args = parser.parse_args()

    def full(path):
        return path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full(args.config), "r") as f:
        cfg = yaml.safe_load(f)
    with open(full(args.audit_config), "r") as f:
        audit_cfg = yaml.safe_load(f)["llm_audit"]
    events = pd.read_csv(full(args.events))
    calls_by_event = load_calls(full(args.calls))
    data_path = cfg["experiment"]["dataset_path"]
    data_path = data_path if os.path.isabs(data_path) else os.path.join(ROOT, data_path)
    traces, metric_rows, errors = reconstruct(
        events, calls_by_event, cfg, audit_cfg, data_path)
    metrics = pd.DataFrame(metric_rows)
    summary = summarize_metrics(metrics)
    roles = role_summary(calls_by_event)
    if len(traces) != len(events):
        errors.append("trace count mismatch")
    if len(set(x["trace_sha256"] for x in traces)) != len(traces):
        errors.append("duplicate reconstructed trace digest")

    outdir = os.path.join(ROOT, "results", "llm_audit")
    with open(os.path.join(outdir, "conversation_trace.jsonl"), "w") as f:
        for item in traces:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    public_traces = [public_trace(item) for item in traces]
    with open(os.path.join(outdir, "conversation_trace_public.jsonl"), "w") as f:
        for item in public_traces:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    metrics.to_csv(os.path.join(outdir, "agentic_event_metrics.csv"), index=False)
    summary.to_csv(os.path.join(outdir, "agentic_summary.csv"), index=False)
    roles.to_csv(os.path.join(outdir, "agent_role_summary.csv"), index=False)
    with open(os.path.join(outdir, "conversation_examples.md"), "w") as f:
        f.write(example_markdown(traces))
    plot_agentic(summary, roles, os.path.join(ROOT, "figures",
                                              "agentic_interaction_metrics"))
    write_latex(os.path.join(ROOT, "paper", "agentic_metrics_table.tex"), summary)
    write_results_tex(os.path.join(ROOT, "paper", "agentic_results.tex"),
                      metrics, summary)
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "events": len(events),
        "reconstructed_traces": len(traces),
        "unique_trace_sha256": len(set(x["trace_sha256"] for x in traces)),
        "new_api_calls": 0,
        "trace_type": "deterministically reconstructed sanitized structured transcript",
        "contains_coordinates": False,
        "contains_terminal_ids": False,
        "contains_api_key": False,
        "public_trace_contains_request_or_response_ids": any(
            ("request_id" in json.dumps(x) or "response_id" in json.dumps(x))
            for x in public_traces),
    }
    with open(os.path.join(outdir, "agentic_trace_validation.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(summary.to_string(index=False))
    print(roles.to_string(index=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
