#!/usr/bin/env python3
"""Run a bounded, real-GPT-4o planner telemetry audit.

This audit samples states from the full Seoul V2X replay. It measures actual
Responses API tokens and wall-clock latency without sending positions or
terminal identifiers. It does not replace the 30-seed network-performance run.
"""

import argparse
import concurrent.futures
import csv
import json
import os
import platform
import socket
import sys
import time
from collections import defaultdict

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.openai_planner import OpenAIResponsesClient, object_schema
from src.simulator import (PolicyState, build_tasks, candidate_plans, choose_plan,
                           evaluate, load_trace, make_infrastructure, radio_options,
                           sha256_file)


RATIONALE_CODES = [
    "lowest_latency", "deadline_margin", "radio_quality", "queue_avoidance",
    "path_efficiency", "proposal_consensus", "no_feasible_plan",
]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True)


def task_at(tasks, ti, j):
    return {
        "class": str(tasks["class"][ti, j]),
        "deadline": float(tasks["deadline"][ti, j]),
        "bits": float(tasks["bits"][ti, j]),
        "tx_w": float(tasks["tx_w"][ti, j]),
        "shadow_ground": float(tasks["shadow_ground"][ti, j]),
        "fading_uav": float(tasks["fading_uav"][ti, j]),
        "fading_leo": float(tasks["fading_leo"][ti, j]),
        "route_jitter": float(tasks["route_jitter"][ti, j]),
        "oneshot_random": float(tasks["oneshot_random"][ti, j]),
        "satellite_tx_w": float(tasks["satellite_tx_w"][ti, j]),
    }


def sample_event_indices(tasks, count, seed):
    """Choose a deterministic class-stratified set of (frame, vehicle) pairs."""
    rng = np.random.default_rng(seed + 48017)
    labels = tasks["class"]
    classes = ["safety", "cooperative", "routine"]
    pools = {}
    for cls in classes:
        pools[cls] = [tuple(x) for x in np.argwhere(labels == cls)]
        rng.shuffle(pools[cls])
    targets = [classes[i % len(classes)] for i in range(count)]
    chosen = []
    offsets = defaultdict(int)
    for cls in targets:
        if offsets[cls] < len(pools[cls]):
            chosen.append(pools[cls][offsets[cls]])
            offsets[cls] += 1
    if len(chosen) < count:
        used = set(chosen)
        rest = [tuple(x) for x in np.argwhere(np.ones_like(labels, dtype=bool))
                if tuple(x) not in used]
        rng.shuffle(rest)
        chosen.extend(rest[:count - len(chosen)])
    return set(chosen)


def choice_schema(choices, field):
    return object_schema({
        field: {"type": "string", "enum": list(choices)},
        "confidence": {"type": "number"},
        "rationale_code": {"type": "string", "enum": RATIONALE_CODES},
    })


def plan_context(radios, state, task, now, cfg):
    plans = candidate_plans(radios, cfg)
    candidates = []
    plan_map = {}
    for idx, plan in enumerate(plans):
        latency, parts = evaluate(plan, radios, state, task, now, cfg)
        plan_id = "p%d" % idx
        item = {
            "plan_id": plan_id,
            "access": plan[0],
            "compute": plan[1],
            "latency_ms": round(1000.0 * latency, 3),
            "tx_ms": round(1000.0 * parts["tx"], 3),
            "path_ms": round(1000.0 * parts["path"], 3),
            "queue_ms": round(1000.0 * parts["wait"], 3),
            "service_ms": round(1000.0 * parts["service"], 3),
            "deadline_met": bool(latency <= task["deadline"]),
        }
        candidates.append(item)
        plan_map[plan_id] = plan
    radio_summary = {}
    for access, value in radios.items():
        radio_summary[access] = {
            "rate_mbps": round(value["rate"] / 1e6, 4),
            "snr_db": round(value["snr"], 3),
            "nearest_index": int(value["index"]),
            "link_distance_m": round(value["distance"], 2),
        }
    task_summary = {
        "class": task["class"],
        "deadline_ms": round(1000.0 * task["deadline"], 3),
        "update_mbit": round(task["bits"] / 1e6, 5),
    }
    return task_summary, radio_summary, candidates, plan_map


def call_agent(client, event_id, role, instructions, context, schema, schema_name):
    result = client.call_json(
        instructions, context, schema_name, schema,
        metadata={"experiment": "aai_cdos_llm_audit", "event": event_id,
                  "agent_role": role})
    row = {k: v for k, v in result.items() if k != "result"}
    row.update({"event_id": event_id, "agent_role": role,
                "decision": result["result"]})
    return result["result"], row


def run_domain_agents(client, workers, event_id, task_summary, radio_summary,
                      candidates, state):
    # The candidate set exposes only the nearest UAV compute node, whereas the
    # simulator state tracks every UAV. Restrict the agent's enum and summary
    # to nodes that can actually be selected in this event.
    compute_nodes = sorted(set(x["compute"] for x in candidates))
    compute_view = []
    for node in compute_nodes:
        related = [x for x in candidates if x["compute"] == node]
        compute_view.append({
            "compute": node,
            "minimum_queue_ms": min(x["queue_ms"] for x in related),
            "service_ms": min(x["service_ms"] for x in related),
        })
    jobs = {
        "oran": (
            "You are the O-RAN agent in AAI-CDOS. Select one access domain from "
            "the measured radio options. Optimize hard-deadline feasibility, then "
            "link rate. Return only the required structured decision.",
            {"task": task_summary, "radio_options": radio_summary},
            choice_schema(["ground", "uav", "leo"], "access"),
            "oran_decision"),
        "compute": (
            "You are the compute agent in AAI-CDOS. Select one exact compute node. "
            "Minimize queue plus service time while respecting the hard deadline. "
            "Return only the required structured decision.",
            {"task": task_summary, "compute_nodes": compute_view},
            choice_schema(compute_nodes, "compute"),
            "compute_decision"),
        "cn": (
            "You are the core-network routing agent in AAI-CDOS. Select one candidate "
            "plan ID using path and end-to-end latency. Treat the deadline as hard. "
            "Return only the required structured decision.",
            {"task": task_summary, "candidate_routes": [
                {k: x[k] for k in ("plan_id", "access", "compute", "path_ms",
                                    "latency_ms", "deadline_met")}
                for x in candidates]},
            choice_schema([x["plan_id"] for x in candidates], "plan_id"),
            "cn_decision"),
    }
    decisions, rows = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for role, (instructions, context, schema, schema_name) in jobs.items():
            future = executor.submit(call_agent, client, event_id, role,
                                     instructions, context, schema, schema_name)
            future_map[future] = role
        for future in concurrent.futures.as_completed(future_map):
            role = future_map[future]
            decision, row = future.result()
            decisions[role] = decision
            rows.append(row)
    return decisions, rows


def run_e2e_agent(client, event_id, task_summary, candidates, proposals,
                  max_rounds):
    plan_ids = [x["plan_id"] for x in candidates]
    schema = object_schema({
        "plan_id": {"type": "string", "enum": plan_ids},
        "expected_deadline_met": {"type": "boolean"},
        "confidence": {"type": "number"},
        "rationale_code": {"type": "string", "enum": RATIONALE_CODES},
    })
    rejected = []
    rows = []
    choice = None
    accepted = False
    feasible = [x["plan_id"] for x in candidates if x["deadline_met"]]
    for round_index in range(1, max_rounds + 1):
        context = {
            "task": task_summary,
            "domain_agent_proposals": proposals,
            "candidate_joint_plans": candidates,
            "coordination_round": round_index,
            "verifier_rejected_plan_ids": rejected,
            "verifier_feasible_plan_ids": feasible,
        }
        instructions = (
            "You are the E2E coordination agent in AAI-CDOS. Reconcile O-RAN, "
            "core-network, and compute proposals into one joint plan. The deterministic "
            "verifier enforces the hard deadline. Prefer a feasible plan with the "
            "largest deadline margin; never repeat a verifier-rejected plan. Return "
            "only the required structured decision.")
        decision, row = call_agent(client, event_id, "e2e_round_%d" % round_index,
                                   instructions, context, schema, "e2e_decision")
        rows.append(row)
        choice = decision["plan_id"]
        selected = next(x for x in candidates if x["plan_id"] == choice)
        accepted = bool(selected["deadline_met"])
        if accepted:
            break
        rejected.append(choice)
    return choice, accepted, len(rows), rejected, rows


def run_one_shot(client, event_id, task_summary, candidates):
    plan_ids = [x["plan_id"] for x in candidates]
    schema = object_schema({
        "plan_id": {"type": "string", "enum": plan_ids},
        "expected_deadline_met": {"type": "boolean"},
        "confidence": {"type": "number"},
        "rationale_code": {"type": "string", "enum": RATIONALE_CODES},
    })
    instructions = (
        "You are the One-Shot LLM baseline. Select one joint plan in a single pass "
        "without domain-agent messages, negotiation, memory, or verifier feedback. "
        "Treat the task deadline as hard. Return only the structured decision.")
    return call_agent(client, event_id, "one_shot", instructions,
                      {"task": task_summary, "candidate_joint_plans": candidates},
                      schema, "one_shot_decision")


def update_state(state, plan, radios, task, now, cfg, vehicle_index, elapsed):
    latency, parts = evaluate(plan, radios, state, task, now, cfg)
    node = plan[1]
    service_start = max(now + parts["tx"] + parts["path"], state.available[node])
    state.available[node] = service_start + parts["service"]
    success = latency <= task["deadline"]
    if success:
        state.age[vehicle_index] = latency
    else:
        state.age[vehicle_index] += elapsed
    state.routes[plan] += 1
    state.memory.append((task["class"], plan, success, latency))
    return latency, success


def run_smoke(client, output_dir):
    schema = object_schema({"status": {"type": "string", "enum": ["ok"]}})
    result = client.call_json(
        "Return the required structured health-check result.",
        {"check": "aai_cdos_responses_api"}, "health_check", schema,
        metadata={"experiment": "aai_cdos_llm_audit", "phase": "smoke"})
    safe = {k: v for k, v in result.items() if k != "result"}
    safe["result"] = result["result"]
    safe["credential_handling"] = "ephemeral_environment_variable; not persisted"
    safe["store"] = False
    write_json(os.path.join(output_dir, "connection_test.json"), safe)
    print("GPT-4o connection test: status=%s tokens=%d latency=%.2fs" %
          (safe["status"], safe["total_tokens"], safe["latency_s"]), flush=True)


def summarize_events(event_rows, call_rows, audit_cfg):
    call_by_event = defaultdict(list)
    for row in call_rows:
        call_by_event[row["event_id"]].append(row)
    for event in event_rows:
        calls = call_by_event[event["event_id"]]
        aai = [x for x in calls if x["agent_role"] != "one_shot"]
        one = [x for x in calls if x["agent_role"] == "one_shot"]
        for prefix, values in (("aai", aai), ("oneshot", one)):
            event[prefix + "_api_calls"] = len(values)
            event[prefix + "_input_tokens"] = sum(x["input_tokens"] for x in values)
            event[prefix + "_output_tokens"] = sum(x["output_tokens"] for x in values)
            event[prefix + "_sum_api_latency_ms"] = round(
                1000.0 * sum(x["latency_s"] for x in values), 3)

    summaries = []
    for vehicles in sorted(set(x["vehicles"] for x in event_rows)):
        rows = [x for x in event_rows if x["vehicles"] == vehicles]
        calls = [x for x in call_rows if x["event_id"].startswith("n%d_" % vehicles)]
        inp = sum(x["input_tokens"] for x in calls)
        out = sum(x["output_tokens"] for x in calls)
        cost = (inp * float(audit_cfg["input_usd_per_million_tokens"]) +
                out * float(audit_cfg["output_usd_per_million_tokens"])) / 1e6
        summaries.append({
            "vehicles": vehicles,
            "events": len(rows),
            "api_calls": len(calls),
            "aai_verifier_acceptance_rate": float(np.mean(
                [x["aai_verifier_accepted"] for x in rows])),
            "aai_selected_deadline_rate": float(np.mean(
                [x["aai_deadline_met"] for x in rows])),
            "oneshot_selected_deadline_rate": float(np.mean(
                [x["oneshot_deadline_met"] for x in rows])),
            "mean_aai_rounds": float(np.mean([x["aai_rounds"] for x in rows])),
            "mean_aai_event_wall_ms": float(np.mean(
                [x["aai_event_wall_ms"] for x in rows])),
            "mean_aai_input_tokens": float(np.mean(
                [x["aai_input_tokens"] for x in rows])),
            "mean_aai_output_tokens": float(np.mean(
                [x["aai_output_tokens"] for x in rows])),
            "mean_oneshot_wall_ms": float(np.mean(
                [x["oneshot_wall_ms"] for x in rows])),
            "mean_oneshot_input_tokens": float(np.mean(
                [x["oneshot_input_tokens"] for x in rows])),
            "mean_oneshot_output_tokens": float(np.mean(
                [x["oneshot_output_tokens"] for x in rows])),
            "input_tokens": inp,
            "output_tokens": out,
            "estimated_standard_cost_usd": cost,
        })
    return summaries


def write_summary_md(path, summaries, total_calls, attempts, model):
    lines = [
        "# Real GPT-4o planner audit", "",
        "This is a bounded, class-stratified telemetry audit over states from the",
        "Seoul V2X replay. It does not replace the 30-seed deterministic network",
        "performance experiment. No coordinates or terminal identifiers were sent.", "",
        "Model requested: `%s`; Responses API `store=false`." % model, "",
        "| Vehicles | Events | API calls | AAI accept | One-shot feasible | AAI wall/event | Est. cost |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in summaries:
        lines.append("| %d | %d | %d | %.1f%% | %.1f%% | %.0f ms | $%.4f |" % (
            x["vehicles"], x["events"], x["api_calls"],
            100.0 * x["aai_verifier_acceptance_rate"],
            100.0 * x["oneshot_selected_deadline_rate"],
            x["mean_aai_event_wall_ms"], x["estimated_standard_cost_usd"]))
    lines += ["", "Completed calls: %d; HTTP request attempts (including retries): %d." %
              (total_calls, attempts), "",
              "Cost is estimated from returned token usage and the prices recorded in",
              "`config/llm_audit.yaml`; billing records remain authoritative.", "",
              "The measured cloud-API planner wall time is longer than the 0.15--1.0 s",
              "application deadlines. These measurements therefore do not support a",
              "synchronous per-event cloud-control claim; deployment would require",
              "asynchronous/precomputed decisions or a substantially lower-latency",
              "serving path/model."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_latex(path, summaries, model):
    lines = [
        "% Auto-generated by scripts/run_llm_audit.py",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Bounded real-%s planner telemetry audit.}" % model.replace("_", "\\_"),
        "  \\label{tab:llm_audit}",
        "  \\begin{tabular}{c|rrrr}",
        "    \\hline",
        "    Vehicles & Events & AAI calls/event & AAI tokens/event & AAI wall (ms) \\\\",
        "    \\hline",
    ]
    for x in summaries:
        aai_calls = 3.0 + x["mean_aai_rounds"]
        aai_tokens = x["mean_aai_input_tokens"] + x["mean_aai_output_tokens"]
        lines.append("    %d & %d & %.2f & %.1f & %.1f \\\\" % (
            x["vehicles"], x["events"], aai_calls, aai_tokens,
            x["mean_aai_event_wall_ms"]))
    lines += ["    \\hline", "  \\end{tabular}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/simulation.yaml")
    parser.add_argument("--audit-config", default="config/llm_audit.yaml")
    parser.add_argument("--output")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--events-per-vehicle-seed", type=int)
    args = parser.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    audit_path = (args.audit_config if os.path.isabs(args.audit_config) else
                  os.path.join(ROOT, args.audit_config))
    cfg = load_yaml(cfg_path)
    audit_cfg = load_yaml(audit_path)["llm_audit"]
    output_dir = args.output or audit_cfg["output_dir"]
    output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    client = OpenAIResponsesClient(
        model=audit_cfg["model"], timeout_s=audit_cfg["timeout_s"],
        max_retries=audit_cfg["max_retries"],
        max_total_requests=audit_cfg["max_total_api_requests"],
        max_output_tokens=audit_cfg["max_output_tokens"])
    if args.smoke:
        run_smoke(client, output_dir)
        return

    runs = int(args.runs or audit_cfg["independent_seeds"])
    events_per = int(args.events_per_vehicle_seed or
                     audit_cfg["events_per_vehicle_seed"])
    frames = int(audit_cfg["trace_frames"])
    data_path = cfg["experiment"]["dataset_path"]
    data_path = data_path if os.path.isabs(data_path) else os.path.join(ROOT, data_path)
    event_rows, call_rows = [], []
    started_all = time.time()
    seed0 = int(cfg["experiment"]["seed_start"])
    priority = {"safety": 0, "cooperative": 1, "routine": 2}

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
            selected = sample_event_indices(tasks, events_per, seed + vehicles)
            last_now = 0.0
            sample_number = 0
            for ti in range(len(trace["times"])):
                now = float(trace["times"][ti])
                elapsed = max(0.0, now - last_now)
                order = sorted(range(vehicles),
                               key=lambda j: (priority[str(tasks["class"][ti, j])], j))
                for j in order:
                    task = task_at(tasks, ti, j)
                    multiplier = float(cfg["orchestration"]["bandwidth_multiplier"][task["class"]])
                    radios = radio_options(now, trace["xy"][ti, j], task, vehicles,
                                           infra, cfg, multiplier)
                    if (ti, j) not in selected:
                        plan, _, _, _, _ = choose_plan(
                            "AAI-CDOS", radios, state, task, now, cfg,
                            tasks["oneshot_noise"][ti, j])
                        update_state(state, plan, radios, task, now, cfg, j, elapsed)
                        continue

                    sample_number += 1
                    event_id = "n%d_s%d_e%d" % (vehicles, seed, sample_number)
                    task_summary, radio_summary, candidates, plan_map = plan_context(
                        radios, state, task, now, cfg)
                    event_started = time.perf_counter()
                    proposals, domain_rows = run_domain_agents(
                        client, int(audit_cfg["parallel_domain_agents"]), event_id,
                        task_summary, radio_summary, candidates, state)
                    aai_choice, accepted, rounds, rejected, e2e_rows = run_e2e_agent(
                        client, event_id, task_summary, candidates, proposals,
                        int(cfg["experiment"]["max_negotiation_rounds"]))
                    aai_wall_ms = 1000.0 * (time.perf_counter() - event_started)
                    call_rows.extend(domain_rows + e2e_rows)

                    fallback = False
                    if not accepted:
                        fallback = True
                        aai_choice = min(candidates, key=lambda x: x["latency_ms"])["plan_id"]
                    aai_candidate = next(x for x in candidates
                                         if x["plan_id"] == aai_choice)

                    one_started = time.perf_counter()
                    one_decision, one_row = run_one_shot(
                        client, event_id, task_summary, candidates)
                    one_wall_ms = 1000.0 * (time.perf_counter() - one_started)
                    call_rows.append(one_row)
                    one_candidate = next(x for x in candidates
                                         if x["plan_id"] == one_decision["plan_id"])

                    chosen_plan = plan_map[aai_choice]
                    actual_latency, actual_success = update_state(
                        state, chosen_plan, radios, task, now, cfg, j, elapsed)
                    event_rows.append({
                        "event_id": event_id,
                        "vehicles": vehicles,
                        "seed": seed,
                        "frame": ti,
                        "task_class": task["class"],
                        "deadline_ms": round(1000.0 * task["deadline"], 3),
                        "aai_plan_id": aai_choice,
                        "aai_access": chosen_plan[0],
                        "aai_compute": chosen_plan[1],
                        "aai_estimated_latency_ms": aai_candidate["latency_ms"],
                        "aai_deadline_met": bool(actual_success),
                        "aai_verifier_accepted": bool(accepted),
                        "aai_rounds": rounds,
                        "aai_rejections": len(rejected),
                        "aai_fallback": fallback,
                        "aai_event_wall_ms": round(aai_wall_ms, 3),
                        "oneshot_plan_id": one_decision["plan_id"],
                        "oneshot_access": one_candidate["access"],
                        "oneshot_compute": one_candidate["compute"],
                        "oneshot_estimated_latency_ms": one_candidate["latency_ms"],
                        "oneshot_deadline_met": bool(one_candidate["deadline_met"]),
                        "oneshot_wall_ms": round(one_wall_ms, 3),
                    })
                    print("event=%s class=%s rounds=%d accepted=%s calls=%d" %
                          (event_id, task["class"], rounds, accepted,
                           client.request_attempts), flush=True)
                    write_csv(os.path.join(output_dir, "events.partial.csv"), event_rows)
                    with open(os.path.join(output_dir, "calls.partial.jsonl"), "w") as f:
                        for row in call_rows:
                            f.write(json.dumps(row, sort_keys=True) + "\n")
                last_now = now

    summaries = summarize_events(event_rows, call_rows, audit_cfg)
    write_csv(os.path.join(output_dir, "events.csv"), event_rows)
    write_csv(os.path.join(output_dir, "summary.csv"), summaries)
    with open(os.path.join(output_dir, "calls.jsonl"), "w") as f:
        for row in call_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    write_summary_md(os.path.join(output_dir, "README.md"), summaries,
                     len(call_rows), client.request_attempts, audit_cfg["model"])
    write_latex(os.path.join(ROOT, "paper", "llm_audit_table.tex"),
                summaries, audit_cfg["model"])
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "audit_type": "bounded class-stratified real-model telemetry audit",
        "not_a_replacement_for": "30-seed deterministic network-performance experiment",
        "requested_model": audit_cfg["model"],
        "returned_models": sorted(set(x["returned_model"] for x in call_rows)),
        "responses_api_store": False,
        "credential_handling": "ephemeral OPENAI_API_KEY environment variable; not persisted",
        "data_sent": "task class/size/deadline and simulated radio/route/queue/compute metrics",
        "data_not_sent": "vehicle coordinates, trace timestamps, terminal identifiers, API key",
        "dataset_path": os.path.realpath(data_path),
        "dataset_sha256": sha256_file(data_path),
        "vehicle_counts": [int(x) for x in cfg["experiment"]["vehicle_counts"]],
        "independent_seeds": runs,
        "events_per_vehicle_seed": events_per,
        "completed_events": len(event_rows),
        "completed_api_calls": len(call_rows),
        "http_request_attempts_including_retries": client.request_attempts,
        "elapsed_s": time.time() - started_all,
        "audit_config": audit_cfg,
    }
    write_json(os.path.join(output_dir, "run_manifest.json"), manifest)
    for name in ("events.partial.csv", "calls.partial.jsonl"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)
    print("done: events=%d calls=%d attempts=%d elapsed=%.1fs" %
          (len(event_rows), len(call_rows), client.request_attempts,
           time.time() - started_all), flush=True)


if __name__ == "__main__":
    main()
