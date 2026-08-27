#!/usr/bin/env python3
"""Run a same-state, real-model AAI-CDOS decision experiment.

Unlike ``run_llm_audit.py``, this experiment executes every compared planner
with the configured OpenAI model.  AAI-CDOS performs domain observation,
parallel domain planning, E2E coordination, deterministic verification,
feedback-driven replanning, execution accounting, and bounded memory updates.
The three baselines also use real model calls but omit the components stated in
their definitions.

The Seoul trace supplies mobility.  Radio, routing, queue, compute, and task
states remain simulated.  All four schemes are evaluated counterfactually on
the same pre-decision simulator state.  Cloud planner wall time is reported
separately and is never added to simulated DT-update latency.
"""

import argparse
import concurrent.futures
import csv
import getpass
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
    "path_efficiency", "proposal_consensus", "verifier_feedback",
    "memory_guided", "no_feasible_plan",
]
SCHEMES = ("AAI-CDOS", "Independent Agents", "One-Shot LLM", "Single Domain")


def load_yaml(path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with open(path, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def write_jsonl(path, rows):
    with open(path, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def task_at(tasks, ti, vehicle):
    return {
        "class": str(tasks["class"][ti, vehicle]),
        "deadline": float(tasks["deadline"][ti, vehicle]),
        "bits": float(tasks["bits"][ti, vehicle]),
        "tx_w": float(tasks["tx_w"][ti, vehicle]),
        "shadow_ground": float(tasks["shadow_ground"][ti, vehicle]),
        "fading_uav": float(tasks["fading_uav"][ti, vehicle]),
        "fading_leo": float(tasks["fading_leo"][ti, vehicle]),
        "route_jitter": float(tasks["route_jitter"][ti, vehicle]),
        "oneshot_random": float(tasks["oneshot_random"][ti, vehicle]),
        "satellite_tx_w": float(tasks["satellite_tx_w"][ti, vehicle]),
    }


def sample_event_indices(tasks, count, seed):
    """Return a deterministic, task-class-stratified event sample."""
    rng = np.random.default_rng(seed + 91021)
    labels = tasks["class"]
    classes = ("safety", "cooperative", "routine")
    pools = {}
    for task_class in classes:
        pools[task_class] = [tuple(x) for x in np.argwhere(labels == task_class)]
        rng.shuffle(pools[task_class])
    selected = []
    offsets = defaultdict(int)
    for index in range(count):
        task_class = classes[index % len(classes)]
        if offsets[task_class] < len(pools[task_class]):
            selected.append(pools[task_class][offsets[task_class]])
            offsets[task_class] += 1
    if len(selected) < count:
        used = set(selected)
        remaining = [tuple(x) for x in np.argwhere(np.ones_like(labels, dtype=bool))
                     if tuple(x) not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[:count - len(selected)])
    return set(selected)


def choice_schema(choices, field):
    return object_schema({
        field: {"type": "string", "enum": list(choices)},
        "confidence": {"type": "number"},
        "rationale_code": {"type": "string", "enum": RATIONALE_CODES},
    })


class MockResponsesClient(object):
    """Schema-respecting local planner used only for pipeline tests."""

    def __init__(self):
        self.request_attempts = 0

    def call_json(self, instructions, input_object, schema_name, schema,
                  metadata=None):
        self.request_attempts += 1
        decision = {}
        for name, spec in schema["properties"].items():
            if "enum" in spec:
                decision[name] = spec["enum"][0]
            elif spec.get("type") == "boolean":
                decision[name] = True
            elif spec.get("type") == "number":
                decision[name] = 0.9
            else:
                decision[name] = "mock"
        return {
            "result": decision,
            "response_id": "mock-response-%d" % self.request_attempts,
            "request_id": "mock-request-%d" % self.request_attempts,
            "returned_model": "mock-schema-planner",
            "status": "completed",
            "service_tier": "local",
            "latency_s": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "retry_index": 0,
        }


def recorded_call(client, event_id, scheme, role, round_index, instructions,
                  context, schema, schema_name):
    result = client.call_json(
        instructions, context, schema_name, schema,
        metadata={
            "experiment": "aai_cdos_actual_agentic",
            "event": event_id,
            "scheme": scheme,
            "agent_role": role,
            "round": round_index,
        })
    row = {key: value for key, value in result.items() if key != "result"}
    row.update({
        "event_id": event_id,
        "scheme": scheme,
        "agent_role": role,
        "coordination_round": round_index,
        "input": context,
        "instructions": instructions,
        "decision": result["result"],
    })
    return result["result"], row


def build_context(radios, state, task, now, cfg):
    plans = candidate_plans(radios, cfg)
    candidates = []
    plan_map = {}
    for index, plan in enumerate(plans):
        latency, parts = evaluate(plan, radios, state, task, now, cfg)
        plan_id = "p%d" % index
        candidates.append({
            "plan_id": plan_id,
            "access": plan[0],
            "compute": plan[1],
            "latency_ms": round(latency * 1000.0, 3),
            "tx_ms": round(parts["tx"] * 1000.0, 3),
            "path_ms": round(parts["path"] * 1000.0, 3),
            "queue_ms": round(parts["wait"] * 1000.0, 3),
            "service_ms": round(parts["service"] * 1000.0, 3),
            "deadline_met": bool(latency <= task["deadline"]),
        })
        plan_map[plan_id] = plan

    radios_public = {}
    for access, value in radios.items():
        radios_public[access] = {
            "rate_mbps": round(value["rate"] / 1e6, 4),
            "snr_db": round(value["snr"], 3),
            "link_distance_m": round(value["distance"], 2),
        }

    compute_nodes = sorted(set(item["compute"] for item in candidates))
    compute_public = []
    for node in compute_nodes:
        related = [item for item in candidates if item["compute"] == node]
        compute_public.append({
            "compute": node,
            "queue_ms": min(item["queue_ms"] for item in related),
            "service_ms": min(item["service_ms"] for item in related),
        })

    routes_public = [
        {key: item[key] for key in ("plan_id", "access", "compute", "path_ms")}
        for item in candidates
    ]
    task_public = {
        "class": task["class"],
        "deadline_ms": round(task["deadline"] * 1000.0, 3),
        "update_mbit": round(task["bits"] / 1e6, 5),
    }
    return task_public, radios_public, compute_public, routes_public, candidates, plan_map


def run_domain_round(client, workers, event_id, scheme, round_index, task,
                     radios, compute_nodes, routes, memory, feedback):
    def role_context(role, **values):
        role_memory = memory.get(role, []) if isinstance(memory, dict) else memory
        context = {
            "task": task,
            "recent_memory": role_memory,
            "verifier_feedback": feedback,
            "coordination_round": round_index,
        }
        context.update(values)
        return context

    jobs = {
        "oran": (
            "You are the O-RAN domain agent. Use only the service intent, radio "
            "observations, bounded memory, and verifier feedback provided to you. "
            "Select an access domain. For a revision round, change the previous "
            "choice when verifier feedback shows that transmission is the blocking "
            "component. Return only the structured decision.",
            role_context("oran", radio_options=radios),
            choice_schema(("ground", "uav", "leo"), "access"),
            "oran_actual_decision"),
        "compute": (
            "You are the computing-infrastructure domain agent. Use only the "
            "service intent, compute observations, bounded memory, and verifier "
            "feedback provided to you. Select one compute node. For a revision "
            "round, avoid a node implicated by excessive queue or service delay. "
            "Return only the structured decision.",
            role_context("compute", compute_nodes=compute_nodes),
            choice_schema([item["compute"] for item in compute_nodes], "compute"),
            "compute_actual_decision"),
        "cn": (
            "You are the core-network routing agent. Use only the service intent, "
            "route observations, bounded memory, and verifier feedback provided to "
            "you. Select one route plan ID. For a revision round, avoid rejected "
            "plans and paths implicated by verifier feedback. Return only the "
            "structured decision.",
            role_context("cn", candidate_routes=routes),
            choice_schema([item["plan_id"] for item in routes], "plan_id"),
            "cn_actual_decision"),
    }
    decisions = {}
    call_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for role, (instructions, context, schema, schema_name) in jobs.items():
            future = executor.submit(
                recorded_call, client, event_id, scheme, role, round_index,
                instructions, context, schema, schema_name)
            future_map[future] = role
        for future in concurrent.futures.as_completed(future_map):
            role = future_map[future]
            decision, row = future.result()
            decisions[role] = decision
            call_rows.append(row)
    return decisions, call_rows


def run_coordinator(client, event_id, round_index, task, proposals, candidates,
                    memory, feedback):
    plan_ids = [item["plan_id"] for item in candidates]
    schema = object_schema({
        "plan_id": {"type": "string", "enum": plan_ids},
        "expected_deadline_met": {"type": "boolean"},
        "confidence": {"type": "number"},
        "rationale_code": {"type": "string", "enum": RATIONALE_CODES},
    })
    instructions = (
        "You are the E2E service coordination agent in AAI-CDOS. Reconcile the "
        "three domain proposals into one joint plan. For each candidate, add its "
        "transmission, path, queue, and service components and treat the service "
        "deadline as hard. The deterministic verifier checks your calculation "
        "after selection. Prefer a plan consistent with the O-RAN and computing "
        "proposals unless verifier feedback requires revision. Use bounded memory "
        "and never repeat a rejected plan. Return only the structured decision.")
    context = {
        "task": task,
        "domain_agent_proposals": proposals,
        "candidate_plan_catalog": [
            {key: item[key] for key in (
                "plan_id", "access", "compute", "tx_ms", "path_ms",
                "queue_ms", "service_ms")}
            for item in candidates
        ],
        "recent_memory": memory,
        "verifier_feedback": feedback,
        "coordination_round": round_index,
    }
    return recorded_call(
        client, event_id, "AAI-CDOS", "e2e", round_index, instructions,
        context, schema, "e2e_actual_decision")


def verifier_feedback(candidate, deadline_ms, rejected_ids):
    return {
        "checked_plan_id": candidate["plan_id"],
        "accepted": bool(candidate["deadline_met"]),
        "estimated_dt_update_latency_ms": candidate["latency_ms"],
        "hard_deadline_ms": deadline_ms,
        "deadline_violation_ms": round(
            max(0.0, candidate["latency_ms"] - deadline_ms), 3),
        "latency_breakdown_ms": {
            "transmission": candidate["tx_ms"],
            "path": candidate["path_ms"],
            "queue": candidate["queue_ms"],
            "service": candidate["service_ms"],
        },
        "rejected_plan_ids": list(rejected_ids),
    }


def run_aai_cdos(client, workers, max_rounds, event_id, task, radios,
                 compute_nodes, routes, candidates, memory):
    call_rows = []
    trace = []
    feedback = None
    rejected = []
    selected = None
    accepted = False
    started = time.perf_counter()
    for round_index in range(1, max_rounds + 1):
        proposals, domain_rows = run_domain_round(
            client, workers, event_id, "AAI-CDOS", round_index, task, radios,
            compute_nodes, routes, memory, feedback)
        decision, coordinator_row = run_coordinator(
            client, event_id, round_index, task, proposals, candidates, memory,
            feedback)
        call_rows.extend(domain_rows)
        call_rows.append(coordinator_row)
        selected = next(item for item in candidates
                        if item["plan_id"] == decision["plan_id"])
        if selected["deadline_met"]:
            accepted = True
            feedback = verifier_feedback(selected, task["deadline_ms"], rejected)
        else:
            rejected.append(selected["plan_id"])
            feedback = verifier_feedback(selected, task["deadline_ms"], rejected)
        feedback["previous_domain_proposals"] = proposals
        feedback["previous_coordinator_decision"] = decision
        trace.append({
            "round": round_index,
            "domain_proposals": proposals,
            "coordinator_decision": decision,
            "verifier": feedback,
        })
        if accepted:
            break

    fallback = False
    if not accepted:
        fallback = True
        selected = min(candidates, key=lambda item: item["latency_ms"])
    return {
        "candidate": selected,
        "rounds": len(trace),
        "rejections": len(rejected),
        "verifier_accepted": accepted,
        "fallback": fallback,
        "decision_wall_ms": round(1000.0 * (time.perf_counter() - started), 3),
        "tool_calls": 3 * len(trace) + len(trace) + 2,
        "trace": trace,
        "calls": call_rows,
    }


def run_single_domain(client, event_id, task, radios, candidates):
    started = time.perf_counter()
    decision, row = recorded_call(
        client, event_id, "Single Domain", "oran", 1,
        "You are a radio-domain-only controller. Select one access domain using "
        "only the service intent and radio observations. You have no E2E route, "
        "compute, memory, negotiation, or verifier information. Return only the "
        "structured decision.",
        {"task": task, "radio_options": radios},
        choice_schema(("ground", "uav", "leo"), "access"),
        "single_domain_actual_decision")
    access = decision["access"]
    matching = [item for item in candidates if item["access"] == access]
    colocated = []
    for item in matching:
        if ((access == "ground" and item["compute"].startswith("ground")) or
                (access == "uav" and item["compute"].startswith("uav")) or
                (access == "leo" and item["compute"].startswith("sat"))):
            colocated.append(item)
    selected = (colocated or matching)[0]
    return {
        "candidate": selected,
        "rounds": 1,
        "rejections": 0,
        "verifier_accepted": False,
        "fallback": False,
        "decision_wall_ms": round(1000.0 * (time.perf_counter() - started), 3),
        "tool_calls": 1,
        "trace": [{"decision": decision}],
        "calls": [row],
    }


def run_independent_agents(client, workers, event_id, task, radios,
                           compute_nodes, routes, candidates, memory):
    started = time.perf_counter()
    decisions, rows = run_domain_round(
        client, workers, event_id, "Independent Agents", 1, task, radios,
        compute_nodes, routes, memory, None)
    access = decisions["oran"]["access"]
    compute = decisions["compute"]["compute"]
    cn_plan = next(item for item in candidates
                   if item["plan_id"] == decisions["cn"]["plan_id"])
    exact = [item for item in candidates
             if item["access"] == access and item["compute"] == compute]
    proposal_conflict = not (
        cn_plan["access"] == access and cn_plan["compute"] == compute)

    # All schemes must share the same valid joint-plan action space. Independent
    # proposals are reconciled only by deterministic disagreement counting; no
    # latency, deadline, verifier, or feedback is used by this baseline.
    def disagreement(item):
        return (
            int(item["access"] != access) +
            int(item["compute"] != compute) +
            int(item["plan_id"] != decisions["cn"]["plan_id"]),
            item["plan_id"],
        )

    selected = min(candidates, key=disagreement)
    return {
        "candidate": selected,
        "cn_candidate": cn_plan,
        "proposal_conflict": proposal_conflict or not exact,
        "rounds": 1,
        "rejections": 0,
        "verifier_accepted": False,
        "fallback": False,
        "decision_wall_ms": round(1000.0 * (time.perf_counter() - started), 3),
        "tool_calls": 3,
        "trace": [{"domain_proposals": decisions}],
        "calls": rows,
    }


def run_one_shot(client, event_id, task, radios, compute_nodes, routes,
                 candidates):
    started = time.perf_counter()
    schema = object_schema({
        "plan_id": {"type": "string",
                    "enum": [item["plan_id"] for item in candidates]},
        "confidence": {"type": "number"},
        "rationale_code": {"type": "string", "enum": RATIONALE_CODES},
    })
    decision, row = recorded_call(
        client, event_id, "One-Shot LLM", "one_shot", 1,
        "You are the One-Shot LLM baseline. Select one valid joint plan ID in a "
        "single response from raw cross-domain observations. "
        "Minimize estimated DT-update latency and treat the service deadline as "
        "hard. "
        "You have no domain-agent messages, bounded memory, tool feedback, "
        "negotiation, or joint-plan verifier. Return only the structured decision.",
        {"task": task, "radio_options": radios, "compute_nodes": compute_nodes,
         "candidate_routes": routes},
        schema, "one_shot_actual_decision")
    return {
        "candidate": next(item for item in candidates
                          if item["plan_id"] == decision["plan_id"]),
        "rounds": 1,
        "rejections": 0,
        "verifier_accepted": False,
        "fallback": False,
        "decision_wall_ms": round(1000.0 * (time.perf_counter() - started), 3),
        "tool_calls": 0,
        "trace": [{"decision": decision}],
        "calls": [row],
    }


def evaluate_result(result, candidates, plan_map, radios, state, task, now, cfg):
    candidate = result.get("candidate")
    if candidate is not None:
        plan = plan_map[candidate["plan_id"]]
        latency_ms = candidate["latency_ms"]
        breakdown = {key: candidate[key] for key in
                     ("tx_ms", "path_ms", "queue_ms", "service_ms")}
    else:
        plan = tuple(result["composed_plan"])
        latency, parts = evaluate(plan, radios, state, task, now, cfg)
        latency_ms = round(latency * 1000.0, 3)
        breakdown = {
            "tx_ms": round(parts["tx"] * 1000.0, 3),
            "path_ms": round(parts["path"] * 1000.0, 3),
            "queue_ms": round(parts["wait"] * 1000.0, 3),
            "service_ms": round(parts["service"] * 1000.0, 3),
        }
    return plan, latency_ms, breakdown


def call_usage(call_rows):
    return {
        "api_calls": len(call_rows),
        "input_tokens": sum(item["input_tokens"] for item in call_rows),
        "output_tokens": sum(item["output_tokens"] for item in call_rows),
        "sum_api_latency_ms": round(1000.0 * sum(
            item["latency_s"] for item in call_rows), 3),
    }


def summarize(rows, call_rows, cfg):
    calls_by_key = defaultdict(list)
    for call in call_rows:
        calls_by_key[(call["event_id"], call["scheme"])].append(call)
    summaries = []
    for vehicles in sorted(set(row["vehicles"] for row in rows)):
        for scheme in SCHEMES:
            subset = [row for row in rows
                      if row["vehicles"] == vehicles and row["scheme"] == scheme]
            if not subset:
                continue
            calls = []
            for row in subset:
                calls.extend(calls_by_key[(row["event_id"], scheme)])
            input_tokens = sum(call["input_tokens"] for call in calls)
            output_tokens = sum(call["output_tokens"] for call in calls)
            cost = (
                input_tokens * float(cfg["input_usd_per_million_tokens"]) +
                output_tokens * float(cfg["output_usd_per_million_tokens"])) / 1e6
            latencies = np.array([row["dt_update_latency_ms"] for row in subset])
            errors = np.array([row["position_error_m"] for row in subset])
            summaries.append({
                "vehicles": vehicles,
                "scheme": scheme,
                "events": len(subset),
                "deadline_success_rate": float(np.mean(
                    [row["deadline_met"] for row in subset])),
                "mean_dt_update_latency_ms": float(np.mean(latencies)),
                "p95_dt_update_latency_ms": float(np.quantile(latencies, 0.95)),
                "mean_dt_age_ms": float(np.mean(
                    [row["dt_age_ms"] for row in subset])),
                "p95_position_error_m": float(np.quantile(errors, 0.95)),
                "mean_decision_wall_ms": float(np.mean(
                    [row["decision_wall_ms"] for row in subset])),
                "mean_rounds": float(np.mean([row["rounds"] for row in subset])),
                "mean_rejections": float(np.mean(
                    [row["rejections"] for row in subset])),
                "fallback_rate": float(np.mean(
                    [row["fallback"] for row in subset])),
                "proposal_conflict_rate": float(np.mean(
                    [row["proposal_conflict"] for row in subset])),
                "api_calls": len(calls),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_standard_cost_usd": cost,
            })
    return summaries


def write_readme(path, summaries, manifest):
    lines = [
        "# Actual-model agentic experiment", "",
        "Every compared planner in this directory used the returned model shown",
        "in `run_manifest.json`. All schemes were evaluated on the same sampled",
        "pre-decision simulator state. Seoul data supplies mobility only; radio,",
        "routing, queue, compute, and task states are simulated.", "",
        "| Vehicles | Scheme | Events | Deadline success | DT latency | Planner wall | Calls | Cost |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {vehicles} | {scheme} | {events} | {success:.1f}% | "
            "{latency:.1f} ms | {wall:.1f} ms | {calls} | ${cost:.4f} |".format(
                vehicles=item["vehicles"], scheme=item["scheme"],
                events=item["events"],
                success=100.0 * item["deadline_success_rate"],
                latency=item["mean_dt_update_latency_ms"],
                wall=item["mean_decision_wall_ms"], calls=item["api_calls"],
                cost=item["estimated_standard_cost_usd"]))
    lines += [
        "", "Important evidence boundary:", "",
        "- `dt_update_latency_ms` excludes cloud planner wall time.",
        "- Planner wall time is measured separately and must be disclosed.",
        "- The experiment is a same-state counterfactual decision benchmark, not",
        "  a packet-level physical testbed deployment.",
        "- Natural-language figure copy may paraphrase structured decisions, but",
        "  must not be presented as verbatim model output.", "",
        "Completed events: %d; completed model calls: %d; request attempts: %d." %
        (manifest["completed_events"], manifest["completed_api_calls"],
         manifest["http_request_attempts_including_retries"]),
    ]
    with open(path, "w") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/simulation.yaml")
    parser.add_argument("--agent-config", default="config/actual_agentic.yaml")
    parser.add_argument("--output")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--events-per-vehicle-seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prompt-api-key", action="store_true",
        help="read the API key without echo and keep it only in this process")
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    agent_path = (args.agent_config if os.path.isabs(args.agent_config) else
                  os.path.join(ROOT, args.agent_config))
    sim_cfg = load_yaml(config_path)
    agent_cfg = load_yaml(agent_path)["actual_agentic"]
    output_dir = args.output or agent_cfg["output_dir"]
    output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.dry_run:
        client = MockResponsesClient()
    else:
        if args.prompt_api_key:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
        client = OpenAIResponsesClient(
            model=agent_cfg["model"], timeout_s=agent_cfg["timeout_s"],
            max_retries=agent_cfg["max_retries"],
            max_total_requests=agent_cfg["max_total_api_requests"],
            max_output_tokens=agent_cfg["max_output_tokens"])
        if args.prompt_api_key:
            os.environ.pop("OPENAI_API_KEY", None)

    runs = int(args.runs or agent_cfg["independent_seeds"])
    events_per = int(args.events_per_vehicle_seed or
                     agent_cfg["events_per_vehicle_seed"])
    frames = int(agent_cfg["trace_frames"])
    max_rounds = int(agent_cfg["max_negotiation_rounds"])
    memory_limit = int(agent_cfg["memory_events"])
    workers = int(agent_cfg["parallel_domain_agents"])
    dataset_path = sim_cfg["experiment"]["dataset_path"]
    dataset_path = (dataset_path if os.path.isabs(dataset_path) else
                    os.path.join(ROOT, dataset_path))

    event_rows = []
    call_rows = []
    event_traces = []
    seed0 = int(sim_cfg["experiment"]["seed_start"])
    started_all = time.time()
    priority = {"safety": 0, "cooperative": 1, "routine": 2}

    for vehicles_value in sim_cfg["experiment"]["vehicle_counts"]:
        vehicles = int(vehicles_value)
        for run_index in range(runs):
            seed = seed0 + run_index
            trace = load_trace(
                dataset_path, vehicles, seed, frames,
                float(sim_cfg["experiment"]["minimum_vehicle_coverage"]),
                sim_cfg["experiment"].get("region_bbox_lonlat"))
            tasks = build_tasks(trace, sim_cfg, seed)
            infrastructure = make_infrastructure(trace, sim_cfg)
            background_state = PolicyState(sim_cfg, vehicles)
            selected_indices = sample_event_indices(tasks, events_per, seed + vehicles)
            aai_memory = []
            independent_memory = {"oran": [], "compute": [], "cn": []}
            sample_number = 0
            last_now = 0.0

            for ti in range(len(trace["times"])):
                now = float(trace["times"][ti])
                elapsed = max(0.0, now - last_now)
                order = sorted(range(vehicles), key=lambda vehicle: (
                    priority[str(tasks["class"][ti, vehicle])], vehicle))
                for vehicle in order:
                    task = task_at(tasks, ti, vehicle)
                    multiplier = float(sim_cfg["orchestration"]["bandwidth_multiplier"][task["class"]])
                    radios = radio_options(
                        now, trace["xy"][ti, vehicle], task, vehicles,
                        infrastructure, sim_cfg, multiplier)
                    is_selected = (ti, vehicle) in selected_indices
                    if not is_selected:
                        background_plan, _, _, _, _ = choose_plan(
                            "AAI-CDOS", radios, background_state, task, now,
                            sim_cfg, tasks["oneshot_noise"][ti, vehicle])
                        latency, parts = evaluate(
                            background_plan, radios, background_state, task, now,
                            sim_cfg)
                        node = background_plan[1]
                        service_start = max(
                            now + parts["tx"] + parts["path"],
                            background_state.available[node])
                        background_state.available[node] = service_start + parts["service"]
                        success = latency <= task["deadline"]
                        if success:
                            background_state.age[vehicle] = latency
                        else:
                            background_state.age[vehicle] += elapsed
                        continue

                    sample_number += 1
                    event_id = "n%d_s%d_e%d" % (vehicles, seed, sample_number)
                    (task_public, radios_public, compute_public, routes_public,
                     candidates, plan_map) = build_context(
                        radios, background_state, task, now, sim_cfg)

                    results = {}
                    results["AAI-CDOS"] = run_aai_cdos(
                        client, workers, max_rounds, event_id, task_public,
                        radios_public, compute_public, routes_public, candidates,
                        aai_memory[-memory_limit:])
                    results["Independent Agents"] = run_independent_agents(
                        client, workers, event_id, task_public, radios_public,
                        compute_public, routes_public, candidates,
                        {role: values[-memory_limit:]
                         for role, values in independent_memory.items()})
                    results["One-Shot LLM"] = run_one_shot(
                        client, event_id, task_public, radios_public,
                        compute_public, routes_public, candidates)
                    results["Single Domain"] = run_single_domain(
                        client, event_id, task_public, radios_public, candidates)

                    for scheme in SCHEMES:
                        result = results[scheme]
                        plan, latency_ms, breakdown = evaluate_result(
                            result, candidates, plan_map, radios,
                            background_state, task, now, sim_cfg)
                        deadline_met = latency_ms <= task_public["deadline_ms"]
                        if deadline_met:
                            dt_age_s = latency_ms / 1000.0
                        else:
                            dt_age_s = float(background_state.age[vehicle]) + elapsed
                        position_error = float(trace["speed"][ti, vehicle]) * dt_age_s
                        usage = call_usage(result["calls"])
                        event_rows.append({
                            "event_id": event_id,
                            "vehicles": vehicles,
                            "seed": seed,
                            "frame": ti,
                            "task_class": task_public["class"],
                            "scheme": scheme,
                            "access": plan[0],
                            "compute": plan[1],
                            "deadline_ms": task_public["deadline_ms"],
                            "dt_update_latency_ms": latency_ms,
                            "tx_ms": breakdown["tx_ms"],
                            "path_ms": breakdown["path_ms"],
                            "queue_ms": breakdown["queue_ms"],
                            "service_ms": breakdown["service_ms"],
                            "deadline_met": bool(deadline_met),
                            "dt_age_ms": round(dt_age_s * 1000.0, 3),
                            "position_error_m": round(position_error, 6),
                            "rounds": result["rounds"],
                            "rejections": result["rejections"],
                            "verifier_accepted": result["verifier_accepted"],
                            "fallback": result["fallback"],
                            "proposal_conflict": bool(result.get(
                                "proposal_conflict", False)),
                            "decision_wall_ms": result["decision_wall_ms"],
                            "tool_calls": result["tool_calls"],
                            "api_calls": usage["api_calls"],
                            "input_tokens": usage["input_tokens"],
                            "output_tokens": usage["output_tokens"],
                            "sum_api_latency_ms": usage["sum_api_latency_ms"],
                        })
                        call_rows.extend(result["calls"])

                    aai_result = results["AAI-CDOS"]
                    aai_candidate = aai_result["candidate"]
                    aai_memory.append({
                        "task_class": task_public["class"],
                        "selected_plan_id": aai_candidate["plan_id"],
                        "access": aai_candidate["access"],
                        "compute": aai_candidate["compute"],
                        "latency_ms": aai_candidate["latency_ms"],
                        "deadline_met": bool(
                            aai_candidate["latency_ms"] <= task_public["deadline_ms"]),
                    })
                    independent_proposals = results["Independent Agents"]["trace"][0][
                        "domain_proposals"]
                    independent_row = next(
                        row for row in reversed(event_rows)
                        if row["event_id"] == event_id and
                        row["scheme"] == "Independent Agents")
                    independent_memory["oran"].append({
                        "task_class": task_public["class"],
                        "access": independent_proposals["oran"]["access"],
                        "deadline_met": independent_row["deadline_met"],
                    })
                    independent_memory["compute"].append({
                        "task_class": task_public["class"],
                        "compute": independent_proposals["compute"]["compute"],
                        "deadline_met": independent_row["deadline_met"],
                    })
                    independent_memory["cn"].append({
                        "task_class": task_public["class"],
                        "plan_id": independent_proposals["cn"]["plan_id"],
                        "deadline_met": independent_row["deadline_met"],
                    })
                    event_traces.append({
                        "event_id": event_id,
                        "vehicles": vehicles,
                        "seed": seed,
                        "frame": ti,
                        "task": task_public,
                        "radio_observation": radios_public,
                        "compute_observation": compute_public,
                        "route_observation": routes_public,
                        "schemes": {
                            scheme: results[scheme]["trace"] for scheme in SCHEMES
                        },
                    })

                    # Advance the common background with the same deterministic
                    # plan regardless of the four counterfactual model decisions.
                    background_plan, _, _, _, _ = choose_plan(
                        "AAI-CDOS", radios, background_state, task, now, sim_cfg,
                        tasks["oneshot_noise"][ti, vehicle])
                    latency, parts = evaluate(
                        background_plan, radios, background_state, task, now,
                        sim_cfg)
                    node = background_plan[1]
                    service_start = max(
                        now + parts["tx"] + parts["path"],
                        background_state.available[node])
                    background_state.available[node] = service_start + parts["service"]
                    if latency <= task["deadline"]:
                        background_state.age[vehicle] = latency
                    else:
                        background_state.age[vehicle] += elapsed

                    print(
                        "event=%s aai_rounds=%d accepted=%s calls=%d" %
                        (event_id, aai_result["rounds"],
                         aai_result["verifier_accepted"], client.request_attempts),
                        flush=True)
                    write_csv(os.path.join(output_dir, "events.partial.csv"), event_rows)
                    write_jsonl(os.path.join(output_dir, "calls.partial.jsonl"), call_rows)
                last_now = now

    summaries = summarize(event_rows, call_rows, agent_cfg)
    write_csv(os.path.join(output_dir, "events.csv"), event_rows)
    write_csv(os.path.join(output_dir, "summary.csv"), summaries)
    write_jsonl(os.path.join(output_dir, "calls.jsonl"), call_rows)
    write_jsonl(os.path.join(output_dir, "conversation_trace.jsonl"), event_traces)

    returned_models = sorted(set(call["returned_model"] for call in call_rows))
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "experiment_type": "same-state actual-model multi-agent decision experiment",
        "dry_run": bool(args.dry_run),
        "requested_model": "mock" if args.dry_run else agent_cfg["model"],
        "returned_models": returned_models,
        "responses_api_store": False,
        "credential_handling": "ephemeral OPENAI_API_KEY environment variable; not persisted",
        "dataset_path": os.path.realpath(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "mobility_source": "authorized Seoul V2X trace",
        "simulated_components": "radio, route, queue, compute, DT task",
        "comparison_state": "common deterministic pre-decision simulator state",
        "planner_wall_time_in_dt_latency": False,
        "vehicle_counts": [int(value) for value in sim_cfg["experiment"]["vehicle_counts"]],
        "independent_seeds": runs,
        "events_per_vehicle_seed": events_per,
        "max_negotiation_rounds": max_rounds,
        "completed_events": len(event_traces),
        "completed_scheme_evaluations": len(event_rows),
        "completed_api_calls": len(call_rows),
        "http_request_attempts_including_retries": client.request_attempts,
        "elapsed_s": time.time() - started_all,
        "agent_config": agent_cfg,
    }
    write_json(os.path.join(output_dir, "run_manifest.json"), manifest)
    write_readme(os.path.join(output_dir, "README.md"), summaries, manifest)

    for filename in ("events.partial.csv", "calls.partial.jsonl"):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            os.remove(path)
    print(
        "done: events=%d evaluations=%d calls=%d attempts=%d elapsed=%.1fs" %
        (len(event_traces), len(event_rows), len(call_rows),
         client.request_attempts, time.time() - started_all),
        flush=True)


if __name__ == "__main__":
    main()
