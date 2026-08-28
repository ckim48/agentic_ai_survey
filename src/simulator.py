import hashlib
import json
import math
import os
import platform
import socket
import time
from collections import Counter, deque

import numpy as np


C = 299792458.0
SCHEMES = ("AAI-CDOS", "Single Domain", "Independent Agents", "One-Shot LLM")


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _interp_column(a):
    x = np.arange(len(a))
    good = np.isfinite(a)
    if not good.any():
        raise ValueError("cannot interpolate an all-NaN position series")
    return np.interp(x, x[good], a[good])


def load_trace(path, num_vehicles, seed, frames, minimum_coverage, region_bbox=None):
    d = np.load(path, allow_pickle=False)
    ids, pos, times = d["ids"], d["pos"].astype(float), d["times"].astype(float)
    nonempty = np.isfinite(pos[:, :, 0]).any(axis=1)
    pos, times = pos[nonempty], times[nonempty]
    coverage = np.isfinite(pos).all(axis=2).mean(axis=0)
    eligible_mask = coverage >= minimum_coverage
    if region_bbox is not None:
        lon0, lat0, lon1, lat1 = [float(x) for x in region_bbox]
        med_lon = np.nanmedian(pos[:, :, 0], axis=0)
        med_lat = np.nanmedian(pos[:, :, 1], axis=0)
        eligible_mask &= ((med_lon >= lon0) & (med_lon <= lon1) &
                          (med_lat >= lat0) & (med_lat <= lat1))
    eligible = np.flatnonzero(eligible_mask)
    if len(eligible) < num_vehicles:
        raise ValueError("only %d vehicles meet coverage %.2f" % (len(eligible), minimum_coverage))

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(eligible, num_vehicles, replace=False))
    if frames > len(times):
        frames = len(times)
    start = int(rng.integers(0, len(times) - frames + 1))
    sl = slice(start, start + frames)
    p = pos[sl][:, chosen, :].copy()
    t = times[sl].copy()
    t -= t[0]
    for j in range(num_vehicles):
        p[:, j, 0] = _interp_column(p[:, j, 0])
        p[:, j, 1] = _interp_column(p[:, j, 1])

    ref_lon = float(np.mean(p[:, :, 0]))
    ref_lat = float(np.mean(p[:, :, 1]))
    x = (p[:, :, 0] - ref_lon) * 111320.0 * math.cos(math.radians(ref_lat))
    y = (p[:, :, 1] - ref_lat) * 110540.0
    xy = np.stack((x, y), axis=2)
    dt = np.diff(t, prepend=t[0])
    dt[0] = np.median(np.diff(t)) if len(t) > 1 else 10.0
    speed = np.zeros((len(t), num_vehicles), dtype=float)
    if len(t) > 1:
        speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=2) / np.maximum(np.diff(t)[:, None], 1e-3)
        speed[0] = speed[1]
    speed = np.clip(speed, 0.0, 45.0)
    return {
        "ids": ids[chosen], "xy": xy, "times": t, "dt": dt, "speed": speed,
        "window_start": start, "raw_frames": int(len(nonempty)),
        "nonempty_frames": int(nonempty.sum()), "eligible_vehicles": int(len(eligible)),
        "reference_lon": ref_lon, "reference_lat": ref_lat,
    }


def make_infrastructure(trace, cfg):
    xy = trace["xy"]
    lo = np.quantile(xy.reshape(-1, 2), 0.10, axis=0)
    hi = np.quantile(xy.reshape(-1, 2), 0.90, axis=0)
    center = (lo + hi) / 2.0
    bs = np.array([
        [lo[0] * 0.70 + center[0] * 0.30, center[1]],
        [hi[0] * 0.70 + center[0] * 0.30, center[1]],
    ])
    nbs = int(cfg["network"]["base_stations"])
    if nbs != 2:
        angles = np.linspace(0, 2 * np.pi, nbs, endpoint=False)
        r = max(float(np.linalg.norm(hi - lo)) * 0.25, 300.0)
        bs = center + np.stack((np.cos(angles), np.sin(angles)), axis=1) * r
    return {"center": center, "bs": bs}


def uav_positions(now, infra, cfg):
    n = int(cfg["network"]["uavs"])
    radius = float(cfg["network"]["uav_orbit_radius_m"])
    speed = float(cfg["network"]["uav_speed_mps"])
    omega = speed / max(radius, 1.0)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + omega * now
    return infra["center"] + radius * np.stack((np.cos(a), np.sin(a)), axis=1)


def build_tasks(trace, cfg, seed):
    rng = np.random.default_rng(seed + 99173)
    T, N = trace["xy"].shape[:2]
    classes = list(cfg["tasks"]["classes"])
    probs = np.array([cfg["tasks"]["classes"][k]["probability"] for k in classes], dtype=float)
    probs /= probs.sum()
    labels = rng.choice(classes, size=(T, N), p=probs)
    deadlines = np.zeros((T, N), dtype=float)
    for k in classes:
        deadlines[labels == k] = float(cfg["tasks"]["classes"][k]["deadline_s"])
    lo, hi = cfg["tasks"]["update_size_mbit"]
    bits = rng.uniform(float(lo), float(hi), size=(T, N)) * 1e6
    p0, p1 = cfg["network"]["vehicle_tx_power_w"]
    tx_w = rng.uniform(float(p0), float(p1), size=(T, N))
    sp0, sp1 = cfg["network"]["satellite_tx_power_w"]
    satellite_tx_w = rng.uniform(float(sp0), float(sp1), size=(T, N))
    shadow_ground = rng.normal(0.0, float(cfg["network"]["ground_shadowing_std_db"]), size=(T, N))
    fading_uav = rng.rayleigh(1.0, size=(T, N))
    fading_leo = rng.rayleigh(1.0, size=(T, N))
    route_jitter = rng.gamma(2.0, 0.0015, size=(T, N))
    oneshot_random = rng.random((T, N))
    oneshot_noise = rng.lognormal(0.0, 0.35, size=(T, N, 8))
    return {
        "class": labels, "deadline": deadlines, "bits": bits, "tx_w": tx_w,
        "satellite_tx_w": satellite_tx_w,
        "shadow_ground": shadow_ground, "fading_uav": fading_uav,
        "fading_leo": fading_leo, "route_jitter": route_jitter,
        "oneshot_random": oneshot_random, "oneshot_noise": oneshot_noise,
    }


class PolicyState(object):
    def __init__(self, cfg, nvehicles):
        self.available = {}
        for i in range(int(cfg["network"]["base_stations"])):
            self.available["ground%d" % i] = 0.0
        for i in range(int(cfg["network"]["uavs"])):
            self.available["uav%d" % i] = 0.0
        self.available["sat0"] = 0.0
        self.age = np.zeros(nvehicles, dtype=float)
        self.memory = deque(maxlen=int(cfg["orchestration"]["memory_events"]))
        self.routes = Counter()


def _dbm(w):
    return 10.0 * math.log10(max(w, 1e-15) * 1000.0)


def _rate(bw, tx_dbm, pathloss_db, noise_density, noise_figure, gain_db=0.0, fading=1.0):
    bw = max(float(bw), 1.0)
    noise_dbm = noise_density + 10.0 * math.log10(bw) + noise_figure
    fading_db = 20.0 * math.log10(max(float(fading), 1e-4))
    snr_db = np.clip(tx_dbm + gain_db + fading_db - pathloss_db - noise_dbm, -20.0, 35.0)
    return bw * math.log2(1.0 + 10.0 ** (snr_db / 10.0)), float(snr_db)


def radio_options(now, pos, task, nvehicles, infra, cfg, bw_multiplier=1.0):
    net = cfg["network"]
    tx_dbm = _dbm(task["tx_w"])
    nd = float(net["noise_density_dbm_hz"])
    nf = float(net["receiver_noise_figure_db"])

    dbs = infra["bs"]
    d_bs = np.linalg.norm(dbs - pos[None, :], axis=1)
    ib = int(np.argmin(d_bs))
    dkm = max(float(d_bs[ib]) / 1000.0, 0.002)
    pl_g = (140.7 + 36.7 * math.log10(dkm) + float(task["shadow_ground"]) +
            float(net.get("ground_extra_loss_db", 0.0)))
    load_g = max(1.0, nvehicles / float(len(dbs)))
    bw_g = float(net["ground_bandwidth_hz"]) / load_g * bw_multiplier
    rg, sg = _rate(bw_g, tx_dbm, pl_g, nd, nf,
                   gain_db=float(net.get("ground_receiver_gain_db", 0.0)))

    up = uav_positions(now, infra, cfg)
    horizontal = np.linalg.norm(up - pos[None, :], axis=1)
    iu = int(np.argmin(horizontal))
    h = float(net["uav_altitude_m"])
    du = math.sqrt(float(horizontal[iu]) ** 2 + h ** 2)
    f_mhz = float(net["air_carrier_hz"]) / 1e6
    pl_u = 32.45 + 20 * math.log10(max(du / 1000.0, 1e-3)) + 20 * math.log10(f_mhz)
    elevation = math.degrees(math.atan2(h, max(float(horizontal[iu]), 1.0)))
    pl_u += 2.0 if elevation >= 30.0 else 12.0
    pl_u += float(net.get("air_extra_loss_db", 0.0))
    load_u = max(1.0, nvehicles / float(len(up)))
    bw_u = float(net["air_bandwidth_hz"]) / load_u * bw_multiplier
    ru, su = _rate(bw_u, tx_dbm, pl_u, nd, nf,
                   gain_db=float(net.get("uav_receiver_gain_db", 0.0)),
                   fading=task["fading_uav"])

    ds = float(net["leo_altitude_m"])
    f_mhz_s = float(net["satellite_carrier_hz"]) / 1e6
    pl_s = (32.45 + 20 * math.log10(ds / 1000.0) +
            20 * math.log10(f_mhz_s) +
            float(net.get("satellite_extra_loss_db", 0.0)))
    bw_s = float(net["satellite_bandwidth_hz"]) / max(float(nvehicles), 1.0) * bw_multiplier
    rs, ss = _rate(bw_s, _dbm(task["satellite_tx_w"]), pl_s, nd, nf,
                   gain_db=float(net["satellite_antenna_gain_db"]), fading=task["fading_leo"])
    options = {
        "ground": {"rate": rg, "snr": sg, "index": ib, "distance": float(d_bs[ib])},
        "uav": {"rate": ru, "snr": su, "index": iu, "distance": du},
        "leo": {"rate": rs, "snr": ss, "index": 0, "distance": ds},
    }
    enabled = net.get("access_enabled", {})
    return {key: value for key, value in options.items()
            if bool(enabled.get(key, True))}


def candidate_plans(radios, cfg):
    plans = []
    if "ground" in radios:
        ib = radios["ground"]["index"]
        plans.append(("ground", "ground%d" % ib))
        for b in range(int(cfg["network"]["base_stations"])):
            if b != ib:
                plans.append(("ground", "ground%d" % b))
        plans.append(("ground", "sat0"))
    if "uav" in radios:
        iu = radios["uav"]["index"]
        plans.append(("uav", "uav%d" % iu))
        for b in range(int(cfg["network"]["base_stations"])):
            plans.append(("uav", "ground%d" % b))
        plans.append(("uav", "sat0"))
    if "leo" in radios:
        plans.append(("leo", "sat0"))
    if not plans:
        raise ValueError("at least one radio access type must be enabled")
    return plans


def node_cpu(node, cfg):
    if node.startswith("ground"):
        return float(cfg["compute"]["ground_edge_cpu_hz"])
    if node.startswith("uav"):
        return float(cfg["compute"]["uav_cpu_hz"])
    return float(cfg["compute"]["satellite_cpu_hz"])


def route_delay(access, node, task, cfg):
    jitter = float(task["route_jitter"])
    if access == "ground" and node.startswith("ground"):
        return 0.003 + jitter
    if access == "uav" and node.startswith("uav"):
        return 0.004 + jitter
    if node.startswith("ground"):
        return 0.009 + jitter
    if node.startswith("uav"):
        return 0.012 + jitter
    prop = 2.0 * float(cfg["network"]["leo_altitude_m"]) / C
    return prop + 0.018 + jitter


def evaluate(plan, radios, state, task, now, cfg, ignore_queue=False):
    access, node = plan
    rate = max(radios[access]["rate"], 1.0)
    tx = float(task["bits"]) / rate
    cycles = float(task["bits"]) * float(cfg["tasks"]["cycles_per_bit"])
    service = cycles / node_cpu(node, cfg)
    wait = 0.0 if ignore_queue else max(0.0, state.available[node] - now)
    path = route_delay(access, node, task, cfg)
    return tx + path + wait + service, {"tx": tx, "path": path, "wait": wait, "service": service}


def choose_plan(scheme, radios, state, task, now, cfg, noise):
    plans = candidate_plans(radios, cfg)
    t0 = time.perf_counter()
    rounds, rejects, tools = 1, 0, 0

    if scheme == "Single Domain":
        access = max(radios, key=lambda a: radios[a]["rate"])
        if access == "ground":
            plan = (access, "ground%d" % radios[access]["index"])
        elif access == "uav":
            plan = (access, "uav%d" % radios[access]["index"])
        else:
            plan = (access, "sat0")
        tools = 1
    elif scheme == "Independent Agents":
        access = max(radios, key=lambda a: radios[a]["rate"])
        nodes = list(state.available)
        node = min(nodes, key=lambda x: max(0.0, state.available[x] - now) +
                   float(task["bits"]) * float(cfg["tasks"]["cycles_per_bit"]) / node_cpu(x, cfg))
        plan = (access, node)
        tools = 4
    elif scheme == "One-Shot LLM":
        scores = []
        for i, p in enumerate(plans):
            lat, _ = evaluate(p, radios, state, task, now, cfg, ignore_queue=True)
            scores.append(lat * float(noise[i % len(noise)]))
        if float(task["oneshot_random"]) < float(cfg["orchestration"]["one_shot_random_plan_probability"]):
            idx = int(np.argmax(noise[:len(plans)])) % len(plans)
        else:
            idx = int(np.argmin(scores))
        plan = plans[idx]
    else:
        # Round 1 mirrors uncoordinated local proposals.
        access = max(radios, key=lambda a: radios[a]["rate"])
        nodes = list(state.available)
        node = min(nodes, key=lambda x: max(0.0, state.available[x] - now) +
                   float(task["bits"]) * float(cfg["tasks"]["cycles_per_bit"]) / node_cpu(x, cfg))
        local = (access, node)
        local_lat, _ = evaluate(local, radios, state, task, now, cfg)
        tools = 6
        if local_lat <= float(task["deadline"]):
            plan = local
        else:
            rejects = 1
            max_rounds = int(cfg["experiment"]["max_negotiation_rounds"])
            if max_rounds <= 1:
                plan = local
            else:
                rounds = 2
                ranked = sorted((evaluate(p, radios, state, task, now, cfg)[0], p) for p in plans)
                feasible = [x for x in ranked if x[0] <= float(task["deadline"])]
                plan = (feasible[0] if feasible else ranked[0])[1]
                tools += 5
                if not feasible and max_rounds > 2:
                    rounds = max_rounds
                    rejects = max(1, rounds - 1)
                    tools += 4 * max(0, rounds - 2)

    planner_ms = (time.perf_counter() - t0) * 1000.0
    return plan, rounds, rejects, tools, planner_ms


def run_scheme(scheme, trace, tasks, cfg, infra):
    T, N = trace["xy"].shape[:2]
    state = PolicyState(cfg, N)
    rows = []
    last_now = 0.0
    for ti in range(T):
        now = float(trace["times"][ti])
        elapsed = max(0.0, now - last_now)
        order = np.arange(N)
        # Stable priority ordering is a deterministic scheduling tool.
        priority = {"safety": 0, "cooperative": 1, "routine": 2}
        if scheme == "AAI-CDOS":
            order = np.array(sorted(order, key=lambda j: (priority[str(tasks["class"][ti, j])], j)))
        for j in order:
            cls = str(tasks["class"][ti, j])
            task = {
                "class": cls,
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
            mult = 1.0
            if scheme == "AAI-CDOS":
                mult = float(cfg["orchestration"]["bandwidth_multiplier"][cls])
            radios = radio_options(now, trace["xy"][ti, j], task, N, infra, cfg, mult)
            plan, rounds, rejects, tools, planner_ms = choose_plan(
                scheme, radios, state, task, now, cfg, tasks["oneshot_noise"][ti, j])
            latency, parts = evaluate(plan, radios, state, task, now, cfg)
            mismatch = 0.0
            if scheme == "Independent Agents":
                colocated = ((plan[0] == "ground" and plan[1].startswith("ground")) or
                             (plan[0] == "uav" and plan[1].startswith("uav")) or
                             (plan[0] == "leo" and plan[1].startswith("sat")))
                if not colocated:
                    # CN chose independently; reconcile a conflicting access/
                    # processing intent without the joint-plan verifier.
                    mismatch = float(cfg["orchestration"]["independent_mismatch_delay_s"])
                    latency += mismatch
            node = plan[1]
            service_start = max(now + parts["tx"] + parts["path"] + mismatch,
                                state.available[node])
            state.available[node] = service_start + parts["service"]
            success = latency <= task["deadline"]
            if success:
                state.age[j] = latency
            else:
                state.age[j] += elapsed
            pos_error = float(trace["speed"][ti, j]) * state.age[j]
            state.routes[plan] += 1
            state.memory.append((cls, plan, success, latency))

            # Proxy accounting only; no claim of GPT token/API latency.
            if scheme == "AAI-CDOS":
                input_tokens = 420 + 170 * rounds + 8 * min(len(state.memory), 32)
                output_tokens = 85 * rounds
            elif scheme == "One-Shot LLM":
                input_tokens, output_tokens = 760, 135
            else:
                input_tokens, output_tokens = 0, 0
            rows.append({
                "success": float(success), "latency": latency, "age": float(state.age[j]),
                "position_error": pos_error, "rounds": rounds, "rejections": rejects,
                "tool_calls": tools, "planner_ms_proxy": planner_ms,
                "input_tokens_proxy": input_tokens, "output_tokens_proxy": output_tokens,
                "access": plan[0], "compute": plan[1].rstrip("0123456789"),
                "task_class": cls,
            })
        last_now = now

    def arr(k):
        return np.array([r[k] for r in rows], dtype=float)

    total = float(len(rows))
    metrics = {
        "events": int(total),
        "deadline_success": float(arr("success").mean()),
        "mean_latency_s": float(arr("latency").mean()),
        "p95_latency_s": float(np.quantile(arr("latency"), 0.95)),
        "mean_dt_age_s": float(arr("age").mean()),
        "p95_position_error_m": float(np.quantile(arr("position_error"), 0.95)),
        "mean_rounds": float(arr("rounds").mean()),
        "mean_rejections": float(arr("rejections").mean()),
        "mean_tool_calls": float(arr("tool_calls").mean()),
        "mean_planner_ms_proxy": float(arr("planner_ms_proxy").mean()),
        "mean_input_tokens_proxy": float(arr("input_tokens_proxy").mean()),
        "mean_output_tokens_proxy": float(arr("output_tokens_proxy").mean()),
    }
    for cls in cfg["tasks"]["classes"]:
        mask = np.array([r["task_class"] == cls for r in rows])
        metrics["deadline_success_%s" % cls] = float(arr("success")[mask].mean())
    route_shares = []
    for (access, compute), count in sorted(state.routes.items()):
        route_shares.append({"access": access, "compute": compute.rstrip("0123456789"),
                             "count": int(count), "share": count / total})
    return metrics, route_shares


def run_one_configuration(cfg, data_path, num_vehicles, seed, frames=None):
    frames = int(frames or cfg["experiment"]["frames_per_run"])
    trace = load_trace(data_path, num_vehicles, seed, frames,
                       float(cfg["experiment"]["minimum_vehicle_coverage"]),
                       cfg["experiment"].get("region_bbox_lonlat"))
    infra = make_infrastructure(trace, cfg)
    tasks = build_tasks(trace, cfg, seed)
    results, routes = [], []
    for scheme in SCHEMES:
        metrics, shares = run_scheme(scheme, trace, tasks, cfg, infra)
        row = {"scheme": scheme, "vehicles": num_vehicles, "seed": seed,
               "frames": int(len(trace["times"])), "window_start": trace["window_start"]}
        row.update(metrics)
        results.append(row)
        for x in shares:
            y = {"scheme": scheme, "vehicles": num_vehicles, "seed": seed}
            y.update(x)
            routes.append(y)
    return results, routes, trace


def runtime_manifest(cfg, data_path):
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": platform.python_version(), "numpy": np.__version__,
        "dataset_path": os.path.realpath(data_path), "dataset_sha256": sha256_file(data_path),
        "planner_mode": cfg["experiment"]["planner_mode"],
        "disclosure": "token and planner-latency columns are deterministic/local proxies, not GPT-4o API telemetry",
        "config": cfg,
    }
