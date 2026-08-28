#!/usr/bin/env python3
"""Generate dedicated digital-twin outcome charts and paper artifacts."""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMES = ["AAI-CDOS", "Single Domain", "Independent Agents", "One-Shot LLM"]
LABELS = {"AAI-CDOS": "AAI-CDOS", "Single Domain": "Single Domain",
          "Independent Agents": "Independent", "One-Shot LLM": "One-Shot LLM"}
STYLES = {
    "AAI-CDOS": dict(color="#2F6FDB", marker="D", linestyle="-", linewidth=1.7),
    "Single Domain": dict(color="#F28E2B", marker="o", linestyle="-", linewidth=1.35),
    "Independent Agents": dict(color="#E15759", marker="s", linestyle="--", linewidth=1.35),
    "One-Shot LLM": dict(color="#59A14F", marker="^", linestyle=":", linewidth=1.55),
}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.2,
        "axes.labelsize": 8.4,
        "axes.titlesize": 9.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 400,
    })


def line_panel(ax, frame, metric, ci, scale, title, ylabel, log=False,
               panel_label=None, annotate_aai=False):
    for scheme in SCHEMES:
        group = frame[frame.scheme == scheme].sort_values("vehicles")
        x = group.vehicles.to_numpy(dtype=float)
        y = scale * group[metric].to_numpy(dtype=float)
        error = scale * group[ci].to_numpy(dtype=float)
        lower = y - error
        if log:
            lower = np.maximum(lower, np.maximum(y * 0.05, 1e-8))
        style = STYLES[scheme]
        ax.fill_between(x, lower, y + error, color=style["color"],
                        alpha=0.10, linewidth=0, zorder=1)
        ax.plot(x, y, label=LABELS[scheme], markersize=4.5,
                markerfacecolor="white", markeredgewidth=0.9,
                zorder=3, **style)
        if annotate_aai and scheme == "AAI-CDOS":
            for x_value, y_value in zip(x, y):
                label = "%.1f" % y_value if y_value >= 10 else "%.2f" % y_value
                ax.annotate(label, (x_value, y_value), xytext=(0, 7),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=6.8, color=style["color"])
    if title:
        ax.set_title(title, pad=4)
    ax.set_xlabel("Number of vehicles", labelpad=2)
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(frame.vehicles.unique()))
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    if panel_label:
        ax.text(0.5, -0.31, panel_label, transform=ax.transAxes,
                ha="center", va="top", fontsize=9.0)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.45,
            color="#B8B8B8", alpha=0.55)
    ax.grid(False, which="minor")
    ax.tick_params(direction="in", length=3.0, width=0.7)
    for spine in ax.spines.values():
        spine.set_color("#3E3E3E")


def environment_panel(ax, frame, metric, ci, scale, ylabel, log=False,
                      panel_label=None):
    conditions = (frame[["condition_index", "condition"]]
                  .drop_duplicates().sort_values("condition_index"))
    condition_labels = conditions.condition.tolist()
    offsets = np.linspace(-0.24, 0.24, len(SCHEMES))
    for scheme, offset in zip(SCHEMES, offsets):
        group = frame[frame.scheme == scheme].sort_values("condition_index")
        x = group.condition_index.to_numpy(dtype=float) + offset
        y = scale * group[metric].to_numpy(dtype=float)
        error = scale * group[ci].to_numpy(dtype=float)
        style = STYLES[scheme]
        ax.errorbar(
            x, y, yerr=error, label=LABELS[scheme], markersize=4.8,
            markerfacecolor="white", markeredgewidth=0.9, capsize=2.4,
            elinewidth=0.9, linewidth=0, color=style["color"],
            marker=style["marker"], zorder=3)
    ax.set_xlabel("Network condition", labelpad=2)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(condition_labels)))
    ax.set_xticklabels(condition_labels, rotation=0)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    if panel_label:
        ax.text(0.5, -0.31, panel_label, transform=ax.transAxes,
                ha="center", va="top", fontsize=9.0)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.45,
            color="#B8B8B8", alpha=0.55)
    ax.grid(False, which="minor")
    ax.tick_params(direction="in", length=3.0, width=0.7)
    for spine in ax.spines.values():
        spine.set_color("#3E3E3E")


def plot_outcomes(frame, output_base):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.35))
    line_panel(axes[0, 0], frame, "deadline_success_mean",
               "deadline_success_ci95", 100.0,
               None, "Deadline satisfaction (%)",
               panel_label="(a) Deadline Satisfaction")
    line_panel(axes[0, 1], frame, "p95_latency_s_mean",
               "p95_latency_s_ci95", 1000.0,
               None, "P95 synchronization latency (ms)", log=True,
               panel_label="(b) P95 Synchronization Latency")
    line_panel(axes[1, 0], frame, "mean_dt_age_s_mean",
               "mean_dt_age_s_ci95", 1.0,
               None, "Mean DT staleness (s)", log=True,
               panel_label="(c) Mean DT Staleness")
    line_panel(axes[1, 1], frame, "p95_position_error_m_mean",
               "p95_position_error_m_ci95", 1.0,
               None, "P95 position error (m)", log=True,
               panel_label="(d) P95 Position Error", annotate_aai=True)

    axes[0, 0].set_ylim(20, 102)
    axes[0, 0].set_yticks([20, 40, 60, 80, 100])
    axes[0, 1].set_ylim(1.2e2, 4.5e4)
    axes[1, 0].set_ylim(1.2e-1, 1.2e2)
    axes[1, 1].set_ylim(8e-1, 2.2e3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.995), frameon=False,
               handlelength=2.6, columnspacing=1.6)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.105, top=0.90,
                        wspace=0.30, hspace=0.58)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_base + "." + ext, bbox_inches="tight",
                    pad_inches=0.035)
    plt.close(fig)

    chart_data = frame[[
        "scheme", "vehicles", "runs", "deadline_success_mean",
        "deadline_success_ci95", "p95_latency_s_mean",
        "p95_latency_s_ci95", "mean_dt_age_s_mean",
        "mean_dt_age_s_ci95", "p95_position_error_m_mean",
        "p95_position_error_m_ci95",
    ]].copy()
    chart_data["scheme"] = pd.Categorical(
        chart_data["scheme"], categories=SCHEMES, ordered=True)
    chart_data = chart_data.sort_values(["vehicles", "scheme"])
    chart_data["scheme"] = chart_data["scheme"].astype(str)
    chart_data = chart_data.rename(columns={
        "deadline_success_mean": "deadline_satisfaction_pct",
        "deadline_success_ci95": "deadline_satisfaction_ci95_pp",
        "p95_latency_s_mean": "p95_sync_latency_ms",
        "p95_latency_s_ci95": "p95_sync_latency_ci95_ms",
        "mean_dt_age_s_mean": "mean_dt_staleness_s",
        "mean_dt_age_s_ci95": "mean_dt_staleness_ci95_s",
        "p95_position_error_m_mean": "p95_position_error_m",
        "p95_position_error_m_ci95": "p95_position_error_ci95_m",
    })
    chart_data["deadline_satisfaction_pct"] *= 100.0
    chart_data["deadline_satisfaction_ci95_pp"] *= 100.0
    chart_data["p95_sync_latency_ms"] *= 1000.0
    chart_data["p95_sync_latency_ci95_ms"] *= 1000.0
    chart_data.to_csv(output_base + ".csv", index=False, float_format="%.6f")


def plot_environment_outcomes(load_frame, environment_frame, output_base):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.35))
    line_panel(axes[0, 0], load_frame, "deadline_success_mean",
               "deadline_success_ci95", 100.0, None,
               "Deadline satisfaction (%)",
               panel_label="(a) Load Scalability")
    line_panel(axes[0, 1], load_frame, "p95_latency_s_mean",
               "p95_latency_s_ci95", 1000.0, None,
               "P95 synchronization latency (ms)", log=True,
               panel_label="(b) Latency Scalability")
    environment_panel(axes[1, 0], environment_frame,
                      "deadline_success_mean", "deadline_success_ci95", 100.0,
                      "Deadline satisfaction (%)",
                      panel_label="(c) Robustness across Environments")
    environment_panel(axes[1, 1], environment_frame,
                      "p95_position_error_m_mean", "p95_position_error_m_ci95", 1.0,
                      "P95 position error (m)", log=True,
                      panel_label="(d) DT Accuracy across Environments")

    axes[0, 0].set_ylim(20, 102)
    axes[0, 0].set_yticks([20, 40, 60, 80, 100])
    axes[0, 1].set_ylim(1.2e2, 4.5e4)
    axes[1, 0].set_ylim(0, 105)
    error_values = environment_frame["p95_position_error_m_mean"].to_numpy(float)
    error_ci = environment_frame["p95_position_error_m_ci95"].to_numpy(float)
    lower = max(0.5, float(np.nanmin(np.maximum(error_values - error_ci, 1e-6))) / 1.5)
    upper = float(np.nanmax(error_values + error_ci)) * 1.7
    axes[1, 1].set_ylim(lower, upper)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.995), frameon=False,
               handlelength=2.6, columnspacing=1.6)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.105, top=0.90,
                        wspace=0.30, hspace=0.58)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_base + "." + ext, bbox_inches="tight",
                    pad_inches=0.035)
    plt.close(fig)

    rows = []
    top_metrics = [
        ("a", "deadline_satisfaction", "deadline_success_mean",
         "deadline_success_ci95", 100.0, "%"),
        ("b", "p95_sync_latency", "p95_latency_s_mean",
         "p95_latency_s_ci95", 1000.0, "ms"),
    ]
    for panel, metric_label, metric, ci, scale, unit in top_metrics:
        for _, row in load_frame.iterrows():
            rows.append({
                "panel": panel, "metric": metric_label, "scheme": row.scheme,
                "vehicles": int(row.vehicles), "condition": "",
                "mean": scale * row[metric], "ci95": scale * row[ci], "unit": unit,
            })
    bottom_metrics = [
        ("c", "deadline_satisfaction", "deadline_success_mean",
         "deadline_success_ci95", 100.0, "%"),
        ("d", "p95_position_error", "p95_position_error_m_mean",
         "p95_position_error_m_ci95", 1.0, "m"),
    ]
    for panel, metric_label, metric, ci, scale, unit in bottom_metrics:
        for _, row in environment_frame.iterrows():
            rows.append({
                "panel": panel, "metric": metric_label, "scheme": row.scheme,
                "vehicles": int(row.vehicles), "condition": row.condition,
                "mean": scale * row[metric], "ci95": scale * row[ci], "unit": unit,
            })
    pd.DataFrame(rows).to_csv(output_base + ".csv", index=False,
                              float_format="%.6f")


def plot_classes(frame, output_base):
    setup_style()
    classes = [("safety", "Safety (150 ms)"),
               ("cooperative", "Cooperative (350 ms)"),
               ("routine", "Routine (1 s)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.6), sharey=True)
    for ax, (cls, title) in zip(axes, classes):
        metric = "deadline_success_%s_mean" % cls
        ci = "deadline_success_%s_ci95" % cls
        line_panel(ax, frame, metric, ci, 100.0, title, "%")
        ax.set_ylim(0, 105)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.03), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    for ext in ("png", "pdf"):
        fig.savefig(output_base + "." + ext, bbox_inches="tight")
    plt.close(fig)


def write_table(path, frame):
    lines = [
        "% Auto-generated by scripts/generate_dt_results.py",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{Digital-twin synchronization outcomes (mean $\\pm$ 95\\% CI over 30 seeds).}",
        "  \\label{tab:dt_outcomes}",
        "  \\begin{tabular}{c|l|rrrr}",
        "    \\hline",
        "    Vehicles & Scheme & Deadline (\\%) & P95 latency (ms) & Mean DT age (s) & P95 error (m) \\\\",
        "    \\hline",
    ]
    for vehicles in sorted(frame.vehicles.unique()):
        block = frame[frame.vehicles == vehicles].set_index("scheme")
        for index, scheme in enumerate(SCHEMES):
            row = block.loc[scheme]
            first = str(int(vehicles)) if index == 0 else ""
            values = [
                "%.2f $\\pm$ %.2f" % (100.0 * row.deadline_success_mean,
                                        100.0 * row.deadline_success_ci95),
                "%.1f $\\pm$ %.1f" % (1000.0 * row.p95_latency_s_mean,
                                        1000.0 * row.p95_latency_s_ci95),
                "%.3f $\\pm$ %.3f" % (row.mean_dt_age_s_mean,
                                        row.mean_dt_age_s_ci95),
                "%.2f $\\pm$ %.2f" % (row.p95_position_error_m_mean,
                                        row.p95_position_error_m_ci95),
            ]
            if scheme == "AAI-CDOS":
                values = ["\\textbf{%s}" % x for x in values]
            lines.append("    %s & %s & %s \\\\" %
                         (first, scheme, " & ".join(values)))
        lines.append("    \\hline")
    lines += ["  \\end{tabular}", "\\end{table*}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def gains(frame):
    rows = []
    for vehicles in sorted(frame.vehicles.unique()):
        block = frame[frame.vehicles == vehicles].set_index("scheme")
        aai = block.loc["AAI-CDOS"]
        baselines = block.drop(index="AAI-CDOS")
        rows.append({
            "vehicles": int(vehicles),
            "deadline_gain_over_best_baseline_pp": 100.0 * (
                aai.deadline_success_mean - baselines.deadline_success_mean.max()),
            "p95_latency_reduction_vs_best_baseline_pct": 100.0 * (
                1.0 - aai.p95_latency_s_mean / baselines.p95_latency_s_mean.min()),
            "dt_age_reduction_vs_best_baseline_pct": 100.0 * (
                1.0 - aai.mean_dt_age_s_mean / baselines.mean_dt_age_s_mean.min()),
            "position_error_reduction_vs_best_baseline_pct": 100.0 * (
                1.0 - aai.p95_position_error_m_mean /
                baselines.p95_position_error_m_mean.min()),
        })
    return pd.DataFrame(rows)


def write_results(path, frame, gain_frame):
    lines = [
        "% Auto-generated by scripts/generate_dt_results.py",
        "\\paragraph{Digital-twin synchronization outcomes.}",
    ]
    for _, gain in gain_frame.iterrows():
        vehicles = int(gain.vehicles)
        aai = frame[(frame.vehicles == vehicles) &
                    (frame.scheme == "AAI-CDOS")].iloc[0]
        lines.append(
            ("With %d vehicles, AAI-CDOS attained %.2f\\%% deadline "
             "satisfaction, %.1f~ms P95 synchronization latency, %.3f~s mean "
             "DT age, and %.2f~m P95 position error. Relative to the strongest "
             "baseline for each metric, this corresponds to a %.2f percentage-"
             "point deadline gain and reductions of %.1f\\%% in P95 latency, "
             "%.1f\\%% in DT age, and %.1f\\%% in position error." %
             (vehicles, 100.0 * aai.deadline_success_mean,
              1000.0 * aai.p95_latency_s_mean, aai.mean_dt_age_s_mean,
              aai.p95_position_error_m_mean,
              gain.deadline_gain_over_best_baseline_pp,
              gain.p95_latency_reduction_vs_best_baseline_pct,
              gain.dt_age_reduction_vs_best_baseline_pct,
              gain.position_error_reduction_vs_best_baseline_pct)))
    lines.append(
        "These policy-level DT outcomes come from the deterministic 30-seed "
        "trace replay. They are kept separate from the 30-event real-GPT-4o "
        "telemetry audit and therefore do not include cloud API wall time in "
        "the network latency metric.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/full/summary.csv")
    parser.add_argument("--environment-input", default="results/environment_sweep/summary.csv")
    args = parser.parse_args()
    path = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)
    frame = pd.read_csv(path)
    errors = []
    if len(frame) != 12:
        errors.append("expected 12 scheme/load rows")
    if set(frame.scheme) != set(SCHEMES):
        errors.append("scheme set mismatch")
    if set(frame.vehicles) != {10, 25, 50}:
        errors.append("vehicle set mismatch")
    if not (frame.runs == 30).all():
        errors.append("not every row has 30 runs")
    required = [
        "deadline_success_mean", "deadline_success_ci95",
        "p95_latency_s_mean", "p95_latency_s_ci95",
        "mean_dt_age_s_mean", "mean_dt_age_s_ci95",
        "p95_position_error_m_mean", "p95_position_error_m_ci95",
        "deadline_success_safety_mean", "deadline_success_safety_ci95",
        "deadline_success_cooperative_mean", "deadline_success_cooperative_ci95",
        "deadline_success_routine_mean", "deadline_success_routine_ci95",
    ]
    if frame[required].isna().any().any():
        errors.append("NaN in required DT metrics")
    outdir = os.path.join(ROOT, "results", "dt_analysis")
    os.makedirs(outdir, exist_ok=True)
    frame[["scheme", "vehicles", "runs"] + required].to_csv(
        os.path.join(outdir, "dt_outcomes_summary.csv"), index=False)
    gain_frame = gains(frame)
    gain_frame.to_csv(os.path.join(outdir, "aai_gains.csv"), index=False)
    plot_outcomes(frame, os.path.join(ROOT, "figures", "dt_outcomes_four_scheme"))
    environment_path = (args.environment_input if os.path.isabs(args.environment_input)
                        else os.path.join(ROOT, args.environment_input))
    environment_report = {"present": os.path.exists(environment_path)}
    if os.path.exists(environment_path):
        environment_frame = pd.read_csv(environment_path)
        environment_errors = []
        if len(environment_frame) != 16:
            environment_errors.append("expected 16 scheme/condition rows")
        if set(environment_frame.scheme) != set(SCHEMES):
            environment_errors.append("environment scheme set mismatch")
        if environment_frame.condition_id.nunique() != 4:
            environment_errors.append("expected four network conditions")
        if set(environment_frame.condition_index) != {0, 1, 2, 3}:
            environment_errors.append("environment condition order mismatch")
        if set(environment_frame.vehicles) != {25}:
            environment_errors.append("environment sweep must fix 25 vehicles")
        if not (environment_frame.runs == 30).all():
            environment_errors.append("not every environment row has 30 runs")
        environment_required = [
            "deadline_success_mean", "deadline_success_ci95",
            "p95_position_error_m_mean", "p95_position_error_m_ci95",
        ]
        if environment_frame[environment_required].isna().any().any():
            environment_errors.append("NaN in environment DT metrics")
        if environment_errors:
            errors.extend(environment_errors)
        else:
            plot_environment_outcomes(
                frame, environment_frame,
                os.path.join(ROOT, "figures", "dt_outcomes_environment_four_scheme"))
        environment_report.update({
            "rows": len(environment_frame),
            "conditions": int(environment_frame.condition_id.nunique()),
            "fixed_vehicle_count": sorted(
                int(value) for value in environment_frame.vehicles.unique()),
            "runs_per_condition_scheme": sorted(
                int(value) for value in environment_frame.runs.unique()),
            "status": "passed" if not environment_errors else "failed",
            "errors": environment_errors,
        })
    plot_classes(frame, os.path.join(ROOT, "figures", "dt_deadline_by_class"))
    write_table(os.path.join(ROOT, "paper", "dt_outcomes_table.tex"), frame)
    write_results(os.path.join(ROOT, "paper", "dt_results.tex"), frame, gain_frame)
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "rows": len(frame),
        "runs_per_scheme_load": 30,
        "vehicle_counts": [10, 25, 50],
        "schemes": SCHEMES,
        "source": "deterministic 30-seed trace-driven network simulation",
        "real_gpt_api_latency_included": False,
        "environment_sweep": environment_report,
    }
    with open(os.path.join(outdir, "validation.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(gain_frame.to_string(index=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
