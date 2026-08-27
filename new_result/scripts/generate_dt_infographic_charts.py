#!/usr/bin/env python3
"""Generate compact DT result charts for the case-study infographic.

The plotted values are policy-level outcomes from the 30-seed deterministic
planner-surrogate experiment. They are intentionally kept separate from the
bounded real-GPT-4o process audit. ReportLab is used so the repository does not
need Matplotlib to reproduce the PDF and PNG.
"""

from math import log10
from pathlib import Path
import shutil
import subprocess

import pandas as pd
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor, white


REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO_ROOT / "results" / "dt_analysis" / "dt_outcomes_summary.csv"
OUTPUT_DIR = REPO_ROOT / "new_result" / "charts"
DATA_DIR = REPO_ROOT / "new_result" / "data"

SCHEMES = ["AAI-CDOS", "Single Domain", "Independent Agents", "One-Shot LLM"]
DISPLAY_NAMES = {
    "AAI-CDOS": "AAI-CDOS (Proposed)",
    "Single Domain": "Single Domain",
    "Independent Agents": "Independent Agents",
    "One-Shot LLM": "One-Shot (surrogate)",
}
COLORS = {
    "AAI-CDOS": HexColor("#356FD4"),
    "Single Domain": HexColor("#F28E2B"),
    "Independent Agents": HexColor("#E15759"),
    "One-Shot LLM": HexColor("#59A14F"),
}
BAR_FILLS = {
    "AAI-CDOS": HexColor("#DCE8FF"),
    "Single Domain": HexColor("#FFE7CC"),
    "Independent Agents": HexColor("#FFE0E0"),
    "One-Shot LLM": HexColor("#E2F2DE"),
}
HATCHES = {
    "AAI-CDOS": "cross",
    "Single Domain": "diag",
    "Independent Agents": "horizontal",
    "One-Shot LLM": "backdiag",
}
MARKERS = {
    "AAI-CDOS": "diamond",
    "Single Domain": "circle",
    "Independent Agents": "square",
    "One-Shot LLM": "triangle",
}
LINE_DASHES = {
    "AAI-CDOS": None,
    "Single Domain": None,
    "Independent Agents": [4, 2],
    "One-Shot LLM": [1.2, 2.0],
}

PAGE_W = 763.0
PAGE_H = 270.0
PANEL_W = 300.0
PANEL_H = 140.0
PANEL_Y = 63.0
LEFT_X = 61.0
RIGHT_X = 448.0
INK = HexColor("#242424")
GRID = HexColor("#D8D8D8")


def add_text(drawing, x, y, text, size=8, anchor="middle", angle=0, bold=False):
    label = String(
        0 if angle else x,
        0 if angle else y,
        text,
        fontName="Times-Bold" if bold else "Times-Roman",
        fontSize=size,
        fillColor=INK,
        textAnchor=anchor,
    )
    if angle:
        group = Group()
        group.add(label)
        group.translate(x, y)
        group.rotate(angle)
        drawing.add(group)
    else:
        drawing.add(label)


def marker(drawing, kind, x, y, color, size=3.1):
    if kind == "circle":
        drawing.add(Circle(x, y, size, fillColor=white, strokeColor=color, strokeWidth=0.9))
    elif kind == "square":
        drawing.add(
            Rect(x - size, y - size, 2 * size, 2 * size, fillColor=white, strokeColor=color, strokeWidth=0.9)
        )
    elif kind == "triangle":
        drawing.add(
            Polygon(
                [x, y + size * 1.15, x - size, y - size, x + size, y - size],
                fillColor=white,
                strokeColor=color,
                strokeWidth=0.9,
            )
        )
    else:
        drawing.add(
            Polygon(
                [x, y + size * 1.2, x - size, y, x, y - size * 1.2, x + size, y],
                fillColor=white,
                strokeColor=color,
                strokeWidth=0.9,
            )
        )


def hatched_bar(drawing, x, y, width, height, scheme):
    """Draw a pale bar with a compact print-safe hatch pattern."""
    if height <= 0:
        return

    color = COLORS[scheme]
    drawing.add(Rect(x, y, width, height, fillColor=BAR_FILLS[scheme], strokeColor=None))

    def diagonal(backward=False):
        spacing = 4.2
        if backward:
            offset = 0.0
            while offset <= height + width:
                u0 = max(0.0, offset - height)
                u1 = min(width, offset)
                if u1 > u0:
                    drawing.add(
                        Line(
                            x + u0,
                            y + offset - u0,
                            x + u1,
                            y + offset - u1,
                            strokeColor=color,
                            strokeWidth=0.42,
                        )
                    )
                offset += spacing
        else:
            offset = -width
            while offset <= height:
                u0 = max(0.0, -offset)
                u1 = min(width, height - offset)
                if u1 > u0:
                    drawing.add(
                        Line(
                            x + u0,
                            y + u0 + offset,
                            x + u1,
                            y + u1 + offset,
                            strokeColor=color,
                            strokeWidth=0.42,
                        )
                    )
                offset += spacing

    pattern = HATCHES[scheme]
    if pattern in {"diag", "cross"}:
        diagonal(False)
    if pattern in {"backdiag", "cross"}:
        diagonal(True)
    if pattern == "horizontal":
        offset = 3.0
        while offset < height:
            drawing.add(
                Line(x, y + offset, x + width, y + offset, strokeColor=color, strokeWidth=0.42)
            )
            offset += 4.2

    drawing.add(Rect(x, y, width, height, fillColor=None, strokeColor=color, strokeWidth=0.7))


def draw_panel(drawing, x0, panel_caption, y_label, log_y, y_min, y_max, y_ticks, rows, mean_col, ci_col):
    y0 = PANEL_Y
    w = PANEL_W
    h = PANEL_H

    def x_map(value):
        return x0 + (value - 10.0) / 40.0 * w

    def y_map(value):
        if log_y:
            return y0 + (log10(value) - log10(y_min)) / (log10(y_max) - log10(y_min)) * h
        return y0 + (value - y_min) / (y_max - y_min) * h

    for value in y_ticks:
        py = y_map(value)
        drawing.add(
            Line(
                x0,
                py,
                x0 + w,
                py,
                strokeColor=GRID,
                strokeWidth=0.45,
                strokeDashArray=[1.2, 1.8],
            )
        )
        tick = f"{int(value):,}"
        add_text(drawing, x0 - 7, py - 2.7, tick, size=8.2, anchor="end")

    for value in (10, 25, 50):
        px = x_map(value)
        drawing.add(
            Line(
                px,
                y0,
                px,
                y0 + h,
                strokeColor=GRID,
                strokeWidth=0.45,
                strokeDashArray=[1.2, 1.8],
            )
        )
        add_text(drawing, px, y0 - 13, str(value), size=8.2)

    drawing.add(Rect(x0, y0, w, h, fillColor=None, strokeColor=INK, strokeWidth=0.7))
    add_text(drawing, x0 + w / 2, y0 - 27, "Connected vehicles", size=8.6)
    add_text(drawing, x0 + w / 2, y0 - 43, panel_caption, size=9.1, bold=True)
    add_text(drawing, x0 - 43, y0 + h / 2, y_label, size=8.6, angle=90)

    for scheme in SCHEMES:
        subset = rows[rows["scheme"] == scheme].sort_values("vehicles")
        points = []
        for record in subset.to_dict("records"):
            mean = float(record[mean_col])
            ci = float(record[ci_col])
            px = x_map(float(record["vehicles"]))
            py = y_map(mean)
            lower = max(y_min, mean - ci)
            upper = min(y_max, mean + ci)
            low_y = y_map(lower)
            high_y = y_map(upper)
            drawing.add(Line(px, low_y, px, high_y, strokeColor=COLORS[scheme], strokeWidth=0.55))
            drawing.add(Line(px - 1.8, low_y, px + 1.8, low_y, strokeColor=COLORS[scheme], strokeWidth=0.55))
            drawing.add(Line(px - 1.8, high_y, px + 1.8, high_y, strokeColor=COLORS[scheme], strokeWidth=0.55))
            points.append((px, py))

        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            drawing.add(
                Line(
                    x1,
                    y1,
                    x2,
                    y2,
                    strokeColor=COLORS[scheme],
                    strokeWidth=1.2,
                    strokeDashArray=LINE_DASHES[scheme],
                )
            )
        for px, py in points:
            marker(drawing, MARKERS[scheme], px, py, COLORS[scheme])


def add_legend(drawing):
    widths = [142, 124, 142, 168]
    total = sum(widths)
    x = (PAGE_W - total) / 2
    y = 250
    for scheme, width in zip(SCHEMES, widths):
        drawing.add(
            Line(
                x,
                y,
                x + 18,
                y,
                strokeColor=COLORS[scheme],
                strokeWidth=1.2,
                strokeDashArray=LINE_DASHES[scheme],
            )
        )
        marker(drawing, MARKERS[scheme], x + 9, y, COLORS[scheme], size=2.8)
        add_text(drawing, x + 24, y - 2.8, DISPLAY_NAMES[scheme], size=8.4, anchor="start")
        x += width


def draw_bar_panel(
    drawing,
    x0,
    panel_caption,
    y_label,
    log_y,
    y_min,
    y_max,
    y_ticks,
    rows,
    mean_col,
    ci_col,
):
    y0 = PANEL_Y
    w = PANEL_W
    h = PANEL_H
    vehicles = [10, 25, 50]
    bar_width = 13.0
    gap = 2.0
    group_width = len(SCHEMES) * bar_width + (len(SCHEMES) - 1) * gap

    def y_map(value):
        if log_y:
            return y0 + (log10(value) - log10(y_min)) / (log10(y_max) - log10(y_min)) * h
        return y0 + (value - y_min) / (y_max - y_min) * h

    for value in y_ticks:
        py = y_map(value)
        drawing.add(
            Line(
                x0,
                py,
                x0 + w,
                py,
                strokeColor=GRID,
                strokeWidth=0.45,
                strokeDashArray=[1.2, 1.8],
            )
        )
        add_text(drawing, x0 - 7, py - 2.7, f"{int(value):,}", size=8.2, anchor="end")

    baseline = y_map(y_min)
    for group_index, vehicle_count in enumerate(vehicles):
        center_x = x0 + w * (group_index + 0.5) / len(vehicles)
        group_x = center_x - group_width / 2
        add_text(drawing, center_x, y0 - 13, str(vehicle_count), size=8.2)

        for scheme_index, scheme in enumerate(SCHEMES):
            record = rows[
                (rows["scheme"] == scheme) & (rows["vehicles"] == vehicle_count)
            ].iloc[0]
            mean = float(record[mean_col])
            ci = float(record[ci_col])
            bar_x = group_x + scheme_index * (bar_width + gap)
            bar_top = y_map(mean)
            hatched_bar(drawing, bar_x, baseline, bar_width, bar_top - baseline, scheme)

            low_y = y_map(max(y_min, mean - ci))
            high_y = y_map(min(y_max, mean + ci))
            center_bar = bar_x + bar_width / 2
            drawing.add(Line(center_bar, low_y, center_bar, high_y, strokeColor=INK, strokeWidth=0.55))
            drawing.add(Line(center_bar - 1.8, low_y, center_bar + 1.8, low_y, strokeColor=INK, strokeWidth=0.55))
            drawing.add(Line(center_bar - 1.8, high_y, center_bar + 1.8, high_y, strokeColor=INK, strokeWidth=0.55))

    drawing.add(Rect(x0, y0, w, h, fillColor=None, strokeColor=INK, strokeWidth=0.7))
    add_text(drawing, x0 + w / 2, y0 - 27, "Connected vehicles", size=8.6)
    add_text(drawing, x0 + w / 2, y0 - 43, panel_caption, size=9.1, bold=True)
    add_text(drawing, x0 - 43, y0 + h / 2, y_label, size=8.6, angle=90)


def add_bar_legend(drawing):
    widths = [142, 124, 142, 168]
    total = sum(widths)
    x = (PAGE_W - total) / 2
    y = 246
    for scheme, width in zip(SCHEMES, widths):
        hatched_bar(drawing, x, y - 3.5, 16, 8, scheme)
        add_text(drawing, x + 22, y - 2.7, DISPLAY_NAMES[scheme], size=8.4, anchor="start")
        x += width


def export_drawing(drawing, stem):
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_prefix = OUTPUT_DIR / stem
    renderPDF.drawToFile(drawing, str(pdf_path))

    pdftocairo = shutil.which("pdftocairo")
    if pdftocairo is None:
        raise RuntimeError("pdftocairo is required to render the PNG preview")
    completed = subprocess.run(
        [pdftocairo, "-png", "-singlefile", "-r", "300", str(pdf_path), str(png_prefix)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    png_path = png_prefix.with_suffix(".png")
    if not png_path.exists() or png_path.stat().st_size < 1000:
        raise RuntimeError(f"pdftocairo failed ({completed.returncode}): {completed.stderr}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(SUMMARY_PATH)
    data = source[source["scheme"].isin(SCHEMES)].copy()
    data["deadline_success_pct"] = 100.0 * data["deadline_success_mean"]
    data["deadline_success_ci95_pct"] = 100.0 * data["deadline_success_ci95"]
    data = data[
        [
            "scheme",
            "vehicles",
            "runs",
            "deadline_success_pct",
            "deadline_success_ci95_pct",
            "p95_position_error_m_mean",
            "p95_position_error_m_ci95",
        ]
    ].sort_values(["scheme", "vehicles"])
    data.to_csv(DATA_DIR / "dt_infographic_metrics.csv", index=False)

    drawing = Drawing(PAGE_W, PAGE_H)
    add_legend(drawing)
    draw_panel(
        drawing,
        LEFT_X,
        "(a) Deadline satisfaction",
        "Deadline satisfaction (%)",
        False,
        20,
        102,
        [20, 40, 60, 80, 100],
        data,
        "deadline_success_pct",
        "deadline_success_ci95_pct",
    )
    draw_panel(
        drawing,
        RIGHT_X,
        "(b) DT synchronization error",
        "Position error (m, log scale)",
        True,
        1,
        2200,
        [1, 10, 100, 1000],
        data,
        "p95_position_error_m_mean",
        "p95_position_error_m_ci95",
    )
    add_text(
        drawing,
        PAGE_W / 2,
        5,
        "Planner-surrogate evaluation; mean and 95% CI over 30 seeds.",
        size=7.7,
    )

    export_drawing(drawing, "dt_infographic_results")

    bar_drawing = Drawing(PAGE_W, PAGE_H)
    add_bar_legend(bar_drawing)
    draw_bar_panel(
        bar_drawing,
        LEFT_X,
        "(a) Deadline satisfaction",
        "Deadline satisfaction (%)",
        False,
        0,
        105,
        [0, 20, 40, 60, 80, 100],
        data,
        "deadline_success_pct",
        "deadline_success_ci95_pct",
    )
    draw_bar_panel(
        bar_drawing,
        RIGHT_X,
        "(b) DT synchronization error",
        "P95 position error (m, log scale)",
        True,
        1,
        2200,
        [1, 10, 100, 1000],
        data,
        "p95_position_error_m_mean",
        "p95_position_error_m_ci95",
    )
    add_text(
        bar_drawing,
        PAGE_W / 2,
        5,
        "Planner-surrogate evaluation; mean and 95% CI over 30 seeds.",
        size=7.7,
    )
    export_drawing(bar_drawing, "dt_infographic_results_bar")


if __name__ == "__main__":
    main()
