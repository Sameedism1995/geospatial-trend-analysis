#!/usr/bin/env python3
"""Generate thesis PNG diagrams: architecture, orchestration, data flow."""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reports" / "thesis_methodology" / "figures"
DPI = 300


def _box(ax, xy, w, h, text, fc, ec="#2c3e50", fontsize=8.5):
    x, y = xy
    r = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(r)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
        color="#1a1a1a",
    )
    return (x + w / 2, y + h), (x + w / 2, y)  # top center, bottom center


def _arrow(ax, p0, p1, color="#34495e"):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.3,
        color=color,
        zorder=1,
    )
    ax.add_patch(arr)


def figure_system_architecture():
    """Layered logical architecture."""
    fig, ax = plt.subplots(figsize=(12, 10), dpi=DPI)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")

    ax.text(
        6,
        9.55,
        "Overall system architecture",
        fontsize=14,
        fontweight="bold",
        ha="center",
        color="#2c3e50",
    )
    ax.text(
        6,
        9.15,
        "Weekly Baltic grid panel: extract → harmonize → feature layer → QA & analytics",
        fontsize=9,
        ha="center",
        style="italic",
        color="#555",
    )

    layers = [
        (
            "Control plane",
            8.05,
            6.95,
            3.85,
            1.05,
            "run_final_pipeline.py\n└─ run_full_pipeline.main()",
            "#e8f4fc",
        ),
        (
            "Data sources",
            0.5,
            5.05,
            11.0,
            1.35,
            "Baseline: modeling_dataset.parquet (vessel grid spine)\n"
            "Earth Engine aux: NO₂ • S1 oil proxy • S2 water quality • optional land NDVI\n"
            "Parallel ingestion: run_ingestion.py + YAML → EMODnet / HELCOM / Sentinel Hub catalogue",
            "#eafaf1",
        ),
        (
            "Harmonization & features",
            0.5,
            3.25,
            11.0,
            1.35,
            "merge_sources (grid_cell_id, week_start_utc) → merged_dataset.parquet\n"
            "feature_engineering + port proximity/exposure → features_ml_ready.parquet\n"
            "Optional: land_sea buffering + interactions (rewrite parquet)",
            "#fef9e7",
        ),
        (
            "Quality & analytics",
            0.5,
            1.45,
            11.0,
            1.25,
            "validation JSON • EDA • correlation • anomaly • coastal_impact_score • visualizations\n"
            "Downstream CLIs: coastal wind transport • exposure indices (consume panel)",
            "#fdebd0",
        ),
        (
            "Persistent outputs",
            0.5,
            0.15,
            11.0,
            1.05,
            "processed/*.parquet  •  data/aux/*.parquet  •  data/validation/*.json\n"
            "outputs/reports/, outputs/plots/, run mirror + FINAL_RUN_SUMMARY.md",
            "#eceff1",
        ),
    ]

    centers = []
    for title, lx, ly, lw, lh, body, fc in layers:
        ax.text(lx + 0.08, ly + lh + 0.06, title, fontsize=10, fontweight="bold", color="#2c3e50")
        _box(ax, (lx, ly), lw, lh, body, fc)
        cx, cy = lx + lw / 2, ly + lh / 2
        centers.append((cx, ly + lh, cx, ly))

    for i in range(len(centers) - 1):
        y_bot = centers[i][3]
        y_top_above = centers[i + 1][1]
        mid_x = centers[i][0]
        _arrow(ax, (mid_x, y_bot - 0.02), (mid_x, y_top_above + 0.02))

    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "system_architecture.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    return path


def figure_pipeline_orchestration():
    """Vertical workflow for run_final_pipeline → run_full_pipeline spine."""
    fig, ax = plt.subplots(figsize=(11, 13), dpi=DPI)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    cx = 5.0

    ax.text(cx, 12.35, "Pipeline orchestration workflow", fontsize=14, fontweight="bold", ha="center", color="#2c3e50")
    ax.text(
        cx,
        11.95,
        "run_final_pipeline.py → snapshot + GEE probe → run_full_pipeline.main()",
        fontsize=9,
        ha="center",
        color="#555",
    )

    def place(cx_, y_top, w, h, text, fc):
        yb = y_top - h
        _box(ax, (cx_ - w / 2, yb), w, h, text, fc)
        return {"cx": cx_, "yb": yb, "yt": y_top, "xm": cx_, "ym": yb + h / 2}

    y = 11.65
    b1 = place(cx, y, 4.8, 0.68, "Config snapshot • GEE init\n(fallback to cached aux if unavailable)", "#d6eaf8")
    y = b1["yb"] - 0.12
    b2 = place(cx, y, 4.8, 0.55, "Load vessels → modeling_dataset spine", "#d6eaf8")
    y = b2["yb"] - 0.12
    b3 = place(cx, y, 4.9, 0.58, "GEE extracts per source → intermediate + data/aux/*.parquet", "#d6eaf8")
    y = b3["yb"] - 0.12
    b4 = place(cx, y, 4.9, 0.52, "merge_sources → merged_dataset.parquet", "#fef9e7")
    y = b4["yb"] - 0.12
    b5 = place(cx, y, 5.0, 0.72, "feature_engineering + port proximity/exposure\n→ features_ml_ready.parquet (v1)", "#fdebd0")
    y = b5["yb"] - 0.12
    b6 = place(cx, y, 3.6, 0.48, "--land-impact ?", "#fadbd8")
    y_side_top = b6["yb"] - 0.08
    bl = place(2.35, y_side_top, 3.0, 0.58, "Land–sea buffering +\ninteractions → rewrite parquet", "#d5f5e3")
    br = place(7.65, y_side_top, 2.4, 0.58, "Skip land\nextension", "#fdebd0")
    merge_top = min(bl["yb"], br["yb"]) - 0.1
    b7 = place(cx, merge_top, 4.9, 0.52, "global_validation → validation JSON", "#ebf5fb")
    y = b7["yb"] - 0.12
    b8 = place(cx, y, 5.0, 0.62, "EDA • correlation • optional stages\nanomaly • coastal_impact_score • visualizations", "#ebf5fb")
    y = b8["yb"] - 0.12
    b9 = place(
        cx,
        y,
        5.0,
        0.68,
        "Mirror run folder • final_dataset_validation\nhub distance-decay • FINAL_RUN_SUMMARY.md",
        "#eceff1",
    )

    chain = [b1, b2, b3, b4, b5, b6]
    for a, b in zip(chain, chain[1:]):
        _arrow(ax, (a["xm"], a["yb"] - 0.01), (b["xm"], b["yt"] + 0.01))

    _arrow(ax, (b6["xm"] - 0.15, b6["yb"] - 0.01), (bl["xm"], bl["yt"] + 0.01))
    _arrow(ax, (b6["xm"] + 0.15, b6["yb"] - 0.01), (br["xm"], br["yt"] + 0.01))
    _arrow(ax, (bl["xm"], bl["yb"] - 0.01), (b7["xm"], b7["yt"] + 0.01))
    _arrow(ax, (br["xm"], br["yb"] - 0.01), (b7["xm"], b7["yt"] + 0.01))

    tail = [b7, b8, b9]
    for a, b in zip(tail, tail[1:]):
        _arrow(ax, (a["xm"], a["yb"] - 0.01), (b["xm"], b["yt"] + 0.01))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pipeline_orchestration_workflow.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    return path


def figure_data_flow_integration():
    """Sources → integration → panel → consumers."""
    fig, ax = plt.subplots(figsize=(14, 9.5), dpi=DPI)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title band (clear margin above diagram)
    ax.text(7, 9.55, "Data flow and integration framework", fontsize=14, fontweight="bold", ha="center", va="top", color="#2c3e50")
    ax.text(
        7,
        9.02,
        "Multi-source join on (grid_cell_id, week_start_utc); parallel catalogue ingestion for provenance",
        fontsize=8.5,
        ha="center",
        va="top",
        color="#555",
        wrap=False,
    )
    ax.text(
        7,
        8.58,
        "(GEE aux cache + processed parquets; optional land–sea extension in main pipeline).",
        fontsize=8,
        ha="center",
        va="top",
        color="#666",
        style="italic",
    )

    # Column headers — horizontal, above boxes (avoids rotation collision with title)
    hdr_y = 6.28
    ax.text(1.85, hdr_y, "Acquisition", fontsize=10, fontweight="bold", ha="center", va="bottom", color="#2c3e50")
    ax.text(5.75, hdr_y, "Integration core", fontsize=10, fontweight="bold", ha="center", va="bottom", color="#2c3e50")
    ax.text(9.75, hdr_y, "Artefacts", fontsize=10, fontweight="bold", ha="center", va="bottom", color="#2c3e50")

    # Left column: external / file sources
    _box(ax, (0.35, 4.9), 3.0, 0.85, "EMODnet GeoTIFFs\n(vessel density sampling)", "#d5f5e3")
    _box(ax, (0.35, 3.68), 3.0, 0.85, "Google Earth Engine\nS5P NO₂ • S1 GRD • S2 SR", "#d6eaf8")
    _box(ax, (0.35, 2.48), 3.0, 0.8, "Ingestion orchestrator\nYAML → EMODnet WMS\nHELCOM XML • SH STAC", "#e8daef")

    # Center: integration (stacked without overlap)
    _box(ax, (4.2, 4.88), 3.2, 0.9, "Intermediate\nvessels + per-source frames", "#fef9e7")
    _box(ax, (4.15, 3.52), 3.3, 0.95, "merge_sources\nouter join + alias normalize", "#fdebd0")
    _box(ax, (4.1, 1.85), 3.4, 1.12, "Feature + port layers\n(+ optional land–sea)\nfeatures_ml_ready.parquet", "#fadbd8")

    # Right: outputs / downstream
    _box(ax, (8.3, 5.02), 2.8, 0.68, "data/aux/*.parquet\n(cached GEE)", "#ebf5fb")
    _box(ax, (8.3, 4.02), 2.8, 0.68, "merged_dataset.parquet", "#ebf5fb")
    _box(ax, (8.15, 2.28), 3.1, 1.08, "Analytics & thesis bundle\ncorrelation • decay • wind\noutputs/reports/", "#eceff1")

    # Arrows: left → center (enter mid-height of top intermediate box)
    _arrow(ax, (2.0, 5.325), (4.18, 5.33))
    _arrow(ax, (2.0, 4.105), (4.18, 5.25))
    _arrow(ax, (2.0, 2.88), (4.18, 5.15))
    # Center vertical chain
    _arrow(ax, (5.8, 4.88), (5.8, 4.49))
    _arrow(ax, (5.8, 3.52), (5.8, 2.99))
    # Center → right
    _arrow(ax, (7.42, 5.36), (8.28, 5.36))
    _arrow(ax, (7.42, 4.0), (8.28, 4.36))
    _arrow(ax, (7.42, 2.55), (8.12, 2.82))

    legend_el = [
        mpatches.Patch(facecolor="#d5f5e3", edgecolor="#2c3e50", label="Land / vessel spine"),
        mpatches.Patch(facecolor="#d6eaf8", edgecolor="#2c3e50", label="Satellite & atmosphere (GEE)"),
        mpatches.Patch(facecolor="#e8daef", edgecolor="#2c3e50", label="Research catalogue ingestion"),
        mpatches.Patch(facecolor="#fadbd8", edgecolor="#2c3e50", label="ML-ready panel"),
    ]
    ax.legend(handles=legend_el, loc="lower center", ncol=4, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.01))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "data_flow_integration_framework.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.35, facecolor="white", edgecolor="none")
    plt.close()
    return path


def main():
    p1 = figure_system_architecture()
    p2 = figure_pipeline_orchestration()
    p3 = figure_data_flow_integration()
    for p in (p1, p2, p3):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
