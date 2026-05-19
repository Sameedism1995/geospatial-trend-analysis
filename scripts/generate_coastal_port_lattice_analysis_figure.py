#!/usr/bin/env python3
"""Publication-style figure for coastal / port-adjacent lattice framing.

Reads lattice polygons from dashboard intermediates when available; overlays
nearest-harbour assignment counts. Does not rerun exposure composition.

Exports::
    outputs/final_thesis_figures/fig_coastal_port_adjacent_lattice_analysis.{png,pdf}
    outputs/final_figures/coastal_port_adjacent_lattice_caption.md

Regenerate::
    python scripts/generate_coastal_port_lattice_analysis_figure.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
GEOJSON_DEFAULT = ROOT / "outputs" / "final_thesis_figures" / "intermediate" / "dashboard_land_exposure_cells.geojson"
COAST_PATH = ROOT / "processed" / "basemap_cache" / "ne_110m_coastline.geojson"
OUT_FIG = ROOT / "outputs" / "final_thesis_figures" / "fig_coastal_port_adjacent_lattice_analysis"
OUT_CAP = ROOT / "outputs" / "final_figures" / "coastal_port_adjacent_lattice_caption.md"
SCRIPT_COAST = ROOT / "scripts" / "generate_geospatial_coastal_exposure_maps.py"

DPI = 300
STUDY_PORTS = ("Turku", "Mariehamn", "Stockholm")


def _load_coastal():
    spec = importlib.util.spec_from_file_location("coastal_mod", SCRIPT_COAST)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    if not GEOJSON_DEFAULT.exists():
        raise FileNotFoundError(
            f"Missing {GEOJSON_DEFAULT}; run scripts/generate_composite_land_exposure_dashboard.py first "
            "or point GEOJSON_DEFAULT to a lattice GeoJSON."
        )

    gdf = gpd.read_file(GEOJSON_DEFAULT)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    gdf["nearest_port_first"] = gdf["nearest_port_first"].fillna("—").astype(str)
    gdf.loc[gdf["nearest_port_first"].str.strip().eq(""), "nearest_port_first"] = "—"

    minx, miny, maxx, maxy = gdf.total_bounds
    pad = max((maxx - minx), (maxy - miny)) * 0.08 + 0.15
    extent = (minx - pad, maxx + pad, miny - pad, maxy + pad)
    view = box(*extent)

    coast_mod = _load_coastal()
    port_gdf = None
    if coast_mod is not None:
        try:
            port_gdf, _ = coast_mod.port_markers_proj()
            port_gdf = port_gdf.to_crs(4326)
        except Exception:
            port_gdf = None

    fig = plt.figure(figsize=(11.2, 8.6), dpi=DPI, facecolor="white")
    gs = gridspec.GridSpec(
        figure=fig,
        nrows=2,
        ncols=1,
        height_ratios=[1.0, 0.22],
        left=0.07,
        right=0.98,
        top=0.93,
        bottom=0.09,
        hspace=0.22,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])

    ax_map.set_facecolor("#f8fafc")
    if COAST_PATH.exists():
        coast = gpd.read_file(COAST_PATH)
        if coast.crs is None:
            coast = coast.set_crs(4326)
        else:
            coast = coast.to_crs(4326)
        try:
            coast = coast.clip(gpd.GeoSeries([view], crs=4326))
            coast.plot(ax=ax_map, color="#334155", linewidth=0.55, alpha=0.85)
        except Exception:
            pass

    # Categorical colours: highlight study ports
    pal = {}
    base_cols = plt.cm.tab20(np.linspace(0.05, 0.95, 20))
    for i, p in enumerate(sorted(gdf["nearest_port_first"].unique())):
        pal[p] = base_cols[i % 20]
    for sp in STUDY_PORTS:
        if sp in pal:
            pal[sp] = dict(Turku="#1d4ed8", Mariehamn="#b45309", Stockholm="#15803d").get(sp, pal[sp])

    gdf["__col"] = gdf["nearest_port_first"].map(lambda x: pal.get(x, "#94a3b8"))
    gdf.plot(ax=ax_map, color=gdf["__col"], edgecolor="#1e293b", linewidth=0.15, alpha=0.88)

    if port_gdf is not None and not port_gdf.empty:
        mark = port_gdf[port_gdf["name"].isin(STUDY_PORTS)]
        if not mark.empty:
            ax_map.scatter(
                mark.geometry.x,
                mark.geometry.y,
                s=120,
                c="#fde047",
                edgecolors="#0f172a",
                linewidths=1.1,
                zorder=22,
                label="study ports",
            )
            for _, rw in mark.iterrows():
                ax_map.annotate(
                    rw["name"],
                    xy=(rw.geometry.x, rw.geometry.y),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8.75,
                    color="#0f172a",
                    fontweight="normal",
                )

    ax_map.set_xlim(extent[0], extent[1])
    ax_map.set_ylim(extent[2], extent[3])
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("Longitude (°)")
    ax_map.set_ylabel("Latitude (°)")
    ax_map.grid(True, linestyle=":", alpha=0.45, color="#cbd5e1")
    ax_map.tick_params(labelsize=9)
    for s in ax_map.spines.values():
        s.set_color("#cbd5e1")

    fig.suptitle(
        "Coastal and port-adjacent lattice analysis",
        fontsize=14.25,
        fontweight="normal",
        color="#0f172a",
        y=0.97,
    )
    ax_map.set_title(
        "Harbour-linked lattice polygons (nearest archive port colouring · observational framing)",
        fontsize=10.15,
        color="#475569",
        pad=11,
        loc="left",
        fontweight="normal",
    )

    counts = (
        gdf[gdf["_core_complete"]] if "_core_complete" in gdf.columns else gdf
    )["nearest_port_first"].value_counts()
    cats = counts.index.tolist()
    cols_b = [pal.get(c, "#94a3b8") for c in cats]
    ypos = np.arange(len(cats), dtype=float)
    ax_bar.barh(ypos, counts.values.astype(int), height=0.62, color=cols_b, edgecolor="#475569", linewidth=0.35)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(cats, fontsize=8.85)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Lattice cell count (complete-driver subset where flagged)", fontsize=9.35, color="#334155")
    ax_bar.set_facecolor("#f8fafc")
    ax_bar.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax_bar.tick_params(labelsize=8.95)
    for s in ax_bar.spines.values():
        s.set_color("#cbd5e1")

    legend_elems = [
        Patch(facecolor=pal[p], edgecolor="#1e293b", linewidth=0.4, label=f"{p} (cells)")
        for p in STUDY_PORTS
        if p in pal and p != "—"
    ]
    if legend_elems:
        ax_map.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    color="none",
                    markerfacecolor="#fde047",
                    markeredgecolor="#0f172a",
                    markersize=10,
                    markeredgewidth=1,
                    label="Study ports",
                ),
                *legend_elems,
            ],
            loc="lower left",
            frameon=True,
            fontsize=8.25,
            framealpha=0.92,
            edgecolor="#e2e8f0",
            facecolor="#ffffff",
            borderpad=0.45,
        )

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.07)
    fig.savefig(OUT_FIG.with_suffix(".pdf"), dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    n_cell = len(gdf)
    n_complete = int(gdf["_core_complete"].sum()) if "_core_complete" in gdf.columns else n_cell
    cap = f"""## Coastal and port-adjacent lattice analysis

**Figure:** `outputs/final_thesis_figures/fig_coastal_port_adjacent_lattice_analysis.png` · PDF sibling

**Caption:** Discrete coastal harbour-adjacent lattice cells are tinted by archived **nearest-port** assignment derived from weekly panel aggregates; shoreline context uses NE 110m coastlines clipped to study extent; gold markers denote **Turku**, **Mariehamn**, and **Stockholm** study ports; lower panel summarizes cell counts (**complete-driver lattice subset**).

**Interpretation:** The map is descriptive geometry for port-linked coastal analysis—not a dispersion or health-impact map. Incomplete-driver cells remain excluded from the count strip when `_core_complete` is available.

**Support:** Lattice features **n = {n_cell}**; complete-driver-flagged subset **n = {n_complete}** (counts by harbour in stripe).

---
Generated by `scripts/generate_coastal_port_lattice_analysis_figure.py`.
"""
    OUT_CAP.parent.mkdir(parents=True, exist_ok=True)
    OUT_CAP.write_text(cap.strip() + "\n", encoding="utf-8")

    print(f"Wrote {OUT_FIG.with_suffix('.png')}")
    print(f"Wrote {OUT_CAP}")


if __name__ == "__main__":
    main()
