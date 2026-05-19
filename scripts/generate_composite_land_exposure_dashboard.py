#!/usr/bin/env python3
"""Thesis-style composite coastal land exposure dashboard.

Reuses lattice construction from ``generate_geospatial_coastal_exposure_maps.py``.
Interpretation stays observational — spatial association only; analysis unchanged.

Exports::
  outputs/final_thesis_figures/thesis_dashboard_refined.{png,pdf}
  outputs/final_thesis_figures/composite_land_exposure_dashboard.{png,pdf}
  outputs/final_thesis_figures/dashboard_caption.md
  outputs/final_thesis_figures/intermediate/dashboard_panel_*.{png,pdf}
  outputs/final_thesis_figures/intermediate/dashboard_land_exposure_cells.{parquet,geojson}

Figure layout (~6000 × 4400 px at 300 DPI): Baltic overview, three port zooms, large
distance-structure panel; unified legend strip; light academic theme.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import interpolate as sp_interpolate
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_COAST = ROOT / "scripts" / "generate_geospatial_coastal_exposure_maps.py"
OUT_DIR = ROOT / "outputs" / "final_thesis_figures"
MID = OUT_DIR / "intermediate"

DPI_MAIN = 300
FIG_W_IN = 6000 / DPI_MAIN
FIG_H_IN = 4400 / DPI_MAIN

# Light academic palette
FIG_BG = "#f4f6f9"
AX_MAP_BG = "#ffffff"
FG_TEXT = "#1e293b"
FG_MUTED = "#64748b"
EDGE_GRID = "#cbd5e1"
CELL_EDGE = "#94a3b8"
CORR_COLOR = "#0f766e"
HOT_EDGE = "#b45309"
PORT_FACE = "#f8fafc"
PORT_EDGE = "#b91c1c"
INCOMPLETE_FILL = "#e2e8f0"

logging.basicConfig(level=logging.INFO, format="%(message)s")

CM = mpl.colormaps.get_cmap("plasma")


def _load_coastal_module():
    spec = importlib.util.spec_from_file_location("coastal_exposure_impl", SCRIPT_COAST)
    if spec is None or spec.loader is None:
        raise ImportError(str(SCRIPT_COAST))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def coord_formatters_merc(yy_ref_for_x_ticks: float, xx_ref_for_y_ticks: float):
    """Formatters converting Web Mercator coordinates to approximate lon/lat labels."""
    try:
        from pyproj import Transformer

        inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        def fx(xv, _pos):
            lon, _ = inv.transform(float(xv), float(yy_ref_for_x_ticks))
            if lon >= -1e-9:
                return f"{lon:.1f}\u202f°E"
            return f"{-lon:.1f}\u202f°W"

        def fy(yv, _pos):
            _, lat = inv.transform(float(xx_ref_for_y_ticks), float(yv))
            return f"{lat:.1f}\u202f°N"

        return FuncFormatter(fx), FuncFormatter(fy)
    except Exception:

        def kmx(xv, _pos):
            return f"{xv / 1000.0:.0f}"

        def kmy(yv, _pos):
            return f"{yv / 1000.0:.0f}"

        return FuncFormatter(kmx), FuncFormatter(kmy)


def clip_gdf_bbox(g3857: gpd.GeoDataFrame, xmin: float, ymin: float, xmax: float, ymax: float) -> gpd.GeoDataFrame:
    return g3857.cx[xmin:xmax, ymin:ymax].copy()


def add_north_arrow(ax_merc: plt.Axes, color: str = FG_TEXT) -> None:
    fx, fy = 0.92, 0.82
    ax_merc.annotate(
        "",
        xy=(fx, fy + 0.05),
        xytext=(fx, fy),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.35, mutation_scale=14),
        zorder=200,
    )
    ax_merc.text(
        fx,
        fy + 0.072,
        "N",
        ha="center",
        transform=ax_merc.transAxes,
        fontsize=10,
        fontweight="normal",
        color=color,
        zorder=200,
    )


def add_scale_bar(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float, km: float) -> None:
    x0 = xmin + (xmax - xmin) * 0.04
    y0 = ymin + (ymax - ymin) * 0.045
    w_m = km * 1000.0
    h = (ymax - ymin) * 0.012
    ax.add_patch(Rectangle((x0, y0), w_m, h, fc=FG_TEXT, ec=EDGE_GRID, lw=0.5, zorder=190, clip_on=False))
    xm = x0 + 0.5 * w_m
    ax.text(
        xm,
        y0 + h + (ymax - ymin) * 0.016,
        f"{km:.0f}\u202fkm",
        ha="center",
        fontsize=8.25,
        color=FG_TEXT,
        zorder=190,
        clip_on=False,
    )


def corridor_outline(ax: plt.Axes, corr3857: gpd.GeoDataFrame, z_base: int = 38) -> None:
    """Subtle dashed outline for corridor mask (no heavy glow)."""
    if corr3857.empty:
        return

    dash = (0, (5.5, 3.5))

    def _one(gser: gpd.GeoSeries) -> None:
        gser.plot(ax=ax, lw=1.15, linestyle=dash, color=CORR_COLOR, alpha=0.88, zorder=z_base + 5)

    try:
        u = corr3857.dissolve()
        geom = u.geometry.iloc[0]
        bd = getattr(geom, "boundary", None)
        if bd is not None and not bd.is_empty:
            _one(gpd.GeoSeries([bd], crs=corr3857.crs))
            return
    except Exception:
        pass
    corr3857.boundary.plot(ax=ax, lw=1.05, linestyle=dash, color=CORR_COLOR, alpha=0.82, zorder=z_base + 4)


def decorate_map_axes(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float, *, scale_km: float | None) -> None:
    ax.set_xlim(xmin * 1.0005, xmax * 1.0005)
    ax.set_ylim(ymin * 1.0005, ymax * 1.0005)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(AX_MAP_BG)

    yr = float(ymin + (ymax - ymin) * 0.5)
    xr = float(xmin + (xmax - xmin) * 0.5)
    fmx, _ = coord_formatters_merc(yr, yr)
    _, fmy = coord_formatters_merc(xr, xr)

    ax.xaxis.set_major_formatter(fmx)
    ax.yaxis.set_major_formatter(fmy)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontsize(7.95)
        lab.set_color(FG_MUTED)
    ax.tick_params(axis="both", colors=FG_MUTED, length=2.8, width=0.65)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(EDGE_GRID)
        spine.set_linewidth(0.75)

    ax.grid(True, linestyle="-", lw=0.4, alpha=0.35, color=EDGE_GRID)

    add_north_arrow(ax)
    if scale_km is not None:
        add_scale_bar(ax, xmin, xmax, ymin, ymax, scale_km)


def composite_colorbar_h(
    ax_cb: mpl.axes.Axes,
    vmin: float,
    vmax: float,
    label: str,
    cmap=mpl.cm.plasma,
) -> None:
    vmin, vmax = float(vmin), float(vmax)
    if vmax <= vmin + 1e-14:
        vmax = vmin + 1e-6
    cmap = cmap or CM
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation="horizontal")
    cbar.ax.tick_params(colors=FG_MUTED, labelsize=8)
    ax_cb.tick_params(colors=FG_MUTED)
    cbar.set_label(label, fontsize=9, color=FG_TEXT, labelpad=6)


def plot_ports(ax: plt.Axes, markers3857: gpd.GeoDataFrame, whitelist: tuple[str, ...] | None, annotate: bool):
    wl = tuple(n.casefold() for n in whitelist) if whitelist else None
    mk = markers3857[markers3857["name"].str.casefold().isin(wl)] if wl else markers3857
    if mk.empty:
        return
    for _, row in mk.iterrows():
        ax.scatter(
            row.geometry.x,
            row.geometry.y,
            s=118,
            c=PORT_FACE,
            edgecolors=PORT_EDGE,
            linewidths=1.55,
            zorder=140,
            clip_on=False,
        )
        if annotate:
            ax.annotate(
                row["name"],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="normal",
                color=FG_TEXT,
                zorder=145,
                path_effects=[mpl.patheffects.withStroke(linewidth=2.25, foreground=AX_MAP_BG)],
                clip_on=False,
            )


def draw_map_panel(
    ax: plt.Axes,
    bbox_merc_x0: float,
    bbox_merc_y0: float,
    bbox_merc_x1: float,
    bbox_merc_y1: float,
    g3857: gpd.GeoDataFrame,
    land3857: gpd.GeoDataFrame,
    port_df: gpd.GeoDataFrame,
    vmin: float,
    vmax: float,
    *,
    title: str,
    port_whitelist: tuple[str, ...] | None,
    annotate_ports: bool,
    scale_km: float | None,
    show_hotspot_hatch: bool,
    draw_colorbar: bool = False,
    title_fontsize: float = 10.85,
    title_pad: float = 10.0,
) -> None:
    xmin, xmax = bbox_merc_x0, bbox_merc_x1
    ymin, ymax = bbox_merc_y0, bbox_merc_y1
    sub_all = clip_gdf_bbox(g3857, xmin, ymin, xmax, ymax)
    clip_gs = gpd.GeoSeries([box(xmin, ymin, xmax, ymax)], crs=g3857.crs)
    sub_land = land3857.clip(clip_gs)

    decorate_map_axes(ax, xmin, xmax, ymin, ymax, scale_km=scale_km)

    sub_land.plot(ax=ax, facecolor="#e9edf5", edgecolor="#384252", lw=0.58, alpha=1.0, zorder=2)

    incomp = sub_all.loc[~sub_all["_core_complete"]]
    if not incomp.empty:
        incomp.plot(
            ax=ax,
            facecolor=INCOMPLETE_FILL,
            edgecolor="#94a3b8",
            lw=0.12,
            hatch="...",
            alpha=0.62,
            zorder=5,
        )

    comp = sub_all.loc[sub_all["_core_complete"]].copy()
    if not comp.empty:
        comp.plot(
            ax=ax,
            column="composite_land_exposure_score",
            cmap=CM,
            vmin=vmin,
            vmax=vmax,
            linewidth=0.12,
            edgecolor=CELL_EDGE,
            legend=False,
            alpha=0.93,
            zorder=12,
            missing_kwds=dict(color="none"),
        )

    corr = sub_all.loc[(sub_all["corridor_flag"] & sub_all["_core_complete"])]
    corridor_outline(ax, corr)

    plot_ports(ax, port_df, port_whitelist, annotate_ports)

    if show_hotspot_hatch and "hotspot_top10pct" in sub_all.columns:
        hotrim = sub_all[sub_all["hotspot_top10pct"] & sub_all["_core_complete"]]
        if not hotrim.empty:
            hotrim.plot(
                ax=ax,
                facecolor="none",
                edgecolor=HOT_EDGE,
                linewidth=0.85,
                hatch="\\\\\\\\",
                zorder=30,
                alpha=0.78,
            )

    if draw_colorbar:
        composite_colorbar_h(ax, vmin, vmax, "Composite exposure (normalized)")
    ax.set_title(title, color=FG_TEXT, fontsize=title_fontsize, fontweight="normal", pad=title_pad, loc="left")


def distance_panel(
    ax: plt.Axes,
    g_agg: pd.DataFrame,
    *,
    comp_col: str = "composite_land_exposure_score",
    bins_km_max: float = 90.0,
    bin_width: float = 4.0,
    min_cells: int = 5,
) -> float | None:
    dd = g_agg[g_agg["_core_complete"]].copy()
    dd = dd.assign(
        dist_km=pd.to_numeric(dd["dist_first"], errors="coerce"),
        comp_score=pd.to_numeric(dd[comp_col], errors="coerce"),
    )
    dd = dd[np.isfinite(dd["dist_km"]) & np.isfinite(dd["comp_score"])].copy()

    xmax = float(np.nanpercentile(dd["dist_km"], 98))
    xmax = float(min(max(xmax, bin_width * 5), bins_km_max))
    bins = np.arange(0.0, xmax + bin_width * 2, bin_width)
    dd["dist_bin"] = pd.cut(dd["dist_km"], bins=bins, include_lowest=True, right=False)

    grp = dd.groupby("dist_bin", observed=False)["comp_score"]
    centers = grp.median().index.map(lambda iv: getattr(iv, "mid", np.nan)).astype(float)
    medians = grp.median().values
    q1 = grp.quantile(0.25).values
    q3 = grp.quantile(0.75).values
    counts = grp.count().values

    valid = (~np.isnan(centers)) & (counts >= min_cells)

    xc = np.asarray(centers[valid], dtype=float)
    ym = np.asarray(medians[valid], dtype=float)
    y_lo = np.asarray(q1, dtype=float)[valid]
    y_hi = np.asarray(q3, dtype=float)[valid]

    ax.set_facecolor(AX_MAP_BG)
    corr = dd["dist_km"].corr(dd["comp_score"]) if len(dd) > 60 else np.nan

    if len(xc) >= 2:
        xf = np.linspace(float(np.nanmin(xc)), float(np.nanmax(xc)), max(240, len(xc) * 24))
        ix = np.argsort(xc)
        xs = xc[ix]
        ys = ym[ix]
        k = max(1, min(3, len(xs) - 1))
        try:
            spline = sp_interpolate.make_interp_spline(xs, ys, k=k)
            y_s = spline(xf)
            y_s = np.clip(y_s, float(np.nanmin(ys)), float(np.nanmax(ys)))
        except Exception:
            y_s = np.interp(xf, xs, ys)
        ax.plot(xf, y_s, color="#3730a3", lw=2.05, alpha=0.92, zorder=5)

    order = np.argsort(xc)
    xc_ord = xc[order]
    y_lo_ord = y_lo[order]
    y_hi_ord = y_hi[order]

    ax.fill_between(xc_ord, y_lo_ord, y_hi_ord, color="#fde047", alpha=0.38, interpolate=True, zorder=2)
    ax.plot(
        xc_ord,
        ym[order],
        marker="s",
        lw=2.05,
        markersize=5.25,
        markerfacecolor="#b45309",
        markeredgecolor="#fcd34d",
        markeredgewidth=0.45,
        color="#b45309",
        zorder=6,
    )

    ax.set_xlabel("Distance to nearest archive port (km)", color=FG_TEXT, fontsize=10.85)
    ax.set_ylabel("Median lattice composite exposure (normalized)", color=FG_TEXT, fontsize=10.85)
    ax.tick_params(colors=FG_MUTED, labelsize=9.15)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE_GRID)
        sp.set_linewidth(0.75)
    ax.grid(True, linestyle=":", alpha=0.45, color=EDGE_GRID)

    rho_ret: float | None = float(corr) if np.isfinite(corr) else None
    if rho_ret is not None:
        ax.text(
            0.975,
            0.065,
            f"Pearson ρ (cells) ≈ {rho_ret:.3f}",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=8.05,
            color=FG_MUTED,
        )

    ax.set_title("Composite exposure by port-distance band", color=FG_TEXT, fontsize=11.65, fontweight="normal", loc="left", pad=11)
    return rho_ret


def draw_hotspot_standalone_dark_light(
    ax: plt.Axes,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    g3857: gpd.GeoDataFrame,
    land3857: gpd.GeoDataFrame,
    port_df: gpd.GeoDataFrame,
    vmin: float,
    vmax: float,
    *,
    title: str | None = None,
) -> None:
    """Optional intermediate export panel (inferno hotspots on Baltic — legacy path). Uses light theme."""

    sub_all = clip_gdf_bbox(g3857, xmin, ymin, xmax, ymax)
    clip_box = box(xmin, ymin, xmax, ymax)
    sub_land = land3857.clip(gpd.GeoSeries([clip_box], crs=g3857.crs))

    decorate_map_axes(ax, xmin, xmax, ymin, ymax, scale_km=200.0)
    sub_land.plot(ax=ax, facecolor="#e9edf5", edgecolor="#384252", lw=0.55, alpha=1.0, zorder=2)

    cmap_muted = mpl.colormaps["plasma"]
    cmap_hot = mpl.colormaps["inferno"]
    base = sub_all[sub_all["_core_complete"] & ~sub_all["hotspot_top10pct"]].copy()
    hot = sub_all[sub_all["hotspot_top10pct"] & sub_all["_core_complete"]].copy()

    vmin_hi, vmax_hi = vmin, vmax
    if not hot.empty:
        hvals = pd.to_numeric(hot["composite_land_exposure_score"], errors="coerce")
        vmin_hi = max(vmin, float(hvals.quantile(0.02)))
        vmax_hi = float(hvals.max())

    if not base.empty:
        base.plot(
            ax=ax,
            column="composite_land_exposure_score",
            cmap=cmap_muted,
            vmin=vmin,
            vmax=vmax,
            linewidth=0.06,
            edgecolor="#aab6c9",
            alpha=0.28,
            zorder=6,
            legend=False,
            missing_kwds=dict(color=AX_MAP_BG),
        )

    if not hot.empty:
        hot.plot(
            ax=ax,
            column="composite_land_exposure_score",
            cmap=cmap_hot,
            vmin=vmin_hi,
            vmax=vmax_hi,
            linewidth=0.1,
            edgecolor="#92400e",
            alpha=0.88,
            zorder=22,
            legend=False,
            missing_kwds=dict(color="none"),
        )

    corridor_outline(ax, sub_all.loc[sub_all["corridor_flag"] & sub_all["_core_complete"]])

    plot_ports(ax, port_df, None, False)

    composite_colorbar_h(
        inset_axes(ax, width="74%", height="8%", loc="lower center", bbox_to_anchor=(0, 0.01, 1, 1), bbox_transform=ax.transAxes),
        vmin_hi,
        vmax_hi if not hot.empty else vmax,
        "Inferno scale · top-decile subset",
        cmap=cmap_hot,
    )
    ttl = title or "Hotspot-focused overview (exported panel)"
    ax.set_title(ttl, color=FG_TEXT, fontsize=11, fontweight="normal", pad=9, loc="left")


def _thesis_rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.65,
            "axes.edgecolor": EDGE_GRID,
            "axes.facecolor": AX_MAP_BG,
            "figure.facecolor": FIG_BG,
            "figure.dpi": DPI_MAIN,
            "savefig.dpi": DPI_MAIN,
            "savefig.facecolor": FIG_BG,
            "text.color": FG_TEXT,
        },
    )


def print_dashboard_validation(coast_mod, g3857_all: gpd.GeoDataFrame, g_agg: pd.DataFrame) -> dict:
    n_tot = len(g_agg)
    n_core = int(g_agg["_core_complete"].astype(bool).sum())
    ex = pd.to_numeric(g_agg.loc[g_agg["_core_complete"], "composite_land_exposure_score"], errors="coerce")
    vmin_ex, vmax_ex = float(np.nanmin(ex.values)), float(np.nanmax(ex.values))
    hot = int((g_agg["hotspot_top10pct"] & g_agg["_core_complete"]).sum())
    print("--- VALIDATION (composite land exposure dashboard) ---")
    print(f"CRS display: {coast_mod.CRS_PLOT} (author geometries {coast_mod.CRS_WGS84})")
    print(f"lattice polygons: {n_tot}")
    print(f"valid composite observations (complete driver stack): {n_core}")
    print(f"top-10pct hotspot cells on composite_land_exposure_score: {hot}")
    print(f"composite score range over valid cells: [{vmin_ex:.5f}, {vmax_ex:.5f}]")
    mei_miss = pd.to_numeric(g_agg["mei_median"], errors="coerce").isna().mean()
    ves_miss = pd.to_numeric(g_agg["ves_median"], errors="coerce").isna().mean()
    no2_miss = pd.to_numeric(g_agg["no2_median"], errors="coerce").isna().mean()
    withheld = (~g_agg["_core_complete"]).mean()
    print(
        "missingness fractions — mei {0:.4f}, vessel_density {1:.4f}, no2_median {2:.4f}, cells withheld composite {3:.4f}".format(
            mei_miss, ves_miss, no2_miss, withheld
        )
    )
    print("Structural missingness preserved; no stochastic gap-fill for ocean-absent composites.")
    sys.stdout.flush()
    return dict(vmin=float(vmin_ex), vmax=float(vmax_ex))


def export_single_panel(fname: Path, dpi: float, draw_fn) -> None:
    MID.mkdir(parents=True, exist_ok=True)
    _thesis_rc()
    fig = plt.figure(figsize=(FIG_W_IN * 0.48, FIG_H_IN * 0.38), dpi=dpi, facecolor=FIG_BG)
    ax = fig.add_axes([0.06, 0.08, 0.92, 0.85])
    draw_fn(ax)
    fig.savefig(fname.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor=FIG_BG, pad_inches=0.08)
    fig.savefig(fname.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor=FIG_BG, pad_inches=0.06)
    plt.close(fig)
    logging.info("saved %s", fname.with_suffix(".png"))


def unified_legend_handles(pct_corr: int) -> list:
    corridor_dash = (0, (5.5, 3.5))
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PORT_FACE,
            markeredgecolor=PORT_EDGE,
            markersize=8.5,
            markeredgewidth=1.2,
            label="Study ports",
        ),
        Line2D([0], [0], linestyle=corridor_dash, lw=2.05, color=CORR_COLOR, label=f"High-density vessel corridor (Baltic ~ P{pct_corr})"),
        Patch(
            facecolor="none",
            edgecolor=HOT_EDGE,
            linewidth=1.05,
            hatch="///",
            alpha=1.0,
            label="Top-decile hotspots (composite; outline hatch)",
        ),
        Patch(facecolor="#fde047", ec="#ca8a04", lw=0, alpha=0.45, label="Distance-bin interquartile band"),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            color="#b45309",
            markersize=6,
            markerfacecolor="#b45309",
            markeredgecolor="#ca8a04",
            markeredgewidth=0.5,
            label="Distance-bin medians",
        ),
        Line2D([0], [0], color="#3730a3", lw=2.0, linestyle="-", label="Smoothed median trend (splines)",
        ),
    ]


def write_dashboard_caption(
    out_path: Path,
    *,
    n_cells: int,
    n_core: int,
    n_hot: int,
    vmin_sc: float,
    vmax_sc: float,
    rho_dist: float | None,
    pct_corr: int,
) -> None:
    rho_txt = f"{rho_dist:.3f}" if rho_dist is not None and rho_dist == rho_dist else "n/a (sparse cells)"
    cap = rf"""## Composite coastal land exposure dashboard (`thesis_dashboard_refined` · `composite_land_exposure_dashboard`)

**Caption:**
Gridded lattice maps show the observational coastal composite exposure score (normalized, plasma colouring) adjacent to shoreline and shipping-corridor outlines; dashed teal lines denote the high-density vessel-density corridor mask (~ Baltic-wide **P{pct_corr}** on median vessel-density). Zoom panels highlight harbour-adjacent structure with subdued top-decile outlines (composite). The distance panel summarizes **median** lattice composite scores and **interquartile ranges** versus distance to the archived nearest port (complete-driver cells only).

**Interpretation:**
Read as descriptive co-location among maritime, atmospheric, and distance descriptors on the masked coastal lattice—not causal coastal impact. Incomplete driver stacks retain structural gap patterns (lighter hatched overlays); inferno-toned exports in `intermediate/` remain available for appendix-style hotspot emphasis.

**Support:** Lattice cells **N = {n_cells}**; composite-eligible (**full driver stack**) **n = {n_core}**; top-decile composite hotspots **n = {n_hot}**; lattice score display range **[{vmin_sc:.4f},\u202f{vmax_sc:.4f}]** (cell-level ρ distance vs composite ≈ **{rho_txt}**, association only).

---
*Generated by* `scripts/generate_composite_land_exposure_dashboard.py` *(analysis path unchanged vs.* `compose_exposure_cells` *upstream).*
"""
    out_path.write_text(cap.strip() + "\n", encoding="utf-8")


def main() -> None:
    coast_mod = _load_coastal_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MID.mkdir(parents=True, exist_ok=True)
    pct_corr = int(getattr(coast_mod, "CORRIDOR_PCT_GLOBAL", 88))

    panel_pd, wind_col = coast_mod.load_weekly_panel()
    g_agg = coast_mod.temporal_cell_table(panel_pd, wind_col)
    g_agg = coast_mod.compose_exposure_cells(g_agg, wind_col)
    g_agg["composite_land_exposure_score"] = g_agg["composite_exposure_score"]

    land_wgs = coast_mod.load_land_gdf_wgs84()
    g_wgs = coast_mod.geometries_for_cells(g_agg)
    g3857_all = g_wgs.to_crs(coast_mod.CRS_PLOT)
    land_plot = land_wgs.to_crs(coast_mod.CRS_PLOT)

    g4326 = g3857_all.to_crs(coast_mod.CRS_WGS84)
    xmin_b, ymin_b, xmax_b, ymax_b = g4326.total_bounds.astype(float).tolist()
    pad_lon = max((xmax_b - xmin_b) * 0.08, 0.06)
    pad_lat = max((ymax_b - ymin_b) * 0.05, 0.04)
    x0_w = max(coast_mod.BALTIC_LON_RANGE[0], xmin_b - pad_lon * 5.8)
    x1_w = min(coast_mod.BALTIC_LON_RANGE[1], xmax_b + pad_lon * 8.8)
    y0_w = max(coast_mod.BALTIC_LAT_RANGE[0], ymin_b - pad_lat)
    y1_w = min(coast_mod.BALTIC_LAT_RANGE[1], ymax_b + pad_lat)

    xmin_m, ymin_m, xmax_m, ymax_m = coast_mod.wgs_bbox_to_xy(x0_w, x1_w, y0_w, y1_w)
    balt_clip_wgs = gpd.GeoSeries([box(x0_w, y0_w, x1_w, y1_w)], crs=coast_mod.CRS_WGS84)
    land_balt = land_plot.clip(balt_clip_wgs.to_crs(coast_mod.CRS_PLOT))

    finite = pd.to_numeric(g3857_all.loc[g3857_all["_core_complete"], "composite_land_exposure_score"], errors="coerce").to_numpy(
        dtype=float
    )
    vmin = float(np.nanpercentile(finite, 5)) if finite.size else 0.0
    vmax = float(np.nanpercentile(finite, 95)) if finite.size else 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmin, vmax = 0.0, 1.0

    port_markers_df, _ = coast_mod.port_markers_proj()

    val = print_dashboard_validation(coast_mod, g3857_all, g_agg)

    dash_parq = MID / "dashboard_land_exposure_cells.parquet"
    dash_gj = MID / "dashboard_land_exposure_cells.geojson"

    tbl_out = g3857_all.to_crs(coast_mod.CRS_WGS84).copy()
    tbl_out["composite_land_exposure_score"] = pd.to_numeric(tbl_out["composite_land_exposure_score"], errors="coerce")
    coast_mod.parquet_export(tbl_out, dash_parq)
    try:
        tbl_out.to_file(dash_gj, driver="GeoJSON")
    except Exception as exc:
        logging.warning("dashboard GeoJSON write failed (%s)", exc)

    _thesis_rc()
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI_MAIN, facecolor=FIG_BG)

    gs = gridspec.GridSpec(
        figure=fig,
        nrows=4,
        ncols=3,
        height_ratios=[1.05, 1.0, 2.08, 0.34],
        left=0.05,
        right=0.99,
        top=0.925,
        bottom=0.034,
        wspace=0.07,
        hspace=0.11,
    )

    _fam = mpl.rcParams["font.family"]
    fam = _fam[0] if isinstance(_fam, (list, tuple)) and _fam else "DejaVu Sans"
    fig.suptitle(
        "Coastal lattice composite exposure",
        fontsize=13.1,
        fontweight="normal",
        color=FG_TEXT,
        y=0.988,
        ha="center",
        fontfamily=str(fam),
    )

    ax_a = fig.add_subplot(gs[0, :])
    draw_map_panel(
        ax_a,
        xmin_m,
        ymin_m,
        xmax_m,
        ymax_m,
        g3857_all,
        land_balt,
        port_markers_df,
        vmin,
        vmax,
        title="Baltic coastal exposure overview",
        port_whitelist=None,
        annotate_ports=True,
        scale_km=220.0,
        show_hotspot_hatch=False,
        draw_colorbar=False,
        title_pad=7.0,
        title_fontsize=11.0,
    )

    zcfg = (
        ("Turku exposure hotspots", ("Turku",), "turku", 55.0),
        ("Mariehamn exposure hotspots", ("Mariehamn",), "mariehamn", 40.0),
        ("Stockholm exposure hotspots", ("Stockholm",), "stockholm", 52.0),
    )

    zoom_blocks = []
    for col, cfg in enumerate(zcfg):
        ax_z = fig.add_subplot(gs[1, col])
        zoom_blocks.append(ax_z)

    for ax_z, cfg in zip(zoom_blocks, zcfg):
        title, plist, slug, km_scl = cfg
        bb = coast_mod._ZOOM_BOX_DEG[slug]
        xm0, ym0, xm1, ym1 = coast_mod.wgs_bbox_to_xy(bb[0], bb[1], bb[2], bb[3])
        z_clip = box(bb[0], bb[2], bb[1], bb[3])
        land_zoom = land_plot.clip(gpd.GeoSeries([z_clip], crs=coast_mod.CRS_WGS84).to_crs(coast_mod.CRS_PLOT))
        draw_map_panel(
            ax_z,
            xm0,
            ym0,
            xm1,
            ym1,
            g3857_all,
            land_zoom,
            port_markers_df,
            vmin,
            vmax,
            title=title,
            port_whitelist=plist,
            annotate_ports=True,
            scale_km=km_scl,
            show_hotspot_hatch=True,
            draw_colorbar=False,
            title_fontsize=10.45,
            title_pad=6.5,
        )

    ax_f = fig.add_subplot(gs[2, :])
    rho_notes = distance_panel(ax_f, g_agg)
    ax_f.tick_params(axis="both", labelsize=9.15)

    leg_ax = fig.add_subplot(gs[3, :])
    leg_ax.set_facecolor(FIG_BG)
    leg_ax.axis("off")
    leg_ax.set_xticks([])
    leg_ax.set_yticks([])

    uni = unified_legend_handles(pct_corr)
    leg_ax.legend(
        handles=uni,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=8.15,
        labelcolor=FG_TEXT,
        handletextpad=0.75,
        columnspacing=1.25,
        borderpad=0.15,
        handlelength=2.35,
        borderaxespad=0.05,
        shadow=False,
    )

    cb_in = inset_axes(
        leg_ax,
        width="90%",
        height="38%",
        loc="lower center",
        bbox_to_anchor=(0.0, 0.02, 1.0, 1.0),
        bbox_transform=leg_ax.transAxes,
        borderpad=0,
    )
    composite_colorbar_h(cb_in, vmin, vmax, "Cell fill colour: composite exposure (normalized; plasma)")
    rho_use = rho_notes

    fig.align_xlabels(zoom_blocks)

    n_tot = len(g_agg)
    n_core = int(g_agg["_core_complete"].astype(bool).sum())
    n_hotspot = int((g_agg["hotspot_top10pct"] & g_agg["_core_complete"]).sum())

    for stem in ("thesis_dashboard_refined", "composite_land_exposure_dashboard"):
        p = OUT_DIR / stem
        fig.savefig(p.with_suffix(".png"), dpi=DPI_MAIN, facecolor=FIG_BG)
        fig.savefig(p.with_suffix(".pdf"), dpi=DPI_MAIN, facecolor=FIG_BG)
        logging.info("saved %s", p.with_suffix(".png"))

    write_dashboard_caption(
        OUT_DIR / "dashboard_caption.md",
        n_cells=n_tot,
        n_core=n_core,
        n_hot=n_hotspot,
        vmin_sc=val["vmin"],
        vmax_sc=val["vmax"],
        rho_dist=rho_use,
        pct_corr=pct_corr,
    )
    plt.close(fig)

    def _pa(ax):
        draw_map_panel(
            ax,
            xmin_m,
            ymin_m,
            xmax_m,
            ymax_m,
            g3857_all,
            land_balt,
            port_markers_df,
            vmin,
            vmax,
            title="Baltic coastal exposure overview",
            port_whitelist=None,
            annotate_ports=True,
            scale_km=220.0,
            show_hotspot_hatch=False,
            draw_colorbar=False,
        )

    for slug, plist, km in (
        ("b_turku", ("Turku",), 55.0),
        ("c_mariehamn", ("Mariehamn",), 40.0),
        ("d_stockholm", ("Stockholm",), 52.0),
    ):
        bb = coast_mod._ZOOM_BOX_DEG[slug.replace("b_", "").replace("c_", "").replace("d_", "")]
        _slug = slug.split("_", 1)[-1]

        def _pz(ax, bd=bb, pl=plist, km_s=km, sk=_slug):
            xm0, ym0, xm1, ym1 = coast_mod.wgs_bbox_to_xy(bd[0], bd[1], bd[2], bd[3])
            z_clip_i = box(bd[0], bd[2], bd[1], bd[3])
            land_zoom_i = land_plot.clip(gpd.GeoSeries([z_clip_i], crs=coast_mod.CRS_WGS84).to_crs(coast_mod.CRS_PLOT))
            draw_map_panel(
                ax,
                xm0,
                ym0,
                xm1,
                ym1,
                g3857_all,
                land_zoom_i,
                port_markers_df,
                vmin,
                vmax,
                title=f"{sk.title()} exposure hotspots",
                port_whitelist=pl,
                annotate_ports=True,
                scale_km=km_s,
                show_hotspot_hatch=True,
                draw_colorbar=False,
            )

        export_single_panel(MID / f"dashboard_panel_{slug}", DPI_MAIN * 1.03, lambda ax, fz=_pz: fz(ax))

    export_single_panel(MID / "dashboard_panel_a_overview", DPI_MAIN * 1.03, lambda ax: _pa(ax))

    export_single_panel(
        MID / "dashboard_panel_e_hotspots_top10pct",
        DPI_MAIN,
        lambda ax: draw_hotspot_standalone_dark_light(
            ax,
            xmin_m,
            ymin_m,
            xmax_m,
            ymax_m,
            g3857_all,
            land_balt,
            port_markers_df,
            vmin,
            vmax,
            title="Baltic top-decile composite (exported panel)",
        ),
    )

    export_single_panel(MID / "dashboard_panel_f_distance_decay", DPI_MAIN * 1.06, lambda ax: distance_panel(ax, g_agg))

    logging.info(
        "val summary vmin=%s vmax=%s hot=%s",
        val["vmin"],
        val["vmax"],
        int((g_agg["hotspot_top10pct"] & g_agg["_core_complete"]).sum()),
    )


if __name__ == "__main__":
    main()
