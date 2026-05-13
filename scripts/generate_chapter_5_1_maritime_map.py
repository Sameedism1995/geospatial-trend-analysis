#!/usr/bin/env python3
"""
Thesis Chapter 5.1 figure: Baltic maritime traffic intensity (vessel_density_t)
as a **discrete grid-cell choropleth** (no triangulated ocean surface), with
Natural Earth coastlines, focal ports, north arrow, scale bar, and legends for
discrete **corridor** / **port-adjacent** edge emphasis.

Outputs (default):
  outputs/final_figures/chapter_5_1_baltic_vessel_density_maritime_exposure.{png,pdf}
  outputs/final_figures/chapter_5_1_baltic_vessel_density_maritime_exposure_caption.md
  outputs/final_figures/chapter_5_1_vessel_density_spatial_summary.csv
  (+ companion CSV exports as documented in script)

Basemap: Natural Earth 110m land (cached under processed/basemap_cache); download
uses SSL context fallback if verification fails (public-domain data only).
"""

from __future__ import annotations

import re
import ssl
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "processed" / "features_ml_ready.parquet"
OUT_DIR = ROOT / "outputs" / "final_figures"
CACHE_DIR = ROOT / "processed" / "basemap_cache"
CACHE_ZIP = CACHE_DIR / "ne_110m_land.zip"

NE_LAND_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"

# Focal ports (WGS84): Turku & Naantali from hub_strategy conventions
PORT_COORDS: dict[str, tuple[float, float]] = {
    "Stockholm": (59.3293, 18.0686),
    "Turku": (60.435, 22.225),
    "Mariehamn": (60.0973, 19.9348),
    "Naantali": (60.4669, 22.0258),
}

STAR_COLOR_BY_PORT: dict[str, str] = {
    "Stockholm": "#0072b2",
    "Turku": "#d55e00",
    "Mariehamn": "#009e73",
    "Naantali": "#cc79a7",
}

# Baltic study window (filters mis-tagged longitudes outside the analytic domain)
BALTIC_LAT_RANGE = (53.8, 67.6)
BALTIC_LON_RANGE = (8.4, 31.2)

# Choropleth edge-highlight rules (exported to summary CSV / caption)
CORRIDOR_PERCENTILE_GLOBAL = 88
PORT_HOT_PERCENTILE_WITHIN_LABEL = 72
PORT_ADJ_DISTANCE_KM_MAX = 42.0


def _download_ne_land() -> bytes:
    for ctx in (ssl.create_default_context(), ssl._create_unverified_context()):
        try:
            req = urllib.request.Request(NE_LAND_URL, headers={"User-Agent": "geospatial-thesis/1.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
                return r.read()
        except Exception:
            continue
    raise RuntimeError("Could not download Natural Earth land shapefile (network/SSL).")


def load_land_gdf() -> gpd.GeoDataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CACHE_ZIP.is_file():
        CACHE_ZIP.write_bytes(_download_ne_land())
    return gpd.read_file(f"zip://{CACHE_ZIP}!ne_110m_land.shp")


# grid_cell_id pattern: "g{res}_r{#}_c{#}" ; res nominal edge length (degrees).
_CELL_RES_RE = re.compile(r"^g(?P<res>[\d.]+)_")


def parse_cell_resolution_deg(grid_cell_id: str, default_deg: float = 0.1) -> float:
    m = _CELL_RES_RE.match(str(grid_cell_id))
    if not m:
        return default_deg
    try:
        return float(m.group("res"))
    except ValueError:
        return default_deg


def square_cell_polygon(lon: float, lat: float, res_deg: float) -> Polygon:
    half = float(res_deg) / 2.0
    return box(float(lon) - half, float(lat) - half, float(lon) + half, float(lat) + half)


def add_north_arrow(ax: plt.Axes, x: float = 0.93, y: float = 0.86) -> None:
    ax.annotate(
        "",
        xy=(x, y + 0.07),
        xytext=(x, y),
        textcoords=ax.transAxes,
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.4, mutation_scale=14, shrinkA=0, shrinkB=0),
        zorder=20,
    )
    ax.text(x, y + 0.095, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold", color="0.1", zorder=20)


def add_scale_bar_km(ax: plt.Axes, km: float, reference_lat: float, loc: tuple[float, float]) -> None:
    """Horizontal scale bar using degree width at ``reference_lat`` (km along parallel)."""
    lat_rad = np.deg2rad(reference_lat)
    deg_lon_per_km = 1.0 / (111.320 * np.cos(lat_rad))

    lon0, lat0 = loc
    width_deg = km * deg_lon_per_km
    height_deg = max(0.03, ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012

    rect = mpl.patches.Rectangle(
        (lon0, lat0),
        width_deg,
        height_deg,
        facecolor="0.12",
        edgecolor="0.12",
        linewidth=0.6,
        zorder=25,
        clip_on=False,
    )
    ax.add_patch(rect)

    xc = lon0 + 0.5 * width_deg
    ax.annotate(f"{km:.0f} km", xy=(xc, lat0), xytext=(0, -(height_deg * 3.8)), textcoords="offset points", ha="center", va="top", fontsize=9, color="0.15", clip_on=False, zorder=25)


def build_panel() -> None:
    if not PARQUET.is_file():
        raise FileNotFoundError(PARQUET)

    df = pd.read_parquet(PARQUET)
    vd_col = "vessel_density_t"
    if vd_col not in df.columns:
        vd_col = "vessel_density"

    agg_spec: dict[str, tuple[str, str]] = {
        "grid_centroid_lat": ("grid_centroid_lat", "first"),
        "grid_centroid_lon": ("grid_centroid_lon", "first"),
        "nearest_port": ("nearest_port", "first"),
        "mean_vessel_density_t": (vd_col, "mean"),
        "max_vessel_density_t": (vd_col, "max"),
    }
    if "distance_to_port_km" in df.columns:
        agg_spec["distance_to_port_km"] = ("distance_to_port_km", "first")
    if "maritime_pressure_index" in df.columns:
        agg_spec["mean_maritime_pressure"] = ("maritime_pressure_index", "mean")
    if "coastal_exposure_score" in df.columns:
        agg_spec["mean_coastal_exposure"] = ("coastal_exposure_score", "mean")

    agg = df.groupby("grid_cell_id", observed=False).agg(**agg_spec).reset_index()
    agg["mean_vessel_density_t"] = pd.to_numeric(agg["mean_vessel_density_t"], errors="coerce")
    if "mean_maritime_pressure" in agg.columns:
        agg["mean_maritime_pressure"] = pd.to_numeric(agg["mean_maritime_pressure"], errors="coerce")
    if "mean_coastal_exposure" in agg.columns:
        agg["mean_coastal_exposure"] = pd.to_numeric(agg["mean_coastal_exposure"], errors="coerce")

    bal = agg["grid_centroid_lat"].between(*BALTIC_LAT_RANGE) & agg["grid_centroid_lon"].between(*BALTIC_LON_RANGE)
    use = bal & agg["mean_vessel_density_t"].notna()
    agg_b = agg.loc[use].copy()

    lon = agg_b["grid_centroid_lon"].to_numpy(dtype=float)
    lat = agg_b["grid_centroid_lat"].to_numpy(dtype=float)
    z = agg_b["mean_vessel_density_t"].to_numpy(dtype=float)

    vmin, vmax_obs = float(z.min()), float(z.max())
    p95, p98 = np.percentile(z, [95, 98])
    vmax_vis = float(min(max(p98 * 1.05, p95), vmax_obs))

    margin = max(0.35, np.ptp(lon) * 0.045)
    lon_min, lon_max = float(lon.min() - margin), float(lon.max() + margin)
    lat_min, lat_max = float(lat.min() - margin), float(lat.max() + margin)

    res_deg_series = agg_b["grid_cell_id"].map(parse_cell_resolution_deg)
    geoms = [
        square_cell_polygon(lc, phi, rd)
        for lc, phi, rd in zip(
            agg_b["grid_centroid_lon"].astype(float),
            agg_b["grid_centroid_lat"].astype(float),
            res_deg_series,
        )
    ]
    gdf = gpd.GeoDataFrame(agg_b.assign(geometry=geoms), geometry="geometry", crs="EPSG:4326")

    p_corr = float(np.percentile(z, CORRIDOR_PERCENTILE_GLOBAL))
    p_within_port = gdf.groupby("nearest_port")["mean_vessel_density_t"].transform(
        lambda s: np.percentile(s, PORT_HOT_PERCENTILE_WITHIN_LABEL)
    )
    gdf["_corridor_edge"] = gdf["mean_vessel_density_t"] >= p_corr
    gdf["_port_within_hot"] = gdf["mean_vessel_density_t"] >= p_within_port
    d_ok = pd.to_numeric(gdf["distance_to_port_km"], errors="coerce") if "distance_to_port_km" in gdf.columns else pd.Series(np.nan, index=gdf.index)
    gdf["_port_near"] = gdf["_port_within_hot"] & d_ok.notna() & (d_ok <= PORT_ADJ_DISTANCE_KM_MAX)

    norm = Normalize(vmin=vmin, vmax=vmax_vis, clip=True)
    cmap = mpl.colormaps.get_cmap("inferno")

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 120,
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.2, 9.4), layout="constrained")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#f5f9fc")

    gdf.plot(
        ax=ax,
        column="mean_vessel_density_t",
        cmap=cmap,
        norm=norm,
        linewidth=0.22,
        edgecolor="#6b7c8f",
        zorder=2,
        legend=False,
    )

    if gdf["_corridor_edge"].any():
        gdf.loc[gdf["_corridor_edge"]].plot(
            ax=ax,
            facecolor="none",
            edgecolor="#0a2540",
            linewidth=1.05,
            zorder=4,
        )

    if gdf["_port_near"].any():
        near_g = gdf.loc[gdf["_port_near"]]
        for pname, grp in near_g.groupby("nearest_port"):
            pc = STAR_COLOR_BY_PORT.get(pname, "#333333")
            grp.plot(ax=ax, facecolor="none", edgecolor=pc, linewidth=1.15, linestyle=(0, (4, 1.2)), zorder=5, alpha=0.95)

    land = load_land_gdf()
    land_local = land.cx[lon_min:lon_max, lat_min:lat_max]
    if not land_local.empty:
        land_local.plot(ax=ax, color="#f0ebe3", edgecolor="#45382e", linewidth=0.4, zorder=6, alpha=1.0)

    port_colors = STAR_COLOR_BY_PORT
    for pname, (plat, plon) in PORT_COORDS.items():
        ax.scatter(
            [plon],
            [plat],
            s=130,
            marker="*",
            c=port_colors.get(pname, "#000000"),
            edgecolors="0.1",
            linewidths=0.5,
            zorder=14,
        )
        dx, dy = {"Stockholm": (-1.1, 0.22), "Turku": (0.35, -0.35), "Mariehamn": (0.25, 0.28), "Naantali": (0.35, 0.22)}.get(
            pname, (0.4, 0.2)
        )
        ax.annotate(
            pname,
            (plon, plat),
            textcoords="offset points",
            xytext=(dx * 12, dy * 12),
            fontsize=9,
            fontweight="semibold",
            color="0.05",
            ha="left",
            path_effects=[mpe.withStroke(linewidth=2.5, foreground="white")],
            zorder=15,
        )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(
        "Baltic study region: discrete grid-cell choropleth of mean vessel-density (ML panel)",
        pad=10,
        fontweight="semibold",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.45, color="0.55", alpha=0.45, zorder=0)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.72, pad=0.09, extend="max")
    cbar.set_label("Mean weekly vessel-density index (EMODnet; time-averaged per grid cell)")

    port_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markersize=11,
            markerfacecolor=port_colors[p],
            markeredgecolor="0.1",
            label=p,
        )
        for p in PORT_COORDS
    ]
    leg_ports = ax.legend(handles=port_handles, loc="upper left", frameon=True, fontsize=8.5, title="Major ports", ncol=2)
    leg_ports.get_frame().set_alpha(0.94)
    ax.add_artist(leg_ports)

    corridor_handle = mlines.Line2D(
        [0],
        [0],
        color="#0a2540",
        lw=2.0,
        solid_capstyle="butt",
        label=f"Corridor cells (global ≥ {CORRIDOR_PERCENTILE_GLOBAL}th pct.; bold tile edges)",
    )
    port_edge_handle = mlines.Line2D(
        [0],
        [0],
        color="#555555",
        lw=1.8,
        linestyle=(0, (4, 1.2)),
        label=(
            "Port-adjacent hotspots (within-port percentile ≥ "
            f"P{PORT_HOT_PERCENTILE_WITHIN_LABEL}, distance ≤ {PORT_ADJ_DISTANCE_KM_MAX:.0f} km; "
            "dashed rims match port hues)"
        ),
    )
    leg_high = ax.legend(
        handles=[corridor_handle, port_edge_handle],
        loc="lower left",
        frameon=True,
        fontsize=7.95,
        title="Discrete emphasis (no smoothing)",
    )
    leg_high.get_frame().set_alpha(0.94)
    ax.add_artist(leg_high)

    ref_lat = float(np.median(lat))
    deg_lon_100km = 100.0 / (111.320 * np.cos(np.deg2rad(ref_lat)))
    add_scale_bar_km(ax, km=100.0, reference_lat=ref_lat, loc=(lon_max - deg_lon_100km - 0.58, lat_min + 0.22))
    add_north_arrow(ax)

    base = OUT_DIR / "chapter_5_1_baltic_vessel_density_maritime_exposure"
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- Summary statistics & tables ---
    rho_lon_val = pd.Series(lon).corr(pd.Series(z), method="spearman") if pd.Series(lon).nunique() > 1 and pd.Series(z).nunique() > 1 else float("nan")
    rho_lat_val = pd.Series(lat).corr(pd.Series(z), method="spearman") if pd.Series(lat).nunique() > 1 and pd.Series(z).nunique() > 1 else float("nan")
    stats_rows = [
        {"metric": "n_grid_cells_baltic_bbox_with_vessel_density", "value": len(agg_b)},
        {"metric": "n_grid_cells_total_parquet", "value": int(df["grid_cell_id"].nunique())},
        {"metric": "n_grid_cells_excluded_outside_bbox_or_null_vd", "value": int(df["grid_cell_id"].nunique()) - len(agg_b)},
        {"metric": "mean_vessel_density_t_time_mean_per_cell", "value": round(float(z.mean()), 6)},
        {"metric": "median_vessel_density_t_time_mean_per_cell", "value": round(float(np.median(z)), 6)},
        {"metric": "std_vessel_density_t_time_mean_per_cell", "value": round(float(z.std(ddof=0)), 6)},
        {"metric": "p90_vessel_density_t_time_mean_per_cell", "value": round(float(np.percentile(z, 90)), 6)},
        {"metric": "p95_vessel_density_t_time_mean_per_cell", "value": round(float(np.percentile(z, 95)), 6)},
        {"metric": "max_vessel_density_t_time_mean_per_cell", "value": round(float(z.max()), 6)},
        {"metric": "choropleth_cell_geometry", "value": "square_from_centroid_plus_minus_half_res_deg_from_grid_cell_id"},
        {"metric": "nominal_cell_resolution_deg", "value": parse_cell_resolution_deg(str(agg_b["grid_cell_id"].iloc[0]))},
        {"metric": "corridor_highlight_global_percentile_threshold", "value": CORRIDOR_PERCENTILE_GLOBAL},
        {"metric": "corridor_highlight_n_cells", "value": int(gdf["_corridor_edge"].sum())},
        {"metric": "port_adjacent_within_port_percentile_threshold", "value": PORT_HOT_PERCENTILE_WITHIN_LABEL},
        {"metric": "port_adjacent_distance_km_max", "value": PORT_ADJ_DISTANCE_KM_MAX},
        {"metric": "port_adjacent_highlight_n_cells", "value": int(gdf["_port_near"].sum())},
        {"metric": "spearman_rho_lon_vs_mean_vd", "value": round(float(rho_lon_val), 4) if rho_lon_val == rho_lon_val else ""},
        {"metric": "spearman_rho_lat_vs_mean_vd", "value": round(float(rho_lat_val), 4) if rho_lat_val == rho_lat_val else ""},
    ]
    pd.DataFrame(stats_rows).to_csv(OUT_DIR / "chapter_5_1_vessel_density_spatial_summary.csv", index=False)

    top = agg_b.nlargest(15, "mean_vessel_density_t")[
        ["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon", "nearest_port", "mean_vessel_density_t", "max_vessel_density_t"]
    ].copy()
    top["rank_overall"] = range(1, len(top) + 1)
    top.to_csv(OUT_DIR / "chapter_5_1_highest_vessel_density_cells.csv", index=False)

    port_blocks = []
    for port in ["Stockholm", "Turku", "Mariehamn", "Naantali"]:
        sub = agg_b.loc[agg_b["nearest_port"] == port].nlargest(8, "mean_vessel_density_t").copy()
        if sub.empty:
            continue
        sub.insert(0, "focal_port_label", port)
        cols_use = ["focal_port_label", "grid_cell_id", "grid_centroid_lat", "grid_centroid_lon", "mean_vessel_density_t"]
        for cx in ["mean_maritime_pressure", "mean_coastal_exposure"]:
            if cx in sub.columns:
                cols_use.append(cx)
        port_blocks.append(sub[cols_use])

    if port_blocks:
        pd.concat(port_blocks, ignore_index=True).to_csv(OUT_DIR / "chapter_5_1_port_adjacent_highest_exposure.csv", index=False)

    n_weeks = int(df["week_start_utc"].nunique()) if "week_start_utc" in df.columns else 0
    top_str = ", ".join(
        f"{r.grid_cell_id} ({r.nearest_port}, mean={r.mean_vessel_density_t:.2f})" for r in top.head(5).itertuples()
    )
    rho_lon_txt = f"{rho_lon_val:.3f}" if rho_lon_val == rho_lon_val else "n/a"
    rho_lat_txt = f"{rho_lat_val:.3f}" if rho_lat_val == rho_lat_val else "n/a"
    res_deg_ex = parse_cell_resolution_deg(str(agg_b["grid_cell_id"].iloc[0]))
    n_corridor_cells = int(gdf["_corridor_edge"].sum())
    n_port_adj_cells = int(gdf["_port_near"].sum())

    caption_md = f"""# Figure — Baltic maritime traffic intensity and exposure context (Chapter 5.1)

## Caption

**Figure 5.1.** Baltic Sea study extent showing maritime traffic intensity on the **actual machine-learning lattice** (`processed/features_ml_ready.parquet`). Each tiled polygon is the nominal **square cell** centred on the archived `grid_centroid_lon` / `grid_centroid_lat`, with edge length inferred from **`grid_cell_id`** (prefix `g0.100` ⇒ **≈ {res_deg_ex:.3f}°** per side in this build). Polygon fill reflects the **weekly time-mean `vessel_density_t`**, averaged across **{n_weeks}** UTC week anchors (**{len(agg_b)}** Baltic-bounded cells retained; malformed off-domain centroids omitted). No continuous ocean interpolation is applied—the figure is strictly a **gridded choropleth**. **Bold navy rims** denote cells at or above the **global {CORRIDOR_PERCENTILE_GLOBAL}th percentile** of those means (discrete corridor emphasis across **{n_corridor_cells}** cells). **Dashed rims** tinted by labelled port denote **within-port hotspots**: cells meeting the assigned `nearest_port` **P{PORT_HOT_PERCENTILE_WITHIN_LABEL}** threshold among co-labelled cells **and** `distance_to_port_km` ≤ **{PORT_ADJ_DISTANCE_KM_MAX:.0f} km** (**{n_port_adj_cells}** cells highlighted). Shorelines derive from **Natural Earth 110m** land polygons. **Stockholm**, **Turku**, **Mariehamn**, and **Naantali** use blue, orange, green, and magenta star markers respectively. Horizontal colour bar, **100 km** scale bar, and north arrow follow standard cartographic marginalia.

## Chapter 5.1 — short interpretation (thesis-ready)

Because each observation in the ML panel ties to a **fixed `grid_cell_id`**, the choropleth makes the **empirical support** of vessel-density covariates transparent: intensity is patchy at the resolution of the sampling mesh rather than an artificial smooth sea surface. Spearman correlation of cell means with longitude is about **{rho_lon_txt}** and with latitude about **{rho_lat_txt}**, underscoring **along-basin heterogeneity** rather than spatial homogeneity. The discrete global-percentile emphasis reveals **elongated high-traffic clusters** consistent with major fairways, while the port-conditioned dashed borders stress **localised adjacency surges** near hub assignments. Together these devices communicate that maritime exposure in the modelling frame is **cell-local and anisotropic**, warranting stratified interpretations relative to Åland-archipelago arcs, Finnish gateway ports, and the Stockholm skerries (**{top_str}** summarise the strongest global cell means).

## Data notes

- Primary attribute: **`mean vessel_density_t` per grid cell**, identical aggregation as used for map coloring.
- No triangulation/KDE smoothing; geometries are rectangles in geographic degrees centred on archival centroids.
- Companion tables: `outputs/final_figures/chapter_5_1_vessel_density_spatial_summary.csv`, `chapter_5_1_highest_vessel_density_cells.csv`, `chapter_5_1_port_adjacent_highest_exposure.csv`, PDF/PNG under the same basename.
- Regenerate: `python3 scripts/generate_chapter_5_1_maritime_map.py`
"""
    (OUT_DIR / "chapter_5_1_baltic_vessel_density_maritime_exposure_caption.md").write_text(caption_md, encoding="utf-8")

    print(base.with_suffix(".png"))


if __name__ == "__main__":
    build_panel()
