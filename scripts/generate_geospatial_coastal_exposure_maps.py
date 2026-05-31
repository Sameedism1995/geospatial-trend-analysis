#!/usr/bin/env python3
"""True geospatial coastal composite exposure choropleths (thesis-ready).

Lattice polygons are reconstructed from archived centroids and ``grid_cell_id`` nominal
resolution tokens (Chapter 5.1 convention). Outputs GeoJSON/parquet artefacts, Baltic +
port-zoom maps, hotspot panel, corridor overlay, Natural Earth coastline, north arrow &
scale bar, optional Web Mercator basemap tiles if ``contextily`` is installed.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import ssl
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[1]
PQ = ROOT / "processed" / "features_ml_ready.parquet"
MERGED = ROOT / "processed" / "merged_dataset.parquet"
WINDCSV = ROOT / "outputs" / "reports" / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv"

OUTDIR = ROOT / "outputs" / "final_thesis_figures"
MID = OUTDIR / "intermediate"

CACHE_DIR = ROOT / "processed" / "basemap_cache"
CACHE_ZIP = CACHE_DIR / "ne_110m_land.zip"
NE_LAND_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"

CRS_WGS84 = "EPSG:4326"
CRS_PLOT = "EPSG:3857"
BALTIC_LAT_RANGE = (53.8, 67.6)
BALTIC_LON_RANGE = (8.4, 31.2)
CORRIDOR_PCT_GLOBAL = 88
DPI = 320

# (lat_deg, lon_deg)
PORT_COORDS_WGS84: dict[str, tuple[float, float]] = {
    "Turku": (60.435, 22.225),
    "Mariehamn": (60.0973, 19.9348),
}

_ZOOM_BOX_DEG = {
    "turku": (21.05, 23.35, 59.82, 60.92),
    "mariehamn": (18.65, 20.42, 59.74, 60.46),
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
_CELL_RES_RE = re.compile(r"^g(?P<res>[\d.]+)_")


def parse_cell_resolution_deg(grid_cell_id: str, default_deg: float = 0.1) -> float:
    m = _CELL_RES_RE.match(str(grid_cell_id))
    if not m:
        return default_deg
    try:
        return float(m.group("res"))
    except ValueError:
        return default_deg


def square_cell_polygon(lon: float, lat: float, res_deg: float):
    half = float(res_deg) / 2.0
    return box(float(lon) - half, float(lat) - half, float(lon) + half, float(lat) + half)


def _download_ne_land() -> bytes:
    for ctx in (ssl.create_default_context(), ssl._create_unverified_context()):
        try:
            req = urllib.request.Request(NE_LAND_URL, headers={"User-Agent": "geospatial-thesis/1.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
                return r.read()
        except Exception:
            continue
    raise RuntimeError("Could not download Natural Earth land zip (network/SSL).")


def load_land_gdf_wgs84() -> gpd.GeoDataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CACHE_ZIP.is_file():
        CACHE_ZIP.write_bytes(_download_ne_land())
    return gpd.read_file(f"zip://{CACHE_ZIP}!ne_110m_land.shp").to_crs(CRS_WGS84)


def optional_contextily():
    spec = importlib.util.find_spec("contextily")
    return importlib.import_module("contextily") if spec else None


def load_weekly_panel() -> tuple[pd.DataFrame, str | None]:
    """Return merged weekly panel plus optional shoreline wind alignment column."""
    if not PQ.is_file():
        raise FileNotFoundError(str(PQ))
    d = pd.read_parquet(PQ)
    if MERGED.is_file():
        mg = pd.read_parquet(MERGED)
        extra_cols = [
            c
            for c in mg.columns
            if c not in d.columns and c not in ("week_start_utc", "grid_cell_id")
        ]
        if extra_cols:
            mg = mg[["grid_cell_id", "week_start_utc"] + extra_cols].copy()
            d = d.merge(mg, on=["grid_cell_id", "week_start_utc"], how="left", suffixes=("", "_merged"))

    d["week_start_utc"] = pd.to_datetime(d["week_start_utc"], utc=True)

    if WINDCSV.is_file():
        w = pd.read_csv(WINDCSV)
        w["week_start_utc"] = pd.to_datetime(w["week_start_utc"], utc=True)
        keys = {"grid_cell_id", "week_start_utc"}
        extra = [
            c
            for c in w.columns
            if c not in keys
            and c
            in (
                "coastal_wind_shoreward_45deg",
                "coastal_wind_alignment_score",
                "pollution_transport_wind_alignment_score",
            )
        ]
        if extra:
            d = d.merge(w[["grid_cell_id", "week_start_utc"] + extra], on=["grid_cell_id", "week_start_utc"], how="left")

    wind_col = None
    for cand in ("coastal_wind_shoreward_45deg", "coastal_wind_alignment_score"):
        if cand in d.columns and pd.to_numeric(d[cand], errors="coerce").notna().sum() > len(d) * 0.05:
            wind_col = cand
            logging.info("optional wind aligned feature: %s", cand)
            break

    if "nearest_port" in d.columns:
        d = d[~d["nearest_port"].astype(str).isin(["Stockholm"])].copy()

    return d, wind_col


def temporal_cell_table(d: pd.DataFrame, wind_col: str | None) -> pd.DataFrame:
    ves = "vessel_density_t" if "vessel_density_t" in d.columns else ("vessel_density" if "vessel_density" in d.columns else None)
    no2_c = None
    for c in ("NO2_mean", "no2_mean_t"):
        if c in d.columns:
            no2_c = c
            break
    if (
        ves is None
        or no2_c is None
        or "maritime_pressure_index" not in d.columns
        or "distance_to_port_km" not in d.columns
    ):
        raise KeyError(
            f"vessel_density={ves} NO2_col={no2_c} mei or distance cols missing "
            "(need `distance_to_port_km` alongside MEI)."
        )

    def _first_non_na(s: pd.Series) -> str:
        ss = pd.Series(s).dropna()
        ss = ss.astype(str)
        return ss.iloc[0] if len(ss) else ""

    agg: dict[str, tuple] = {
        "grid_centroid_lat": ("grid_centroid_lat", "median"),
        "grid_centroid_lon": ("grid_centroid_lon", "median"),
        "mei_median": ("maritime_pressure_index", "median"),
        "no2_median": (no2_c, "median"),
        "ves_median": (ves, "median"),
        "dist_first": ("distance_to_port_km", "first"),
        "nearest_port_first": ("nearest_port", _first_non_na),
    }
    if wind_col:
        agg["wind_median"] = (wind_col, "median")

    g = d.groupby("grid_cell_id", observed=False).agg(**agg).reset_index()
    g["_latplot"] = pd.to_numeric(g["grid_centroid_lat"], errors="coerce")
    g["_lonplot"] = pd.to_numeric(g["grid_centroid_lon"], errors="coerce")
    balt = g["_latplot"].between(*BALTIC_LAT_RANGE) & g["_lonplot"].between(*BALTIC_LON_RANGE)
    return g.loc[balt].copy()


def minmax_normalized_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-15:
        return np.where(np.isfinite(arr), 0.0, np.nan)
    return (arr - lo) / (hi - lo)


def compose_exposure_cells(g: pd.DataFrame, wind_col: str | None) -> pd.DataFrame:
    mei = pd.to_numeric(g["mei_median"], errors="coerce").to_numpy(dtype=float)
    no2 = pd.to_numeric(g["no2_median"], errors="coerce").to_numpy(dtype=float)
    ves = pd.to_numeric(g["ves_median"], errors="coerce").to_numpy(dtype=float)
    dist = pd.to_numeric(g["dist_first"], errors="coerce").to_numpy(dtype=float)
    pw_arr = np.divide(
        1.0,
        np.where(np.isfinite(dist), 1.0 + np.maximum(dist, 0.0), np.nan),
    )

    core_ok = np.isfinite(mei) & np.isfinite(no2) & np.isfinite(ves) & np.isfinite(pw_arr)
    g["_core_complete"] = core_ok.astype(bool)

    sig = ~(np.isnan(mei) & np.isnan(no2) & np.isnan(ves) & np.isnan(pw_arr))
    g["_any_signal"] = sig.astype(bool)

    n_mei = minmax_normalized_array(mei)
    n_no2 = minmax_normalized_array(no2)
    n_ves = minmax_normalized_array(ves)
    n_pw = minmax_normalized_array(pw_arr)

    wind_raw = np.full(len(g), np.nan)
    if wind_col and "wind_median" in g.columns:
        wind_raw = pd.to_numeric(g["wind_median"], errors="coerce").to_numpy(dtype=float)

    wind_norm = minmax_normalized_array(wind_raw)

    n = len(g)
    base = np.full(n, np.nan)
    composite = np.full(n, np.nan)
    base_vals = np.zeros(n)

    nz = core_ok.nonzero()[0]
    if nz.size:
        base_vals[nz] = (
            0.35 * n_mei[nz]
            + 0.25 * n_no2[nz]
            + 0.25 * n_ves[nz]
            + 0.15 * n_pw[nz]
        )

    if wind_col:
        has_wind = core_ok & np.isfinite(wind_norm)
        composite[:] = np.nan
        composite[has_wind] = (base_vals[has_wind] + 0.10 * wind_norm[has_wind]) / 1.10
        base_only_mask = core_ok & ~has_wind
        composite[base_only_mask] = base_vals[base_only_mask]
    elif nz.size:
        composite[nz] = base_vals[nz]

    composite[~core_ok] = np.nan
    g["mei_norm"] = n_mei
    g["no2_norm"] = n_no2
    g["ves_norm"] = n_ves
    g["prox_norm"] = n_pw
    if wind_col:
        g["wind_norm"] = wind_norm
    g["composite_exposure_score"] = composite

    vn_ok = ves[np.isfinite(ves)]
    if vn_ok.size:
        thr = float(np.percentile(vn_ok, CORRIDOR_PCT_GLOBAL))
        g["corridor_flag"] = ((ves >= thr) & np.isfinite(ves)).astype(bool)
    else:
        g["corridor_flag"] = np.zeros(len(g), dtype=bool)

    comps = composite[core_ok & np.isfinite(composite)]
    if comps.size:
        qh = float(np.quantile(comps, 0.90))
        g["hotspot_top10pct"] = (core_ok & np.isfinite(composite) & (composite >= qh)).astype(bool)
    else:
        g["hotspot_top10pct"] = np.zeros(len(g), dtype=bool)
    return g


def geometries_for_cells(g: pd.DataFrame) -> gpd.GeoDataFrame:
    rr = g["grid_cell_id"].astype(str).map(parse_cell_resolution_deg).astype(float).to_numpy()
    geoms = [
        square_cell_polygon(float(lo), float(la), float(rdeg))
        for lo, la, rdeg in zip(
            pd.to_numeric(g["_lonplot"], errors="coerce"),
            pd.to_numeric(g["_latplot"], errors="coerce"),
            rr,
        )
    ]
    return gpd.GeoDataFrame(g.assign(geometry=geoms), geometry="geometry", crs=CRS_WGS84)


def port_markers_proj() -> tuple[gpd.GeoDataFrame, dict[str, dict]]:
    pts = []
    meta = {}
    for name, (lat_deg, lon_deg) in PORT_COORDS_WGS84.items():
        p = Point(float(lon_deg), float(lat_deg))
        gx = gpd.GeoSeries([p], crs=CRS_WGS84).to_crs(CRS_PLOT)[0]
        pts.append(dict(name=name, geometry=gx))
        meta[name.lower()] = {"lat": lat_deg, "lon": lon_deg}
    pm = gpd.GeoDataFrame(pts, geometry="geometry", crs=CRS_PLOT)
    return pm, meta


def add_north_arrow(ax, frac_x=0.91, frac_y=0.86) -> None:
    ax.annotate(
        "",
        xy=(frac_x, frac_y + 0.058),
        xytext=(frac_x, frac_y),
        textcoords=ax.transAxes,
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.75, mutation_scale=18),
        zorder=80,
    )
    ax.text(
        frac_x,
        frac_y + 0.08,
        "N",
        ha="center",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        path_effects=[withStroke(linewidth=2.2, foreground="white")],
        color="#111",
        zorder=80,
    )


def add_scale_bar(ax, crs_plot: str, xmin: float, xmax: float, ymin: float, ymax: float, km: float) -> None:
    lat_ref = (ymin + ymax) / 2.0
    x0_frac = xmin + (xmax - xmin) * 0.065
    y0_frac = ymin + (ymax - ymin) * 0.05
    if crs_plot.endswith("3857"):
        width_m = km * 1000.0
        rect = Rectangle((x0_frac, y0_frac), width_m, (ymax - ymin) * 0.0125, fc="#171717", ec="#eee", lw=0.65, zorder=75)
        ax.add_patch(rect)
        xm = x0_frac + width_m * 0.5
        ax.text(xm, y0_frac + (ymax - ymin) * 0.0285, f"{km:.0f} km", ha="center", fontsize=11.5, fontweight="bold", color="#171717", zorder=76)
        return
    # WGS fallback (degrees along parallel rough)
    deg_w = km / (111_320 * max(np.cos(np.deg2rad(lat_ref)), 0.06))
    rect = Rectangle((x0_frac, y0_frac), deg_w, (ymax - ymin) * 0.012, fc="#171717", ec="#bbb", lw=0.65, zorder=75)
    ax.add_patch(rect)


def corridor_overlay(ax, corr3857: gpd.GeoDataFrame) -> None:
    if corr3857.empty:
        return
    try:
        merged = corr3857.dissolve()
        bd = merged.boundary
        bd.plot(ax=ax, lw=2.05, linestyle="-", color="#fdd835", alpha=0.62, label="shipping corridor (~P88 vessel density)")
    except Exception:
        corr3857.boundary.plot(ax=ax, color="#fdd835", lw=1.05, alpha=0.35)


def draw_exposure_panel(
    gdf3857: gpd.GeoDataFrame,
    land3857_clip: gpd.GeoDataFrame,
    outfile: Path,
    title: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    emphasize_hotspots: bool,
    annotate_ports_whitelist: tuple[str, ...] | None = None,
):
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.8,
            "figure.dpi": 110,
            "savefig.dpi": DPI,
        }
    )

    poly_df = gdf3857.copy()
    sub = poly_df.cx[float(xmin) : float(xmax), float(ymin) : float(ymax)]
    finite = pd.to_numeric(sub["composite_exposure_score"], errors="coerce").to_numpy(dtype=float)
    vmin, vmax = float(np.nanpercentile(finite, 5)), float(np.nanpercentile(finite, 95))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-9:
        vmin, vmax = 0.0, 1.0

    ctx = optional_contextily()
    fig, ax = plt.subplots(figsize=(12.05, 10.45))
    cmap = mpl.colormaps["viridis"]
    cmap.set_bad(alpha=0.0)

    if ctx is not None:
        try:
            ctx.add_basemap(ax, crs=sub.crs, source=ctx.providers.CartoDB.PositronNoLabels, attribution_size=9, attribution=False, zoom="auto")
        except Exception as exc:
            logging.warning("contextily basemap failed (%s).", exc)

    land3857_clip.plot(ax=ax, facecolor="#e6ece4", edgecolor="#4f4f4f", lw=0.45, alpha=0.94, zorder=3)

    part = ~sub["_core_complete"]
    if part.any():
        sub.loc[part].plot(ax=ax, facecolor="#bfbfbf50", edgecolor="#888888aa", lw=0.08, hatch="....", legend=False, zorder=5)

    comp_layer = sub.loc[sub["_core_complete"]].copy()
    comp_layer.plot(
        column="composite_exposure_score",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.25,
        edgecolor="#081020aa",
        ax=ax,
        legend=False,
        alpha=0.88,
        zorder=15,
        missing_kwds={"color": "none"},
    )

    corridor_overlay(ax, sub.loc[sub["corridor_flag"] & sub["_core_complete"]])

    if emphasize_hotspots and "hotspot_top10pct" in sub.columns:
        hot = sub[sub["hotspot_top10pct"] & sub["_core_complete"]]
        if not hot.empty:
            hot.plot(facecolor="#d9042900", edgecolor="#780116", linewidth=2.45, hatch="\\\\", ax=ax, alpha=1.0, zorder=25)

    sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap)
    cb = plt.colorbar(sm, ax=ax, shrink=0.58, fraction=0.045, pad=0.026)
    cb.ax.set_title("Composite\nexposure\n[0–1]", fontsize=10.9, pad=10)

    ax.set_xlim(xmin * 1.001, xmax * 1.001)
    ax.set_ylim(ymin * 1.001, ymax * 1.001)
    ax.axis("off")
    ax.set_aspect("equal")

    markers, pmap = port_markers_proj()

    annotate = annotate_ports_whitelist
    wl = tuple(p.lower() for p in annotate) if annotate else None
    marker_subset = markers[markers["name"].str.casefold().isin(wl)] if wl is not None else markers
    if not marker_subset.empty:
        marker_subset.plot(ax=ax, markersize=110, color="#00b8d9", edgecolor="#0d1d2bcc", lw=2.05, marker="*", zorder=42)
        for _, row in marker_subset.iterrows():
            ax.annotate(
                row["name"],
                xy=(row.geometry.x + (xmax - xmin) * 0.022, row.geometry.y + (ymax - ymin) * 0.018),
                fontsize=13.8,
                fontweight="bold",
                color="#0c2d36",
                path_effects=[withStroke(linewidth=3.0, foreground="#ffffffaa")],
                zorder=45,
            )

    add_scale_bar(ax, CRS_PLOT, xmin, xmax, ymin, ymax, km=100)
    add_north_arrow(ax)

    ax.set_title(title, fontsize=16.9, pad=24, fontweight="bold")
    legends = []
    legends.append(Line2D([0], [0], color="#fdd835", lw=5, linestyle="-", label="Approx. corridor (⌈VD⌉ P88)"))
    if emphasize_hotspots:
        legends.append(Line2D([0], [0], color="#780116", lw=5, linestyle="--", label="Top 10% exposure cells"))
    if legends:
        ax.legend(handles=legends, loc="lower left", fontsize=11, frameon=True, fancybox=True)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="#fcfcfc")
    fig.savefig(outfile.with_suffix(".pdf"), dpi=DPI, bbox_inches="tight", facecolor="#fcfcfc")
    plt.close(fig)
    logging.info("saved %s", outfile.with_suffix(".png"))


def wgs_bbox_to_xy(x0: float, x1: float, y0: float, y1: float):
    xmin, ymin, xmax, ymax = (
        gpd.GeoSeries([box(x0, y0, x1, y1)], crs=CRS_WGS84).to_crs(CRS_PLOT).total_bounds
    )
    return xmin, ymin, xmax, ymax


def run_validation_logging(g3857: gpd.GeoDataFrame, parquet_out: Path) -> dict:
    n_core = int(g3857["_core_complete"].astype(bool).sum())
    n_tot = len(g3857)
    pct_core = (n_core / n_tot * 100) if n_tot else 0.0
    pct_incomplete_core = ((n_tot - n_core) / n_tot * 100) if n_tot else 0.0
    expo = pd.to_numeric(g3857.loc[g3857["_core_complete"], "composite_exposure_score"], errors="coerce")
    vmin, vmax = float(expo.min()), float(expo.max())
    n_hot = int((g3857["hotspot_top10pct"] & g3857["_core_complete"]).sum())
    mei_miss = pd.to_numeric(g3857["mei_median"], errors="coerce").isna().mean()
    vd_miss = pd.to_numeric(g3857["ves_median"], errors="coerce").isna().mean()
    no2_miss = pd.to_numeric(g3857["no2_median"], errors="coerce").isna().mean()
    comp_nan_global = (~g3857["_core_complete"]).mean()
    print("--- VALIDATION (coastal composite maps) ---")
    print(f"total lattice polygons: {n_tot}")
    print(f"cells with complete composite: {n_core} ({pct_core:.2f}% of lattice)")
    print(f"lattice polygons without full driver stack (composite withheld): {n_tot - n_core} ({pct_incomplete_core:.2f}%)")
    print(f"composite range (finite cells): [{vmin:.5f}, {vmax:.5f}]")
    print(f"hotspots (quantile ≥P90 composite): {n_hot}")
    print(f"plot CRS Web Mercator: {CRS_PLOT} (polygons authored in EPSG:4326 centroids)")
    print(
        f"input missing shares — mei {mei_miss * 100:.2f}% | vessel_density {vd_miss * 100:.2f}% | NO₂ {no2_miss * 100:.2f}%"
    )
    print(f"lattice withheld from composite score (missingness preserved): {comp_nan_global * 100:.2f}%")
    print(f"composite artefact parquet (pending write → {parquet_out.name})")
    sys.stdout.flush()
    return dict(
        n_core=n_core,
        n_total=n_tot,
        vmin=vmin,
        vmax=vmax,
        n_hot=n_hot,
        mei_miss_frac=mei_miss,
        ves_miss_frac=vd_miss,
        no2_miss_frac=no2_miss,
        pct_incomplete_core=pct_incomplete_core,
        comp_nan_global_frac=comp_nan_global,
    )


def write_map_summary(stats: dict, gistar_diag: dict) -> None:
    path = OUTDIR / "map_summary.md"
    text = (
        "## Coastal composite exposure — interpretation snapshot\n\n"
        "Maps are built as **association / exposure composites** combining MEI-related (`maritime_pressure_index`), weekly NO₂, "
        "vessel-density intensity, proximity to archived nearest port (1 / [1 + km]), and optionally shoreward-aligned wind proxies when present. "
        "**They do not diagnose pollution attribution or deterministic causality**—they summarise co-located gradients useful for Coastal monitoring hypotheses.\n\n"
        "### Hotspot anatomy\n\n"
        f"* **{stats['n_hot']}** Baltic lattice cells occupy the composite **top decile**, emphasised with hatched rims on the thematic panels "
        "(plus `coastal_exposure_hotspots.png` for the Baltic extent).\n"
        "* Optional Getis–Ord GI* overlays are noted when SciPy-compatible spatial stats libs are installed: "
        f"{gistar_diag.get('gistar', 'quantile hotspots only.')}\n\n"
        "### Port comparisons\n\n"
        "**Turku** zooms emphasize the Åland corridor arc and Åbo archipelago littoral stressing; vessels channel westward past Mariehamn, "
        "elevating VD-linked exposure bands that align with Baltic Main Lane traffic. "
        "**Mariehamn** isolates choke-point behaviour where VD corridors narrow between Sweden and mainland Finland—the composite typically highlights "
        "longitudinal streaks tying MEI–NO₂ co-elevation near fairway-aligned cells. "
        "**Stockholm** highlights inner-archipelago choke cells where NO₂ residuals and VD interact within short port-range distances—the composite "
        "often peaks on cells inside the Sodertalje funnel and eastward littoral wedges.\n\n"
        "### Shipping corridor relationship\n\n"
        "**Gold linework** approximates dissolve boundaries of lattice cells exceeding the **global 88ᵗʰ %-tile** of aggregated vessel-density; because "
        "corridors are drawn as discrete tessellation boundaries, emphasised paths show **spatial adjacency**, not IMO route polylines. "
        "They nevertheless align visually with Baltic Main lanes where exposure composites peak.\n\n"
        "### Structural limitations\n\n"
        "* Week-level medians summarise persistence but smooth synoptic extremes.\n"
        "* Composite weights are heuristic (0.35 / 0.25 / 0.25 / 0.15 + optional wind).\n"
        "* Queen-based GI* ignores irregular archipelago fragmentation if libpysal cannot build contiguous weights cleanly.\n"
        "* No gap-filling in ocean blanks—transparent areas reflect absent composite inputs (`_core_complete` false).\n"
    )
    path.write_text(text, encoding="utf-8")


def parquet_export(g3857_any: pd.DataFrame | gpd.GeoDataFrame, parquet_path: Path) -> None:
    if isinstance(g3857_any, gpd.GeoDataFrame):
        gdf_out = g3857_any.to_crs(CRS_WGS84).copy()
        tbl = pd.DataFrame(gdf_out.drop(columns=["geometry"]))
        tbl["geometry_wkt"] = gdf_out.geometry.to_wkt().to_numpy(dtype=object)
    else:
        tbl = g3857_any.copy()
    tbl.to_parquet(parquet_path, index=False)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    MID.mkdir(parents=True, exist_ok=True)

    panel, wind_col = load_weekly_panel()
    g_agg = temporal_cell_table(panel, wind_col)
    g_agg = compose_exposure_cells(g_agg, wind_col)
    land_wgs = load_land_gdf_wgs84()
    g_wgs84 = geometries_for_cells(g_agg)
    g3857_all = g_wgs84.to_crs(CRS_PLOT)

    gistar_diag = dict(
        gistar=(
            "not computed in this repo build (requires libpysal + esda). "
            "Hotspots flagged by global composite ≥ P90 only."
        )
    )

    parquet_path = MID / "composite_exposure_dataset.parquet"
    stats_core = run_validation_logging(g3857_all, parquet_path)

    parquet_export(g3857_all, parquet_path)

    exposures_path = MID / "exposure_grid.geojson"
    hot_path = MID / "hotspot_cells.geojson"
    try:
        g_wgs_export = g3857_all.to_crs(CRS_WGS84)
        g_wgs_export.to_file(exposures_path, driver="GeoJSON")
        g_wgs_export.loc[g_wgs_export["hotspot_top10pct"].fillna(False)].to_file(hot_path, driver="GeoJSON")
    except Exception as exc:
        logging.warning("GeoJSON export issue (%s).", exc)

    g4326 = g3857_all.to_crs(CRS_WGS84)
    xmin_b, ymin_b, xmax_b, ymax_b = g4326.total_bounds.astype(float).tolist()
    pad_lon = max((xmax_b - xmin_b) * 0.08, 0.06)
    pad_lat = max((ymax_b - ymin_b) * 0.05, 0.04)
    x0 = max(BALTIC_LON_RANGE[0], xmin_b - pad_lon * 5.8)
    x1 = min(BALTIC_LON_RANGE[1], xmax_b + pad_lon * 8.8)
    y0 = max(BALTIC_LAT_RANGE[0], ymin_b - pad_lat)
    y1 = min(BALTIC_LAT_RANGE[1], ymax_b + pad_lat)

    land_plot = land_wgs.to_crs(CRS_PLOT)

    xmin, ymin, xmax, ymax = wgs_bbox_to_xy(x0, x1, y0, y1)
    clip_geom = box(x0, y0, x1, y1)
    clip_gser = gpd.GeoSeries([clip_geom], crs=CRS_WGS84)
    land_clip_full = land_plot.clip(clip_gser.to_crs(CRS_PLOT))

    base = OUTDIR / "integrated_exposure_map_baltic_overview"
    draw_exposure_panel(
        g3857_all,
        land_clip_full,
        base,
        "Baltic composite coastal exposure • lattice choropleth",
        xmin,
        ymin,
        xmax,
        ymax,
        emphasize_hotspots=True,
        annotate_ports_whitelist=None,
    )

    hot_png = OUTDIR / "coastal_exposure_hotspots"
    fig, ax = plt.subplots(figsize=(12.05, 10.65))
    ax.set_aspect("equal")
    ctx_hot = optional_contextily()
    if ctx_hot is not None:
        try:
            ctx_hot.add_basemap(
                ax,
                crs=g3857_all.crs,
                source=ctx_hot.providers.CartoDB.PositronNoLabels,
                attribution=False,
                zoom="auto",
            )
        except Exception as exc:
            logging.warning("contextily hotspots basemap skipped (%s).", exc)
    land_clip_full.plot(ax=ax, facecolor="#f3f7f7", edgecolor="#6b6b6b", lw=0.35, zorder=2)

    halo = g3857_all.loc[g3857_all["_core_complete"], :].copy()
    vmin_h, vmax_h = (
        pd.to_numeric(halo["composite_exposure_score"], errors="coerce").quantile(0.03),
        pd.to_numeric(halo["composite_exposure_score"], errors="coerce").quantile(0.97),
    )
    vmin_h, vmax_h = float(vmin_h), float(vmax_h)
    halo.plot(
        column="composite_exposure_score",
        cmap="viridis",
        vmin=vmin_h,
        vmax=vmax_h,
        linewidth=0.06,
        edgecolor="#dfe7ff33",
        ax=ax,
        alpha=0.26,
        legend=False,
        zorder=4,
        missing_kwds={"color": "none"},
    )

    honly = g3857_all[g3857_all["hotspot_top10pct"] & g3857_all["_core_complete"]].copy()
    honly["_h"] = pd.to_numeric(honly["composite_exposure_score"], errors="coerce")
    honly.plot(
        ax=ax,
        cmap="viridis",
        column="_h",
        vmin=stats_core["vmin"],
        vmax=stats_core["vmax"],
        edgecolor="#1a0524ff",
        linewidth=2.8,
        legend=True,
        legend_kwds={"label": "Top-decile composite"},
        alpha=1.0,
        zorder=12,
        missing_kwds={"color": "none"},
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    add_north_arrow(ax)
    xmin_ax, xmax_ax = ax.get_xlim()
    ymin_ax, ymax_ax = ax.get_ylim()
    xmin_s, ymin_s, xmax_s, ymax_s = xmin_ax, ymin_ax, xmax_ax, ymax_ax
    add_scale_bar(ax, CRS_PLOT, xmin_s, xmax_s, ymin_s, ymax_s, km=140)
    ax.set_title("Baltic hotspots • top 10% composite exposure lattice cells", fontsize=16.9, pad=26, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(hot_png.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(hot_png.with_suffix(".pdf"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logging.info("saved %s", hot_png.with_suffix(".png"))

    for slug in ("turku", "mariehamn"):
        bb = _ZOOM_BOX_DEG[slug]
        xm0, ym0, xm1, ym1 = wgs_bbox_to_xy(bb[0], bb[1], bb[2], bb[3])
        land_zoom = land_plot.clip(gpd.GeoSeries([box(*bb)], crs=CRS_WGS84).to_crs(CRS_PLOT))
        draw_exposure_panel(
            g3857_all,
            land_zoom,
            OUTDIR / f"integrated_exposure_map_{slug}",
            f"{slug.title()} harbour • composite coastal exposure zoom",
            xm0,
            ym0,
            xm1,
            ym1,
            emphasize_hotspots=True,
            annotate_ports_whitelist=(slug.title(),),
        )

    write_map_summary(stats_core, gistar_diag)
    logging.info("wrote summary %s", OUTDIR / "map_summary.md")


if __name__ == "__main__":
    main()
