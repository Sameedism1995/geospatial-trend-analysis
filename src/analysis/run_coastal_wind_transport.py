#!/usr/bin/env python3
"""
Coastal wind-transport exposure: landward bearing from NE coast geometry, weekly wind (Open‑Meteo
ERA5 archive or --wind-csv), and pollution hotspot corridors (vessel | NO2 | oil quantile hotspots).

Associations / directional exposure only — not causal transport proof.

Writes:
  outputs/reports/run_coastal_wind_transport/coastal_wind_alignment_features.csv
  outputs/reports/run_coastal_wind_transport/coastal_wind_exposure_summary.csv
  outputs/visualizations/run_coastal_wind_transport/coastal_pollution_transport_map.html
  outputs/figures/run_coastal_wind_transport/*.png
  outputs/reports/coastal_wind_transport_interpretation.md

Optional: --augment-parquet merges exposure columns into a copy of the base panel.

Run:
  python3 src/analysis/run_coastal_wind_transport.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.run_land_pollution_drivers_wind import (  # noqa: E402
    apply_wind_to_dataframe,
    assign_shipping_band_tight,
    attach_nearest_high_vessel_seed,
    ensure_coast_distance,
    fetch_open_meteo_cluster_wind,
    initial_bearing_deg,
    normalized_user_wind,
    smallest_angle_deg,
    week_key_monday_start,
    wind_to_direction_deg,
)
from features.coastal_wind_geometry import (  # noqa: E402
    attach_nearest_multivariate_hotspot,
    grid_nearest_coast_reference_table,
    nearest_geodesic_reference,
)
from features.land_sea_buffering import BufferingConfig  # noqa: E402
from human_impact_distance_analysis import (  # noqa: E402
    load_coastline_points,
    load_land_boundary_points,
)

RUN = "run_coastal_wind_transport"
REPORTS = _ROOT / "outputs" / "reports" / RUN
FIGURES = _ROOT / "outputs" / "figures" / RUN
VIZ = _ROOT / "outputs" / "visualizations" / RUN
INTERPRET_MD = _ROOT / "outputs" / "reports" / "coastal_wind_transport_interpretation.md"

LINKED_PATH = _ROOT / "outputs/reports/run_nearest_land_ndvi_linkage/nearest_land_ndvi_linked_dataset.parquet"
DEFAULT_BASE = _ROOT / "final_run_stockholm_fixed_20260505_1356/processed/features_ml_ready.parquet"
COAST_KM = 30.0
BANDS = ["0-3 km", "3-7 km", "7-15 km", "15-30 km"]
COS_45 = math.cos(math.radians(45.0))


def _haversine_km_vec(lat: np.ndarray, lon: np.ndarray, plat: float, plon: float) -> np.ndarray:
    """Great-circle distance (km) from arrays of (lat,lon) to a point (degrees)."""
    rlat1 = np.radians(lat.astype(float))
    rlon1 = np.radians(lon.astype(float))
    rlat2 = math.radians(plat)
    rlon2 = math.radians(plon)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * math.cos(rlat2) * (np.sin(dlon / 2.0) ** 2)
    h = np.clip(h, 0.0, 1.0)
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(h))


def mask_within_any_port_km(df: pd.DataFrame, ports_path: Path, km: float) -> pd.Series:
    """True where centroid is within `km` of any port in CSV (WGS84)."""
    if km <= 0 or not ports_path.is_file():
        return pd.Series(False, index=df.index)
    pt = pd.read_csv(ports_path)
    if not {"latitude", "longitude"}.issubset(pt.columns):
        return pd.Series(False, index=df.index)
    lat = pd.to_numeric(df["grid_centroid_lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(df["grid_centroid_lon"], errors="coerce").to_numpy(dtype=float)
    mind = np.full(len(df), np.inf, dtype=float)
    for _, row in pt.iterrows():
        plat, plon = float(row["latitude"]), float(row["longitude"])
        d = _haversine_km_vec(lat, lon, plat, plon)
        mind = np.minimum(mind, d)
    return pd.Series(np.isfinite(mind) & (mind <= km), index=df.index)


def safe_corr_spearman(df: pd.DataFrame, a: str, b: str) -> tuple[float, float, int]:
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 8:
        return float("nan"), float("nan"), int(m.sum())
    r, p = stats.spearmanr(x.loc[m], y.loc[m])
    return float(r), float(p), int(m.sum())


def destination_point_km(lat_deg: float, lon_deg: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    """Geodesic offset (short distance) for wind arrow display."""
    R = 6371.0088
    br = math.radians(bearing_deg)
    φ1, λ1 = math.radians(lat_deg), math.radians(lon_deg)
    dr = dist_km / R
    φ2 = math.asin(math.sin(φ1) * math.cos(dr) + math.cos(φ1) * math.sin(dr) * math.cos(br))
    λ2 = λ1 + math.atan2(
        math.sin(br) * math.sin(dr) * math.cos(φ1),
        math.cos(dr) - math.sin(φ1) * math.sin(φ2),
    )
    return math.degrees(φ2), (math.degrees(λ2) + 540.0) % 360.0 - 180.0


def build_wind_clusters(panel_df: pd.DataFrame, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    uniq_g = (
        panel_df.drop_duplicates("grid_cell_id")[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]]
        .dropna()
    )
    n_clust = min(8, max(1, len(uniq_g) // 25))
    clus_lat = np.array([float(uniq_g["grid_centroid_lat"].median())])
    clus_lon = np.array([float(uniq_g["grid_centroid_lon"].median())])
    cmap_series = pd.Series(0, dtype=int, index=uniq_g["grid_cell_id"].astype(str))
    grid_to_cluster = pd.Series(0, index=df.index, dtype=int)
    if len(uniq_g) >= 2 and n_clust > 1:
        km = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
        lab = km.fit_predict(uniq_g[["grid_centroid_lat", "grid_centroid_lon"]].to_numpy(dtype=float))
        cmap_series = pd.Series(lab.astype(int), index=uniq_g["grid_cell_id"].astype(str))
        grid_to_cluster = df["grid_cell_id"].astype(str).map(cmap_series).reindex(df.index).fillna(0).astype(int)
        clus_lat = km.cluster_centers_[:, 0].astype(float)
        clus_lon = km.cluster_centers_[:, 1].astype(float)
    elif len(uniq_g) >= 1:
        grid_to_cluster = df["grid_cell_id"].astype(str).map(cmap_series).reindex(df.index).fillna(0).astype(int)
    return clus_lat, clus_lon, cmap_series, grid_to_cluster


def attach_wind(
    df: pd.DataFrame,
    panel_df: pd.DataFrame,
    cmap_series: pd.Series,
    grid_to_cluster: pd.Series,
    wind_csv: Path | None,
    ne_meta: dict[str, Any],
) -> tuple[pd.DataFrame, bool, pd.DataFrame]:
    panel_uniq = panel_df.drop_duplicates("grid_cell_id")["grid_cell_id"].astype(str)
    ws = pd.to_datetime(df["week_start_utc"], utc=True).min()
    we = pd.to_datetime(df["week_start_utc"], utc=True).max()
    start_d = ws.date().isoformat() if pd.notna(ws) else "2023-01-01"
    end_d = we.date().isoformat() if pd.notna(we) else "2023-12-31"

    wind_long = pd.DataFrame()
    wind_user_loaded = False
    if wind_csv and Path(wind_csv).is_file():
        wu_raw = pd.read_csv(Path(wind_csv))
        wu = normalized_user_wind(wu_raw)
        if not wu.empty and {"grid_cell_id", "wind_u_mean", "wind_v_mean", "week_start_utc"}.issubset(wu.columns):
            wu["grid_cell_id"] = wu["grid_cell_id"].astype(str)
            meta = {**ne_meta, "source": "user_csv"}
            wu["_meta_json"] = json.dumps(meta)
            wm = week_key_monday_start(df["week_start_utc"])
            wind_merge = df[["grid_cell_id"]].copy()
            wind_merge["_wm"] = wm.values
            wind_merge["_idx"] = np.arange(len(wind_merge))
            wmu = wu.copy()
            wmu["_wm"] = week_key_monday_start(wmu["week_start_utc"])
            m2 = wind_merge.merge(
                wmu.drop(columns=["week_start_utc"], errors="ignore"),
                left_on=["grid_cell_id", "_wm"],
                right_on=["grid_cell_id", "_wm"],
                how="left",
            )
            m2 = m2.sort_values("_idx")
            df["wind_cluster_id"] = grid_to_cluster.values
            for col in (
                "wind_u_mean",
                "wind_v_mean",
                "wind_speed_mean",
                "wind_direction_to_degrees",
                "wind_direction_from_degrees",
            ):
                df[col] = m2[col].values if col in m2.columns else np.nan
            df["wind_direction_degrees"] = df["wind_direction_from_degrees"]
            df["wind_data_source"] = "user_csv"
            wind_user_loaded = True

    if not wind_user_loaded:
        clus_lat, clus_lon, _, _ = build_wind_clusters(panel_df, df)
        print(f"[INFO] Fetching Open‑Meteo ERA5 archive ({start_d}–{end_d}), {len(clus_lat)} wind cluster(s)...")
        wind_long = fetch_open_meteo_cluster_wind(
            cluster_lat=np.asarray(clus_lat, dtype=float),
            cluster_lon=np.asarray(clus_lon, dtype=float),
            start_date=start_d,
            end_date=end_d,
        )
        cmap_lookup = cmap_series.to_dict()
        wf_list_inner: list[pd.DataFrame] = []
        if not wind_long.empty:
            for gid_str in panel_uniq.values:
                cid = int(cmap_lookup.get(gid_str, 0))
                subw = wind_long.loc[wind_long["cluster_id"] == cid].copy()
                if subw.empty:
                    continue
                subw.insert(0, "grid_cell_id", gid_str)
                wf_list_inner.append(subw.drop(columns=["cluster_id"], errors="ignore"))
        wf_expanded = pd.concat(wf_list_inner, ignore_index=True) if wf_list_inner else pd.DataFrame()
        if wf_expanded.empty:
            df = apply_wind_to_dataframe(df, grid_to_cluster, pd.DataFrame())
        else:
            wf_expanded["_meta_json"] = json.dumps({**ne_meta, "source": "open_meteo_era5_archive"})
            df = apply_wind_to_dataframe(df, grid_to_cluster, wind_long)

    return df, wind_user_loaded, wind_long


def summary_rows_append(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    rows.append(dict(**kwargs))


def run_stats(Pan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for target in ("local_no2_excess", "weekly_no2_anomaly"):
        if target not in Pan.columns:
            continue
        r, p, n = safe_corr_spearman(Pan, target, "coastal_wind_alignment_score")
        summary_rows_append(
            rows,
            test=f"spearman_{target}_vs_coastal_wind_alignment",
            statistic="spearman_r",
            value=r,
            p_value=p,
            n=n,
            notes="Receptor landward bearing vs wind toward (see methodology).",
        )

    r2, p2, n2 = safe_corr_spearman(Pan, "local_no2_excess", "pollution_transport_wind_alignment_score")
    summary_rows_append(
        rows,
        test="spearman_local_no2_excess_vs_pollution_transport_alignment",
        statistic="spearman_r",
        value=r2,
        p_value=p2,
        n=n2,
        notes="Hotspot→coast bearing vs wind toward at receptor (weekly hotspot pool).",
    )

    vd = pd.to_numeric(Pan["vessel_density_t"], errors="coerce")
    ca = pd.to_numeric(Pan["coastal_wind_alignment_score"], errors="coerce")
    y = pd.to_numeric(Pan["local_no2_excess"], errors="coerce")
    prod = vd * ca
    m = prod.notna() & y.notna()
    if int(m.sum()) >= 12:
        r, p = stats.spearmanr(prod.loc[m], y.loc[m])
        n = int(m.sum())
        summary_rows_append(
            rows,
            test="spearman_no2_vs_vessel_times_coastal_alignment",
            statistic="spearman_r",
            value=float(r),
            p_value=float(p),
            n=n,
            notes="Naive maritime×shoreward‑wind coupling (not a formal interaction test).",
        )

    hi_v = vd >= vd.quantile(0.9)
    sub_hi = Pan.loc[hi_v & ca.notna() & y.notna()].copy()
    if len(sub_hi) >= 12:
        r, p, n = safe_corr_spearman(sub_hi, "local_no2_excess", "coastal_wind_alignment_score")
        summary_rows_append(
            rows,
            test="spearman_no2_vs_coastal_alignment_within_high_vessel_p90",
            statistic="spearman_r",
            value=r,
            p_value=p,
            n=n,
            notes="Restrict to high vessel-density rows (weekly cross-section pooled).",
        )

    ox = pd.to_numeric(Pan["oil_slick_probability_t"], errors="coerce") if "oil_slick_probability_t" in Pan.columns else pd.Series(dtype=float)
    ox_al = ox * pd.to_numeric(Pan["coastal_wind_alignment_score"], errors="coerce")
    if "nearest_land_ndvi_mean" in Pan.columns:
        nv = pd.to_numeric(Pan["nearest_land_ndvi_mean"], errors="coerce")
        m3 = ox_al.notna() & nv.notna()
        if int(m3.sum()) >= 12:
            r, p = stats.spearmanr(ox_al.loc[m3], nv.loc[m3])
            nn = int(m3.sum())
            summary_rows_append(
                rows,
                test="spearman_ndvi_vs_oil_x_coastal_alignment",
                statistic="spearman_r",
                value=float(r),
                p_value=float(p),
                n=nn,
                notes="Oil×shorewind vs NDVI (exploratory land response; confounded).",
            )

    for band in BANDS:
        slab = Pan.loc[Pan["shipping_distance_band_tight"] == band]
        if len(slab) < 8:
            continue
        shore = pd.to_numeric(slab["coastal_wind_alignment_score"], errors="coerce") >= COS_45
        yb = pd.to_numeric(slab["local_no2_excess"], errors="coerce")
        a = yb.loc[shore].dropna().to_numpy(dtype=float)
        b = yb.loc[~shore].dropna().to_numpy(dtype=float)
        if len(a) >= 4 and len(b) >= 4:
            p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            summary_rows_append(
                rows,
                test=f"mannwhitney_no2_excess_shoreward_wind_vs_not_band_{band.replace(' ', '_')}",
                statistic="mean_diff_shoreward_minus_other",
                value=float(np.mean(a) - np.mean(b)),
                p_value=p,
                n=int(len(a) + len(b)),
                notes="Shoreward wind: coastal_wind_alignment_score >= cos(45°).",
            )

    df_decay = []
    for band in BANDS:
        slab = Pan.loc[Pan["shipping_distance_band_tight"] == band]
        x = pd.to_numeric(slab["local_no2_excess"], errors="coerce").dropna()
        if len(x):
            df_decay.append(
                {
                    "distance_band": band,
                    "n": len(x),
                    "mean_local_no2_excess": float(x.mean()),
                    "median_local_no2_excess": float(x.median()),
                    "mean_coastal_wind_alignment": float(
                        pd.to_numeric(slab["coastal_wind_alignment_score"], errors="coerce").mean(),
                    ),
                },
            )
    if df_decay:
        dd = pd.DataFrame(df_decay)
        for _, r in dd.iterrows():
            summary_rows_append(
                rows,
                test=f"decay_profile_mean_no2_band_{str(r['distance_band']).replace(' ', '_')}",
                statistic="mean_local_no2_excess",
                value=float(r["mean_local_no2_excess"]),
                p_value=np.nan,
                n=int(r["n"]),
                notes="Pooled mean NO2 excess by shipping-distance band (not causal).",
            )

    return pd.DataFrame(rows)


def write_interpretation_md(summary: pd.DataFrame, Pan: pd.DataFrame, meta: dict[str, Any]) -> None:
    def _rp(p: Path) -> str:
        try:
            return str(p.relative_to(_ROOT))
        except ValueError:
            return str(p)

    n = len(Pan)
    ngrid = Pan["grid_cell_id"].nunique()
    nweek = Pan["week_start_utc"].nunique()

    findings: list[str] = []
    if summary is not None and len(summary):
        sg = summary[summary["statistic"] == "spearman_r"].copy()
        sg = sg[pd.to_numeric(sg["p_value"], errors="coerce") < 0.05].sort_values("p_value")
        for _, r in sg.head(6).iterrows():
            findings.append(
                f"- **{r['test']}**: rho={float(r['value']):.3f}, p={float(r['p_value']):.3g}, n={int(r['n'])}.",
            )

    lines = [
        "# Coastal wind transport — interpretation (thesis draft)",
        "",
        "## Purpose",
        "Directional exposure and association framing for maritime signals, coastal wind, NO2 and oil proxies. Not proof of deposition, health harm, or emission attribution.",
        "",
        "## Methodology (summary)",
        "- **Landward bearing (receptor):** bearing from each grid centroid to the nearest Natural Earth coastline sample (or land-boundary fallback). Compared to wind toward from weekly mean u/v (Open-Meteo ERA5 archive or `--wind-csv`). `coastal_wind_alignment_score = cos(smallest angle difference)`.",
        "- **Hotspot corridor:** each week, top-decile vessel density, NO2, or oil slick proxy define a combined hotspot pool; each row gets the nearest hotspot. Bearing from hotspot to nearest coastline sample yields `pollution_transport_wind_alignment_score` vs wind toward at the receptor.",
        "- **Panel:** same coastal mask as `run_land_pollution_drivers_wind` (distance to coast <=30 km; shipping distance <30 km). 0.1 deg grid via `grid_cell_id`.",
        "",
        "## Wind conventions (validation)",
        "- Open-Meteo `wind_direction_10m` is meteorological FROM; u east, v north; wind toward = atan2(u,v) deg from N.",
        "- Initial bearing: forward azimuth point1 to point2, clockwise from north.",
        "- Angles use smallest arc in [0,180] before cosine.",
        "",
        "## Assumptions",
        "- Weekly aggregation is informative for NO2 and proxies.",
        "- Cluster-representative wind approximates local flow.",
        "- Nearest coastline sample approximates landward normal for marine-heavy cells; coarse for some land cells.",
        "",
        "## Limitations",
        "- Confounding (urban NO2, weather, sensors, SAR speckle).",
        "- 110m shoreline simplification.",
        "- Scalar cosine cannot prove plume paths.",
        "- **Negative correlation is not evidence against transport**; it can reflect mixing, sources, or geometry.",
        "",
        "## Quantitative snapshots",
        "",
        f"- Coastal panel rows: **{n:,}**, grids **{ngrid}**, weeks **{nweek}**.",
        f"- Wind source: **{meta.get('wind_source', '?')}**.",
        "",
        "## Strongest statistical signals (exploratory)",
        "",
    ]
    if findings:
        lines.extend(findings)
    else:
        lines.append("- (No p<0.05 Spearman rows in summary.)")
    lines.extend(
        [
            "",
            "### Full test table (`coastal_wind_exposure_summary.csv`)",
            "",
        ],
    )
    if summary is not None and len(summary):
        try:
            table = summary.head(22).to_markdown(index=False)
        except ImportError:
            table = summary.head(22).to_string(index=False)
        lines.append(table)
        lines.append("")
    lines.extend(
        [
            "## Strongest cautions",
            "- Association / exposure language only; avoid causal transport claims.",
            "",
            "## Files (repo-relative)",
            f"- `{_rp(REPORTS / 'coastal_wind_alignment_features.csv')}`",
            f"- `{_rp(REPORTS / 'coastal_wind_exposure_summary.csv')}`",
            f"- `{_rp(VIZ / 'coastal_pollution_transport_map.html')}`",
            f"- `{_rp(FIGURES)}/` — includes `coastal_pollution_transport_context_map.png` (quiver + shoreline alignment coloring).",
            "- Example augment: `python3 src/analysis/run_coastal_wind_transport.py --augment-parquet outputs/processed/features_ml_ready_coastal_wind.parquet`",
            "",
        ],
    )
    INTERPRET_MD.parent.mkdir(parents=True, exist_ok=True)
    INTERPRET_MD.write_text("\n".join(lines), encoding="utf-8")


def build_folium_map(
    Pan_u: pd.DataFrame,
    coast_lat: np.ndarray,
    coast_lon: np.ndarray,
    rep_week: pd.Timestamp,
) -> None:
    try:
        import folium
        from folium.plugins import AntPath, Fullscreen
    except ImportError:
        print("[WARN] folium not installed; skip HTML map.")
        return

    mlat = float(Pan_u["grid_centroid_lat"].median())
    mlon = float(Pan_u["grid_centroid_lon"].median())
    m = folium.Map(location=[mlat, mlon], zoom_start=7, tiles="CartoDB positron")

    step = max(1, len(coast_lat) // 400)
    pts = [(float(lat), float(lon)) for lat, lon in zip(coast_lat[::step], coast_lon[::step])]
    AntPath(pts, color="#003366", weight=2, opacity=0.6, dash_array=[10, 20]).add_to(m)

    gm_cells = Pan_u.drop_duplicates("grid_cell_id")[
        ["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]
    ].assign(grid_cell_id=lambda x: x["grid_cell_id"].astype(str))

    wk_panel = Pan_u.copy()
    wk_panel["week_start_utc"] = pd.to_datetime(wk_panel["week_start_utc"], utc=True).dt.normalize()
    rw = pd.to_datetime(rep_week, utc=True).normalize()
    one = wk_panel[wk_panel["week_start_utc"] == rw].copy()
    if one.empty:
        cols_med = [
            c
            for c in (
                "coastal_wind_alignment_score",
                "pollution_transport_wind_alignment_score",
                "wind_u_mean",
                "wind_v_mean",
                "wind_speed_mean",
            )
            if c in Pan_u.columns and pd.api.types.is_numeric_dtype(Pan_u[c])
        ]
        if not cols_med:
            cols_med = ["coastal_wind_alignment_score"] if "coastal_wind_alignment_score" in Pan_u.columns else []
        one = Pan_u.groupby("grid_cell_id", as_index=False)[cols_med].median(numeric_only=True)
        one["grid_cell_id"] = one["grid_cell_id"].astype(str)
        one = one.merge(gm_cells, on="grid_cell_id", how="left")
    else:
        one = one.copy()
        one["grid_cell_id"] = one["grid_cell_id"].astype(str)

    vmin = float(np.nanpercentile(one["coastal_wind_alignment_score"].to_numpy(dtype=float), 5))
    vmax = float(np.nanpercentile(one["coastal_wind_alignment_score"].to_numpy(dtype=float), 95))
    if not np.isfinite(vmin):
        vmin, vmax = -1.0, 1.0

    for _, r in one.drop_duplicates("grid_cell_id").iterrows():
        sc = float(r.get("coastal_wind_alignment_score", np.nan))
        if not np.isfinite(sc):
            color = "#888888"
        else:
            t = (sc - vmin) / (vmax - vmin + 1e-9)
            t = max(0.0, min(1.0, t))
            color = plt.cm.RdYlGn(t)[:3]
            color = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
        folium.CircleMarker(
            location=[float(r["grid_centroid_lat"]), float(r["grid_centroid_lon"])],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.75,
            popup=(
                f"{r['grid_cell_id']}<br/>coastal_align={sc:.2f}<br/>"
                f"transport_align={float(r.get('pollution_transport_wind_alignment_score', np.nan)):.2f}"
            ),
        ).add_to(m)

        u = float(r.get("wind_u_mean", np.nan))
        v = float(r.get("wind_v_mean", np.nan))
        spd = float(r.get("wind_speed_mean", np.nan))
        if np.isfinite(u) and np.isfinite(v) and np.isfinite(spd):
            wto = float(wind_to_direction_deg(np.array([u]), np.array([v]))[0])
            la, lo = float(r["grid_centroid_lat"]), float(r["grid_centroid_lon"])
            la2, lo2 = destination_point_km(la, lo, wto, min(25.0, 8.0 + 0.15 * spd))
            folium.PolyLine([(la, lo), (la2, lo2)], color="#1f78b4", weight=2, opacity=0.85).add_to(m)

    snap = one.drop_duplicates("grid_cell_id")
    if "no2_mean_t" in snap.columns and pd.to_numeric(snap["no2_mean_t"], errors="coerce").notna().sum() > 5:
        thr_n = float(pd.to_numeric(snap["no2_mean_t"], errors="coerce").quantile(0.9))
        for _, r in snap.iterrows():
            nv = pd.to_numeric(r.get("no2_mean_t"), errors="coerce")
            if np.isfinite(nv) and nv >= thr_n:
                folium.CircleMarker(
                    location=[float(r["grid_centroid_lat"]), float(r["grid_centroid_lon"])],
                    radius=6,
                    color="darkorange",
                    weight=2,
                    fill=False,
                    popup="NO2 hotspot (p90 this week snapshot)",
                ).add_to(m)
    if "oil_slick_probability_t" in snap.columns and pd.to_numeric(snap["oil_slick_probability_t"], errors="coerce").notna().sum() > 5:
        thr_o = float(pd.to_numeric(snap["oil_slick_probability_t"], errors="coerce").quantile(0.9))
        for _, r in snap.iterrows():
            ov = pd.to_numeric(r.get("oil_slick_probability_t"), errors="coerce")
            if np.isfinite(ov) and ov >= thr_o:
                folium.CircleMarker(
                    location=[float(r["grid_centroid_lat"]), float(r["grid_centroid_lon"])],
                    radius=7,
                    color="crimson",
                    weight=2,
                    fill=False,
                    popup="Oil slick proxy hotspot (p90 this week snapshot)",
                ).add_to(m)

    vm = Pan_u.groupby("grid_cell_id", as_index=False)["vessel_density_t"].median()
    hi_v_thr = float(vm["vessel_density_t"].quantile(0.9))
    hi_cells = vm.loc[pd.to_numeric(vm["vessel_density_t"], errors="coerce") >= hi_v_thr].copy()
    hi_cells["grid_cell_id"] = hi_cells["grid_cell_id"].astype(str)
    hi_cells = hi_cells.merge(gm_cells, on="grid_cell_id", how="left")
    hi_cells = hi_cells.dropna(subset=["grid_centroid_lat"])
    for _, r in hi_cells.head(120).iterrows():
        folium.RegularPolygonMarker(
            location=[float(r["grid_centroid_lat"]), float(r["grid_centroid_lon"])],
            number_of_sides=3,
            radius=7,
            color="navy",
            fill_color="cyan",
            fill_opacity=0.7,
            popup="High vessel-density cell (spatial median over weeks)",
        ).add_to(m)

    fg = folium.FeatureGroup(name="Legend")
    fg.add_child(
        folium.Marker(
            [mlat + 0.35, mlon + 0.35],
            icon=folium.DivIcon(
                html=(
                    "<div style='font-size:11px;width:300px;background:#fff;padding:6px;"
                    "border:1px solid #ccc'>Teal dashed path: NE coast sampling. "
                    "Colored discs: shoreline alignment. Blue lines: mean wind toward. "
                    "Cyan triangles: high vessel-density cells. Orange rings: weekly NO2 p90. "
                    "Red rings: weekly oil-slick-proxy p90.</div>"
                ),
            ),
        ),
    )
    fg.add_to(m)

    Fullscreen().add_to(m)
    VIZ.mkdir(parents=True, exist_ok=True)
    m.save(str(VIZ / "coastal_pollution_transport_map.html"))
    print(f"[OK] saved {VIZ / 'coastal_pollution_transport_map.html'}")


def plot_coastal_context_map(
    Pan: pd.DataFrame,
    coast_lat: np.ndarray,
    coast_lon: np.ndarray,
) -> None:
    """Matplotlib context map: shoreline sample, grids colored by mean coastal alignment, wind quivers."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    byg = Pan.groupby("grid_cell_id")[
        ["coastal_wind_alignment_score", "grid_centroid_lat", "grid_centroid_lon"]
    ].agg(
        coastal_wind_alignment_mean=("coastal_wind_alignment_score", "mean"),
        grid_centroid_lat=("grid_centroid_lat", "first"),
        grid_centroid_lon=("grid_centroid_lon", "first"),
    )
    uw = Pan.groupby("grid_cell_id")[["wind_u_mean", "wind_v_mean"]].mean().reset_index()
    plat = pd.merge(byg.reset_index(), uw, on="grid_cell_id")
    lon = plat["grid_centroid_lon"].to_numpy(dtype=float)
    lat = plat["grid_centroid_lat"].to_numpy(dtype=float)
    sc = plat["coastal_wind_alignment_mean"].to_numpy(dtype=float)
    u = plat["wind_u_mean"].to_numpy(dtype=float)
    v = plat["wind_v_mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 8.2))
    step = max(1, len(coast_lat) // 300)
    ax.plot(coast_lon[::step], coast_lat[::step], "-", color="#003366", lw=1.2, alpha=0.65, label="Coast sampling")
    sca = ax.scatter(lon, lat, c=sc, cmap="RdYlGn", vmin=-1, vmax=1, s=55, edgecolors="k", linewidths=0.4, zorder=4)
    plt.colorbar(sca, ax=ax, label="Mean coastal_wind_alignment_score", shrink=0.66)
    scale = 0.09
    ax.quiver(
        lon,
        lat,
        u * scale,
        v * scale,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        color="navy",
        alpha=0.75,
        zorder=3,
        label="Mean wind u,v (+ east, + north)",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Coastal pollution transport context\n(association-only; directional exposure)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "coastal_pollution_transport_context_map.png", dpi=160)
    plt.close(fig)


def plot_figures(Pan: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ca = pd.to_numeric(Pan["coastal_wind_alignment_score"], errors="coerce")
    if ca.notna().any():
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(ca.dropna(), bins=36, kde=True, ax=ax, color="darkgreen")
        ax.axvline(COS_45, color="crimson", ls="--", label="cos(45 deg) shoreward threshold")
        ax.set_title("Coastal wind alignment score (receptor landward)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "coastal_wind_alignment_hist.png", dpi=160)
        plt.close(fig)

    y = pd.to_numeric(Pan["local_no2_excess"], errors="coerce")
    m = ca.notna() & y.notna()
    if int(m.sum()) > 30:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        hb = ax.hexbin(ca.loc[m], y.loc[m], gridsize=30, cmap="viridis", mincnt=1)
        plt.colorbar(hb, ax=ax, label="count")
        r, p, n = safe_corr_spearman(Pan.loc[m], "coastal_wind_alignment_score", "local_no2_excess")
        ax.text(
            0.02,
            0.98,
            f"Spearman rho={r:.3f}\np={p:.3g}\nn={n}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
        ax.set_xlabel("coastal_wind_alignment_score")
        ax.set_ylabel("local_no2_excess")
        ax.set_title("NO2 excess vs coastal (landward) wind alignment")
        fig.tight_layout()
        fig.savefig(FIGURES / "no2_vs_coastal_wind_alignment_hexbin.png", dpi=160)
        plt.close(fig)

    ox = pd.to_numeric(Pan["oil_slick_probability_t"], errors="coerce") if "oil_slick_probability_t" in Pan.columns else pd.Series(index=Pan.index, dtype=float)
    m3 = ca.notna() & ox.notna()
    if int(m3.sum()) > 30:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        hb = ax.hexbin(ca.loc[m3], ox.loc[m3], gridsize=28, cmap="OrRd", mincnt=1)
        plt.colorbar(hb, ax=ax, label="count")
        r2, p2, nn = safe_corr_spearman(Pan.loc[m3], "coastal_wind_alignment_score", "oil_slick_probability_t")
        ax.text(
            0.02,
            0.98,
            f"Spearman rho={r2:.3f}\np={p2:.3g}\nn={nn}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
        ax.set_xlabel("coastal_wind_alignment_score")
        ax.set_ylabel("oil_slick_probability_t")
        ax.set_title("Oil slick proxy vs coastal wind alignment")
        fig.tight_layout()
        fig.savefig(FIGURES / "oil_vs_coastal_wind_alignment_hexbin.png", dpi=160)
        plt.close(fig)

    rows = []
    for band in BANDS:
        slab = Pan.loc[Pan["shipping_distance_band_tight"] == band]
        x = pd.to_numeric(slab["local_no2_excess"], errors="coerce").dropna()
        rows.append({"band": band, "mean": float(x.mean()) if len(x) else np.nan, "n": len(x)})
    if any(np.isfinite(r["mean"]) for r in rows):
        dd = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.bar(np.arange(len(dd)), dd["mean"], color="steelblue", alpha=0.85)
        ax.set_xticks(np.arange(len(dd)))
        ax.set_xticklabels(dd["band"], rotation=15)
        ax.set_ylabel("Mean local NO2 excess")
        ax.set_title("Inland decay context: NO2 excess by shipping-distance band (pooled)")
        fig.tight_layout()
        fig.savefig(FIGURES / "no2_excess_by_shipping_band_coastal_wind_run.png", dpi=160)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--linked-path", type=Path, default=LINKED_PATH)
    ap.add_argument("--ne-cache", type=Path, default=_ROOT / "data" / "aux" / "natural_earth_coast_cache")
    ap.add_argument("--wind-csv", type=Path, default=None)
    ap.add_argument("--hotspot-quantile", type=float, default=0.90)
    ap.add_argument("--augment-parquet", type=Path, default=None, help="Write merged full panel to this path")
    ap.add_argument(
        "--union-port-buffers-km",
        type=float,
        default=0.0,
        help="If >0, OR the export panel with all rows within this distance (km) of any port in --ports-csv "
        "(recovers wind rows for port buffer cells not in the coastal×shipping grid). Fetches real ERA5 via "
        "existing Open-Meteo path; does not invent wind values.",
    )
    ap.add_argument(
        "--ports-csv",
        type=Path,
        default=_ROOT / "data" / "aux" / "baltic_ports.csv",
        help="Port coordinates for --union-port-buffers-km.",
    )
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)

    base = Path(args.input) if args.input else DEFAULT_BASE
    if not base.is_file():
        print(f"[FATAL] missing base parquet {base}")
        return 1

    linked_ok = args.linked_path.is_file()
    if linked_ok:
        print(f"[INFO] Loading linked parquet: {args.linked_path}")
        df = pd.read_parquet(args.linked_path)
    else:
        print(f"[INFO] Using base parquet: {base}")
        df = pd.read_parquet(base)

    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    df["grid_cell_id"] = df["grid_cell_id"].astype(str)
    df = ensure_coast_distance(df, Path(args.ne_cache))

    need = [
        "vessel_density_t",
        "distance_to_nearest_high_vessel_density_cell",
        "oil_slick_probability_t",
        "no2_mean_t",
        "grid_centroid_lat",
        "grid_centroid_lon",
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"[FATAL] missing columns: {miss}")
        return 1

    cfg = BufferingConfig()
    df = attach_nearest_high_vessel_seed(df, q=float(cfg.high_activity_quantile))
    df = attach_nearest_multivariate_hotspot(
        df,
        quantile=float(args.hotspot_quantile),
    )

    dc = pd.to_numeric(df["distance_to_coast_km"], errors="coerce")
    ds = pd.to_numeric(df["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    panel_m = dc.notna() & (dc <= COAST_KM) & ds.notna() & (ds >= 0) & (ds < 30)
    base_panel_mask = panel_m.copy()
    ukm = float(getattr(args, "union_port_buffers_km", 0.0) or 0.0)
    if ukm > 0:
        port_ring = mask_within_any_port_km(df, Path(args.ports_csv), ukm)
        added = int((port_ring & ~base_panel_mask).sum())
        panel_m = base_panel_mask | port_ring
        print(
            f"[INFO] union_port_buffers_km={ukm} (ports {args.ports_csv}): "
            f"OR panel adds {added} rows beyond coastal×shipping panel; total panel rows {int(panel_m.sum())}",
        )
    df["shipping_distance_band_tight"] = assign_shipping_band_tight(pd.Series(ds, index=df.index))

    no2 = pd.to_numeric(df["no2_mean_t"], errors="coerce")
    wm = df.groupby(df["week_start_utc"].dt.normalize())["no2_mean_t"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").mean(),
    )
    df["weekly_no2_anomaly"] = no2 - wm
    ow = df["week_start_utc"].dt.normalize()
    ref_band = df["shipping_distance_band_tight"] == "15-30 km"
    band_mean = (
        df.loc[ref_band].groupby(df.loc[ref_band, "week_start_utc"].dt.normalize())["no2_mean_t"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    )
    df["local_no2_excess"] = no2 - ow.map(band_mean.get)

    coast_pts = load_coastline_points(Path(args.ne_cache)) or load_land_boundary_points(Path(args.ne_cache))
    if coast_pts is None:
        print("[FATAL] could not load coastline / land boundary for bearing geometry")
        return 1
    coast_lat, coast_lon = coast_pts

    uniq = (
        df.loc[panel_m, ["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]]
        .drop_duplicates("grid_cell_id")
        .dropna()
    )
    coast_tab = grid_nearest_coast_reference_table(
        uniq,
        coast_lat=np.asarray(coast_lat, dtype=float),
        coast_lon=np.asarray(coast_lon, dtype=float),
    )
    df = df.merge(coast_tab, on="grid_cell_id", how="left")

    gla = pd.to_numeric(df["grid_centroid_lat"], errors="coerce").to_numpy()
    glo = pd.to_numeric(df["grid_centroid_lon"], errors="coerce").to_numpy()
    rla = pd.to_numeric(df["nearest_coast_ref_lat"], errors="coerce").to_numpy()
    rlo = pd.to_numeric(df["nearest_coast_ref_lon"], errors="coerce").to_numpy()
    df["bearing_cell_to_coast_deg"] = initial_bearing_deg(gla, glo, rla, rlo)

    hsla = pd.to_numeric(df["pollution_hotspot_lat"], errors="coerce").to_numpy()
    hslo = pd.to_numeric(df["pollution_hotspot_lon"], errors="coerce").to_numpy()
    _, hsc_rla, hsc_rlo = nearest_geodesic_reference(hsla, hslo, coast_lat, coast_lon)
    valid_hs = np.isfinite(hsla) & np.isfinite(hslo)
    bear_hs = np.full(len(df), np.nan)
    bear_hs[valid_hs] = initial_bearing_deg(
        hsla[valid_hs],
        hslo[valid_hs],
        hsc_rla[valid_hs].astype(float),
        hsc_rlo[valid_hs].astype(float),
    )
    df["bearing_hotspot_to_coast_deg"] = bear_hs

    panel_df = df.loc[panel_m].copy()
    clus_lat, clus_lon, cmap_series, grid_to_cluster = build_wind_clusters(panel_df, df)
    ne_meta = {"clusters": int(len(clus_lat)), "bearing_method": "cell_to_nearest_ne_coast_sample"}

    df, wind_ul, wind_long = attach_wind(
        df,
        panel_df,
        cmap_series,
        grid_to_cluster,
        args.wind_csv,
        ne_meta,
    )
    meta_out = {
        "wind_source": ("user_csv" if wind_ul else ("open_meteo_era5_archive" if len(wind_long) else "none")),
        **ne_meta,
    }

    u_w = pd.to_numeric(df["wind_u_mean"], errors="coerce").to_numpy()
    v_w = pd.to_numeric(df["wind_v_mean"], errors="coerce").to_numpy()
    wind_to = wind_to_direction_deg(u_w, v_w)

    delta_coast = smallest_angle_deg(
        pd.to_numeric(df["bearing_cell_to_coast_deg"], errors="coerce").to_numpy(),
        wind_to,
    )
    df["coastal_wind_angle_diff_deg"] = delta_coast
    df["coastal_wind_alignment_score"] = np.cos(np.radians(delta_coast))
    df["coastal_wind_shoreward_45deg"] = (df["coastal_wind_alignment_score"] >= COS_45).astype(float)

    hs_bear = pd.to_numeric(df["bearing_hotspot_to_coast_deg"], errors="coerce").to_numpy()
    delta_tr = smallest_angle_deg(hs_bear, wind_to)
    df["pollution_transport_angle_diff_deg"] = delta_tr
    df["pollution_transport_wind_alignment_score"] = np.cos(np.radians(delta_tr))

    cols_out = [
        "grid_cell_id",
        "week_start_utc",
        "nearest_coast_ref_lat",
        "nearest_coast_ref_lon",
        "nearest_coast_ref_distance_km",
        "bearing_cell_to_coast_deg",
        "bearing_hotspot_to_coast_deg",
        "pollution_hotspot_lat",
        "pollution_hotspot_lon",
        "pollution_hotspot_type",
        "wind_u_mean",
        "wind_v_mean",
        "wind_speed_mean",
        "wind_direction_to_degrees",
        "coastal_wind_angle_diff_deg",
        "coastal_wind_alignment_score",
        "coastal_wind_shoreward_45deg",
        "pollution_transport_angle_diff_deg",
        "pollution_transport_wind_alignment_score",
        "distance_to_coast_km",
        "shipping_distance_band_tight",
        "vessel_density_t",
        "local_no2_excess",
        "weekly_no2_anomaly",
        "oil_slick_probability_t",
    ]
    if "nearest_land_ndvi_mean" in df.columns:
        cols_out.append("nearest_land_ndvi_mean")
    have = [c for c in cols_out if c in df.columns]

    csv_path = REPORTS / "coastal_wind_alignment_features.csv"
    df.loc[panel_m, have].to_csv(csv_path, index=False)
    print(f"[OK] wrote {csv_path}")

    Pan = df.loc[panel_m].copy()
    summ = run_stats(Pan)
    summ.to_csv(REPORTS / "coastal_wind_exposure_summary.csv", index=False)
    print(f"[OK] wrote {REPORTS / 'coastal_wind_exposure_summary.csv'}")

    plot_figures(Pan)
    plot_coastal_context_map(Pan, np.asarray(coast_lat, dtype=float), np.asarray(coast_lon, dtype=float))

    weeks_sorted = sorted(Pan["week_start_utc"].dropna().unique())
    rep_week = pd.Timestamp(weeks_sorted[len(weeks_sorted) // 2]) if weeks_sorted else Pan["week_start_utc"].iloc[0]
    build_folium_map(Pan, np.asarray(coast_lat, dtype=float), np.asarray(coast_lon, dtype=float), rep_week)

    write_interpretation_md(summ, Pan, meta_out)

    if args.augment_parquet:
        aug_cols = [
            "coastal_wind_alignment_score",
            "bearing_cell_to_coast_deg",
            "coastal_wind_angle_diff_deg",
            "coastal_wind_shoreward_45deg",
            "pollution_transport_wind_alignment_score",
            "bearing_hotspot_to_coast_deg",
            "pollution_hotspot_lat",
            "pollution_hotspot_lon",
            "pollution_hotspot_type",
            "nearest_coast_ref_distance_km",
        ]
        add = df[["grid_cell_id", "week_start_utc"] + [c for c in aug_cols if c in df.columns]].copy()
        base_df = pd.read_parquet(base)
        base_df["grid_cell_id"] = base_df["grid_cell_id"].astype(str)
        base_df["week_start_utc"] = pd.to_datetime(base_df["week_start_utc"], utc=True).dt.normalize()
        add["week_start_utc"] = pd.to_datetime(add["week_start_utc"], utc=True).dt.normalize()
        merged = base_df.merge(
            add,
            on=["grid_cell_id", "week_start_utc"],
            how="left",
            suffixes=("", "_cw"),
        )
        Path(args.augment_parquet).parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(Path(args.augment_parquet), index=False)
        print(f"[OK] wrote augmented parquet {args.augment_parquet} rows={len(merged)}")

    print("Done. See coastal_wind_transport_interpretation.md for thesis framing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
