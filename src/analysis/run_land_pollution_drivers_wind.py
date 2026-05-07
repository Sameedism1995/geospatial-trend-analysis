#!/usr/bin/env python3
"""
Land-side pollution driver analysis: NO₂, oil slick proxy, weekly wind, shipping exposure.

Writes ONLY under outputs/{reports,figures,visualizations}/run_land_pollution_drivers_wind/

Default wind: Open-Meteo ERA5 archive at regional cluster centroids (see wind_features_weekly.csv metadata).

Run:
  python3 src/analysis/run_land_pollution_drivers_wind.py
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request

import certifi
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from features.land_sea_buffering import BufferingConfig  # noqa: E402
from human_impact_distance_analysis import (  # noqa: E402
    distance_to_coast_km_for_grids,
    load_coastline_points,
    load_land_boundary_points,
)

RUN = "run_land_pollution_drivers_wind"
REPORTS = _ROOT / "outputs" / "reports" / RUN
FIGURES = _ROOT / "outputs" / "figures" / RUN
VIZ = _ROOT / "outputs" / "visualizations" / RUN

LINKED_PATH = _ROOT / "outputs/reports/run_nearest_land_ndvi_linkage/nearest_land_ndvi_linked_dataset.parquet"
COAST_KM = 30.0
BANDS = ["0-3 km", "3-7 km", "7-15 km", "15-30 km"]
EARTH_R = 6371.0088


def ensure_coast_distance(df: pd.DataFrame, cache: Path) -> pd.DataFrame:
    out = df.copy()
    if (
        "distance_to_coast_km" in out.columns
        and float(out["distance_to_coast_km"].notna().mean()) > 0.99
    ):
        return out
    uniq = (
        out[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]]
        .drop_duplicates("grid_cell_id")
        .dropna(subset=["grid_centroid_lat", "grid_centroid_lon"])
    )
    pts = load_coastline_points(cache) or load_land_boundary_points(cache)
    if pts is None:
        out["distance_to_coast_km"] = np.nan
        return out
    lat0, lon0 = pts
    dc = distance_to_coast_km_for_grids(
        uniq["grid_centroid_lat"].to_numpy(dtype=float),
        uniq["grid_centroid_lon"].to_numpy(dtype=float),
        lat0,
        lon0,
    )
    out["distance_to_coast_km"] = (
        out["grid_cell_id"].astype(str).map(pd.Series(dc, index=uniq["grid_cell_id"].astype(str)))
    )
    return out


def assign_shipping_band_tight(dship: pd.Series) -> pd.Series:
    x = pd.to_numeric(dship, errors="coerce")
    out = pd.Series(np.nan, index=dship.index, dtype=object)
    out.loc[(x >= 0) & (x < 3)] = BANDS[0]
    out.loc[(x >= 3) & (x < 7)] = BANDS[1]
    out.loc[(x >= 7) & (x < 15)] = BANDS[2]
    out.loc[(x >= 15) & (x < 30)] = BANDS[3]
    return out


def haversine_km_broadcast(
    q_lat: np.ndarray,
    q_lon: np.ndarray,
    s_lat: np.ndarray,
    s_lon: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(q_lat)[:, None]
    lon1 = np.radians(q_lon)[:, None]
    lat2 = np.radians(s_lat)[None, :]
    lon2 = np.radians(s_lon)[None, :]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def attach_nearest_high_vessel_seed(
    df: pd.DataFrame,
    *,
    vessel_col: str = "vessel_density_t",
    lat_col: str = "grid_centroid_lat",
    lon_col: str = "grid_centroid_lon",
    week_col: str = "week_start_utc",
    q: float = 0.90,
) -> pd.DataFrame:
    """Match BufferingConfig.high_activity_quantile: nearest seed centroid per row."""
    out = df.copy()
    n = len(out)
    slat = np.full(n, np.nan)
    slon = np.full(n, np.nan)
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[vessel_col] = pd.to_numeric(out[vessel_col], errors="coerce")
    out[week_col] = pd.to_datetime(out[week_col], utc=True, errors="coerce")

    for wk, idx in out.groupby(week_col).groups.items():
        qix = np.asarray(idx)
        sub = out.loc[qix]
        xy = sub[lat_col].notna() & sub[lon_col].notna()
        if not xy.any():
            continue
        vv = sub.loc[xy, vessel_col].dropna()
        if vv.empty:
            continue
        thr = float(vv.quantile(q))
        seed_m = xy & (sub[vessel_col] >= thr)
        if not seed_m.any():
            continue
        seed_lat = sub.loc[seed_m, lat_col].to_numpy(dtype=float)
        seed_lon = sub.loc[seed_m, lon_col].to_numpy(dtype=float)
        qlat = out.loc[qix, lat_col].to_numpy(dtype=float)
        qlon = out.loc[qix, lon_col].to_numpy(dtype=float)
        d = haversine_km_broadcast(qlat, qlon, seed_lat, seed_lon)
        j = np.argmin(d, axis=1)
        slat[qix] = seed_lat[j]
        slon[qix] = seed_lon[j]
    out["nearest_vessel_seed_lat"] = slat
    out["nearest_vessel_seed_lon"] = slon
    return out


def initial_bearing_deg(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Clockwise degrees from North, point1 → point2."""
    φ1 = np.radians(lat1)
    φ2 = np.radians(lat2)
    Δλ = np.radians(lon2 - lon1)
    y = np.sin(Δλ) * np.cos(φ2)
    x = np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(Δλ)
    θ = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return θ


def smallest_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def wind_uv_from_open_meteo(from_deg: np.ndarray, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological FROM direction; u east, v north."""
    r = np.radians(from_deg)
    u = -speed * np.sin(r)
    v = -speed * np.cos(r)
    return u, v


def wind_to_direction_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0


def fetch_open_meteo_cluster_wind(
    cluster_lat: np.ndarray,
    cluster_lon: np.ndarray,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Hourly → weekly mean u,v,speed per cluster; returns long table cluster_id, week_start_utc."""
    rows: list[dict[str, Any]] = []
    base = "https://archive-api.open-meteo.com/v1/archive"
    for cid, (lat, lon) in enumerate(zip(cluster_lat, cluster_lon)):
        params = urllib.parse.urlencode(
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": "wind_speed_10m,wind_direction_10m",
                "timezone": "UTC",
            }
        )
        url = f"{base}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "geospatial-trend-analysis/1.0"})
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        try:
            with urllib.request.urlopen(req, timeout=120, context=ssl_ctx) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Open-Meteo fetch failed cluster {cid}: {exc}")
            continue
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        ws = hourly.get("wind_speed_10m") or []
        wd = hourly.get("wind_direction_10m") or []
        if len(times) < 24:
            continue
        dfh = pd.DataFrame({"time": pd.to_datetime(times, utc=True), "ws": ws, "wd": wd})
        dfh = dfh.dropna(subset=["ws", "wd"])
        u, v = wind_uv_from_open_meteo(dfh["wd"].to_numpy(dtype=float), dfh["ws"].to_numpy(dtype=float))
        dfh["u"], dfh["v"] = u, v
        dfh["week_start_utc"] = (dfh["time"] - pd.to_timedelta(dfh["time"].dt.dayofweek, unit="D")).dt.normalize()
        wk = (
            dfh.groupby("week_start_utc", as_index=False)
            .agg(wind_u_mean=("u", "mean"), wind_v_mean=("v", "mean"), wind_speed_mean=("ws", "mean"))
        )
        wk["cluster_id"] = cid
        spd = wk["wind_speed_mean"].to_numpy()
        uu = wk["wind_u_mean"].to_numpy()
        vv = wk["wind_v_mean"].to_numpy()
        wk["wind_direction_to_degrees"] = wind_to_direction_deg(uu, vv)
        wk["wind_direction_from_degrees"] = (wk["wind_direction_to_degrees"] + 180.0) % 360.0
        rows.append(wk)
        time.sleep(0.35)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def week_key_monday_start(series: pd.Series) -> pd.Series:
    """Map any timestamp to the UTC Monday-start of its calendar week (matches Open-Meteo weekly bins)."""
    t = pd.to_datetime(series, utc=True, errors="coerce").dt.normalize()
    return (t - pd.to_timedelta(t.dt.dayofweek, unit="D")).dt.normalize()


def apply_wind_to_dataframe(df: pd.DataFrame, grid_to_cluster: pd.Series, wk_clean: pd.DataFrame) -> pd.DataFrame:
    """Attach cluster-weekly ERA5-derived wind to full panel."""
    out = df.copy()
    cid = pd.to_numeric(grid_to_cluster.reindex(out.index), errors="coerce").fillna(0).astype(int)
    out["wind_cluster_id"] = cid.values
    out["_wk"] = week_key_monday_start(out["week_start_utc"])
    if wk_clean is None or wk_clean.empty:
        for col in ("wind_u_mean", "wind_v_mean", "wind_speed_mean", "wind_direction_to_degrees", "wind_direction_from_degrees"):
            out[col] = np.nan
        out["wind_direction_degrees"] = np.nan
        out["wind_data_source"] = "none"
        out.drop(columns=["_wk"], inplace=True)
        return out
    wm = wk_clean.copy()
    wm["_wk"] = week_key_monday_start(wm["week_start_utc"])
    keep = wm.drop(columns=["week_start_utc"], errors="ignore")
    merged = out.merge(
        keep,
        left_on=["wind_cluster_id", "_wk"],
        right_on=["cluster_id", "_wk"],
        how="left",
    )
    merged.drop(columns=["_wk"], inplace=True)
    merged["wind_direction_degrees"] = merged["wind_direction_from_degrees"]
    merged["wind_data_source"] = "open_meteo_era5_archive_cluster_mean"
    return merged


def normalized_user_wind(wf: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    for c in wf.columns:
        lc = str(c).lower().strip()
        if lc in {"u10", "u", "wind_u", "wind_u_mean"}:
            colmap[c] = "wind_u_mean"
        if lc in {"v10", "v", "wind_v", "wind_v_mean"}:
            colmap[c] = "wind_v_mean"
        if lc in {"ws", "wind_speed_mean", "wind_speed"}:
            colmap[c] = "wind_speed_mean"
    out = wf.rename(columns=colmap)
    out["week_start_utc"] = pd.to_datetime(out["week_start_utc"], utc=True, errors="coerce")
    if "wind_u_mean" not in out.columns or "wind_v_mean" not in out.columns:
        return pd.DataFrame()
    u = pd.to_numeric(out["wind_u_mean"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(out["wind_v_mean"], errors="coerce").to_numpy(dtype=float)
    if "wind_speed_mean" not in out.columns:
        out["wind_speed_mean"] = np.sqrt(u * u + v * v)
    else:
        spd = pd.to_numeric(out["wind_speed_mean"], errors="coerce").to_numpy(dtype=float)
        fil = ~np.isfinite(spd)
        spd[fil] = np.sqrt(u[fil] ** 2 + v[fil] ** 2)
        out["wind_speed_mean"] = spd
    out["wind_direction_to_degrees"] = wind_to_direction_deg(u, v)
    out["wind_direction_from_degrees"] = (out["wind_direction_to_degrees"] + 180.0) % 360.0
    return out


def aligned_compare(
    df: pd.DataFrame,
    outcome: str,
    flag: str,
) -> dict[str, Any]:
    if flag not in df.columns:
        raise KeyError(flag)
    x = pd.to_numeric(df[outcome], errors="coerce")
    al = pd.to_numeric(df[flag], errors="coerce").eq(1.0)
    non = pd.to_numeric(df[flag], errors="coerce").eq(0.0)
    a = x.loc[al].dropna().to_numpy(dtype=float)
    b = x.loc[non].dropna().to_numpy(dtype=float)

    def _cohens_d(hi: np.ndarray, lo: np.ndarray) -> float:
        if len(hi) < 2 or len(lo) < 2:
            return float("nan")
        v1, v2 = float(np.var(hi, ddof=1)), float(np.var(lo, ddof=1))
        n1, n2 = len(hi), len(lo)
        pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1))
        if pooled == 0:
            return float("nan")
        return float((np.mean(hi) - np.mean(lo)) / pooled)

    row: dict[str, Any] = {
        "outcome": outcome,
        "n_aligned_true": int(len(a)),
        "n_aligned_false": int(len(b)),
        "mean_aligned_true": float(np.mean(a)) if len(a) else np.nan,
        "mean_aligned_false": float(np.mean(b)) if len(b) else np.nan,
        "median_aligned_true": float(np.median(a)) if len(a) else np.nan,
        "median_aligned_false": float(np.median(b)) if len(b) else np.nan,
        "cohens_d_true_minus_false": _cohens_d(a, b),
    }
    if len(a) >= 3 and len(b) >= 3:
        row["welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        row["mann_whitney_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    else:
        row["welch_p"] = np.nan
        row["mann_whitney_p"] = np.nan
    return row


def safe_corr_spearman(df: pd.DataFrame, a: str, b: str) -> tuple[float, float, int]:
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 8:
        return float("nan"), float("nan"), int(m.sum())
    r, p = stats.spearmanr(x.loc[m], y.loc[m])
    return float(r), float(p), int(m.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None, help="Override base parquet path")
    ap.add_argument("--linked-path", type=Path, default=LINKED_PATH)
    ap.add_argument("--ne-cache", type=Path, default=_ROOT / "data" / "aux" / "natural_earth_coast_cache")
    ap.add_argument("--wind-csv", type=Path, default=None, help="Optional precomputed wind CSV (grid_cell_id, week_start_utc, u, v, ...)")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)

    linked_ok = args.linked_path.is_file()
    base = Path(
        args.input
        if args.input
        else _ROOT / "final_run_stockholm_fixed_20260505_1356/processed/features_ml_ready.parquet"
    )
    if not base.is_file():
        print(f"[FATAL] missing {base}")
        return 1

    if linked_ok:
        print(f"[INFO] Loading linked parquet: {args.linked_path}")
        df = pd.read_parquet(args.linked_path)
    else:
        print(f"[INFO] Linked dataset not found — using base: {base}")
        df = pd.read_parquet(base)

    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    df["grid_cell_id"] = df["grid_cell_id"].astype(str)
    df = ensure_coast_distance(df, Path(args.ne_cache))

    need = [
        "vessel_density_t",
        "distance_to_nearest_high_vessel_density_cell",
        "oil_slick_probability_t",
        "detection_score",
        "no2_mean_t",
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"[FATAL] missing columns: {miss}")
        return 1
    cfg = BufferingConfig()
    df = attach_nearest_high_vessel_seed(df, q=float(cfg.high_activity_quantile))

    # --- STEP 5 panel prep (coastal + seeds) ---
    dc = pd.to_numeric(df["distance_to_coast_km"], errors="coerce")
    ds = pd.to_numeric(df["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    panel_m = dc.notna() & (dc <= COAST_KM) & ds.notna() & (ds >= 0) & (ds < 30)
    df["shipping_distance_band_tight"] = assign_shipping_band_tight(pd.Series(ds, index=df.index))

    # --- STEP 4 NO₂ excess ---
    no2 = pd.to_numeric(df["no2_mean_t"], errors="coerce")
    wm = df.groupby(df["week_start_utc"].dt.normalize())["no2_mean_t"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").mean()
    )
    df["weekly_no2_anomaly"] = no2 - wm

    ow = df["week_start_utc"].dt.normalize()
    ref_band = df["shipping_distance_band_tight"] == "15-30 km"
    band_mean = (
        df.loc[ref_band].groupby(df.loc[ref_band, "week_start_utc"].dt.normalize())["no2_mean_t"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    )
    df["local_no2_excess"] = no2 - ow.map(band_mean.get)

    gmean = df.groupby("grid_cell_id")["no2_mean_t"].transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
    df["grid_no2_anomaly"] = no2 - gmean

    df["land_no2"] = no2
    df["land_no2_anomaly"] = df["weekly_no2_anomaly"]

    no2_feat = df.loc[panel_m, ["grid_cell_id", "week_start_utc", "no2_mean_t", "weekly_no2_anomaly", "local_no2_excess", "grid_no2_anomaly", "shipping_distance_band_tight"]].copy()
    no2_feat.to_csv(REPORTS / "no2_excess_features.csv", index=False)

    # --- STEP 2 Wind ---
    panel_df = df.loc[panel_m].copy()
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

    ws = pd.to_datetime(df["week_start_utc"], utc=True).min()
    we = pd.to_datetime(df["week_start_utc"], utc=True).max()
    start_d = ws.date().isoformat() if pd.notna(ws) else "2023-01-01"
    end_d = we.date().isoformat() if pd.notna(we) else "2023-12-31"

    wind_long = pd.DataFrame()
    meta: dict[str, Any] = {"clusters": int(n_clust), "source": "unknown"}
    wind_user_loaded = False

    if args.wind_csv and Path(args.wind_csv).is_file():
        wu_raw = pd.read_csv(Path(args.wind_csv))
        wu = normalized_user_wind(wu_raw)
        if not wu.empty and {"grid_cell_id", "wind_u_mean", "wind_v_mean", "week_start_utc"}.issubset(wu.columns):
            wu["grid_cell_id"] = wu["grid_cell_id"].astype(str)
            meta["source"] = "user_csv"
            wu["_meta_json"] = json.dumps(meta)
            wu.to_csv(REPORTS / "wind_features_weekly.csv", index=False)
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
            for col in [
                "wind_u_mean",
                "wind_v_mean",
                "wind_speed_mean",
                "wind_direction_to_degrees",
                "wind_direction_from_degrees",
            ]:
                df[col] = m2[col].values if col in m2.columns else np.nan
            df["wind_direction_degrees"] = df["wind_direction_from_degrees"]
            df["wind_data_source"] = "user_csv"
            wind_user_loaded = True
        else:
            print("[WARN] --wind-csv present but missing grid_cell_id, week_start_utc and u/v columns; fetching Open‑Meteo instead.")

    if not wind_user_loaded:
        meta["source"] = "open_meteo_era5_archive"
        print(f"[INFO] Fetching Open-Meteo ERA5 archive ({start_d}–{end_d}), {len(clus_lat)} wind cluster(s)...")
        wind_long = fetch_open_meteo_cluster_wind(
            cluster_lat=np.asarray(clus_lat, dtype=float),
            cluster_lon=np.asarray(clus_lon, dtype=float),
            start_date=start_d,
            end_date=end_d,
        )
        wf_list_inner: list[pd.DataFrame] = []
        cmap_lookup = cmap_series.to_dict()
        if not wind_long.empty:
            for gid_str in uniq_g["grid_cell_id"].astype(str).values:
                cid = int(cmap_lookup.get(gid_str, 0))
                subw = wind_long.loc[wind_long["cluster_id"] == cid].copy()
                if subw.empty:
                    continue
                subw.insert(0, "grid_cell_id", gid_str)
                wf_list_inner.append(subw.drop(columns=["cluster_id"], errors="ignore"))
        wf_expanded = pd.concat(wf_list_inner, ignore_index=True) if wf_list_inner else pd.DataFrame()
        if wf_expanded.empty:
            pd.DataFrame(
                [{**meta, "note": "no wind fetched — check network / dates or supply --wind-csv"}],
            ).to_csv(REPORTS / "wind_features_weekly.csv", index=False)
            df = apply_wind_to_dataframe(df, grid_to_cluster, pd.DataFrame())
        else:
            wf_expanded["_meta_json"] = json.dumps(meta)
            wf_expanded.to_csv(REPORTS / "wind_features_weekly.csv", index=False)
            df = apply_wind_to_dataframe(df, grid_to_cluster, wind_long)

    lat1 = pd.to_numeric(df["nearest_vessel_seed_lat"], errors="coerce").to_numpy()
    lon1 = pd.to_numeric(df["nearest_vessel_seed_lon"], errors="coerce").to_numpy()
    lat2 = pd.to_numeric(df["grid_centroid_lat"], errors="coerce").to_numpy()
    lon2 = pd.to_numeric(df["grid_centroid_lon"], errors="coerce").to_numpy()
    bearing = initial_bearing_deg(lat1, lon1, lat2, lon2)
    u_w = pd.to_numeric(df["wind_u_mean"], errors="coerce").to_numpy()
    v_w = pd.to_numeric(df["wind_v_mean"], errors="coerce").to_numpy()
    wind_to = wind_to_direction_deg(u_w, v_w)
    delta = smallest_angle_deg(bearing, wind_to)
    df["bearing_lane_centroid_deg"] = bearing
    df["wind_alignment_angle_diff_deg"] = delta
    df["wind_alignment_score"] = np.cos(np.radians(delta))
    wa_int = pd.Series(np.nan, index=df.index, dtype=float)
    finite_d = np.isfinite(delta)
    wa_int.loc[finite_d] = (delta[finite_d] <= 45.0).astype(float)
    df["wind_aligned_to_land"] = wa_int
    cat = np.full(len(df), "", dtype=object)
    finite = np.isfinite(delta)
    cat[finite & (delta <= 45)] = "aligned"
    cat[finite & (delta > 45) & (delta <= 135)] = "crosswind"
    cat[finite & (delta > 135)] = "opposite"
    df["wind_alignment_category"] = cat

    want_align = [
            "grid_cell_id",
            "week_start_utc",
            "nearest_vessel_seed_lat",
            "nearest_vessel_seed_lon",
            "bearing_lane_centroid_deg",
            "wind_u_mean",
            "wind_v_mean",
            "wind_speed_mean",
            "wind_direction_to_degrees",
            "wind_direction_from_degrees",
            "wind_alignment_angle_diff_deg",
            "wind_alignment_score",
            "wind_aligned_to_land",
            "wind_alignment_category",
    ]
    have = [c for c in want_align if c in df.columns]
    df.loc[panel_m, have].to_csv(REPORTS / "wind_alignment_features.csv", index=False)

    # STEP 6 & 7
    Pan = df.loc[panel_m].copy()
    PanM = Pan.copy()
    corr_rows: list[dict[str, Any]] = []

    no2_win = Pan[Pan["wind_aligned_to_land"].notna()].copy()
    no2_rows = []
    for outc in ["weekly_no2_anomaly", "local_no2_excess", "grid_no2_anomaly"]:
        no2_rows.append(aligned_compare(no2_win.dropna(subset=[outc]), outc, "wind_aligned_to_land"))
    pd.DataFrame(no2_rows).to_csv(REPORTS / "no2_wind_aligned_land_impact.csv", index=False)

    oil_win = Pan[Pan["wind_aligned_to_land"].notna()].copy()
    oil_rows = []
    for outc in ["oil_slick_probability_t", "detection_score"]:
        if outc in oil_win.columns:
            oil_rows.append(aligned_compare(oil_win.dropna(subset=[outc]), outc, "wind_aligned_to_land"))
    pd.DataFrame(oil_rows).to_csv(REPORTS / "oil_slick_wind_coastal_risk.csv", index=False)

    # STEP 8 models
    vd = pd.to_numeric(PanM["vessel_density_t"], errors="coerce").fillna(0).to_numpy(dtype=float)
    oi = pd.to_numeric(PanM["oil_slick_probability_t"], errors="coerce").fillna(0).to_numpy(dtype=float)
    al = pd.to_numeric(PanM["wind_alignment_score"], errors="coerce").fillna(0).to_numpy(dtype=float)
    PanM["oil_x_align"] = oi * al
    PanM["vessel_x_align"] = vd * al
    targ = pd.to_numeric(PanM["local_no2_excess"], errors="coerce")
    for c in [
        "vessel_density_t",
        "distance_to_nearest_high_vessel_density_cell",
        "oil_slick_probability_t",
        "detection_score",
        "wind_speed_mean",
        "wind_alignment_score",
    ]:
        if c not in PanM.columns:
            continue
        r, p, n = safe_corr_spearman(PanM, "local_no2_excess", c)
        corr_rows.append({"feature": c, "target": "local_no2_excess", "spearman_r": r, "p_value": p, "n": n})
    r2, p2, n2 = safe_corr_spearman(PanM, "weekly_no2_anomaly", "wind_alignment_score")
    corr_rows.append(
        {"feature": "wind_alignment_score", "target": "weekly_no2_anomaly", "spearman_r": r2, "p_value": p2, "n": n2}
    )
    r_oil, p_oil, n_oil = safe_corr_spearman(PanM, "oil_slick_probability_t", "wind_alignment_score")
    corr_rows.append(
        {
            "feature": "wind_alignment_score",
            "target": "oil_slick_probability_t",
            "spearman_r": r_oil,
            "p_value": p_oil,
            "n": n_oil,
        }
    )
    for c in ["vessel_density_t", "distance_to_nearest_high_vessel_density_cell", "wind_speed_mean", "detection_score"]:
        if c not in PanM.columns:
            continue
        ro, po, no = safe_corr_spearman(PanM, "oil_slick_probability_t", c)
        corr_rows.append({"feature": c, "target": "oil_slick_probability_t", "spearman_r": ro, "p_value": po, "n": no})

    pd.DataFrame(corr_rows).to_csv(REPORTS / "land_pollution_driver_correlation.csv", index=False)

    Xmat = PanM[
        ["vessel_density_t", "distance_to_nearest_high_vessel_density_cell", "oil_slick_probability_t", "detection_score", "wind_speed_mean", "wind_alignment_score", "distance_to_coast_km"]
    ].apply(pd.to_numeric, errors="coerce")
    Xmat["oil_x_align"] = PanM["oil_x_align"].to_numpy(dtype=float)
    Xmat["vessel_x_align"] = PanM["vessel_x_align"].to_numpy(dtype=float)
    y_vec = targ.to_numpy(dtype=float)
    finite = np.isfinite(y_vec) & (~Xmat.isna().any(axis=1).to_numpy())
    coef_md = ""
    imp_rows: list[dict[str, Any]] = []
    if int(finite.sum()) > 200:
        scaler = StandardScaler()
        Xm = scaler.fit_transform(Xmat.loc[finite].to_numpy(dtype=float))
        yf = y_vec[finite]
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xm, yf)
        for name, coef in zip(Xmat.columns, ridge.coef_):
            imp_rows.append({"model": "ridge_standardized", "feature": name, "coefficient": float(coef), "score_r2_train": float(ridge.score(Xm, yf))})

        rf = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=12, min_samples_leaf=4, n_jobs=-1)
        rf.fit(Xm, yf)
        for name, imp in zip(Xmat.columns, rf.feature_importances_):
            imp_rows.append({"model": "random_forest", "feature": name, "importance_gini": float(imp)})
        try:
            hgb = HistGradientBoostingRegressor(max_depth=6, random_state=42, max_iter=200)
            hgb.fit(Xm, yf)
            h_imp = getattr(hgb, "feature_importances_", None)
            if h_imp is None:
                h_imp = np.full(len(Xmat.columns), np.nan)
            for name, imp in zip(Xmat.columns, h_imp):
                imp_rows.append(
                    {"model": "hist_gradient_boosting", "feature": name, "importance": float(imp)}
                )
        except Exception:  # noqa: BLE001
            pass
        coef_md = (
            f"Ridge standardized coefficients on **local_no2_excess**, n={int(finite.sum())}. "
            f"Train R² (ridge): {ridge.score(Xm, yf):.4f}; RF/HGB fitted for feature ranking only."
        )
    else:
        coef_md = f"Insufficient complete rows for multivariate models (n_complete={int(finite.sum())})."

    pd.DataFrame(imp_rows).to_csv(REPORTS / "land_pollution_driver_feature_importance.csv", index=False)

    # STEP optional NDVI
    ndvi_rows = []
    if "nearest_land_ndvi_mean" in Pan.columns:
        nv = pd.to_numeric(Pan["nearest_land_ndvi_mean"], errors="coerce")
        lx = pd.to_numeric(Pan["local_no2_excess"], errors="coerce")
        ox = pd.to_numeric(Pan["oil_slick_probability_t"], errors="coerce")

        mx = lx.median(skipna=True)
        if np.isfinite(mx):
            hi, lo = nv.loc[lx >= mx].dropna(), nv.loc[(lx.notna()) & (lx < mx)].dropna()
            hh, ll = hi.to_numpy(dtype=float), lo.to_numpy(dtype=float)
            if len(hh) >= 3 and len(ll) >= 3:
                ndvi_rows.append(
                    {
                        "split": "high_vs_low_local_no2_excess_median_split",
                        "n_high": len(hh),
                        "n_low": len(ll),
                        "mean_group_high_ndvi": float(np.mean(hh)),
                        "mean_group_low_ndvi": float(np.mean(ll)),
                        "mann_whitney_p": float(stats.mannwhitneyu(hh, ll, alternative="two-sided").pvalue),
                    }
                )

        mo = ox.median(skipna=True)
        if np.isfinite(mo):
            hi, lo = nv.loc[ox >= mo].dropna(), nv.loc[(ox.notna()) & (ox < mo)].dropna()
            hh, ll = hi.to_numpy(dtype=float), lo.to_numpy(dtype=float)
            if len(hh) >= 3 and len(ll) >= 3:
                ndvi_rows.append(
                    {
                        "split": "high_vs_low_oil_slick_median_split",
                        "n_high": len(hh),
                        "n_low": len(ll),
                        "mean_group_high_ndvi": float(np.mean(hh)),
                        "mean_group_low_ndvi": float(np.mean(ll)),
                        "mann_whitney_p": float(stats.mannwhitneyu(hh, ll, alternative="two-sided").pvalue),
                    }
                )

        aligned = Pan["wind_aligned_to_land"].eq(1.0)
        not_aligned = Pan["wind_aligned_to_land"].eq(0.0)
        a = nv.loc[aligned].dropna().to_numpy(dtype=float)
        b = nv.loc[not_aligned].dropna().to_numpy(dtype=float)
        if len(a) >= 3 and len(b) >= 3:
            ndvi_rows.append(
                {
                    "split": "wind_aligned_true_vs_false",
                    "n_aligned": len(a),
                    "n_not_aligned": len(b),
                    "mean_ndvi_aligned": float(np.mean(a)),
                    "mean_ndvi_not_aligned": float(np.mean(b)),
                    "mann_whitney_p": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
                }
            )

    pd.DataFrame(ndvi_rows).to_csv(REPORTS / "ndvi_supporting_land_response.csv", index=False)
    # --- Plots ---
    def band_decay_plot(outcome: str, fname: str, title: str) -> None:
        rows = []
        for bl in BANDS:
            slab = Pan.loc[Pan["shipping_distance_band_tight"] == bl]
            x = pd.to_numeric(slab[outcome], errors="coerce").dropna()
            rows.append(
                {
                    "band": bl,
                    "mean": float(x.mean()) if len(x) else np.nan,
                    "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                    "n": len(x),
                }
            )
        dd = pd.DataFrame(rows)
        if dd["mean"].isna().all():
            return
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        xi = np.arange(len(dd))
        ax.bar(xi, dd["mean"], yerr=dd["std"], capsize=3, color="steelblue", alpha=0.85)
        ax.set_xticks(xi)
        ax.set_xticklabels(dd["band"], rotation=15)
        ax.set_title(title)
        ax.set_ylabel(outcome)
        fig.tight_layout()
        fig.savefig(FIGURES / fname, dpi=160)
        plt.close(fig)

    band_decay_plot("local_no2_excess", "no2_excess_by_shipping_band.png", "Local NO₂ excess by shipping distance band")
    band_decay_plot("oil_slick_probability_t", "oil_slick_by_shipping_band.png", "Oil slick proxy by shipping distance band")

    def aligned_box(y: str, fname: str, title: str) -> None:
        sub = Pan[Pan["wind_aligned_to_land"].notna()].copy()
        sub["_g"] = np.where(sub["wind_aligned_to_land"].eq(1.0), "Wind toward land (±45°)", "Not aligned")
        sub[y] = pd.to_numeric(sub[y], errors="coerce")
        sub = sub.dropna(subset=[y])
        if sub.empty:
            return
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.boxplot(data=sub, x="_g", y=y, ax=ax, hue="_g", palette="Set2", legend=False)
        ax.set_title(title)
        ax.set_xlabel("")
        fig.tight_layout()
        fig.savefig(FIGURES / fname, dpi=160)
        plt.close(fig)

    aligned_box("local_no2_excess", "no2_excess_wind_aligned_vs_not.png", "Local NO₂ excess: wind aligned vs not")
    aligned_box("oil_slick_probability_t", "oil_slick_wind_aligned_vs_not.png", "Oil slick proxy: wind aligned vs not")

    if imp_rows:
        imdf = pd.DataFrame(imp_rows)
        hgb = imdf[imdf["model"] == "hist_gradient_boosting"]
        if not hgb.empty:
            hgb = hgb.sort_values("importance", ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(hgb["feature"], hgb["importance"], color="teal")
            ax.invert_yaxis()
            ax.set_title("HistGradientBoosting feature importance\n(target: local_no2_excess)")
            fig.tight_layout()
            fig.savefig(FIGURES / "feature_importance_no2_excess.png", dpi=160)
            plt.close(fig)

    # Map high-risk
    try:
        oil = pd.to_numeric(Pan["oil_slick_probability_t"], errors="coerce")
        ex = pd.to_numeric(Pan["local_no2_excess"], errors="coerce")
        hi_oil = oil >= oil.quantile(0.9)
        hi_ex = ex >= ex.quantile(0.9)
        risk = hi_oil & hi_ex & (Pan["wind_aligned_to_land"].eq(1.0))
        uniq_r = Pan.loc[risk].drop_duplicates("grid_cell_id")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            pd.to_numeric(Pan["grid_centroid_lon"], errors="coerce"),
            pd.to_numeric(Pan["grid_centroid_lat"], errors="coerce"),
            s=8,
            alpha=0.35,
            c="silver",
            label="Coastal panel",
        )
        if not uniq_r.empty:
            ax.scatter(
                uniq_r["grid_centroid_lon"],
                uniq_r["grid_centroid_lat"],
                s=40,
                c="crimson",
                edgecolors="k",
                label="High risk (oil p90 ∩ NO₂ excess p90 ∩ wind aligned)",
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
        ax.set_title("High-risk coastal cells (potential landward pathway context)")
        fig.tight_layout()
        fig.savefig(FIGURES / "high_risk_coastal_cells_map.png", dpi=160)
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass

    try:
        import folium

        sub = Pan.drop_duplicates("grid_cell_id").sample(min(200, Pan["grid_cell_id"].nunique()), random_state=3)
        m = folium.Map(
            location=[float(sub["grid_centroid_lat"].mean()), float(sub["grid_centroid_lon"].mean())],
            zoom_start=7,
            tiles="CartoDB positron",
        )
        for _, r in sub.iterrows():
            folium.CircleMarker(
                location=[float(r["grid_centroid_lat"]), float(r["grid_centroid_lon"])],
                radius=4,
                color="red" if r.get("wind_aligned_to_land") == 1.0 else "blue",
                fill=True,
                fill_opacity=0.65,
                popup=str(r["grid_cell_id"]),
            ).add_to(m)
        m.save(str(VIZ / "coastal_panel_wind_aligned_map.html"))
    except Exception:
        pass

    imgs = sorted(FIGURES.glob("*.png"))
    gall = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Land pollution drivers — wind</title></head><body><h2>Figures</h2><ul>"]
    for im in imgs:
        gall.append(f"<li>{im.name}<br/><img src='../figures/{RUN}/{im.name}' style='max-width:900px'></li>")
    gall.append("</ul></body></html>")
    VIZ.mkdir(parents=True, exist_ok=True)
    (VIZ / "figures_gallery.html").write_text("\n".join(gall), encoding="utf-8")

    # Research summary verdict
    def nanmin_pair(a: float, b: float) -> float:
        vals = [x for x in (a, b) if np.isfinite(x)]
        return float(np.nanmin(vals)) if vals else float("nan")

    def no2_alignment_strength(pdf: pd.DataFrame) -> bool:
        if pdf.empty:
            return False
        for _, r in pdf.iterrows():
            d = r.get("cohens_d_true_minus_false", np.nan)
            p = nanmin_pair(float(r.get("welch_p", np.nan)), float(r.get("mann_whitney_p", np.nan)))
            mt, mf = r.get("mean_aligned_true", np.nan), r.get("mean_aligned_false", np.nan)
            if not np.isfinite(d) or not np.isfinite(mt) or not np.isfinite(mf) or not np.isfinite(p):
                continue
            if mt > mf and p < 0.05 and abs(float(d)) > 0.05:
                return True
        return False

    def oil_alignment_strength(pdf: pd.DataFrame) -> bool:
        """Any significant contrast for oil/detection under aligned wind (exploratory)."""
        if pdf.empty:
            return False
        for _, r in pdf.iterrows():
            d = r.get("cohens_d_true_minus_false", np.nan)
            p = nanmin_pair(float(r.get("welch_p", np.nan)), float(r.get("mann_whitney_p", np.nan)))
            if np.isfinite(d) and np.isfinite(p) and p < 0.05 and abs(float(d)) > 0.06:
                return True
        return False

    pdf_no2 = pd.DataFrame(no2_rows)
    pdf_oil = pd.DataFrame(oil_rows)
    no2_sig = no2_alignment_strength(pdf_no2)
    oil_sig = oil_alignment_strength(pdf_oil)
    wind_missing = float(Pan["wind_speed_mean"].notna().mean()) < 0.05

    wind_failure = (not wind_user_loaded) and wind_long.empty
    if wind_missing:
        verdict = "INCONCLUSIVE — negligible wind coverage for coastal panel rows."
    elif wind_failure:
        verdict = "INCONCLUSIVE — wind could not be populated (offline fetch failed and no --wind-csv)."
    elif no2_sig and oil_sig:
        verdict = "STRONG — NO₂ excess and oil/detection contrasts line up with aligned-land wind summaries (still non-causal)."
    elif no2_sig or oil_sig:
        verdict = "MODERATE — mixed support: at least one focal test meets p<0.05 with non-trivial Cohen's d; urban/regional NO₂ and SAR noise remain confounders."
    else:
        verdict = "WEAK — exploratory evidence only on this weekly / cluster-wind aggregation."

    summ = f"""# Land pollution drivers (wind-aware) — summary

Associations only; **not causal proof** of land deposition.

## Framing (careful wording)
- **Drivers associated with land-side pollution** = statistical association of NO₂ anomaly/excess with maritime and wind fields.  
- **Wind-supported transport evidence** = coherence between **wind alignment** (lane→cell bearing vs wind *to* direction) and elevated NO₂ or oil proxy.  
- **Potential coastal exposure pathway** — not direct shoreline proof.

## Analytic takeaways  
- **NO₂ excess vs wind aligned:** see `no2_wind_aligned_land_impact.csv`.  
- **Oil coastal-risk vs wind aligned:** `oil_slick_wind_coastal_risk.csv` — *potential transport / risk*, not proven deposition.  
- **Driver ranking:** `land_pollution_driver_correlation.csv`, `land_pollution_driver_feature_importance.csv`.

## Data notes  
- Wind source: **{meta.get("source", "?")}** — default Open‑Meteo **ERA5 archive** at **{meta.get("clusters", "?")}** spatial cluster(s) over the coastal panel; weekly means from hourly samples. (**Note:** archived `wind_speed_10m` is typically reported in **km/h**; direction-based alignment uses u/v-derived **bearing**, which is unit-consistent.)
- Alignment uses the **nearest high-vessel-density seed centroid** each week (P90 threshold, same spirit as `land_sea_buffering`).

## NDVI  
See `ndvi_supporting_land_response.csv` — **supporting context only**.

## Verdict

**{verdict}**

{coef_md}

---
Artifacts: `outputs/reports/{RUN}/`, `outputs/figures/{RUN}/`, `outputs/visualizations/{RUN}/`.
"""

    md_path = REPORTS / "research_summary.md"
    md_path.write_text(summ.replace("{RUN}", RUN), encoding="utf-8")
    (REPORTS / "land_pollution_driver_model_summary.md").write_text(coef_md + "\n\nSee `research_summary.md` for context.\n")

    wcov = float(Pan["wind_speed_mean"].notna().mean()) if len(Pan) else 0.0

    cdf_all = pd.DataFrame(corr_rows)
    cdf_ne = cdf_all[(cdf_all["target"] == "local_no2_excess") & cdf_all["spearman_r"].notna()]
    strongest_no2 = cdf_ne.loc[cdf_ne["spearman_r"].abs().idxmax()].to_dict() if len(cdf_ne) else {}
    cdf_oi = cdf_all[(cdf_all["target"] == "oil_slick_probability_t") & cdf_all["spearman_r"].notna()].copy()
    cdf_oi = cdf_oi[cdf_oi["spearman_r"].abs() < 0.999]
    strongest_oil = cdf_oi.loc[cdf_oi["spearman_r"].abs().idxmax()].to_dict() if len(cdf_oi) else {}

    print("=== run_land_pollution_drivers_wind ===")
    print(f"Rows (full merged): {len(df):,}")
    print(f"Coastal land-impact panel rows: {len(Pan):,}")
    print(f"Wind numeric coverage on panel (frac non-null wind_speed_mean): {wcov:.3f}")
    if strongest_no2:
        print(
            f"Strongest NO₂-excess driver (Spearman): {strongest_no2.get('feature')} r={float(strongest_no2.get('spearman_r')):.4f}",
        )
    if strongest_oil:
        print(
            f"Strongest oil-proxy association (Spearman): {strongest_oil.get('feature')} r={float(strongest_oil.get('spearman_r')):.4f}",
        )
    print(f"Wind alignment strengthens NO₂ excess signal (directional test): {'YES' if no2_sig else 'NO'}")
    print(f"Wind alignment / oil coastal-risk contrast (exploratory): {'YES' if oil_sig else 'NO'}")
    print(f"Final evidence verdict: {verdict}")
    print("\nSee outputs/reports/run_land_pollution_drivers_wind/ for CSVs and figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
