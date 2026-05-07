"""
Coastal landward bearings and multivariate pollution hotspots for wind-transport analysis.

Nearest coast reference uses Natural Earth coastline / land-boundary samples (same cache as
human_impact_distance_analysis). Bearings are clockwise degrees from north (0-360).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_R_KM = 6371.0088


def _ll_rad(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.c_[np.radians(lat.astype(float)), np.radians(lon.astype(float))]


def nearest_geodesic_reference(
    query_lat: np.ndarray,
    query_lon: np.ndarray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each query point, return (distance_km, matched_ref_lat, matched_ref_lon)
    to the closest reference point (haversine / great-circle).
    """
    q = _ll_rad(np.asarray(query_lat).ravel(), np.asarray(query_lon).ravel())
    r = _ll_rad(np.asarray(ref_lat).ravel(), np.asarray(ref_lon).ravel())
    if len(r) == 0:
        n = len(q)
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    tree = BallTree(r, metric="haversine")
    dist_rad, idx = tree.query(q, k=1)
    dist_km = dist_rad[:, 0] * EARTH_R_KM
    j = idx[:, 0]
    return dist_km, ref_lat[j], ref_lon[j]


def grid_nearest_coast_reference_table(
    uniq_grids: pd.DataFrame,
    *,
    grid_id_col: str = "grid_cell_id",
    lat_col: str = "grid_centroid_lat",
    lon_col: str = "grid_centroid_lon",
    coast_lat: np.ndarray,
    coast_lon: np.ndarray,
) -> pd.DataFrame:
    """One row per grid_cell_id with nearest coastline sample and distance."""
    u = uniq_grids.drop_duplicates(subset=[grid_id_col]).dropna(subset=[lat_col, lon_col])
    if u.empty:
        return pd.DataFrame(
            columns=[
                grid_id_col,
                "nearest_coast_ref_lat",
                "nearest_coast_ref_lon",
                "nearest_coast_ref_distance_km",
            ],
        )
    la = u[lat_col].to_numpy(dtype=float)
    lo = u[lon_col].to_numpy(dtype=float)
    dkm, rla, rlo = nearest_geodesic_reference(la, lo, coast_lat, coast_lon)
    out = pd.DataFrame(
        {
            grid_id_col: u[grid_id_col].astype(str).values,
            "nearest_coast_ref_lat": rla,
            "nearest_coast_ref_lon": rlo,
            "nearest_coast_ref_distance_km": dkm,
        }
    )
    return out


def attach_nearest_multivariate_hotspot(
    df: pd.DataFrame,
    *,
    week_col: str = "week_start_utc",
    lat_col: str = "grid_centroid_lat",
    lon_col: str = "grid_centroid_lon",
    vessel_col: str = "vessel_density_t",
    no2_col: str = "no2_mean_t",
    oil_col: str = "oil_slick_probability_t",
    quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Each week, cells with vessel >= q, OR no2 >= q, OR oil >= q (weekly quantiles) are hotspots.
    For each row, attach the *nearest* qualifying hotspot cell's centroid and a type tag.
    """
    out = df.copy()
    n = len(out)
    out[week_col] = pd.to_datetime(out[week_col], utc=True, errors="coerce")
    for c in (vessel_col, no2_col, oil_col, lat_col, lon_col):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    hs_lat = np.full(n, np.nan)
    hs_lon = np.full(n, np.nan)
    hs_types = np.full(n, "", dtype=object)

    for wk, idx in out.groupby(week_col, sort=False).groups.items():
        qix = np.asarray(idx, dtype=int)
        sub = out.iloc[qix]
        m = sub[lat_col].notna() & sub[lon_col].notna()
        if not m.any():
            continue
        vv = sub.loc[m, vessel_col].dropna()
        nn = sub.loc[m, no2_col].dropna()
        oo = sub.loc[m, oil_col].dropna()
        thr_v = float(vv.quantile(quantile)) if len(vv) >= 5 else np.inf
        thr_n = float(nn.quantile(quantile)) if len(nn) >= 5 else np.inf
        thr_o = float(oo.quantile(quantile)) if len(oo) >= 5 else np.inf

        hot_mask = m & (
            (sub[vessel_col] >= thr_v)
            | (sub[no2_col] >= thr_n)
            | (sub[oil_col] >= thr_o)
        )
        if not hot_mask.any():
            continue

        seed_lat = sub.loc[hot_mask, lat_col].to_numpy(dtype=float)
        seed_lon = sub.loc[hot_mask, lon_col].to_numpy(dtype=float)
        seed_idx = sub.index[hot_mask].to_numpy()

        for i in qix:
            gla = float(out.at[i, lat_col]) if pd.notna(out.at[i, lat_col]) else np.nan
            glo = float(out.at[i, lon_col]) if pd.notna(out.at[i, lon_col]) else np.nan
            if not (np.isfinite(gla) and np.isfinite(glo)):
                continue
            d = haversine_km_broadcast(
                np.array([gla]),
                np.array([glo]),
                seed_lat,
                seed_lon,
            )[0]
            j = int(np.argmin(d))
            win = int(seed_idx[j])
            hs_lat[i] = float(out.at[win, lat_col])
            hs_lon[i] = float(out.at[win, lon_col])
            tags = []
            if pd.notna(out.at[win, vessel_col]) and float(out.at[win, vessel_col]) >= thr_v:
                tags.append("vessel")
            if pd.notna(out.at[win, no2_col]) and float(out.at[win, no2_col]) >= thr_n:
                tags.append("no2")
            if pd.notna(out.at[win, oil_col]) and float(out.at[win, oil_col]) >= thr_o:
                tags.append("oil")
            hs_types[i] = "+".join(tags) if tags else "hotspot"

    out["pollution_hotspot_lat"] = hs_lat
    out["pollution_hotspot_lon"] = hs_lon
    out["pollution_hotspot_type"] = hs_types
    return out


def haversine_km_broadcast(
    q_lat: np.ndarray,
    q_lon: np.ndarray,
    s_lat: np.ndarray,
    s_lon: np.ndarray,
) -> np.ndarray:
    """Pairwise distances (km): each query vs each seed."""
    lat1 = np.radians(q_lat)[:, None]
    lon1 = np.radians(q_lon)[:, None]
    lat2 = np.radians(s_lat)[None, :]
    lon2 = np.radians(s_lon)[None, :]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
