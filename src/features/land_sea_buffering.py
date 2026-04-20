"""
Land–Sea buffering features (LAND IMPACT EXTENSION LAYER, Step 2).

Purely geometric post-processing on the already-merged feature table. Does NOT touch
any data-extraction module. Produces, for every (grid_cell_id, week_start_utc):

    distance_to_nearest_high_vessel_density_cell   (kilometres, per week)
    coastal_exposure_band                           ({"0-10km", "10-50km", "50+km"})
    coastal_exposure_score                          (float in [0, 1])

The exposure score is a piecewise-monotone decay function of distance:

    0–10 km   → 1.00           (HIGH exposure; direct coastal/coastal-adjacent cells)
    10–50 km  → 0.50           (MEDIUM exposure; regional atmospheric/hydrological reach)
    50+ km    → 0.10 * exp(-(d-50)/100)  (CONTROL baseline; gentle decay to ~0)

`high_vessel_density` seeds are identified, per week, as cells whose vessel_density
exceeds the weekly P90 (configurable). Distance is computed with a vectorised
haversine in kilometres against the per-week pool of seed cells. Cells in weeks
with no seed (e.g., missing AIS) receive NaN distance and coastal_exposure_score=0.0.

Inputs (expected columns in the input DataFrame):
    grid_cell_id, week_start_utc,
    grid_centroid_lat (or centroid_lat / lat),
    grid_centroid_lon (or centroid_lon / lon / lng),
    vessel_density    (or vessel_density_t)

Output: new columns appended (originals preserved):
    distance_to_nearest_high_vessel_density_cell
    coastal_exposure_band
    coastal_exposure_score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

LOGGER = logging.getLogger("land_sea_buffering")


@dataclass(frozen=True)
class BufferingConfig:
    high_activity_quantile: float = 0.90
    high_km: float = 10.0
    medium_km: float = 50.0
    score_high: float = 1.0
    score_medium: float = 0.5
    score_control_base: float = 0.1
    score_control_decay_km: float = 100.0


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised pairwise haversine distance (km). Broadcasts lat1/lon1 against lat2/lon2."""
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = lat2_r - lat1_r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _exposure_score(distance_km: np.ndarray, cfg: BufferingConfig) -> np.ndarray:
    """Piecewise exposure score with gentle exponential decay beyond the medium band."""
    out = np.zeros_like(distance_km, dtype=float)
    finite = np.isfinite(distance_km)
    if not finite.any():
        return out

    d = distance_km
    high = finite & (d <= cfg.high_km)
    medium = finite & (d > cfg.high_km) & (d <= cfg.medium_km)
    far = finite & (d > cfg.medium_km)

    out[high] = cfg.score_high
    out[medium] = cfg.score_medium
    out[far] = cfg.score_control_base * np.exp(-(d[far] - cfg.medium_km) / cfg.score_control_decay_km)
    return out


def _band(distance_km: np.ndarray, cfg: BufferingConfig) -> np.ndarray:
    labels = np.full(distance_km.shape, "unknown", dtype=object)
    finite = np.isfinite(distance_km)
    labels[finite & (distance_km <= cfg.high_km)] = "0-10km"
    labels[finite & (distance_km > cfg.high_km) & (distance_km <= cfg.medium_km)] = "10-50km"
    labels[finite & (distance_km > cfg.medium_km)] = "50+km"
    return labels


def compute_coastal_exposure(
    df: pd.DataFrame,
    *,
    config: BufferingConfig | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Append coastal-exposure features to the input DataFrame (a copy).

    Does not mutate the input. If required columns are missing, returns the input
    unchanged (with a warning log) so the broader pipeline remains non-blocking.
    """
    cfg = config or BufferingConfig()
    logger = logger or LOGGER

    if df.empty:
        logger.warning("[LAND-SEA-BUFFER] Empty dataframe — no features computed")
        return df.copy()

    lat_col = _pick_col(df, ["grid_centroid_lat", "centroid_lat", "latitude", "lat"])
    lon_col = _pick_col(df, ["grid_centroid_lon", "centroid_lon", "longitude", "lon", "lng"])
    vessel_col = _pick_col(df, ["vessel_density", "vessel_density_t"])
    week_col = _pick_col(df, ["week_start_utc", "week_start", "week"])

    missing = [
        name for name, val in
        (("lat", lat_col), ("lon", lon_col), ("vessel_density", vessel_col), ("week", week_col))
        if val is None
    ]
    if missing:
        logger.warning("[LAND-SEA-BUFFER] Missing required columns: %s — skipping", missing)
        out = df.copy()
        for col, default in (
            ("distance_to_nearest_high_vessel_density_cell", np.nan),
            ("coastal_exposure_band", "unknown"),
            ("coastal_exposure_score", 0.0),
        ):
            out[col] = default
        return out

    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[vessel_col] = pd.to_numeric(out[vessel_col], errors="coerce")
    out[week_col] = pd.to_datetime(out[week_col], utc=True, errors="coerce")

    dist = np.full(len(out), np.nan, dtype=float)
    out_idx_by_week: dict[pd.Timestamp, np.ndarray] = {}
    for wk, idx in out.groupby(week_col).groups.items():
        out_idx_by_week[wk] = np.asarray(idx)

    n_weeks_total = len(out_idx_by_week)
    n_weeks_with_seeds = 0

    for wk, idx in out_idx_by_week.items():
        sub = out.loc[idx]
        has_xy = sub[lat_col].notna() & sub[lon_col].notna()
        if not has_xy.any():
            continue

        vessel_vals = sub.loc[has_xy, vessel_col]
        non_null_vessel = vessel_vals.dropna()
        if non_null_vessel.empty:
            continue

        threshold = float(non_null_vessel.quantile(cfg.high_activity_quantile))
        if not np.isfinite(threshold):
            continue

        seed_mask = (sub[vessel_col] >= threshold) & has_xy
        if not seed_mask.any():
            continue
        n_weeks_with_seeds += 1

        seed_lat = sub.loc[seed_mask, lat_col].to_numpy()
        seed_lon = sub.loc[seed_mask, lon_col].to_numpy()

        q_idx = np.asarray(idx)
        q_lat = out.loc[q_idx, lat_col].to_numpy()
        q_lon = out.loc[q_idx, lon_col].to_numpy()

        # Pairwise haversine: (n_query, n_seed).
        d = _haversine_km(
            q_lat[:, None],
            q_lon[:, None],
            seed_lat[None, :],
            seed_lon[None, :],
        )
        d_min = np.nanmin(d, axis=1)
        dist[q_idx] = d_min

    out["distance_to_nearest_high_vessel_density_cell"] = dist
    out["coastal_exposure_band"] = _band(dist, cfg)
    out["coastal_exposure_score"] = _exposure_score(dist, cfg)

    logger.info(
        "[LAND-SEA-BUFFER] weeks processed=%d/%d with-seeds=%d | mean_distance_km=%.2f | non-null=%d/%d",
        n_weeks_with_seeds,
        n_weeks_total,
        n_weeks_with_seeds,
        float(np.nanmean(dist)) if np.isfinite(dist).any() else float("nan"),
        int(np.isfinite(dist).sum()),
        int(len(dist)),
    )
    return out


def run_land_sea_buffering(
    features: pd.DataFrame,
    *,
    logger: logging.Logger | None = None,
    config: BufferingConfig | None = None,
) -> pd.DataFrame:
    """Public entry point used by the pipeline orchestrator."""
    return compute_coastal_exposure(features, config=config, logger=logger)


__all__ = [
    "BufferingConfig",
    "compute_coastal_exposure",
    "run_land_sea_buffering",
]
