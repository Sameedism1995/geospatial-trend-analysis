"""Nearest-port distances for grid centroids (thesis port exposure layer)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
DEFAULT_GRID_RES_DEG = 0.1
# Baltic Sea approximate centre — last-resort centroid if grid id / coords missing
_FALLBACK_LAT, _FALLBACK_LON = 59.5, 20.0

LOGGER = logging.getLogger("port_proximity")


def _project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def grid_centroid_from_cell_id(grid_id: str | float, res_deg: float = DEFAULT_GRID_RES_DEG) -> tuple[float | None, float | None]:
    if grid_id is None or (isinstance(grid_id, float) and pd.isna(grid_id)):
        return (None, None)
    gid = str(grid_id).strip()
    if gid == "unmapped":
        return (None, None)
    try:
        parts = gid.split("_")
        row = int(parts[1][1:])
        col = int(parts[2][1:])
        lat = (row + 0.5) * res_deg - 90.0
        lon = (col + 0.5) * res_deg - 180.0
        return (float(lat), float(lon))
    except (IndexError, ValueError):
        return (None, None)


def _haversine_km_pairs(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Shape (n, m) pairwise haversine km between n points (lat1/lon1) and m ports (lat2/lon2)."""
    lat1_r = np.radians(lat1)[:, np.newaxis]
    lon1_r = np.radians(lon1)[:, np.newaxis]
    lat2_r = np.radians(lat2)[np.newaxis, :]
    lon2_r = np.radians(lon2)[np.newaxis, :]
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    return (2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(np.float64)


def _resolve_coords(df: pd.DataFrame, logger: logging.Logger | None) -> tuple[np.ndarray, np.ndarray]:
    lat = pd.to_numeric(df.get("grid_centroid_lat"), errors="coerce").to_numpy(dtype=np.float64)
    lon = pd.to_numeric(df.get("grid_centroid_lon"), errors="coerce").to_numpy(dtype=np.float64)
    need_fill = np.isnan(lat) | np.isnan(lon)
    if need_fill.any() and "grid_cell_id" in df.columns:
        gids = df["grid_cell_id"].to_numpy()
        for i in np.where(need_fill)[0]:
            la, lo = grid_centroid_from_cell_id(gids[i])
            if la is not None and lo is not None:
                lat[i], lon[i] = la, lo
                need_fill[i] = False
    if need_fill.any():
        if logger:
            logger.warning(
                "[PORT PROXIMITY] %d rows missing centroids; using fallback (%.3f, %.3f)",
                int(need_fill.sum()),
                _FALLBACK_LAT,
                _FALLBACK_LON,
            )
        lat[need_fill] = _FALLBACK_LAT
        lon[need_fill] = _FALLBACK_LON
    return lat, lon


def compute_port_proximity(
    df: pd.DataFrame,
    *,
    ports_path: Path | None = None,
    project_root: Path | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Append nearest_port and distance_to_port_km using data/aux/baltic_ports.csv.

    Expects grid_centroid_lat / grid_centroid_lon (fills from grid_cell_id when possible).
    """
    log = logger or LOGGER
    if df.empty:
        return df
    root = project_root if project_root is not None else _project_root_from_here()
    path = ports_path if ports_path is not None else root / "data" / "aux" / "baltic_ports.csv"
    if not path.exists():
        log.error("[PORT PROXIMITY] Missing ports file: %s", path)
        raise FileNotFoundError(f"Ports dataset not found: {path}")

    ports = pd.read_csv(path)
    if not {"latitude", "longitude"}.issubset(ports.columns):
        cols = list(ports.columns)
        raise ValueError(f"baltic_ports.csv requires latitude/longitude columns; got {cols}")
    name_col = "port_name" if "port_name" in ports.columns else ports.columns[0]
    plat = pd.to_numeric(ports["latitude"], errors="coerce").to_numpy(dtype=np.float64)
    plon = pd.to_numeric(ports["longitude"], errors="coerce").to_numpy(dtype=np.float64)
    names = ports[name_col].astype(str).to_numpy()

    mask = np.isfinite(plat) & np.isfinite(plon)
    plat, plon, names = plat[mask], plon[mask], names[mask]
    if len(names) == 0:
        raise ValueError("[PORT PROXIMITY] No valid port coordinates in CSV.")

    lat, lon = _resolve_coords(df, log)
    dist_m = _haversine_km_pairs(lat, lon, plat, plon)
    idx = np.argmin(dist_m, axis=1)
    row_idx = np.arange(len(df))
    out = df.copy()
    out["nearest_port"] = names[idx]
    out["distance_to_port_km"] = dist_m[row_idx, idx].astype(np.float64)

    if out["nearest_port"].isna().any():
        raise ValueError("[PORT PROXIMITY] nearest_port assignment produced nulls.")
    if not np.isfinite(out["distance_to_port_km"].to_numpy()).all():
        raise ValueError("[PORT PROXIMITY] distance_to_port_km must be finite for all rows.")

    log.info(
        "[PORT PROXIMITY] Assigned nearest ports for %d rows (min_dist_km=%.3f max=%.3f)",
        len(out),
        float(out["distance_to_port_km"].min()),
        float(out["distance_to_port_km"].max()),
    )
    return out
