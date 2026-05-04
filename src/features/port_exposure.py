"""Port-centric exposure score from vessel intensity and proximity."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("port_exposure")


def _pick_vessel_density(df: pd.DataFrame) -> pd.Series | None:
    for col in ("vessel_density", "vessel_density_t", "density_total", "density_total_log"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


def add_port_exposure_score(df: pd.DataFrame, *, logger: logging.Logger | None = None) -> pd.DataFrame:
    """
    port_exposure_score = vessel_density / (1 + distance_to_port_km)

    Requires distance_to_port_km from compute_port_proximity.
    """
    log = logger or LOGGER
    if df.empty:
        return df
    if "distance_to_port_km" not in df.columns:
        log.warning("[PORT EXPOSURE] distance_to_port_km missing; skipping port_exposure_score")
        df = df.copy()
        df["port_exposure_score"] = np.nan
        return df

    out = df.copy()
    d = pd.to_numeric(out["distance_to_port_km"], errors="coerce")
    denom = 1.0 + d
    vessel = _pick_vessel_density(out)
    if vessel is None:
        log.warning("[PORT EXPOSURE] No vessel density column; port_exposure_score set to NaN")
        out["port_exposure_score"] = np.nan
        return out
    out["port_exposure_score"] = vessel / denom
    n_ok = int(out["port_exposure_score"].notna().sum())
    log.info("[PORT EXPOSURE] port_exposure_score non-null rows: %d / %d", n_ok, len(out))
    return out
