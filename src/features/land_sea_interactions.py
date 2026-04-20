"""
Land–Sea interaction features (LAND IMPACT EXTENSION LAYER, Step 3).

Given a merged grid×week dataframe that already contains:
    - vessel_density        (maritime activity proxy)
    - NO2_mean              (atmospheric transfer proxy)
    - ndvi_mean             (land vegetation response proxy)

this module appends three normalised indices and three interaction terms:

    maritime_pressure_index      = min-max(vessel_density)
    atmospheric_transfer_index   = min-max(NO2_mean)
    land_response_index          = min-max(ndvi_mean)

    vessel_x_no2                 = vessel_density * NO2_mean
    no2_x_ndvi                   = NO2_mean * ndvi_mean
    vessel_x_ndvi_lag{1,2,3}     = vessel_density(t-k) * ndvi_mean(t), per grid

The lagged vessel×ndvi term is computed within each grid cell (grouped on
grid_cell_id, sorted on week_start_utc) so the shift respects temporal ordering
and does not leak across cells.

This module reads no files and writes no files. The orchestrator passes the
in-memory feature frame and persists the enriched frame.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("land_sea_interactions")

LAG_WEEKS_DEFAULT: tuple[int, ...] = (1, 2, 3)


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo = s.min(skipna=True)
    hi = s.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_land_sea_interactions(
    df: pd.DataFrame,
    *,
    lags: Iterable[int] = LAG_WEEKS_DEFAULT,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Return a copy of `df` with land-sea indices and interaction terms appended.

    Missing inputs are handled gracefully: if e.g. ndvi_mean is absent, the NDVI-
    dependent columns are filled with NaN rather than raising.
    """
    logger = logger or LOGGER
    if df.empty:
        logger.warning("[LAND-SEA-INTER] Empty dataframe — no interactions computed")
        return df.copy()

    out = df.copy()

    vessel_col = _first_present(out, ["vessel_density", "vessel_density_t"])
    no2_col = _first_present(out, ["NO2_mean", "no2_mean_t"])
    ndvi_col = _first_present(out, ["ndvi_mean"])
    grid_col = _first_present(out, ["grid_cell_id"])
    week_col = _first_present(out, ["week_start_utc", "week_start", "week"])

    vessel = pd.to_numeric(out[vessel_col], errors="coerce") if vessel_col else pd.Series(np.nan, index=out.index)
    no2 = pd.to_numeric(out[no2_col], errors="coerce") if no2_col else pd.Series(np.nan, index=out.index)
    ndvi = pd.to_numeric(out[ndvi_col], errors="coerce") if ndvi_col else pd.Series(np.nan, index=out.index)

    out["maritime_pressure_index"] = _minmax(vessel)
    out["atmospheric_transfer_index"] = _minmax(no2)
    out["land_response_index"] = _minmax(ndvi)

    out["vessel_x_no2"] = vessel * no2
    out["no2_x_ndvi"] = no2 * ndvi

    if grid_col and week_col:
        sorter_week = pd.to_datetime(out[week_col], utc=True, errors="coerce")
        order_idx = np.lexsort((sorter_week.values, out[grid_col].astype(str).values))
        inverse = np.argsort(order_idx)
        ordered_vessel = vessel.values[order_idx]
        ordered_grid = out[grid_col].astype(str).values[order_idx]
        for lag in lags:
            shifted = pd.Series(ordered_vessel).shift(lag).values
            same_grid = pd.Series(ordered_grid).shift(lag).values == ordered_grid
            shifted = np.where(same_grid, shifted, np.nan)
            unsorted = shifted[inverse]
            out[f"vessel_x_ndvi_lag{lag}"] = unsorted * ndvi.values
    else:
        for lag in lags:
            out[f"vessel_x_ndvi_lag{lag}"] = np.nan
        logger.warning("[LAND-SEA-INTER] Missing grid/week columns — lagged interactions set to NaN")

    non_null = {
        col: int(pd.to_numeric(out[col], errors="coerce").notna().sum())
        for col in [
            "maritime_pressure_index",
            "atmospheric_transfer_index",
            "land_response_index",
            "vessel_x_no2",
            "no2_x_ndvi",
            *[f"vessel_x_ndvi_lag{lag}" for lag in lags],
        ]
    }
    logger.info("[LAND-SEA-INTER] Added interaction features; non-null counts=%s", non_null)
    return out


__all__ = ["add_land_sea_interactions", "LAG_WEEKS_DEFAULT"]
