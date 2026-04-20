"""
Shared helpers for Google Earth Engine grid-week extractors.

Observational only: no interpolation, no imputation of missing remote-sensing values.
Buffers define aggregation regions; statistics come from GEE reducers only.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from utils.ee_init import safe_initialize_ee

# Default ±deg half-width around each grid centroid (rectangle sampling region)
DEFAULT_BUFFER_DEG_NO2 = 0.1
DEFAULT_BUFFER_DEG_S1 = 0.1
# Study-area bounding box padding (degrees) beyond min/max grid coordinates
STUDY_BBOX_PAD_DEG = 0.1


def load_modeling_grids_and_weeks(path: Path) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Unique grid centroids and sorted distinct week starts (UTC) from modeling parquet."""
    df = pd.read_parquet(
        path,
        columns=["grid_cell_id", "week_start_utc", "grid_centroid_lat", "grid_centroid_lon"],
    )
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    grids = (
        df.groupby("grid_cell_id", as_index=False)
        .agg(
            grid_centroid_lat=("grid_centroid_lat", "first"),
            grid_centroid_lon=("grid_centroid_lon", "first"),
        )
        .dropna(subset=["grid_centroid_lat", "grid_centroid_lon"])
    )
    weeks = sorted(df["week_start_utc"].dropna().unique())
    return grids, list(weeks)


def study_bbox(grids: pd.DataFrame, ee, pad_deg: float = STUDY_BBOX_PAD_DEG) -> Any:
    """Axis-aligned rectangle covering all grid points + pad (for filterBounds / regional stats)."""
    lat = grids["grid_centroid_lat"].astype(float)
    lon = grids["grid_centroid_lon"].astype(float)
    return ee.Geometry.Rectangle(
        [
            float(lon.min() - pad_deg),
            float(lat.min() - pad_deg),
            float(lon.max() + pad_deg),
            float(lat.max() + pad_deg),
        ]
    )


def build_grid_feature_collection(grids: pd.DataFrame, buffer_deg: float, ee) -> Any:
    """One axis-aligned rectangle per grid_cell_id: centroid ± buffer_deg (degrees)."""
    feats = []
    for _, r in grids.iterrows():
        lon, lat = float(r["grid_centroid_lon"]), float(r["grid_centroid_lat"])
        geom = ee.Geometry.Rectangle(
            [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
        )
        feats.append(ee.Feature(geom, {"grid_cell_id": r["grid_cell_id"]}))
    return ee.FeatureCollection(feats)


def iter_week_utc_bounds(weeks: list[pd.Timestamp]) -> Iterator[tuple[pd.Timestamp, Any, Any]]:
    """
    For each ISO week anchor, yield (week_start, week_start_datetime, week_end_datetime)
    with [start, end) = [week_start_utc, week_start_utc + 7 days).
    """
    for wk in weeks:
        if pd.isna(wk):
            continue
        start = pd.Timestamp(wk).to_pydatetime()
        end = start + timedelta(days=7)
        yield pd.Timestamp(wk), start, end


def week_filter_dates_ee(ee, start_dt, end_dt) -> tuple[Any, Any]:
    """Earth Engine Date pair for filterDate(start, end)."""
    return (
        ee.Date(pd.Timestamp(start_dt).isoformat()),
        ee.Date(pd.Timestamp(end_dt).isoformat()),
    )


def initialize_earth_engine(ee) -> None:
    """
    Authenticate first: `earthengine authenticate`
    Set GOOGLE_CLOUD_PROJECT or EE_PROJECT if your Cloud project is required.
    """
    safe_initialize_ee(ee)
