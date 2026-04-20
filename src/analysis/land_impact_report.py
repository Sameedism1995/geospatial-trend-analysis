"""
Land Impact final report (LAND IMPACT EXTENSION LAYER, Step 6).

Emits `outputs/reports/land_impact_analysis.csv` — one row per grid_cell_id ×
week_start_utc — with the minimum columns required by the spec:

    grid_cell_id
    week_start_utc
    ndvi_mean
    vessel_density
    NO2_mean
    coastal_exposure_score
    lagged_correlation          (best |Spearman| across our land-sea pairs)

We also include a small set of helpful context columns when available:
    coastal_exposure_band, land_response_index, atmospheric_transfer_index,
    maritime_pressure_index, distance_to_nearest_high_vessel_density_cell.

Reads only from the in-memory features frame and from an optional lag-summary
CSV; does not re-extract any remote-sensing data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("land_impact_report")

KEY_COLS = ["grid_cell_id", "week_start_utc"]

REQUIRED_COLS = [
    "grid_cell_id",
    "week_start_utc",
    "ndvi_mean",
    "vessel_density",
    "NO2_mean",
    "coastal_exposure_score",
    "lagged_correlation",
]

CONTEXT_COLS = [
    "coastal_exposure_band",
    "distance_to_nearest_high_vessel_density_cell",
    "maritime_pressure_index",
    "atmospheric_transfer_index",
    "land_response_index",
    "grid_centroid_lat",
    "grid_centroid_lon",
]


def _best_lagged_correlation(lag_summary_path: Path) -> float:
    """Return the strongest |Spearman| across configured land-sea pairs, or NaN."""
    if not lag_summary_path.exists():
        return float("nan")
    try:
        df = pd.read_csv(lag_summary_path)
        if "best_spearman" not in df.columns or df.empty:
            return float("nan")
        best = df["best_spearman"].abs().max(skipna=True)
        return float(best) if np.isfinite(best) else float("nan")
    except Exception:  # noqa: BLE001
        return float("nan")


def build_land_impact_table(
    features: pd.DataFrame,
    *,
    lag_summary_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Construct the final per-grid-week land-impact analysis table."""
    logger = logger or LOGGER
    if features.empty:
        logger.warning("[LAND-IMPACT-REPORT] Empty features — emitting empty table")
        return pd.DataFrame(columns=REQUIRED_COLS)

    df = features.copy()
    for col in REQUIRED_COLS + CONTEXT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    best_r = _best_lagged_correlation(Path(lag_summary_path)) if lag_summary_path else float("nan")
    df["lagged_correlation"] = best_r

    keep = REQUIRED_COLS + [c for c in CONTEXT_COLS if c not in REQUIRED_COLS and c in df.columns]
    out = df[keep].copy()
    out = out.dropna(subset=KEY_COLS, how="any").drop_duplicates(subset=KEY_COLS, keep="last")
    out = out.sort_values(KEY_COLS).reset_index(drop=True)
    return out


def run_land_impact_report(
    features: pd.DataFrame,
    *,
    reports_dir: Path,
    lag_summary_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """Public entry point. Writes `land_impact_analysis.csv` and returns its path."""
    logger = logger or LOGGER
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "land_impact_analysis.csv"

    table = build_land_impact_table(features, lag_summary_path=lag_summary_path, logger=logger)
    table.to_csv(out_path, index=False)

    non_null: dict[str, Any] = {}
    for col in REQUIRED_COLS:
        if col in table.columns:
            non_null[col] = int(pd.to_numeric(table[col], errors="coerce").notna().sum())
    logger.info(
        "[LAND-IMPACT-REPORT] Wrote %s | rows=%d | non_null_counts=%s",
        out_path,
        len(table),
        non_null,
    )
    return out_path


__all__ = ["run_land_impact_report", "build_land_impact_table", "REQUIRED_COLS", "CONTEXT_COLS"]
