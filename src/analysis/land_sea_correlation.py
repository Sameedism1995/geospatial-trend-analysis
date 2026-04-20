"""
Lagged land–sea correlation analysis (LAND IMPACT EXTENSION LAYER, Step 4).

For each configured driver→response pair, computes:
    * Pearson and Spearman correlation at lags 1..MAX_LAG (weeks)
    * Best lag (by |Spearman|), best value, and a rank-significance proxy

Driver→response pairs (configurable via the `pairs` argument):
    vessel_density → ndvi_mean          (1..MAX_LAG weeks)
    NO2_mean       → ndvi_mean          (1..MAX_LAG weeks)

Lagging convention: `driver[t-k]` vs `response[t]`, k ∈ {1..MAX_LAG}. Shifts are
performed per `grid_cell_id` so they never leak across cells.

Outputs:
    outputs/reports/land_sea_lagged_correlations.csv   (long format, one row per pair×lag)
    outputs/reports/land_sea_lag_summary.csv           (one row per pair with best lag)

This module is read-only on the filesystem during computation; it writes only to the
`reports_dir` you pass in. It is NON-BLOCKING: any internal error returns an empty
summary instead of raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("land_sea_correlation")

MAX_LAG_DEFAULT = 8


@dataclass(frozen=True)
class LagPair:
    driver: str
    response: str
    label: str


DEFAULT_PAIRS: tuple[LagPair, ...] = (
    LagPair(driver="vessel_density", response="ndvi_mean", label="vessel_density -> ndvi_mean"),
    LagPair(driver="NO2_mean", response="ndvi_mean", label="NO2_mean -> ndvi_mean"),
)


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pair_columns(df: pd.DataFrame, driver: str, response: str) -> tuple[str | None, str | None]:
    driver_candidates = {
        "vessel_density": ["vessel_density", "vessel_density_t"],
        "NO2_mean": ["NO2_mean", "no2_mean_t"],
    }
    response_candidates = {
        "ndvi_mean": ["ndvi_mean"],
    }
    drv = _first_present(df, driver_candidates.get(driver, [driver]))
    rsp = _first_present(df, response_candidates.get(response, [response]))
    return drv, rsp


def _shift_per_grid(df: pd.DataFrame, column: str, lag: int, grid_col: str, week_col: str) -> pd.Series:
    sorted_df = df.sort_values([grid_col, week_col])
    shifted = sorted_df.groupby(grid_col)[column].shift(lag)
    return shifted.reindex(df.index)


def _significance_label(abs_r: float, n_obs: int) -> str:
    if not np.isfinite(abs_r) or n_obs < 30:
        return "insufficient"
    if abs_r >= 0.5:
        return "strong"
    if abs_r >= 0.3:
        return "moderate"
    if abs_r >= 0.15:
        return "weak"
    return "negligible"


def compute_lagged_correlations(
    df: pd.DataFrame,
    *,
    pairs: Iterable[LagPair] = DEFAULT_PAIRS,
    max_lag: int = MAX_LAG_DEFAULT,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (long_table, summary_table) of land-sea lagged correlations."""
    logger = logger or LOGGER
    long_rows: list[dict] = []
    summary_rows: list[dict] = []

    if df.empty:
        logger.warning("[LAND-SEA-CORR] Empty dataframe — returning empty frames")
        return pd.DataFrame(), pd.DataFrame()

    grid_col = _first_present(df, ["grid_cell_id"])
    week_col = _first_present(df, ["week_start_utc", "week_start", "week"])
    if grid_col is None or week_col is None:
        logger.warning("[LAND-SEA-CORR] Missing grid/week columns — skipping")
        return pd.DataFrame(), pd.DataFrame()

    base = df.copy()
    base[week_col] = pd.to_datetime(base[week_col], utc=True, errors="coerce")
    base = base.dropna(subset=[grid_col, week_col])

    for pair in pairs:
        drv, rsp = _pair_columns(base, pair.driver, pair.response)
        if drv is None or rsp is None:
            logger.warning(
                "[LAND-SEA-CORR] Skipping pair %s: driver/response column not found (driver=%s, response=%s)",
                pair.label,
                drv,
                rsp,
            )
            continue

        best = {"lag": None, "spearman": np.nan, "pearson": np.nan, "n": 0}
        for lag in range(1, max_lag + 1):
            shifted = _shift_per_grid(base, drv, lag, grid_col, week_col)
            response_vals = pd.to_numeric(base[rsp], errors="coerce")
            paired = pd.DataFrame({"driver_lag": pd.to_numeric(shifted, errors="coerce"), "response": response_vals}).dropna()
            n = int(len(paired))
            if n < 10:
                long_rows.append({
                    "pair": pair.label,
                    "driver": pair.driver,
                    "response": pair.response,
                    "lag_weeks": int(lag),
                    "pearson": np.nan,
                    "spearman": np.nan,
                    "n": n,
                    "significance": "insufficient",
                })
                continue

            pearson = float(paired["driver_lag"].corr(paired["response"], method="pearson"))
            spearman = float(paired["driver_lag"].corr(paired["response"], method="spearman"))
            abs_s = abs(spearman) if np.isfinite(spearman) else 0.0
            long_rows.append({
                "pair": pair.label,
                "driver": pair.driver,
                "response": pair.response,
                "lag_weeks": int(lag),
                "pearson": pearson,
                "spearman": spearman,
                "n": n,
                "significance": _significance_label(abs_s, n),
            })
            if np.isfinite(spearman) and abs_s > (abs(best["spearman"]) if np.isfinite(best["spearman"]) else 0.0):
                best = {"lag": int(lag), "spearman": spearman, "pearson": pearson, "n": n}

        summary_rows.append({
            "pair": pair.label,
            "driver": pair.driver,
            "response": pair.response,
            "best_lag_weeks": best["lag"],
            "best_spearman": best["spearman"],
            "best_pearson": best["pearson"],
            "n_at_best_lag": best["n"],
            "significance_rank": _significance_label(abs(best["spearman"]) if np.isfinite(best["spearman"]) else 0.0, best["n"]),
        })

    long_df = pd.DataFrame(long_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="best_spearman", key=lambda s: s.abs(), ascending=False, na_position="last"
    ).reset_index(drop=True)

    logger.info(
        "[LAND-SEA-CORR] Computed lagged correlations: %d pair×lag rows over %d pairs",
        len(long_df),
        len(summary_df),
    )
    return long_df, summary_df


def run_land_sea_correlation(
    features: pd.DataFrame,
    *,
    reports_dir: Path,
    max_lag: int = MAX_LAG_DEFAULT,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """Public entry point used by orchestrator / CLI. Writes CSVs and returns paths."""
    logger = logger or LOGGER
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    long_df, summary_df = compute_lagged_correlations(features, max_lag=max_lag, logger=logger)
    long_path = reports_dir / "land_sea_lagged_correlations.csv"
    summary_path = reports_dir / "land_sea_lag_summary.csv"
    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    logger.info("[LAND-SEA-CORR] Wrote %s", long_path)
    logger.info("[LAND-SEA-CORR] Wrote %s", summary_path)
    return {"long": long_path, "summary": summary_path}


__all__ = [
    "LagPair",
    "DEFAULT_PAIRS",
    "compute_lagged_correlations",
    "run_land_sea_correlation",
    "MAX_LAG_DEFAULT",
]
