from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "correlations": 0.4,
    "lag_effects": 0.2,
    "exposure": 0.2,
    "anomalies": 0.2,
}

# Per-cell temporal stats need at least this many non-null weeks to be trusted.
MIN_WEEKS_PER_CELL_FOR_CORR = 8
MAX_LAG_WEEKS = 4
ROLLING_WINDOW_WEEKS = 12

# Time-varying driver / response candidates for the per-cell correlation
# components. Tried in order; the first pair where BOTH columns vary inside
# the same cell wins. `vessel_density` is intentionally NOT a candidate —
# in this pipeline it is a static spatial proxy (one value per grid cell), so
# any per-cell correlation involving it would be undefined.
PER_CELL_PAIR_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("NO2_mean", "detection_score"),
    ("NO2_mean", "oil_slick_probability_t"),
    ("NO2_mean", "fai_mean"),
)


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def _build_exposure_component(df: pd.DataFrame, logger) -> tuple[pd.Series, list[str]]:
    """Resolve exposure component using whichever columns actually exist."""
    parts: list[pd.Series] = []
    used: list[str] = []

    # Direct exposure score (already 0-1 in our pipeline).
    if "coastal_exposure_score" in df.columns:
        s = pd.to_numeric(df["coastal_exposure_score"], errors="coerce")
        if s.notna().any():
            parts.append(_minmax(s))
            used.append("coastal_exposure_score")

    # Distance-to-vessel-hotspot: invert so closer = higher exposure.
    if "distance_to_nearest_high_vessel_density_cell" in df.columns:
        s = pd.to_numeric(
            df["distance_to_nearest_high_vessel_density_cell"], errors="coerce"
        )
        if s.notna().any():
            parts.append(1.0 - _minmax(s))
            used.append("distance_to_nearest_high_vessel_density_cell(inv)")

    # Legacy port/lane distances (older pipelines).
    for col in ("distance_to_port", "distance_to_port_km", "distance_to_lane", "distance_to_shipping_km"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                parts.append(1.0 - _minmax(s))
                used.append(f"{col}(inv)")

    if not parts:
        logger.info("[COASTAL IMPACT] exposure component: no resolvable columns")
        return pd.Series(np.nan, index=df.index), used

    combined = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return combined, used


def _select_pair(df: pd.DataFrame) -> tuple[str, str] | None:
    """Pick the first candidate pair where both columns are time-varying."""
    for driver, response in PER_CELL_PAIR_CANDIDATES:
        if driver in df.columns and response in df.columns:
            return driver, response
    return None


def _rolling_abs_spearman(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    """Trailing-window |Spearman| for two aligned series."""
    pair = pd.concat([s1, s2], axis=1).dropna()
    if pair.empty:
        return pd.Series(np.nan, index=s1.index)
    ranks = pair.rank(method="average")
    r1 = ranks.iloc[:, 0]
    r2 = ranks.iloc[:, 1]
    # Pearson on ranks within a window == Spearman.
    mean1 = r1.rolling(window, min_periods=MIN_WEEKS_PER_CELL_FOR_CORR).mean()
    mean2 = r2.rolling(window, min_periods=MIN_WEEKS_PER_CELL_FOR_CORR).mean()
    cov = (r1 * r2).rolling(window, min_periods=MIN_WEEKS_PER_CELL_FOR_CORR).mean() - mean1 * mean2
    var1 = (r1 * r1).rolling(window, min_periods=MIN_WEEKS_PER_CELL_FOR_CORR).mean() - mean1 ** 2
    var2 = (r2 * r2).rolling(window, min_periods=MIN_WEEKS_PER_CELL_FOR_CORR).mean() - mean2 ** 2
    denom = (var1 * var2).clip(lower=0).pow(0.5).replace(0, np.nan)
    rho = cov / denom
    out = rho.abs().reindex(s1.index)
    return out


def _build_per_cell_correlation(df: pd.DataFrame, logger) -> pd.Series:
    """Hybrid per-cell |Spearman| component.

    For each cell, every week first tries the trailing-`ROLLING_WINDOW_WEEKS`
    rolling |Spearman| of (driver, response). When the rolling window can't
    be computed (e.g. low-coverage detection_score weeks), falls back to that
    cell's full-history |Spearman| scalar. This preserves temporal variation
    where the data supports it AND keeps broad row coverage.
    """
    pair = _select_pair(df)
    if pair is None or not {"grid_cell_id", "week_start_utc"}.issubset(df.columns):
        logger.info("[COASTAL IMPACT] corr component: no usable per-cell pair found")
        return pd.Series(np.nan, index=df.index)
    driver, response = pair

    work = df[["grid_cell_id", "week_start_utc", driver, response]].copy()
    work["week_start_utc"] = pd.to_datetime(work["week_start_utc"], errors="coerce", utc=True)
    work[driver] = pd.to_numeric(work[driver], errors="coerce")
    work[response] = pd.to_numeric(work[response], errors="coerce")
    work = work.sort_values(["grid_cell_id", "week_start_utc"])

    # Per-cell full-history scalar fallback.
    cell_scalar: dict[Any, float] = {}
    for cell_id, group in work.groupby("grid_cell_id", dropna=False):
        valid = group.dropna(subset=[driver, response])
        if (
            len(valid) < MIN_WEEKS_PER_CELL_FOR_CORR
            or valid[driver].nunique() <= 1
            or valid[response].nunique() <= 1
        ):
            cell_scalar[cell_id] = np.nan
            continue
        try:
            rho = float(valid[[driver, response]].corr(method="spearman").iloc[0, 1])
        except Exception:
            rho = np.nan
        cell_scalar[cell_id] = abs(rho) if not pd.isna(rho) else np.nan

    rolled = (
        work.groupby("grid_cell_id", dropna=False, group_keys=False)
        .apply(
            lambda g: _rolling_abs_spearman(g[driver], g[response], ROLLING_WINDOW_WEEKS),
            include_groups=False,
        )
    )

    work["rolling_rho"] = rolled.reindex(work.index).values
    work["scalar_rho"] = work["grid_cell_id"].map(cell_scalar).astype(float).values
    work["combined_rho"] = work["rolling_rho"].where(work["rolling_rho"].notna(), work["scalar_rho"])

    series = pd.Series(np.nan, index=df.index, dtype=float)
    series.loc[work.index] = work["combined_rho"].values

    n_cells_with_scalar = int(sum(1 for v in cell_scalar.values() if not pd.isna(v)))
    n_rows_with_rolling = int(work["rolling_rho"].notna().sum())
    n_rows_total = int(work["combined_rho"].notna().sum())
    logger.info(
        "[COASTAL IMPACT] corr component: %s vs %s | %d cells with scalar |Spearman|, "
        "%d rows have rolling-%dw signal, %d rows total non-null after fallback",
        driver,
        response,
        n_cells_with_scalar,
        n_rows_with_rolling,
        ROLLING_WINDOW_WEEKS,
        n_rows_total,
    )
    return series


def _build_per_cell_lag(df: pd.DataFrame, logger) -> pd.Series:
    """Best |lag-corr| between driver and response inside each cell.

    Per-row value is the cell's best |corr| over lags 1..MAX_LAG_WEEKS,
    broadcast to every week of that cell.
    """
    pair = _select_pair(df)
    if pair is None or not {"grid_cell_id", "week_start_utc"}.issubset(df.columns):
        logger.info("[COASTAL IMPACT] lag component: no usable per-cell pair found")
        return pd.Series(np.nan, index=df.index)
    driver, response = pair

    work = df[["grid_cell_id", "week_start_utc", driver, response]].copy()
    work["week_start_utc"] = pd.to_datetime(work["week_start_utc"], errors="coerce", utc=True)
    work[driver] = pd.to_numeric(work[driver], errors="coerce")
    work[response] = pd.to_numeric(work[response], errors="coerce")
    work = work.sort_values(["grid_cell_id", "week_start_utc"])

    cell_lag: dict[Any, float] = {}
    for cell_id, group in work.groupby("grid_cell_id", dropna=False):
        valid = group.dropna(subset=[driver, response])
        if len(valid) < MIN_WEEKS_PER_CELL_FOR_CORR + MAX_LAG_WEEKS:
            cell_lag[cell_id] = np.nan
            continue
        if valid[driver].nunique() <= 1 or valid[response].nunique() <= 1:
            cell_lag[cell_id] = np.nan
            continue
        best = 0.0
        d = valid[driver].reset_index(drop=True)
        r = valid[response].reset_index(drop=True)
        for lag in range(1, MAX_LAG_WEEKS + 1):
            if len(d) - lag < MIN_WEEKS_PER_CELL_FOR_CORR:
                continue
            try:
                paired = pd.concat(
                    [d.iloc[:-lag].reset_index(drop=True), r.iloc[lag:].reset_index(drop=True)],
                    axis=1,
                ).dropna()
                if len(paired) < MIN_WEEKS_PER_CELL_FOR_CORR:
                    continue
                if paired.iloc[:, 0].nunique() <= 1 or paired.iloc[:, 1].nunique() <= 1:
                    continue
                c = float(paired.corr(method="spearman").iloc[0, 1])
                if not pd.isna(c):
                    best = max(best, abs(c))
            except Exception:
                continue
        cell_lag[cell_id] = best if best > 0 else np.nan

    series = df["grid_cell_id"].map(cell_lag).astype(float)
    n_valid_cells = int(sum(1 for v in cell_lag.values() if not pd.isna(v)))
    logger.info(
        "[COASTAL IMPACT] lag component: %d cells produced best |lag-corr| (%s → %s, lags 1-%d)",
        n_valid_cells,
        driver,
        response,
        MAX_LAG_WEEKS,
    )
    return series


def run_coastal_impact_score(
    df: pd.DataFrame,
    feature_registry: dict[str, list[dict[str, Any]]],  # kept for compatibility
    logger,
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    _ = feature_registry
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    reports_dir = Path("outputs/reports")
    plots_dir = Path("outputs/plots")
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    base_cols = [
        c
        for c in ["grid_cell_id", "week_start_utc", "grid_centroid_lat", "grid_centroid_lon"]
        if c in df.columns
    ]
    out = df[base_cols].copy() if base_cols else pd.DataFrame(index=df.index)

    out["corr_component"] = _build_per_cell_correlation(df, logger).values
    out["lag_component"] = _build_per_cell_lag(df, logger).values
    exposure_series, exposure_sources = _build_exposure_component(df, logger)
    out["exposure_component"] = exposure_series.values

    anomaly_path = reports_dir / "anomaly_scores.csv"
    out["anomaly_component"] = np.nan
    if anomaly_path.exists() and {"grid_cell_id", "week_start_utc"}.issubset(set(out.columns)):
        try:
            adf = pd.read_csv(anomaly_path)
            if {"grid_cell_id", "week_start_utc", "anomaly_score"}.issubset(set(adf.columns)):
                adf["week_start_utc"] = pd.to_datetime(adf["week_start_utc"], utc=True, errors="coerce")
                tmp = out.copy()
                tmp["week_start_utc"] = pd.to_datetime(tmp["week_start_utc"], utc=True, errors="coerce")
                merged = tmp.merge(
                    adf[["grid_cell_id", "week_start_utc", "anomaly_score"]],
                    on=["grid_cell_id", "week_start_utc"],
                    how="left",
                )
                out["anomaly_component"] = _minmax(merged["anomaly_score"]).values
        except Exception as exc:  # noqa: BLE001
            logger.warning("[COASTAL IMPACT] Unable to merge anomaly scores: %s", exc)

    component_to_col = {
        "correlations": "corr_component",
        "lag_effects": "lag_component",
        "exposure": "exposure_component",
        "anomalies": "anomaly_component",
    }

    active = [k for k, c in component_to_col.items() if out[c].notna().any()]
    inactive = [k for k in component_to_col if k not in active]
    weight_mass_active = sum(weights.get(k, 0.0) for k in active)
    logger.info(
        "[COASTAL IMPACT] active components: %s | inactive: %s | active_weight_mass=%.2f | exposure_sources=%s",
        active,
        inactive,
        weight_mass_active,
        exposure_sources or "none",
    )

    score_parts = []
    for key, col in component_to_col.items():
        normalized = _minmax(out[col]) if out[col].notna().any() else pd.Series(np.nan, index=out.index)
        score_parts.append((weights.get(key, 0.0), normalized))

    weighted_sum = pd.Series(0.0, index=out.index)
    weight_den = pd.Series(0.0, index=out.index)
    for w, s in score_parts:
        valid = s.notna()
        weighted_sum.loc[valid] += float(w) * s.loc[valid]
        weight_den.loc[valid] += float(w)

    out["coastal_impact_score"] = weighted_sum / weight_den.replace(0, np.nan)
    out = (
        out.sort_values("coastal_impact_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    out.to_csv(reports_dir / "coastal_impact_score.csv", index=False)

    component_coverage = {
        col: float(out[col].notna().mean() * 100.0)
        for col in component_to_col.values()
    }
    logger.info(
        "[COASTAL IMPACT] component coverage %% (rows non-null): %s",
        {k: round(v, 1) for k, v in component_coverage.items()},
    )

    lat_col = next(
        (c for c in ["grid_centroid_lat", "centroid_lat", "lat", "latitude"] if c in out.columns),
        None,
    )
    lon_col = next(
        (c for c in ["grid_centroid_lon", "centroid_lon", "lon", "longitude", "lng"] if c in out.columns),
        None,
    )
    if lat_col and lon_col:
        sp = out[[lat_col, lon_col, "coastal_impact_score"]].copy()
        sp[lat_col] = pd.to_numeric(sp[lat_col], errors="coerce")
        sp[lon_col] = pd.to_numeric(sp[lon_col], errors="coerce")
        sp["coastal_impact_score"] = pd.to_numeric(sp["coastal_impact_score"], errors="coerce")
        sp = sp.dropna()
        if not sp.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            sc = ax.scatter(
                sp[lon_col], sp[lat_col], c=sp["coastal_impact_score"], cmap="viridis", s=18, alpha=0.85
            )
            ax.set_title("Coastal Impact Score map")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(sc, ax=ax, label="Coastal impact score")
            fig.tight_layout()
            fig.savefig(plots_dir / "coastal_impact_map.png", dpi=220)
            plt.close(fig)

    logger.info("[COASTAL IMPACT] Wrote score report: %s", reports_dir / "coastal_impact_score.csv")
    return out
