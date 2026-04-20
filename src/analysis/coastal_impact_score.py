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


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def _load_scalar_signals(reports_dir: Path) -> tuple[float | None, float | None]:
    corr_scalar = None
    lag_scalar = None

    corr_eval = reports_dir / "correlation_evaluation.csv"
    if corr_eval.exists():
        try:
            cdf = pd.read_csv(corr_eval)
            vals = pd.to_numeric(cdf.get("abs_pearson"), errors="coerce")
            if vals.notna().any():
                corr_scalar = float(vals.max())
        except Exception:
            pass

    lag_file = reports_dir / "lagged_correlations.csv"
    if lag_file.exists():
        try:
            ldf = pd.read_csv(lag_file)
            best = pd.to_numeric(ldf.get("best_correlation"), errors="coerce")
            if best.notna().any():
                lag_scalar = float(best.abs().max())
            elif "correlation" in ldf.columns:
                lag_scalar = float(pd.to_numeric(ldf["correlation"], errors="coerce").abs().max())
        except Exception:
            pass
    return corr_scalar, lag_scalar


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

    base_cols = [c for c in ["grid_cell_id", "week_start_utc", "grid_centroid_lat", "grid_centroid_lon"] if c in df.columns]
    out = df[base_cols].copy() if base_cols else pd.DataFrame(index=df.index)

    corr_scalar, lag_scalar = _load_scalar_signals(reports_dir)
    if corr_scalar is not None:
        out["corr_component"] = float(corr_scalar)
    else:
        out["corr_component"] = np.nan

    if lag_scalar is not None:
        out["lag_component"] = float(lag_scalar)
    else:
        out["lag_component"] = np.nan

    port_col = next((c for c in ["distance_to_port", "distance_to_port_km"] if c in df.columns), None)
    lane_col = next((c for c in ["distance_to_lane", "distance_to_shipping_km"] if c in df.columns), None)
    exposure_parts: list[pd.Series] = []
    if port_col:
        exposure_parts.append(1.0 - _minmax(df[port_col]))
    if lane_col:
        exposure_parts.append(1.0 - _minmax(df[lane_col]))
    out["exposure_component"] = pd.concat(exposure_parts, axis=1).mean(axis=1) if exposure_parts else np.nan

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

    score_parts = []
    for key, col in [
        ("correlations", "corr_component"),
        ("lag_effects", "lag_component"),
        ("exposure", "exposure_component"),
        ("anomalies", "anomaly_component"),
    ]:
        normalized = _minmax(out[col]) if out[col].notna().any() else pd.Series(np.nan, index=out.index)
        score_parts.append((weights.get(key, 0.0), normalized))

    weighted_sum = pd.Series(0.0, index=out.index)
    weight_den = pd.Series(0.0, index=out.index)
    for w, s in score_parts:
        valid = s.notna()
        weighted_sum.loc[valid] += float(w) * s.loc[valid]
        weight_den.loc[valid] += float(w)

    out["coastal_impact_score"] = weighted_sum / weight_den.replace(0, np.nan)
    out = out.sort_values("coastal_impact_score", ascending=False, na_position="last").reset_index(drop=True)
    out.to_csv(reports_dir / "coastal_impact_score.csv", index=False)

    lat_col = next((c for c in ["grid_centroid_lat", "centroid_lat", "lat", "latitude"] if c in out.columns), None)
    lon_col = next((c for c in ["grid_centroid_lon", "centroid_lon", "lon", "longitude", "lng"] if c in out.columns), None)
    if lat_col and lon_col:
        sp = out[[lat_col, lon_col, "coastal_impact_score"]].copy()
        sp[lat_col] = pd.to_numeric(sp[lat_col], errors="coerce")
        sp[lon_col] = pd.to_numeric(sp[lon_col], errors="coerce")
        sp["coastal_impact_score"] = pd.to_numeric(sp["coastal_impact_score"], errors="coerce")
        sp = sp.dropna()
        if not sp.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            sc = ax.scatter(sp[lon_col], sp[lat_col], c=sp["coastal_impact_score"], cmap="viridis", s=18, alpha=0.85)
            ax.set_title("Coastal Impact Score map")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(sc, ax=ax, label="Coastal impact score")
            fig.tight_layout()
            fig.savefig(plots_dir / "coastal_impact_map.png", dpi=220)
            plt.close(fig)

    logger.info("[COASTAL IMPACT] Wrote score report: %s", reports_dir / "coastal_impact_score.csv")
    return out
