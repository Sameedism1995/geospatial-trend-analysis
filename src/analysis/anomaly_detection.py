from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _resolve_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "vessel_density",
        "vessel_density_t",
        "total_density",
        "NO2_mean",
        "no2_mean_t",
        "NO2_variability",
        "no2_std_t",
        "ndci",
        "ndti",
        "ndwi",
        "fai",
        "sst",
    ]
    cols = [c for c in candidates if c in df.columns]
    return list(dict.fromkeys(cols))


def _spatial_zscore(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return pd.Series(np.nan, index=df.index)
    z = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, np.nan)
    return z.abs().mean(axis=1, skipna=True)


def run_anomaly_detection(
    df: pd.DataFrame,
    feature_registry: dict[str, list[dict[str, Any]]],  # kept for registry compatibility
    logger,
    *,
    contamination: float = 0.08,
) -> pd.DataFrame:
    _ = feature_registry
    reports_dir = Path("outputs/reports")
    plots_dir = Path("outputs/plots")
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    base_cols = [c for c in ["grid_cell_id", "week_start_utc", "grid_centroid_lat", "grid_centroid_lon"] if c in df.columns]
    out = df[base_cols].copy() if base_cols else pd.DataFrame(index=df.index)
    features = _resolve_columns(df)

    if not features:
        out["anomaly_score"] = np.nan
        out["anomaly_label"] = "normal"
        out.to_csv(reports_dir / "anomaly_scores.csv", index=False)
        logger.warning("[ANOMALY] Skipped Isolation Forest: no required features found.")
        return out

    x = df[features].apply(pd.to_numeric, errors="coerce")
    x_filled = x.fillna(x.median(numeric_only=True))
    if x_filled.isna().all(axis=None):
        out["anomaly_score"] = _spatial_zscore(df)
        out["anomaly_label"] = np.where(out["anomaly_score"].fillna(0) >= 2.0, "anomalous", "normal")
        out.to_csv(reports_dir / "anomaly_scores.csv", index=False)
        logger.warning("[ANOMALY] Isolation Forest unavailable due to all-NaN features; used z-score fallback.")
        return out

    try:
        model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        model.fit(x_filled)
        raw_score = -model.score_samples(x_filled)  # higher = more anomalous
    except Exception as exc:  # noqa: BLE001
        logger.exception("[ANOMALY] Isolation Forest failed, using z-score fallback: %s", exc)
        raw_score = _spatial_zscore(df).fillna(0).values

    z_score = _spatial_zscore(df).fillna(0).values
    anomaly_score = 0.7 * pd.Series(raw_score, index=df.index) + 0.3 * pd.Series(z_score, index=df.index)
    threshold = float(anomaly_score.quantile(0.92)) if anomaly_score.notna().any() else np.inf

    out["anomaly_score"] = anomaly_score
    out["anomaly_label"] = np.where(out["anomaly_score"] >= threshold, "anomalous", "normal")
    out.to_csv(reports_dir / "anomaly_scores.csv", index=False)

    lat_col = next((c for c in ["grid_centroid_lat", "centroid_lat", "lat", "latitude"] if c in df.columns), None)
    lon_col = next((c for c in ["grid_centroid_lon", "centroid_lon", "lon", "longitude", "lng"] if c in df.columns), None)
    if lat_col and lon_col:
        tmp = pd.DataFrame(
            {
                "lat": pd.to_numeric(df[lat_col], errors="coerce"),
                "lon": pd.to_numeric(df[lon_col], errors="coerce"),
                "score": pd.to_numeric(out["anomaly_score"], errors="coerce"),
            }
        ).dropna()
        if not tmp.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            sc = ax.scatter(tmp["lon"], tmp["lat"], c=tmp["score"], cmap="magma", s=16, alpha=0.8)
            ax.set_title("Spatial anomaly map (Isolation Forest + z-score fallback)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(sc, ax=ax, label="Anomaly score")
            fig.tight_layout()
            fig.savefig(plots_dir / "anomaly_map.png", dpi=220)
            plt.close(fig)

    logger.info("[ANOMALY] Wrote anomaly scores: %s", reports_dir / "anomaly_scores.csv")
    return out
