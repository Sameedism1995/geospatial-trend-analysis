from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# Candidate-list resolver: each inner list is a fallback chain for one logical
# feature. The first column that actually exists in the dataframe wins. This
# keeps the anomaly model robust to renaming between extraction stages
# (`ndci` vs `ndci_mean`, `NO2_variability` vs `no2_std_t`, etc.).
FEATURE_CANDIDATE_CHAINS: tuple[tuple[str, ...], ...] = (
    ("vessel_density", "vessel_density_t", "total_density"),
    ("NO2_mean", "no2_mean_t"),
    ("NO2_variability", "no2_std_t"),
    ("ndci_mean", "ndci_median", "ndci"),
    ("ndti_mean", "ndti_median", "ndti"),
    ("ndwi_mean", "ndwi_median", "ndwi"),
    ("fai_mean", "fai_median", "fai"),
    ("b11_mean", "b11_median", "b11"),
    ("sst", "sst_mean"),
    ("detection_score", "oil_slick_probability_t"),
)

MIN_NON_NULL_RATIO = 0.0  # keep all rows; missing features filled with column median
MIN_WEEKS_PER_CELL_FOR_TEMPORAL = 12
TEMPORAL_Z_THRESHOLD = 2.0


def _resolve_columns(df: pd.DataFrame) -> list[str]:
    """Resolve each candidate chain to the first column present in `df`."""
    resolved: list[str] = []
    for chain in FEATURE_CANDIDATE_CHAINS:
        for cand in chain:
            if cand in df.columns:
                resolved.append(cand)
                break
    # de-dupe while preserving order
    return list(dict.fromkeys(resolved))


def _spatial_zscore(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return pd.Series(np.nan, index=df.index)
    z = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, np.nan)
    return z.abs().mean(axis=1, skipna=True)


def _temporal_anomaly_report(
    base: pd.DataFrame,
    anomaly_score: pd.Series,
    logger,
) -> pd.DataFrame:
    """Compute per-cell temporal z-score of the anomaly score.

    For every grid cell with ≥ MIN_WEEKS_PER_CELL_FOR_TEMPORAL non-null weeks,
    standardise the anomaly score against THAT cell's own history. This
    surfaces "unusual for this place at this time" rather than "always-busy
    cell", which is what the spatial IF score reports.
    """
    if "grid_cell_id" not in base.columns:
        logger.warning("[ANOMALY] Temporal z-score skipped: missing grid_cell_id")
        return pd.DataFrame()

    df = base.copy()
    df["anomaly_score"] = pd.to_numeric(anomaly_score, errors="coerce").values

    grp = df.groupby("grid_cell_id", dropna=False)["anomaly_score"]
    counts = grp.transform("count")
    means = grp.transform("mean")
    stds = grp.transform("std")

    z = (df["anomaly_score"] - means) / stds.replace(0, np.nan)
    eligible = counts >= MIN_WEEKS_PER_CELL_FOR_TEMPORAL
    z = z.where(eligible)

    df["temporal_z_score"] = z
    df["temporal_anomaly_label"] = np.where(
        z.abs() >= TEMPORAL_Z_THRESHOLD, "anomalous", "normal"
    )
    df.loc[~eligible, "temporal_anomaly_label"] = "insufficient_history"

    cells_with_history = int((counts >= MIN_WEEKS_PER_CELL_FOR_TEMPORAL).any())
    n_cells_eligible = int(df.loc[eligible, "grid_cell_id"].nunique())
    n_anom = int((df["temporal_anomaly_label"] == "anomalous").sum())
    logger.info(
        "[ANOMALY-TEMPORAL] eligible_cells=%d any_history=%s anomalous_rows=%d",
        n_cells_eligible,
        bool(cells_with_history),
        n_anom,
    )
    return df


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

    base_cols = [
        c
        for c in ["grid_cell_id", "week_start_utc", "grid_centroid_lat", "grid_centroid_lon"]
        if c in df.columns
    ]
    out = df[base_cols].copy() if base_cols else pd.DataFrame(index=df.index)
    features = _resolve_columns(df)
    logger.info("[ANOMALY] Resolved %d feature columns: %s", len(features), features)

    if not features:
        out["anomaly_score"] = np.nan
        out["anomaly_label"] = "normal"
        out.to_csv(reports_dir / "anomaly_scores.csv", index=False)
        logger.warning("[ANOMALY] Skipped Isolation Forest: no required features found.")
        return out

    x = df[features].apply(pd.to_numeric, errors="coerce")
    non_null_ratio = x.notna().mean(axis=1)
    logger.info(
        "[ANOMALY] Feature coverage: %d rows × %d features | mean per-row coverage=%.1f%%",
        len(x),
        len(features),
        float(non_null_ratio.mean() * 100.0),
    )

    medians = x.median(numeric_only=True)
    x_filled = x.fillna(medians)
    if x_filled.isna().all(axis=None):
        out["anomaly_score"] = _spatial_zscore(df)
        out["anomaly_label"] = np.where(
            out["anomaly_score"].fillna(0) >= 2.0, "anomalous", "normal"
        )
        out.to_csv(reports_dir / "anomaly_scores.csv", index=False)
        logger.warning("[ANOMALY] Isolation Forest unavailable; used z-score fallback.")
        return out

    anomaly_score = pd.Series(np.nan, index=df.index, dtype=float)
    try:
        model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42
        )
        model.fit(x_filled)
        raw_score = -model.score_samples(x_filled)  # higher = more anomalous
        z_score_full = _spatial_zscore(df).fillna(0).values
        blended = 0.7 * pd.Series(raw_score, index=x_filled.index) + 0.3 * pd.Series(
            z_score_full, index=x_filled.index
        )
        anomaly_score.loc[x_filled.index] = blended
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[ANOMALY] Isolation Forest failed, using z-score fallback: %s", exc
        )
        anomaly_score = _spatial_zscore(df)

    threshold = (
        float(anomaly_score.quantile(0.92))
        if anomaly_score.notna().any()
        else float("inf")
    )
    out["anomaly_score"] = anomaly_score
    out["anomaly_label"] = np.where(
        out["anomaly_score"] >= threshold, "anomalous", "normal"
    )
    out.to_csv(reports_dir / "anomaly_scores.csv", index=False)
    logger.info(
        "[ANOMALY] Wrote spatial-outlier scores (n=%d, anomalous=%d): %s",
        int(out["anomaly_score"].notna().sum()),
        int((out["anomaly_label"] == "anomalous").sum()),
        reports_dir / "anomaly_scores.csv",
    )

    temporal_df = _temporal_anomaly_report(out[base_cols], anomaly_score, logger)
    if not temporal_df.empty:
        temporal_path = reports_dir / "anomaly_scores_temporal.csv"
        temporal_df.to_csv(temporal_path, index=False)
        logger.info(
            "[ANOMALY-TEMPORAL] Wrote per-cell z-score anomalies: %s", temporal_path
        )

    lat_col = next(
        (c for c in ["grid_centroid_lat", "centroid_lat", "lat", "latitude"] if c in df.columns),
        None,
    )
    lon_col = next(
        (c for c in ["grid_centroid_lon", "centroid_lon", "lon", "longitude", "lng"] if c in df.columns),
        None,
    )
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
            ax.set_title("Spatial anomaly map (Isolation Forest blend)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(sc, ax=ax, label="Anomaly score")
            fig.tight_layout()
            fig.savefig(plots_dir / "anomaly_map.png", dpi=220)
            plt.close(fig)

    return out
