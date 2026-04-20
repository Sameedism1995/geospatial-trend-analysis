from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _resolve_category_columns(feature_registry: dict[str, list[dict[str, Any]]], df: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for category, entries in feature_registry.items():
        cols: list[str] = []
        for entry in entries:
            col = entry.get("resolved_column")
            if col and col in df.columns:
                cols.append(col)
        out[category] = list(dict.fromkeys(cols))
    return out


def _numeric_non_constant(df: pd.DataFrame, columns: list[str]) -> list[str]:
    kept: list[str] = []
    for col in columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if not vals.notna().any():
            continue
        if float(vals.var(skipna=True)) == 0.0:
            continue
        kept.append(col)
    return kept


def _pairwise_weekly_temporal(
    df: pd.DataFrame,
    pair_name: str,
    x_col: str,
    y_col: str,
    time_col: str,
    window_size: int = 4,
) -> pd.DataFrame:
    work = df[[time_col, x_col, y_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    work = work.dropna(subset=[time_col])
    work["week"] = work[time_col].dt.tz_localize(None).dt.to_period("W").astype(str)
    rows: list[dict[str, Any]] = []
    for week, grp in work.groupby("week", sort=True):
        pair = grp[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 3:
            continue
        corr = float(pair[x_col].corr(pair[y_col], method="pearson"))
        rows.append({"feature_pair": pair_name, "week": week, "correlation_value": corr})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("week").reset_index(drop=True)
    out["rolling_mean"] = out["correlation_value"].rolling(window=window_size, min_periods=1).mean()
    out["rolling_std"] = out["correlation_value"].rolling(window=window_size, min_periods=1).std().fillna(0.0)
    return out


def _lag_response_curve(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    time_col: str,
    grid_col: str,
    lags: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[[grid_col, time_col, x_col, y_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
    work = work.dropna(subset=[grid_col, time_col])
    work = work.sort_values([grid_col, time_col]).reset_index(drop=True)
    work["week"] = work[time_col].dt.tz_localize(None).dt.to_period("W").astype(str)

    rows: list[dict[str, Any]] = []
    weekly_peak_records: list[dict[str, Any]] = []

    lag_week_corrs: dict[int, pd.Series] = {}
    for lag in lags:
        shifted = work[[grid_col, "week", x_col, y_col]].copy()
        shifted["_y_future"] = shifted.groupby(grid_col)[y_col].shift(-lag)
        pair = shifted[[x_col, "_y_future"]].apply(pd.to_numeric, errors="coerce").dropna()
        corr = float(pair[x_col].corr(pair["_y_future"], method="pearson")) if len(pair) >= 3 else np.nan
        rows.append(
            {
                "feature_x": x_col,
                "feature_y": y_col,
                "lag": lag,
                "correlation": corr,
            }
        )
        week_corr = (
            shifted.groupby("week", sort=True)
            .apply(
                lambda g: (
                    g[[x_col, "_y_future"]]
                    .apply(pd.to_numeric, errors="coerce")
                    .dropna()
                    .pipe(lambda d: d[x_col].corr(d["_y_future"], method="pearson") if len(d) >= 3 else np.nan)
                )
            )
            .astype(float)
        )
        lag_week_corrs[lag] = week_corr

    out = pd.DataFrame(rows)
    if out.empty:
        return out, pd.DataFrame()

    peak_idx = out["correlation"].abs().idxmax()
    best_lag = int(out.loc[peak_idx, "lag"])
    best_corr = float(out.loc[peak_idx, "correlation"]) if pd.notna(out.loc[peak_idx, "correlation"]) else np.nan
    best_sign = (
        "positive"
        if pd.notna(best_corr) and best_corr > 0
        else ("negative" if pd.notna(best_corr) and best_corr < 0 else "neutral")
    )

    for week in sorted(set().union(*[s.index.tolist() for s in lag_week_corrs.values()])):
        vals: list[tuple[int, float]] = []
        for lag in lags:
            val = lag_week_corrs[lag].get(week, np.nan)
            if pd.notna(val):
                vals.append((lag, float(val)))
        if not vals:
            continue
        wk_best = max(vals, key=lambda t: abs(t[1]))
        weekly_peak_records.append({"week": week, "best_lag_week": int(wk_best[0]), "best_corr_week": float(wk_best[1])})

    weekly_peak = pd.DataFrame(weekly_peak_records)
    if weekly_peak.empty:
        lag_position_std = np.nan
        consistency_score = np.nan
        mean_corr_windows = np.nan
    else:
        lag_position_std = float(weekly_peak["best_lag_week"].std()) if len(weekly_peak) > 1 else 0.0
        same_sign = weekly_peak["best_corr_week"].apply(lambda v: np.sign(v) == np.sign(best_corr) if pd.notna(best_corr) else False)
        consistency_score = float(same_sign.mean() * 100.0)
        mean_corr_windows = float(weekly_peak["best_corr_week"].mean())

    out["is_peak"] = out["lag"] == best_lag
    out["consistency_score"] = consistency_score
    out["best_lag"] = best_lag
    out["best_correlation"] = best_corr
    out["sign"] = best_sign
    out["mean_correlation_time_windows"] = mean_corr_windows
    out["lag_position_std"] = lag_position_std
    return out, weekly_peak


def _interpret_response(best_lag: int, best_corr: float) -> str:
    if pd.isna(best_corr) or abs(best_corr) < 0.1:
        return "no response"
    if best_lag <= 1:
        return "immediate coupling"
    return "delayed response"


def _plot_lag_response_curve(curve_df: pd.DataFrame, output_dir: Path) -> None:
    if curve_df.empty:
        return
    feature_x = str(curve_df.iloc[0]["feature_x"])
    feature_y = str(curve_df.iloc[0]["feature_y"])
    peak = curve_df[curve_df["is_peak"]]
    if peak.empty:
        return
    peak_row = peak.iloc[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve_df["lag"], curve_df["correlation"], marker="o", linewidth=1.6)
    ax.scatter([peak_row["lag"]], [peak_row["correlation"]], s=90)
    ax.annotate(
        f"peak lag={int(peak_row['lag'])}, r={float(peak_row['correlation']):.3f}",
        (peak_row["lag"], peak_row["correlation"]),
        textcoords="offset points",
        xytext=(6, 8),
    )
    ax.set_xlabel("Lag (weeks)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Lag response: {feature_x} -> {feature_y}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    fig.tight_layout()
    file_name = f"lag_response_{feature_x}_to_{feature_y}".replace("/", "_").replace(" ", "_")
    fig.savefig(output_dir / f"{file_name}.png", dpi=220)
    plt.close(fig)


def _run_ml_baseline(
    df: pd.DataFrame,
    category_columns: dict[str, list[str]],
    reports_dir: Path,
    logger,
) -> None:
    target_candidates = ["NO2_mean", "no2_mean_t", "ndti", "ndwi", "detection_score", "oil_slick_probability_t"]
    target = _pick_first_existing(df, target_candidates)
    if target is None:
        logger.warning("[ML BASELINE] Skipped: no suitable target column found")
        pd.DataFrame(columns=["feature", "importance_score"]).to_csv(
            reports_dir / "feature_importance_baseline.csv",
            index=False,
        )
        return

    input_cols = (
        category_columns.get("maritime_activity", [])
        + category_columns.get("exposure", [])
        + category_columns.get("water_quality", [])
        + category_columns.get("atmospheric", [])
    )
    input_cols = [c for c in list(dict.fromkeys(input_cols)) if c != target and c in df.columns]
    input_cols = _numeric_non_constant(df, input_cols)
    if not input_cols:
        logger.warning("[ML BASELINE] Skipped: no usable input features")
        pd.DataFrame(columns=["feature", "importance_score"]).to_csv(
            reports_dir / "feature_importance_baseline.csv",
            index=False,
        )
        return

    model_df = df[input_cols + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(model_df) < 20:
        logger.warning("[ML BASELINE] Skipped: insufficient rows after cleanup")
        pd.DataFrame(columns=["feature", "importance_score"]).to_csv(
            reports_dir / "feature_importance_baseline.csv",
            index=False,
        )
        return

    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as exc:  # noqa: BLE001
        logger.warning("[ML BASELINE] Skipped: sklearn unavailable (%s)", exc)
        pd.DataFrame(columns=["feature", "importance_score"]).to_csv(
            reports_dir / "feature_importance_baseline.csv",
            index=False,
        )
        return

    x = model_df[input_cols]
    y = model_df[target]
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x, y)
    out = pd.DataFrame({"feature": input_cols, "importance_score": model.feature_importances_}).sort_values(
        "importance_score",
        ascending=False,
    )
    out.to_csv(reports_dir / "feature_importance_baseline.csv", index=False)
    logger.info("[ML BASELINE] Feature importance computed for target: %s", target)


def run_scientific_validation(df: pd.DataFrame, feature_registry: dict[str, list[dict[str, Any]]], logger) -> None:
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.warning("[SCIENTIFIC VALIDATION] Skipped: empty dataframe")
        pd.DataFrame(columns=["feature_pair", "week", "correlation_value", "rolling_mean", "rolling_std"]).to_csv(
            reports_dir / "temporal_stability.csv",
            index=False,
        )
        pd.DataFrame(columns=["feature_x", "feature_y", "lag", "correlation", "is_peak", "consistency_score"]).to_csv(
            reports_dir / "causal_lag_analysis.csv",
            index=False,
        )
        pd.DataFrame(
            columns=[
                "feature_x",
                "feature_y",
                "best_lag",
                "best_correlation",
                "sign",
                "mean_correlation_time_windows",
                "lag_position_std",
                "consistency_score",
                "interpretation_label",
            ]
        ).to_csv(
            reports_dir / "causal_lag_summary.csv",
            index=False,
        )
        pd.DataFrame(columns=["feature_x", "feature_y", "lag", "correlation", "best_lag"]).to_csv(
            reports_dir / "lagged_correlations.csv",
            index=False,
        )
        pd.DataFrame(columns=["feature", "importance_score"]).to_csv(
            reports_dir / "feature_importance_baseline.csv",
            index=False,
        )
        return

    category_columns = _resolve_category_columns(feature_registry, df)
    time_col = _pick_first_existing(df, ["week_start_utc", "week_start", "week", "timestamp", "date"])
    grid_col = _pick_first_existing(df, ["grid_cell_id", "grid_id", "cell_id"])

    vessel_col = _pick_first_existing(df, ["vessel_density", "vessel_density_t"])
    no2_col = _pick_first_existing(df, ["NO2_mean", "no2_mean_t"])
    ndwi_col = _pick_first_existing(df, ["ndwi"])
    ndti_col = _pick_first_existing(df, ["ndti"])
    sentinel_col = _pick_first_existing(df, ["detection_score", "oil_slick_probability_t"])

    temporal_frames: list[pd.DataFrame] = []
    lag_frames: list[pd.DataFrame] = []
    lag_summary_rows: list[dict[str, Any]] = []
    if vessel_col and time_col:
        temporal_targets = [
            ("vessel_density vs no2_mean", no2_col),
            ("vessel_density vs ndwi", ndwi_col),
            ("vessel_density vs ndti", ndti_col),
            ("vessel_density vs sentinel_disturbance", sentinel_col),
        ]
        for pair_name, target_col in temporal_targets:
            if target_col is None:
                continue
            frame = _pairwise_weekly_temporal(df, pair_name, vessel_col, target_col, time_col)
            if not frame.empty:
                temporal_frames.append(frame)
                logger.info("[TEMPORAL STABILITY] %s stability computed", pair_name)

    if vessel_col and time_col and grid_col:
        for target_col in [no2_col, ndwi_col, ndti_col, sentinel_col]:
            if target_col is None:
                continue
            lag_df, weekly_peak = _lag_response_curve(df, vessel_col, target_col, time_col, grid_col, lags=[0, 1, 2, 3, 4, 5])
            if lag_df.empty:
                continue
            lag_frames.append(lag_df)
            best_row = lag_df[lag_df["is_peak"]].iloc[0]
            lag_summary_rows.append(
                {
                    "feature_x": vessel_col,
                    "feature_y": target_col,
                    "best_lag": int(best_row["best_lag"]),
                    "best_correlation": float(best_row["best_correlation"]),
                    "sign": str(best_row["sign"]),
                    "mean_correlation_time_windows": float(best_row["mean_correlation_time_windows"])
                    if pd.notna(best_row["mean_correlation_time_windows"])
                    else np.nan,
                    "lag_position_std": float(best_row["lag_position_std"]) if pd.notna(best_row["lag_position_std"]) else np.nan,
                    "consistency_score": float(best_row["consistency_score"]) if pd.notna(best_row["consistency_score"]) else np.nan,
                    "interpretation_label": _interpret_response(int(best_row["best_lag"]), float(best_row["best_correlation"])),
                }
            )
            logger.info(
                "[LAG ANALYSIS] Best lag for %s \u2192 %s = %s weeks",
                vessel_col,
                target_col,
                int(best_row["best_lag"]),
            )
            # Optional spatial-cluster style metric when grid id exists: weekly best-lag variability already captured.
            _ = weekly_peak

    temporal_out = (
        pd.concat(temporal_frames, ignore_index=True)
        if temporal_frames
        else pd.DataFrame(columns=["feature_pair", "week", "correlation_value", "rolling_mean", "rolling_std"])
    )
    temporal_out.to_csv(reports_dir / "temporal_stability.csv", index=False)

    lag_out = (
        pd.concat(lag_frames, ignore_index=True)
        if lag_frames
        else pd.DataFrame(
            columns=[
                "feature_x",
                "feature_y",
                "lag",
                "correlation",
                "is_peak",
                "consistency_score",
                "best_lag",
                "best_correlation",
                "sign",
                "mean_correlation_time_windows",
                "lag_position_std",
            ]
        )
    )
    lag_summary = (
        pd.DataFrame(lag_summary_rows)
        if lag_summary_rows
        else pd.DataFrame(
            columns=[
                "feature_x",
                "feature_y",
                "best_lag",
                "best_correlation",
                "sign",
                "mean_correlation_time_windows",
                "lag_position_std",
                "consistency_score",
                "interpretation_label",
            ]
        )
    )
    lag_out[["feature_x", "feature_y", "lag", "correlation", "is_peak", "consistency_score"]].to_csv(
        reports_dir / "causal_lag_analysis.csv",
        index=False,
    )
    lag_summary.to_csv(reports_dir / "causal_lag_summary.csv", index=False)
    # Backward-compatible output retained.
    lag_out.to_csv(reports_dir / "lagged_correlations.csv", index=False)

    plot_dir = Path("outputs/plots/lag_response_curves")
    for pair in lag_frames:
        _plot_lag_response_curve(pair, plot_dir)

    _run_ml_baseline(df, category_columns, reports_dir, logger)
