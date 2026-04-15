from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pick_first(columns: list[str], candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _pick_by_keywords(columns: list[str], include: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    for col in columns:
        low = col.lower()
        if all(token in low for token in include) and not any(token in low for token in exclude):
            return col
    return None


def infer_schema(df: pd.DataFrame) -> dict[str, str | None]:
    cols = df.columns.astype(str).tolist()
    schema: dict[str, str | None] = {
        "grid_col": _pick_first(cols, ["grid_cell_id", "grid_id", "cell_id", "tile_id", "id"]),
        "time_col": _pick_first(cols, ["week_start_utc", "week_start", "week", "timestamp", "datetime", "date", "time"]),
        "lat_col": _pick_first(cols, ["grid_centroid_lat", "centroid_lat", "latitude", "lat", "y"]),
        "lon_col": _pick_first(cols, ["grid_centroid_lon", "centroid_lon", "longitude", "lon", "lng", "x"]),
    }

    schema["vv_mean_col"] = _pick_first(
        cols,
        ["vv_mean", "sigma0_vv_mean", "sentinel1_vv_mean", "vv_db_mean"],
    )
    schema["vh_mean_col"] = _pick_first(
        cols,
        ["vh_mean", "sigma0_vh_mean", "sentinel1_vh_mean", "vh_db_mean"],
    )
    schema["vv_std_col"] = _pick_first(cols, ["vv_std", "sigma0_vv_std", "sentinel1_vv_std", "vv_db_std"])
    schema["vh_std_col"] = _pick_first(cols, ["vh_std", "sigma0_vh_std", "sentinel1_vh_std", "vh_db_std"])

    if schema["vv_mean_col"] is None:
        schema["vv_mean_col"] = _pick_by_keywords(cols, ["vv"], ["std", "count", "ratio"])
    if schema["vh_mean_col"] is None:
        schema["vh_mean_col"] = _pick_by_keywords(cols, ["vh"], ["std", "count", "ratio"])
    if schema["vv_std_col"] is None:
        schema["vv_std_col"] = _pick_by_keywords(cols, ["vv", "std"])
    if schema["vh_std_col"] is None:
        schema["vh_std_col"] = _pick_by_keywords(cols, ["vh", "std"])

    schema["ratio_col"] = _pick_first(cols, ["vv_vh_ratio", "vv_to_vh_ratio", "ratio_vv_vh"])
    if schema["ratio_col"] is None:
        schema["ratio_col"] = _pick_by_keywords(cols, ["vv", "vh", "ratio"])

    schema["detection_score_col"] = _pick_first(
        cols,
        ["detection_score", "oil_slick_probability_t", "oil_probability", "spill_score", "score"],
    )
    if schema["detection_score_col"] is None:
        schema["detection_score_col"] = _pick_by_keywords(cols, ["detect", "score"])

    schema["heuristic_label_col"] = _pick_first(
        cols,
        ["heuristic_label", "oil_label", "label", "is_oil", "positive_detection"],
    )
    if schema["heuristic_label_col"] is None:
        schema["heuristic_label_col"] = _pick_by_keywords(cols, ["label"])

    return schema


def _numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _time_window(df: pd.DataFrame, time_col: str | None) -> pd.Series:
    if time_col is None or time_col not in df.columns:
        return pd.Series(index=df.index, dtype="object")
    t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    return t.dt.tz_localize(None).dt.to_period("W").astype(str)


def compute_stats(df: pd.DataFrame, schema: dict[str, str | None]) -> dict[str, Any]:
    vv = _numeric(df, schema["vv_mean_col"])
    vh = _numeric(df, schema["vh_mean_col"])
    vv_std = _numeric(df, schema["vv_std_col"])
    vh_std = _numeric(df, schema["vh_std_col"])

    if schema["ratio_col"] is not None and schema["ratio_col"] in df.columns:
        ratio = _numeric(df, schema["ratio_col"])
    else:
        denom = vh.replace(0, np.nan)
        ratio = vv / denom

    score = _numeric(df, schema["detection_score_col"])
    label_raw = df[schema["heuristic_label_col"]] if schema["heuristic_label_col"] in df.columns else pd.Series(dtype=float)
    label = pd.to_numeric(label_raw, errors="coerce") if not label_raw.empty else pd.Series(dtype=float)
    if label.empty:
        label = pd.Series(np.nan, index=df.index, dtype=float)

    nan_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    n_rows = int(len(df))

    key_for_grid_missing = [c for c in [schema["vv_mean_col"], schema["vh_mean_col"], schema["detection_score_col"]] if c]
    grid_missing_pct = None
    if schema["grid_col"] and key_for_grid_missing:
        tmp = df[[schema["grid_col"], *key_for_grid_missing]].copy()
        by_grid_missing = (
            tmp.groupby(schema["grid_col"], dropna=False)[key_for_grid_missing]
            .apply(lambda g: g.isna().any(axis=1).any())
        )
        grid_missing_pct = float(by_grid_missing.mean() * 100.0) if len(by_grid_missing) else None

    positive_pct = None
    if label.notna().any():
        positive_pct = float((label > 0).mean() * 100.0)
    elif score.notna().any():
        positive_pct = float((score > 0).mean() * 100.0)

    time_window = _time_window(df, schema["time_col"])
    weekly_stats: dict[str, Any] = {}
    if time_window.notna().any():
        tmp = pd.DataFrame({"time_window": time_window, "vv": vv, "vh": vh}).dropna(subset=["time_window"])
        agg = tmp.groupby("time_window", sort=True, as_index=False).agg(vv_mean=("vv", "mean"), vh_mean=("vh", "mean"))
        weekly_stats = {
            "n_time_windows": int(len(agg)),
            "vv_mean_variation_std": float(agg["vv_mean"].std()) if "vv_mean" in agg else None,
            "vh_mean_variation_std": float(agg["vh_mean"].std()) if "vh_mean" in agg else None,
        }
    else:
        agg = pd.DataFrame(columns=["time_window", "vv_mean", "vh_mean"])

    spatial_stats: dict[str, Any] = {}
    if schema["grid_col"]:
        g = pd.DataFrame({schema["grid_col"]: df[schema["grid_col"]], "vv": vv, "vh": vh})
        var_grid = g.groupby(schema["grid_col"], dropna=False).agg(vv_var=("vv", "var"), vh_var=("vh", "var")).fillna(0.0)
        var_grid["combined_var"] = var_grid[["vv_var", "vh_var"]].mean(axis=1)
        top_high = var_grid.sort_values("combined_var", ascending=False).head(10)
        top_stable = var_grid.sort_values("combined_var", ascending=True).head(10)
        spatial_stats = {
            "n_grids": int(var_grid.shape[0]),
            "top_10_highest_variance_grids": top_high.reset_index().to_dict(orient="records"),
            "top_10_most_stable_grids": top_stable.reset_index().to_dict(orient="records"),
        }

    # Detection distribution for label if available
    label_distribution: dict[str, int] = {}
    if label.notna().any():
        counts = label.fillna(-1).value_counts(dropna=False).sort_index()
        label_distribution = {str(k): int(v) for k, v in counts.items()}

    core = {
        "data_quality": {
            "total_rows": n_rows,
            "nan_percentage_per_feature": {k: float(v) for k, v in nan_pct.to_dict().items()},
            "missing_grid_coverage_percent": grid_missing_pct,
        },
        "feature_statistics": {
            "vv_mean": float(vv.mean()) if vv.notna().any() else None,
            "vv_std": float(vv_std.mean()) if vv_std.notna().any() else None,
            "vh_mean": float(vh.mean()) if vh.notna().any() else None,
            "vh_std": float(vh_std.mean()) if vh_std.notna().any() else None,
            "vv_vh_ratio_mean": float(ratio.mean()) if ratio.notna().any() else None,
            "vv_vh_ratio_std": float(ratio.std()) if ratio.notna().any() else None,
            "vv_min": float(vv.min()) if vv.notna().any() else None,
            "vv_max": float(vv.max()) if vv.notna().any() else None,
            "vh_min": float(vh.min()) if vh.notna().any() else None,
            "vh_max": float(vh.max()) if vh.notna().any() else None,
        },
        "detection_structure": {
            "detection_score_col": schema["detection_score_col"],
            "heuristic_label_col": schema["heuristic_label_col"],
            "detection_score_mean": float(score.mean()) if score.notna().any() else None,
            "detection_score_std": float(score.std()) if score.notna().any() else None,
            "label_distribution": label_distribution,
            "positive_detection_percent": positive_pct,
        },
        "temporal_stability": weekly_stats,
        "spatial_structure": spatial_stats,
        "weekly_aggregation": agg.to_dict(orient="records"),
        "schema_used": schema,
    }
    return core


def _save_hist(series: pd.Series, title: str, xlabel: str, out_path: Path, bins: int = 60) -> None:
    vals = series.dropna()
    if vals.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(vals, bins=bins, edgecolor="black", linewidth=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def create_plots(df: pd.DataFrame, schema: dict[str, str | None], stats: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vv = _numeric(df, schema["vv_mean_col"])
    vh = _numeric(df, schema["vh_mean_col"])

    # VV and VH histogram in one figure.
    if vv.notna().any() or vh.notna().any():
        plt.figure(figsize=(11, 5))
        if vv.notna().any():
            plt.hist(vv.dropna(), bins=60, alpha=0.6, label="VV", edgecolor="black", linewidth=0.4)
        if vh.notna().any():
            plt.hist(vh.dropna(), bins=60, alpha=0.6, label="VH", edgecolor="black", linewidth=0.4)
        plt.title("Distribution of VV and VH")
        plt.xlabel("Backscatter Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "hist_vv_vh.png", dpi=300)
        plt.close()

    ratio_col = schema["ratio_col"]
    ratio = _numeric(df, ratio_col) if ratio_col else vv / vh.replace(0, np.nan)
    _save_hist(ratio, "VV/VH Ratio Distribution", "VV/VH Ratio", out_dir / "hist_vv_vh_ratio.png")

    # Time series of weekly VV_mean and VH_mean.
    weekly = pd.DataFrame(stats.get("weekly_aggregation", []))
    if not weekly.empty:
        plt.figure(figsize=(12, 6))
        if "vv_mean" in weekly:
            plt.plot(weekly["time_window"], weekly["vv_mean"], marker="o", markersize=3, linewidth=1.2, label="VV mean")
        if "vh_mean" in weekly:
            plt.plot(weekly["time_window"], weekly["vh_mean"], marker="o", markersize=3, linewidth=1.2, label="VH mean")
        plt.title("Weekly Mean VV and VH")
        plt.xlabel("Time Window (Weekly)")
        plt.ylabel("Mean Backscatter")
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "timeseries_weekly_vv_vh.png", dpi=300)
        plt.close()

    # Heatmap-style aggregation as spatial scatter if lat/lon exists.
    lat_col, lon_col = schema["lat_col"], schema["lon_col"]
    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        tmp = df[[lat_col, lon_col]].copy()
        tmp["intensity"] = vv
        tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
        tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
        tmp = tmp.dropna(subset=[lat_col, lon_col, "intensity"])
        if not tmp.empty:
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(tmp[lon_col], tmp[lat_col], c=tmp["intensity"], cmap="inferno", s=18, alpha=0.9)
            plt.colorbar(scatter, label="Mean VV intensity")
            plt.title("Grid-level Mean Intensity (VV)")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.tight_layout()
            plt.savefig(out_dir / "heatmap_grid_mean_intensity.png", dpi=300)
            plt.close()

    score = _numeric(df, schema["detection_score_col"])
    _save_hist(score, "Detection Score Distribution", "Detection Score", out_dir / "detection_score_distribution.png")

    nan_pct = pd.Series(stats["data_quality"]["nan_percentage_per_feature"]).sort_values(ascending=False)
    if not nan_pct.empty:
        top = nan_pct.head(30)
        plt.figure(figsize=(12, 6))
        plt.bar(top.index.astype(str), top.values)
        plt.title("NaN Percentage by Feature (Top 30)")
        plt.xlabel("Feature")
        plt.ylabel("NaN %")
        plt.xticks(rotation=75, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "nan_percentage_per_feature.png", dpi=300)
        plt.close()


def build_observations(stats: dict[str, Any]) -> tuple[list[str], list[str], bool]:
    obs: list[str] = []
    anomalies: list[str] = []

    quality = stats["data_quality"]
    feats = stats["feature_statistics"]
    detect = stats["detection_structure"]
    temporal = stats.get("temporal_stability", {})

    total = quality.get("total_rows", 0)
    obs.append(f"Dataset contains {total:,} rows.")

    nan_map = pd.Series(quality.get("nan_percentage_per_feature", {}))
    if not nan_map.empty:
        top_missing = nan_map.sort_values(ascending=False).head(3)
        obs.append(
            "Highest missingness features: "
            + ", ".join(f"{k} ({v:.1f}%)" for k, v in top_missing.items())
            + "."
        )

    if feats.get("vv_vh_ratio_mean") is not None:
        obs.append(
            f"VV/VH ratio mean is {feats['vv_vh_ratio_mean']:.4f} (std {feats['vv_vh_ratio_std']:.4f})."
        )

    if detect.get("positive_detection_percent") is not None:
        obs.append(
            f"Positive detection rate is {detect['positive_detection_percent']:.2f}%."
        )

    if temporal.get("vv_mean_variation_std") is not None and temporal.get("vh_mean_variation_std") is not None:
        obs.append(
            "Temporal variation (std of weekly means): "
            f"VV={temporal['vv_mean_variation_std']:.4f}, VH={temporal['vh_mean_variation_std']:.4f}."
        )

    max_nan = float(nan_map.max()) if not nan_map.empty else 0.0
    grid_missing = quality.get("missing_grid_coverage_percent")
    pos_pct = detect.get("positive_detection_percent")

    if max_nan > 70:
        anomalies.append(f"Very high missingness detected in at least one feature ({max_nan:.1f}%).")
    if grid_missing is not None and grid_missing > 50:
        anomalies.append(f"High grid-level missing coverage ({grid_missing:.1f}% grids affected).")
    if pos_pct is not None and (pos_pct < 0.1 or pos_pct > 99.9):
        anomalies.append(f"Detection class imbalance is extreme ({pos_pct:.3f}% positives).")

    ready = not anomalies and max_nan < 40
    return obs, anomalies, ready


def write_summary_markdown(stats: dict[str, Any], summary_path: Path) -> None:
    observations, anomalies, ready = build_observations(stats)

    q = stats["data_quality"]
    f = stats["feature_statistics"]
    d = stats["detection_structure"]
    t = stats.get("temporal_stability", {})
    s = stats.get("spatial_structure", {})

    lines = [
        "# EDA Summary Report",
        "",
        "## Key Statistics",
        f"- Total rows: {q.get('total_rows', 0):,}",
        f"- Missing grid coverage (%): {q.get('missing_grid_coverage_percent')}",
        f"- VV mean / std: {f.get('vv_mean')} / {f.get('vv_std')}",
        f"- VH mean / std: {f.get('vh_mean')} / {f.get('vh_std')}",
        f"- VV/VH ratio mean / std: {f.get('vv_vh_ratio_mean')} / {f.get('vv_vh_ratio_std')}",
        f"- Detection positive rate (%): {d.get('positive_detection_percent')}",
        f"- Weekly VV mean variation (std): {t.get('vv_mean_variation_std')}",
        f"- Weekly VH mean variation (std): {t.get('vh_mean_variation_std')}",
        f"- Number of grids: {s.get('n_grids')}",
        "",
        "## Key Observations",
    ]
    lines.extend([f"- {o}" for o in observations] or ["- No observations generated."])
    lines.extend(["", "## Anomalies Detected"])
    lines.extend([f"- {a}" for a in anomalies] or ["- None detected by automatic checks."])
    lines.extend(
        [
            "",
            "## Dataset Readiness",
            (
                "- Dataset appears ready for ML based on current EDA checks."
                if ready
                else "- Dataset is not fully ML-ready; review anomalies and missingness before training."
            ),
        ]
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run(input_path: Path, output_dir: Path, summary_path: Path, weekly_breakdown: bool = True) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_parquet(input_path)
    schema = infer_schema(df)
    stats = compute_stats(df, schema)
    if not weekly_breakdown:
        stats["weekly_aggregation"] = []
        stats["temporal_stability"] = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    create_plots(df, schema, stats, output_dir)
    write_summary_markdown(stats, summary_path)

    stats_path = output_dir / "eda_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"[OK] EDA stats saved: {stats_path}")
    print(f"[OK] EDA summary saved: {summary_path}")
    print(f"[OK] Plots saved in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic EDA report for ML-ready feature parquet.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("processed/features_ml_ready.parquet"),
        help="Path to ML-ready parquet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eda"),
        help="Directory for EDA plots and stats.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/eda_summary.md"),
        help="Path to markdown summary report.",
    )
    parser.add_argument(
        "--no-weekly-breakdown",
        action="store_true",
        help="Disable per-week aggregation statistics.",
    )
    args = parser.parse_args()
    run(args.input, args.output_dir, args.summary_path, weekly_breakdown=not args.no_weekly_breakdown)


if __name__ == "__main__":
    main()
