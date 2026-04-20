from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.ion()


def _pick_first(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _pick_by_keywords(columns: list[str], include: list[str], exclude: list[str] | None = None) -> str | None:
    excluded = exclude or []
    for column in columns:
        lowered = column.lower()
        if all(token in lowered for token in include) and not any(token in lowered for token in excluded):
            return column
    return None


def infer_schema(df: pd.DataFrame) -> dict[str, str | None]:
    columns = df.columns.astype(str).tolist()
    schema: dict[str, str | None] = {
        "time_col": _pick_first(columns, ["week_start_utc", "week_start", "week", "timestamp", "datetime", "date", "time"]),
        "detection_score_col": _pick_first(columns, ["detection_score", "oil_slick_probability_t", "detection_prob", "score"]),
        "vv_mean_col": _pick_first(columns, ["VV_mean", "vv_mean", "sigma0_vv_mean", "vv_db_mean"]),
        "vh_mean_col": _pick_first(columns, ["VH_mean", "vh_mean", "sigma0_vh_mean", "vh_db_mean"]),
        "vv_vh_ratio_col": _pick_first(columns, ["VV_VH_ratio", "vv_vh_ratio", "vv_to_vh_ratio"]),
        "no2_col": _pick_first(columns, ["NO2_mean", "no2_mean_t", "no2_mean"]),
        "vessel_col": _pick_first(columns, ["vessel_density", "vessel_density_t", "density_total"]),
    }

    if schema["detection_score_col"] is None:
        schema["detection_score_col"] = _pick_by_keywords(columns, ["detect", "score"])
    if schema["vv_mean_col"] is None:
        schema["vv_mean_col"] = _pick_by_keywords(columns, ["vv"], ["std", "ratio", "count"])
    if schema["vh_mean_col"] is None:
        schema["vh_mean_col"] = _pick_by_keywords(columns, ["vh"], ["std", "ratio", "count"])
    if schema["vv_vh_ratio_col"] is None:
        schema["vv_vh_ratio_col"] = _pick_by_keywords(columns, ["vv", "vh", "ratio"])
    if schema["no2_col"] is None:
        schema["no2_col"] = _pick_by_keywords(columns, ["no2"], ["std", "trend"])
    if schema["vessel_col"] is None:
        schema["vessel_col"] = _pick_by_keywords(columns, ["vessel", "density"])

    return schema


def _numeric_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


WATER_QUALITY_MEAN_COLS: tuple[str, ...] = (
    "ndwi_mean",
    "ndti_mean",
    "ndci_mean",
    "fai_mean",
    "b11_mean",
)


def build_analysis_frame(df: pd.DataFrame, schema: dict[str, str | None]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["detection_score"] = _numeric_series(df, schema["detection_score_col"])
    out["VV_mean"] = _numeric_series(df, schema["vv_mean_col"])
    out["VH_mean"] = _numeric_series(df, schema["vh_mean_col"])

    if schema["vv_vh_ratio_col"] and schema["vv_vh_ratio_col"] in df.columns:
        out["VV_VH_ratio"] = _numeric_series(df, schema["vv_vh_ratio_col"])
    else:
        out["VV_VH_ratio"] = out["VV_mean"] / out["VH_mean"].replace(0, np.nan)

    out["NO2_mean"] = _numeric_series(df, schema["no2_col"])
    out["vessel_density"] = _numeric_series(df, schema["vessel_col"])

    # Auto-include Sentinel-2 water-quality mean signals when present.
    for wq_col in WATER_QUALITY_MEAN_COLS:
        if wq_col in df.columns:
            out[wq_col] = _numeric_series(df, wq_col)

    if schema["time_col"] and schema["time_col"] in df.columns:
        time_parsed = pd.to_datetime(df[schema["time_col"]], errors="coerce", utc=True)
        out["time_window"] = time_parsed.dt.tz_localize(None).dt.to_period("W").astype(str)
    else:
        out["time_window"] = pd.Series(index=df.index, dtype="object")

    return out


def available_feature_columns(analysis_df: pd.DataFrame) -> list[str]:
    features = [
        "detection_score",
        "VV_mean",
        "VH_mean",
        "VV_VH_ratio",
        "NO2_mean",
        "vessel_density",
        *[c for c in WATER_QUALITY_MEAN_COLS if c in analysis_df.columns],
    ]
    return [column for column in features if column in analysis_df.columns and analysis_df[column].notna().any()]


def select_numeric_features(analysis_df: pd.DataFrame) -> tuple[list[str], dict[str, list[str]], list[str]]:
    candidates = [c for c in analysis_df.columns if c != "time_window"]
    excluded: dict[str, list[str]] = {
        "non_numeric": [],
        "all_nan": [],
        "zero_variance": [],
    }
    selected: list[str] = []
    numeric_found: list[str] = []

    for column in candidates:
        series = pd.to_numeric(analysis_df[column], errors="coerce")
        if not series.notna().any():
            # If conversion produced all NaN, column can be genuinely numeric-but-empty or non-numeric.
            # Classify using original dtype for clearer logging.
            if pd.api.types.is_numeric_dtype(analysis_df[column]):
                excluded["all_nan"].append(column)
            else:
                excluded["non_numeric"].append(column)
            continue
        numeric_found.append(column)
        if float(series.var(skipna=True)) == 0.0:
            excluded["zero_variance"].append(column)
            continue
        selected.append(column)

    return selected, excluded, numeric_found


def compute_correlation_tables(analysis_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = analysis_df[features].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    return pearson, spearman


def _pair_correlation(df: pd.DataFrame, left: str, right: str, method: str) -> float:
    subset = df[[left, right]].dropna()
    if len(subset) < 3:
        return np.nan
    return float(subset[left].corr(subset[right], method=method))


def evaluate_correlations(analysis_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if "detection_score" not in features:
        return pd.DataFrame()

    evaluated_rows: list[dict[str, Any]] = []
    total_rows = len(analysis_df)
    windows = analysis_df["time_window"].dropna().unique().tolist()

    for feature in [col for col in features if col != "detection_score"]:
        overlap = analysis_df[["detection_score", feature]].dropna()
        coverage_pct = (len(overlap) / total_rows * 100.0) if total_rows else 0.0
        pearson_value = _pair_correlation(analysis_df, "detection_score", feature, "pearson")
        spearman_value = _pair_correlation(analysis_df, "detection_score", feature, "spearman")

        per_window: list[float] = []
        sign_values: list[int] = []
        for window in windows:
            window_df = analysis_df.loc[analysis_df["time_window"] == window, ["detection_score", feature]].dropna()
            if len(window_df) < 3:
                continue
            corr = window_df["detection_score"].corr(window_df[feature], method="spearman")
            if pd.notna(corr):
                per_window.append(float(corr))
                if corr > 0:
                    sign_values.append(1)
                elif corr < 0:
                    sign_values.append(-1)

        stability_std = float(np.std(per_window)) if per_window else np.nan
        stable_sign = len(set(sign_values)) <= 1 and len(sign_values) > 0

        informative = coverage_pct > 60.0 and abs(spearman_value) > 0.1 and stable_sign
        weak = coverage_pct > 30.0 and abs(spearman_value) > 0.05
        if informative:
            label = "informative"
        elif weak:
            label = "weak"
        else:
            label = "spurious"

        evaluated_rows.append(
            {
                "pair": f"detection_score_vs_{feature}",
                "coverage_percent": round(coverage_pct, 3),
                "pearson": pearson_value,
                "spearman": spearman_value,
                "abs_spearman": abs(spearman_value) if pd.notna(spearman_value) else np.nan,
                "time_windows_used": len(per_window),
                "time_corr_std": stability_std,
                "stable_sign_across_time": stable_sign,
                "classification": label,
            }
        )

    return pd.DataFrame(evaluated_rows).sort_values(by="abs_spearman", ascending=False, na_position="last")


def _finalize_plot(fig: plt.Figure, save_path: Path, show_plots: bool) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    if show_plots:
        plt.show(block=True)
    plt.close(fig)


def plot_heatmap(corr_df: pd.DataFrame, output_dir: Path, show_plots: bool) -> None:
    if corr_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = corr_df.values
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    ax.set_title("Spearman Correlation Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    _finalize_plot(fig, output_dir / "correlation_heatmap.png", show_plots)


def plot_scatter_pairs(
    analysis_df: pd.DataFrame,
    output_dir: Path,
    show_plots: bool,
    pairs: list[tuple[str, str]] | None = None,
) -> None:
    pairs = pairs or [
        ("detection_score", "vessel_density"),
        ("detection_score", "VV_VH_ratio"),
        ("detection_score", "NO2_mean"),
    ]
    for left, right in pairs:
        subset = analysis_df[[left, right]].dropna()
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(subset[right], subset[left], alpha=0.5, s=12)
        ax.set_xlabel(right)
        ax.set_ylabel(left)
        ax.set_title(f"{left} vs {right}")
        _finalize_plot(fig, output_dir / f"scatter_{left}_vs_{right}.png", show_plots)


def plot_time_correlations(analysis_df: pd.DataFrame, features: list[str], output_dir: Path, show_plots: bool) -> None:
    if "detection_score" not in features:
        return
    windows = sorted(analysis_df["time_window"].dropna().unique().tolist())
    if not windows:
        return

    rows: list[dict[str, Any]] = []
    candidates = [col for col in features if col != "detection_score"]
    for window in windows:
        window_df = analysis_df.loc[analysis_df["time_window"] == window]
        for feature in candidates:
            pair = window_df[["detection_score", feature]].dropna()
            if len(pair) < 3:
                continue
            corr = pair["detection_score"].corr(pair[feature], method="spearman")
            rows.append({"time_window": window, "feature": feature, "spearman": corr})

    chart_df = pd.DataFrame(rows)
    if chart_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for feature in sorted(chart_df["feature"].unique().tolist()):
        subset = chart_df[chart_df["feature"] == feature]
        ax.plot(subset["time_window"], subset["spearman"], marker="o", linewidth=1.3, label=feature)
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax.set_title("Spearman Correlation per Time Window")
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Correlation")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(loc="best")
    _finalize_plot(fig, output_dir / "time_window_correlation.png", show_plots)


def _upper_pairs(corr_df: pd.DataFrame) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(corr_df.index):
        for j, right in enumerate(corr_df.columns):
            if j <= i:
                continue
            value = corr_df.loc[left, right]
            if pd.isna(value):
                continue
            pairs.append((left, right, float(value)))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return pairs


def _lookup_label(evaluation: pd.DataFrame, left: str, right: str) -> str:
    if evaluation.empty:
        return "spurious"
    pair_a = f"{left}_vs_{right}"
    pair_b = f"{right}_vs_{left}"
    match = evaluation[evaluation["pair"].isin([pair_a, pair_b])]
    if match.empty:
        return "spurious"
    return str(match.iloc[0]["classification"])


def classify_pair(
    analysis_df: pd.DataFrame,
    left: str,
    right: str,
    pair_name: str,
) -> pd.DataFrame:
    subset = analysis_df[[left, right]].dropna()
    total_rows = len(analysis_df)
    coverage_pct = (len(subset) / total_rows * 100.0) if total_rows else 0.0
    pearson_value = float(subset[left].corr(subset[right], method="pearson")) if len(subset) >= 3 else np.nan
    spearman_value = float(subset[left].corr(subset[right], method="spearman")) if len(subset) >= 3 else np.nan

    windows = analysis_df["time_window"].dropna().unique().tolist()
    per_window: list[float] = []
    sign_values: list[int] = []
    for window in windows:
        w = analysis_df.loc[analysis_df["time_window"] == window, [left, right]].dropna()
        if len(w) < 3:
            continue
        corr = w[left].corr(w[right], method="spearman")
        if pd.notna(corr):
            per_window.append(float(corr))
            if corr > 0:
                sign_values.append(1)
            elif corr < 0:
                sign_values.append(-1)
    stability_std = float(np.std(per_window)) if per_window else np.nan
    stable_sign = len(set(sign_values)) <= 1 and len(sign_values) > 0
    informative = coverage_pct > 60.0 and abs(spearman_value) > 0.1 and stable_sign
    weak = coverage_pct > 30.0 and abs(spearman_value) > 0.05
    if informative:
        label = "informative"
    elif weak:
        label = "weak"
    else:
        label = "spurious"

    return pd.DataFrame(
        [
            {
                "pair": pair_name,
                "coverage_percent": round(coverage_pct, 3),
                "pearson": pearson_value,
                "spearman": spearman_value,
                "abs_spearman": abs(spearman_value) if pd.notna(spearman_value) else np.nan,
                "time_windows_used": len(per_window),
                "time_corr_std": stability_std,
                "stable_sign_across_time": stable_sign,
                "classification": label,
            }
        ]
    )


def print_summary(pearson: pd.DataFrame, spearman: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    print("[CORRELATION SUMMARY]")
    if evaluation.empty:
        print("[WARN] No evaluation rows produced (missing detection_score or comparable peers).")

    pearson_pairs = _upper_pairs(pearson) if not pearson.empty else []
    spearman_pairs = _upper_pairs(spearman) if not spearman.empty else []

    print("Top Pearson correlations:")
    if not pearson_pairs:
        print("  [WARN] Pearson matrix is empty.")
    else:
        for left, right, value in pearson_pairs[:5]:
            label = _lookup_label(evaluation, left, right)
            print(f"  {left} vs {right} -> {value:.4f} ({label})")

    print("Top Spearman correlations:")
    if not spearman_pairs:
        print("  [WARN] Spearman matrix is empty.")
    else:
        for left, right, value in spearman_pairs[:5]:
            label = _lookup_label(evaluation, left, right)
            print(f"  {left} vs {right} -> {value:.4f} ({label})")

    print("Priority key relationships:")
    for left, right in [
        ("detection_score", "vessel_density"),
        ("detection_score", "VV_VH_ratio"),
        ("detection_score", "NO2_mean"),
    ]:
        if left not in pearson.columns or right not in pearson.columns:
            print(f"  {left} vs {right} -> unavailable (missing column)")
            continue
        p_val = pearson.loc[left, right]
        s_val = spearman.loc[left, right] if right in spearman.columns else np.nan
        label = _lookup_label(evaluation, left, right)
        print(f"  {left} vs {right} -> pearson={p_val:.4f}, spearman={s_val:.4f} ({label})")

    if evaluation.empty:
        return
    for _, row in evaluation.iterrows():
        if row["coverage_percent"] <= 60.0:
            print(f"[WARN] Low coverage for {row['pair']}: {row['coverage_percent']:.1f}%")
        if not bool(row["stable_sign_across_time"]):
            print(f"[WARN] Sign instability across time windows for {row['pair']}.")


def run(
    input_path: Path,
    reports_dir: Path,
    plots_dir: Path,
    show_plots: bool = False,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_parquet(input_path)
    schema = infer_schema(df)
    analysis_df = build_analysis_frame(df, schema)
    selected_features, excluded, numeric_found = select_numeric_features(analysis_df)
    preferred_features = [f for f in available_feature_columns(analysis_df) if f in selected_features]
    for column in selected_features:
        if column not in preferred_features:
            preferred_features.append(column)

    print(f"[CORRELATION][DEBUG] total columns loaded: {len(df.columns)}")
    print(f"[CORRELATION][DEBUG] numeric columns found: {numeric_found}")
    print(f"[CORRELATION][DEBUG] excluded non-numeric: {excluded['non_numeric']}")
    print(f"[CORRELATION][DEBUG] excluded all-NaN: {excluded['all_nan']}")
    print(f"[CORRELATION][DEBUG] excluded zero-variance: {excluded['zero_variance']}")
    print(f"[CORRELATION][DEBUG] final selected features: {preferred_features}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if len(preferred_features) < 2:
        print("[CORRELATION] Skipped: insufficient numeric features")
        print("[CORRELATION] Skipped due to insufficient features")
        pd.DataFrame().to_csv(reports_dir / "pearson_correlation.csv")
        pd.DataFrame().to_csv(reports_dir / "spearman_correlation.csv")
        pd.DataFrame().to_csv(reports_dir / "correlation_evaluation.csv", index=False)
        return

    if len(preferred_features) == 2:
        left, right = preferred_features
        numeric = analysis_df[[left, right]].apply(pd.to_numeric, errors="coerce")
        pearson = numeric.corr(method="pearson")
        spearman = numeric.corr(method="spearman")
        pair_name = f"{left}_vs_{right}"
        evaluation = classify_pair(analysis_df, left, right, pair_name)
        pearson.to_csv(reports_dir / "pearson_correlation.csv")
        spearman.to_csv(reports_dir / "spearman_correlation.csv")
        evaluation.to_csv(reports_dir / "correlation_evaluation.csv", index=False)
        plot_scatter_pairs(analysis_df, plots_dir, show_plots=show_plots, pairs=[(left, right)])
        print_summary(pearson, spearman, evaluation)
        print(f"\n[OK] Pairwise mode used for features: {left}, {right}")
        print(f"[OK] Pearson saved: {reports_dir / 'pearson_correlation.csv'}")
        print(f"[OK] Spearman saved: {reports_dir / 'spearman_correlation.csv'}")
        print(f"[OK] Evaluation saved: {reports_dir / 'correlation_evaluation.csv'}")
        print(f"[OK] Correlation plots saved: {plots_dir}")
        return

    pearson, spearman = compute_correlation_tables(analysis_df, preferred_features)
    evaluation = evaluate_correlations(analysis_df, preferred_features)
    pearson.to_csv(reports_dir / "pearson_correlation.csv")
    spearman.to_csv(reports_dir / "spearman_correlation.csv")
    evaluation.to_csv(reports_dir / "correlation_evaluation.csv", index=False)
    plot_heatmap(spearman, plots_dir, show_plots=show_plots)
    plot_scatter_pairs(analysis_df, plots_dir, show_plots=show_plots)
    plot_time_correlations(analysis_df, preferred_features, plots_dir, show_plots=show_plots)
    print_summary(pearson, spearman, evaluation)
    print(f"\n[OK] Pearson saved: {reports_dir / 'pearson_correlation.csv'}")
    print(f"[OK] Spearman saved: {reports_dir / 'spearman_correlation.csv'}")
    print(f"[OK] Evaluation saved: {reports_dir / 'correlation_evaluation.csv'}")
    print(f"[OK] Correlation plots saved: {plots_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation analysis and visualization for ML-ready dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("processed/features_ml_ready.parquet"),
        help="Input parquet dataset path.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory for correlation CSV outputs.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("outputs/plots/correlations"),
        help="Directory for saved correlation plots.",
    )
    parser.add_argument(
        "--show-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show plots interactively while also saving them (default: false).",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Disable live plot display and save only.",
    )
    args = parser.parse_args()

    show_plots = bool(args.show_plots) and not args.save_only
    run(
        input_path=args.input,
        reports_dir=args.reports_dir,
        plots_dir=args.plots_dir,
        show_plots=show_plots,
    )


if __name__ == "__main__":
    main()
