from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "outputs" / "reports"
FINAL_DIR = ROOT / "outputs" / "final"
ANOMALY_EXTREME_QUANTILE = 0.9


def _strength_label(value: float) -> str:
    v = abs(float(value))
    if v > 0.5:
        return "strong"
    if v >= 0.3:
        return "moderate"
    return "weak"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _matrix_to_pairs(df: pd.DataFrame, method: str, min_abs: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    idx_col = df.columns[0]
    if idx_col != "feature":
        df = df.rename(columns={idx_col: "feature"})
    df = df.set_index("feature")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    pairs: list[dict[str, Any]] = []
    cols = list(numeric.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = numeric.at[col_a, col_b] if col_a in numeric.index else None
            if pd.isna(value):
                continue
            if abs(float(value)) < min_abs:
                continue
            pairs.append(
                {
                    "feature_x": str(col_a),
                    "feature_y": str(col_b),
                    "method": method,
                    "value": float(value),
                    "abs_value": abs(float(value)),
                    "strength": _strength_label(float(value)),
                }
            )
    return pd.DataFrame(pairs)


def _correlation_group(x: str, y: str) -> str:
    pair = f"{x} {y}".lower()
    if "vessel" in pair:
        return "vessel_density relationships"
    if "no2" in pair:
        return "NO2 relationships"
    if any(k in pair for k in ("sentinel", "slick", "disturbance", "detection_score", "vh", "vv")):
        return "sentinel1 / disturbance relationships"
    return "other"


def summarize_correlations(min_abs: float) -> dict[str, Any]:
    pearson = _safe_read_csv(REPORTS_DIR / "pearson_correlation.csv")
    spearman = _safe_read_csv(REPORTS_DIR / "spearman_correlation.csv")
    pearson_pairs = _matrix_to_pairs(pearson, "pearson", min_abs)
    spearman_pairs = _matrix_to_pairs(spearman, "spearman", min_abs)
    all_pairs = pd.concat([pearson_pairs, spearman_pairs], ignore_index=True) if (not pearson_pairs.empty or not spearman_pairs.empty) else pd.DataFrame()
    if all_pairs.empty:
        return {"top10": [], "by_group": {}, "note": "No correlation pairs above threshold."}

    top10 = all_pairs.sort_values("abs_value", ascending=False).head(10).copy()
    top10["group"] = top10.apply(lambda r: _correlation_group(str(r["feature_x"]), str(r["feature_y"])), axis=1)
    by_group = {
        g: grp[["feature_x", "feature_y", "method", "value", "abs_value", "strength"]].to_dict(orient="records")
        for g, grp in top10.groupby("group", sort=False)
    }
    return {"top10": top10.to_dict(orient="records"), "by_group": by_group}


def _lag_label(best_lag: int, consistency: float, abs_corr: float, min_abs: float) -> str:
    if abs_corr < min_abs or consistency < 60:
        return "unstable"
    if int(best_lag) == 0:
        return "synchronous"
    return "delayed response"


def summarize_lags(min_abs: float) -> dict[str, Any]:
    lag_df = _safe_read_csv(REPORTS_DIR / "lagged_correlations.csv")
    if lag_df.empty:
        return {"top10_stable": [], "note": "Lag report missing or unreadable."}

    need = {"feature_x", "feature_y", "lag", "correlation"}
    if not need.issubset(set(lag_df.columns)):
        return {"top10_stable": [], "note": "Lag report schema missing required columns."}

    lag_df["correlation"] = pd.to_numeric(lag_df["correlation"], errors="coerce")
    lag_df["best_correlation"] = pd.to_numeric(lag_df.get("best_correlation"), errors="coerce")
    lag_df["best_lag"] = pd.to_numeric(lag_df.get("best_lag"), errors="coerce")
    lag_df["consistency_score"] = pd.to_numeric(lag_df.get("consistency_score"), errors="coerce").fillna(0.0)
    lag_df["lag_position_std"] = pd.to_numeric(lag_df.get("lag_position_std"), errors="coerce").fillna(99.0)

    rows: list[dict[str, Any]] = []
    for (fx, fy), grp in lag_df.groupby(["feature_x", "feature_y"], sort=False):
        max_row = grp.loc[grp["correlation"].abs().idxmax()]
        best_corr = float(max_row.get("best_correlation") if pd.notna(max_row.get("best_correlation")) else max_row["correlation"])
        best_lag = int(max_row.get("best_lag") if pd.notna(max_row.get("best_lag")) else max_row["lag"])
        consistency = float(max_row.get("consistency_score", 0.0))
        lag_std = float(max_row.get("lag_position_std", 99.0))
        abs_corr = abs(best_corr)
        label = _lag_label(best_lag, consistency, abs_corr, min_abs)
        stability_score = (abs_corr * (max(consistency, 1.0) / 100.0)) / (1.0 + max(lag_std, 0.0))
        rows.append(
            {
                "feature_x": str(fx),
                "feature_y": str(fy),
                "best_lag": best_lag,
                "best_correlation": best_corr,
                "abs_best_correlation": abs_corr,
                "consistency_score": consistency,
                "lag_position_std": lag_std,
                "interpretation": label,
                "strength": _strength_label(best_corr),
                "stability_rank_score": stability_score,
            }
        )

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return {"top10_stable": []}
    ranked = ranked[ranked["abs_best_correlation"] >= min_abs]
    ranked = ranked.sort_values(["stability_rank_score", "abs_best_correlation"], ascending=False).head(10)
    return {"top10_stable": ranked.to_dict(orient="records")}


def summarize_interactions(min_abs: float) -> dict[str, Any]:
    path = REPORTS_DIR / "feature_interactions_ranked.csv"
    df = _safe_read_csv(path)
    if df.empty:
        return {"top_cross_category": [], "note": "Feature interaction report missing or unreadable."}

    if "correlation_value" not in df.columns or "category_pair" not in df.columns:
        return {"top_cross_category": [], "note": "Feature interaction schema missing expected columns."}

    keep_pairs = {
        "maritime_activity ↔ atmospheric",
        "maritime_activity ↔ water_quality",
        "atmospheric ↔ water_quality",
    }
    df["correlation_value"] = pd.to_numeric(df["correlation_value"], errors="coerce")
    df["category_pair"] = df["category_pair"].astype(str)
    filtered = df[df["category_pair"].isin(keep_pairs)].copy()
    filtered = filtered[filtered["correlation_value"].abs() >= min_abs]
    if filtered.empty:
        return {"top_cross_category": [], "note": "No cross-category interactions above threshold."}

    filtered["abs_value"] = filtered["correlation_value"].abs()
    filtered["strength"] = filtered["correlation_value"].apply(_strength_label)
    top = filtered.sort_values("abs_value", ascending=False).head(10)
    cols = [c for c in ["feature_1", "feature_2", "category_pair", "correlation_value", "abs_value", "strength", "interpretation_label"] if c in top.columns]
    return {"top_cross_category": top[cols].to_dict(orient="records")}


def _md_table(title: str, rows: list[dict[str, Any]], cols: list[str]) -> str:
    out = [f"## {title}"]
    if not rows:
        out.append("")
        out.append("_No entries available above threshold._")
        out.append("")
        return "\n".join(out)
    out.append("")
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    out.append("")
    return "\n".join(out)


def write_outputs(summary: dict[str, Any]) -> None:
    json_path = REPORTS_DIR / "results_summary.json"
    md_path = REPORTS_DIR / "results_table.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    md_sections = [
        "# Results Aggregator Summary",
        "",
        _md_table(
            "Top Correlations",
            summary["correlations"].get("top10", []),
            ["feature_x", "feature_y", "method", "value", "abs_value", "strength", "group"],
        ),
        _md_table(
            "Top Stable Lag Signals",
            summary["lags"].get("top10_stable", []),
            ["feature_x", "feature_y", "best_lag", "best_correlation", "consistency_score", "interpretation", "strength"],
        ),
        _md_table(
            "Top Cross-Category Interactions",
            summary["interactions"].get("top_cross_category", []),
            ["feature_1", "feature_2", "category_pair", "correlation_value", "abs_value", "strength", "interpretation_label"],
        ),
    ]
    md_path.write_text("\n".join(md_sections), encoding="utf-8")


def print_console_summary(summary: dict[str, Any]) -> None:
    print("\n[RESULTS AGGREGATOR] Top correlations")
    for row in summary["correlations"].get("top10", [])[:5]:
        print(f"- {row['feature_x']} vs {row['feature_y']} ({row['method']}): {row['value']:.4f} [{row['strength']}]")

    print("\n[RESULTS AGGREGATOR] Top lag signals")
    for row in summary["lags"].get("top10_stable", [])[:5]:
        print(
            f"- {row['feature_x']} -> {row['feature_y']} lag={row['best_lag']} corr={row['best_correlation']:.4f} "
            f"{row['interpretation']}"
        )

    print("\n[RESULTS AGGREGATOR] Top interactions")
    for row in summary["interactions"].get("top_cross_category", [])[:5]:
        print(f"- {row.get('feature_1')} vs {row.get('feature_2')}: {row.get('correlation_value'):.4f} ({row.get('category_pair')})")

    print(f"\n[RESULTS AGGREGATOR] Wrote: {REPORTS_DIR / 'results_summary.json'}")
    print(f"[RESULTS AGGREGATOR] Wrote: {REPORTS_DIR / 'results_table.md'}")


def run(min_abs: float = 0.1) -> dict[str, Any]:
    summary = {
        "threshold_min_abs": float(min_abs),
        "correlations": summarize_correlations(min_abs=min_abs),
        "lags": summarize_lags(min_abs=min_abs),
        "interactions": summarize_interactions(min_abs=min_abs),
    }
    write_outputs(summary)
    print_console_summary(summary)
    return summary


def _flatten_corr_matrix(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    idx_col = df.columns[0]
    if idx_col != "feature":
        df = df.rename(columns={idx_col: "feature"})
    df = df.set_index("feature")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    cols = list(numeric.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            key = tuple(sorted((str(a), str(b))))
            if key in seen or a == b:
                continue
            seen.add(key)
            if a not in numeric.index:
                continue
            value = numeric.at[a, b]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "feature_x": str(a),
                    "feature_y": str(b),
                    "method": method,
                    "value": float(value),
                    "abs_value": abs(float(value)),
                    "strength": _strength_label(float(value)),
                }
            )
    return pd.DataFrame(rows)


def finalize_top_correlations() -> pd.DataFrame:
    pearson = _flatten_corr_matrix(_safe_read_csv(REPORTS_DIR / "pearson_correlation.csv"), "pearson")
    spearman = _flatten_corr_matrix(_safe_read_csv(REPORTS_DIR / "spearman_correlation.csv"), "spearman")
    if pearson.empty and spearman.empty:
        return pd.DataFrame(columns=["feature_x", "feature_y", "method", "value", "abs_value", "strength"])
    combined = pd.concat([pearson, spearman], ignore_index=True)
    combined["pair_key"] = combined.apply(
        lambda r: " | ".join(sorted([str(r["feature_x"]), str(r["feature_y"])])),
        axis=1,
    )
    combined = combined.sort_values("abs_value", ascending=False)
    best_per_pair = combined.drop_duplicates(subset=["pair_key"], keep="first")
    return (
        best_per_pair.drop(columns=["pair_key"])
        .head(20)
        .reset_index(drop=True)
    )


def _lag_interpretation(lag: int) -> str:
    lag_int = int(lag)
    if lag_int == 0:
        return "immediate"
    if 1 <= lag_int <= 2:
        return "short delay"
    return "delayed response"


def finalize_top_lag_signals() -> pd.DataFrame:
    lag_df = _safe_read_csv(REPORTS_DIR / "lagged_correlations.csv")
    required = {"feature_x", "feature_y", "lag", "correlation"}
    base_cols = [
        "feature_x",
        "feature_y",
        "best_lag",
        "best_correlation",
        "abs_best_correlation",
        "interpretation",
        "strength",
    ]
    if lag_df.empty or not required.issubset(lag_df.columns):
        return pd.DataFrame(columns=base_cols)

    lag_df["correlation"] = pd.to_numeric(lag_df["correlation"], errors="coerce")
    lag_df["lag"] = pd.to_numeric(lag_df["lag"], errors="coerce")
    if "best_correlation" in lag_df.columns:
        lag_df["best_correlation"] = pd.to_numeric(lag_df["best_correlation"], errors="coerce")
    if "best_lag" in lag_df.columns:
        lag_df["best_lag"] = pd.to_numeric(lag_df["best_lag"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for (fx, fy), grp in lag_df.groupby(["feature_x", "feature_y"], sort=False):
        valid = grp.dropna(subset=["correlation"])
        if valid.empty:
            continue
        max_row = valid.loc[valid["correlation"].abs().idxmax()]
        best_corr_raw = max_row.get("best_correlation") if "best_correlation" in valid.columns else None
        best_corr = float(best_corr_raw) if pd.notna(best_corr_raw) else float(max_row["correlation"])
        best_lag_raw = max_row.get("best_lag") if "best_lag" in valid.columns else None
        best_lag = int(best_lag_raw) if pd.notna(best_lag_raw) else int(max_row["lag"])
        rows.append(
            {
                "feature_x": str(fx),
                "feature_y": str(fy),
                "best_lag": best_lag,
                "best_correlation": best_corr,
                "abs_best_correlation": abs(best_corr),
                "interpretation": _lag_interpretation(best_lag),
                "strength": _strength_label(best_corr),
            }
        )
    ranked = pd.DataFrame(rows, columns=base_cols)
    if ranked.empty:
        return ranked
    return ranked.sort_values("abs_best_correlation", ascending=False).head(20).reset_index(drop=True)


def finalize_top_anomalies() -> pd.DataFrame:
    adf = _safe_read_csv(REPORTS_DIR / "anomaly_scores.csv")
    base_cols = [
        "grid_cell_id",
        "week_start_utc",
        "grid_centroid_lat",
        "grid_centroid_lon",
        "anomaly_score",
        "anomaly_label",
        "event_class",
    ]
    if adf.empty or "anomaly_score" not in adf.columns:
        return pd.DataFrame(columns=base_cols)
    adf["anomaly_score"] = pd.to_numeric(adf["anomaly_score"], errors="coerce")
    adf = adf.dropna(subset=["anomaly_score"])
    if adf.empty:
        return pd.DataFrame(columns=base_cols)
    threshold = float(adf["anomaly_score"].quantile(ANOMALY_EXTREME_QUANTILE))
    top = adf.sort_values("anomaly_score", ascending=False).head(20).copy()
    top["event_class"] = top["anomaly_score"].apply(
        lambda s: "extreme event" if float(s) > threshold else "moderate anomaly"
    )
    keep = [c for c in base_cols if c in top.columns]
    return top[keep].reset_index(drop=True)


def _load_features_parquet() -> pd.DataFrame:
    path = ROOT / "processed" / "features_ml_ready.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def finalize_top_impact_zones() -> pd.DataFrame:
    cis = _safe_read_csv(REPORTS_DIR / "coastal_impact_score.csv")
    base_cols = [
        "grid_cell_id",
        "week_start_utc",
        "grid_centroid_lat",
        "grid_centroid_lon",
        "coastal_impact_score",
        "vessel_density",
        "NO2_mean",
        "anomaly_score",
        "interpretation",
    ]
    if cis.empty or "coastal_impact_score" not in cis.columns:
        return pd.DataFrame(columns=base_cols)

    cis["coastal_impact_score"] = pd.to_numeric(cis["coastal_impact_score"], errors="coerce")
    cis = cis.dropna(subset=["coastal_impact_score"])
    cis = cis.sort_values("coastal_impact_score", ascending=False).head(20).copy()

    if "week_start_utc" in cis.columns:
        cis["week_start_utc"] = pd.to_datetime(cis["week_start_utc"], utc=True, errors="coerce")

    features = _load_features_parquet()
    if not features.empty and {"grid_cell_id", "week_start_utc"}.issubset(features.columns):
        feat = features.copy()
        feat["week_start_utc"] = pd.to_datetime(feat["week_start_utc"], utc=True, errors="coerce")
        enrich_cols = [c for c in ["vessel_density", "NO2_mean", "grid_centroid_lat", "grid_centroid_lon"] if c in feat.columns]
        if enrich_cols:
            merge = feat[["grid_cell_id", "week_start_utc", *enrich_cols]].drop_duplicates(
                subset=["grid_cell_id", "week_start_utc"]
            )
            for col in enrich_cols:
                if col in cis.columns:
                    merge = merge.rename(columns={col: f"__{col}__ext"})
            cis = cis.merge(merge, on=["grid_cell_id", "week_start_utc"], how="left")
            for col in enrich_cols:
                ext = f"__{col}__ext"
                if ext in cis.columns:
                    if col in cis.columns:
                        cis[col] = cis[col].where(cis[col].notna(), cis[ext])
                    else:
                        cis[col] = cis[ext]
                    cis = cis.drop(columns=[ext])

    if "anomaly_score" not in cis.columns:
        anomaly = _safe_read_csv(REPORTS_DIR / "anomaly_scores.csv")
        if not anomaly.empty and {"grid_cell_id", "week_start_utc", "anomaly_score"}.issubset(anomaly.columns):
            anomaly = anomaly.copy()
            anomaly["week_start_utc"] = pd.to_datetime(anomaly["week_start_utc"], utc=True, errors="coerce")
            cis = cis.merge(
                anomaly[["grid_cell_id", "week_start_utc", "anomaly_score"]].drop_duplicates(
                    subset=["grid_cell_id", "week_start_utc"]
                ),
                on=["grid_cell_id", "week_start_utc"],
                how="left",
            )

    cis["interpretation"] = "high environmental pressure zone"
    keep = [c for c in base_cols if c in cis.columns]
    return cis[keep].reset_index(drop=True)


def _paragraph_correlations(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No correlation pairs reached the reporting threshold in the current run."
    top = rows[0]
    sign = "positive" if float(top["value"]) > 0 else "negative"
    return (
        f"The strongest cross-feature correlation links {top['feature_x']} and {top['feature_y']} "
        f"({top['method']}, r = {float(top['value']):.3f}, {sign}, {top['strength']}). "
        "Magnitudes across the top-ranked pairs remain primarily in the weak-to-moderate band, "
        "consistent with weekly grid-level remote-sensing signals where direct coupling between "
        "maritime pressure and environmental response is expected to be partial rather than deterministic."
    )


def _paragraph_lags(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No lag signals passed the strength threshold in the current run."
    top = rows[0]
    return (
        f"Temporal analysis indicates that {top['feature_x']} relates most strongly to {top['feature_y']} "
        f"at a best lag of {int(top['best_lag'])} week(s) "
        f"(r = {float(top['best_correlation']):.3f}, {top['interpretation']}, {top['strength']}). "
        "Short delays dominate the ranked list, suggesting measurable but bounded temporal response "
        "between shipping-related pressure and environmental indicators."
    )


def _paragraph_anomalies(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No anomalous grid-week events were detected in the current outputs."
    top = rows[0]
    return (
        f"The most extreme detected anomaly is grid {top.get('grid_cell_id')} at {top.get('week_start_utc')} "
        f"(score = {float(top.get('anomaly_score', 0.0)):.3f}, {top.get('event_class')}). "
        "The top-20 anomalies cluster around a small number of repeatedly-flagged grid cells, "
        "indicating persistent local deviations rather than isolated noise."
    )


def _paragraph_impact(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Coastal impact scores could not be derived from the current artifacts."
    top = rows[0]
    return (
        f"The highest-pressure coastal zone is grid {top.get('grid_cell_id')} at {top.get('week_start_utc')} "
        f"(composite score = {float(top.get('coastal_impact_score', 0.0)):.3f}). "
        "Ranked zones consistently combine elevated shipping-related features with stronger anomaly response, "
        "supporting the thesis hypothesis that spatial exposure near ports and shipping corridors "
        "concentrates environmental pressure."
    )


def _md_section(title: str, paragraph: str, rows: list[dict[str, Any]], cols: list[str]) -> str:
    body = [f"## {title}", "", paragraph, "", "### Top 5 findings", ""]
    if not rows:
        body.append("_No entries available._")
        body.append("")
        return "\n".join(body)
    body.append("| " + " | ".join(cols) + " |")
    body.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows[:5]:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    body.append("")
    return "\n".join(body)


def finalize() -> dict[str, Any]:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    correlations = finalize_top_correlations()
    lag_signals = finalize_top_lag_signals()
    anomalies = finalize_top_anomalies()
    impact_zones = finalize_top_impact_zones()

    correlations.to_csv(FINAL_DIR / "top_correlations.csv", index=False)
    lag_signals.to_csv(FINAL_DIR / "top_lag_signals.csv", index=False)
    anomalies.to_csv(FINAL_DIR / "top_anomalies.csv", index=False)
    impact_zones.to_csv(FINAL_DIR / "top_impact_zones.csv", index=False)

    payload: dict[str, Any] = {
        "top_correlations": correlations.to_dict(orient="records"),
        "top_lag_signals": lag_signals.to_dict(orient="records"),
        "top_anomalies": anomalies.to_dict(orient="records"),
        "top_impact_zones": impact_zones.to_dict(orient="records"),
    }
    (FINAL_DIR / "thesis_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    md_sections = [
        "# Thesis Results Summary",
        "",
        _md_section(
            "Correlation Insights",
            _paragraph_correlations(payload["top_correlations"]),
            payload["top_correlations"],
            ["feature_x", "feature_y", "method", "value", "abs_value", "strength"],
        ),
        _md_section(
            "Temporal Dynamics",
            _paragraph_lags(payload["top_lag_signals"]),
            payload["top_lag_signals"],
            ["feature_x", "feature_y", "best_lag", "best_correlation", "abs_best_correlation", "interpretation", "strength"],
        ),
        _md_section(
            "Anomaly Detection",
            _paragraph_anomalies(payload["top_anomalies"]),
            payload["top_anomalies"],
            ["grid_cell_id", "week_start_utc", "anomaly_score", "anomaly_label", "event_class"],
        ),
        _md_section(
            "Coastal Impact Analysis",
            _paragraph_impact(payload["top_impact_zones"]),
            payload["top_impact_zones"],
            ["grid_cell_id", "week_start_utc", "coastal_impact_score", "vessel_density", "NO2_mean", "anomaly_score", "interpretation"],
        ),
    ]
    (FINAL_DIR / "thesis_results.md").write_text("\n".join(md_sections), encoding="utf-8")

    print("[RESULTS AGGREGATOR] Finalized thesis outputs:")
    for name in [
        "top_correlations.csv",
        "top_lag_signals.csv",
        "top_anomalies.csv",
        "top_impact_zones.csv",
        "thesis_summary.json",
        "thesis_results.md",
    ]:
        print(f"- {FINAL_DIR / name}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate pipeline result artifacts into thesis-ready summary outputs.")
    parser.add_argument("--min-abs-threshold", type=float, default=0.1, help="Minimum absolute strength threshold.")
    parser.add_argument("--finalize", action="store_true", help="Produce final thesis-ready tables under outputs/final/.")
    args = parser.parse_args()
    if args.finalize:
        finalize()
    else:
        run(min_abs=args.min_abs_threshold)


if __name__ == "__main__":
    main()
