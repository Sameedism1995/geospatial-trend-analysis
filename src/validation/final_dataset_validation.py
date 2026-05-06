"""Final dataset validation for the thesis pipeline.

Runs a battery of sanity checks on the ML-ready feature parquet:
- per-column missing %, with high-missing flag (>30%)
- inf / -inf detection (replaced with NaN for the report)
- distribution stats (mean / std / min / max / coverage)
- temporal consistency (no missing weeks in the panel grid)
- spatial consistency (no duplicate grid-week pairs)
- correlation sanity (flag |corr| > 0.98 between numeric features)
- coverage (per-column % non-null) — NDVI is allowed to be low (coastal-only)

Run:
    python3 src/validation/final_dataset_validation.py \
        --dataset final_run/processed/features_ml_ready.parquet \
        --out-dir final_run/validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("final_dataset_validation")

HIGH_MISSING_THRESHOLD = 30.0  # percent
HIGH_CORR_THRESHOLD = 0.98
COASTAL_ONLY_FEATURES = {"ndvi_mean", "ndvi_median", "ndvi_std", "land_response_index"}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            try:
                out[c] = pd.to_numeric(out[c])
            except (ValueError, TypeError):
                continue
    return out


def missing_value_report(df: pd.DataFrame) -> dict[str, Any]:
    miss = df.isna().mean() * 100.0
    high = miss[miss > HIGH_MISSING_THRESHOLD].to_dict()
    return {
        "missing_percent_per_column": {k: round(float(v), 4) for k, v in miss.to_dict().items()},
        "high_missing_columns": {k: round(float(v), 4) for k, v in high.items()},
        "high_missing_threshold_percent": HIGH_MISSING_THRESHOLD,
    }


def inf_value_report(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {"total_inf_values": 0, "per_column": {}}
    inf_mask = np.isinf(numeric)
    per_col = inf_mask.sum().to_dict()
    per_col_nonzero = {k: int(v) for k, v in per_col.items() if v}
    return {
        "total_inf_values": int(inf_mask.sum().sum()),
        "per_column": per_col_nonzero,
    }


def distribution_report(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    numeric = df.select_dtypes(include=[np.number])
    out: dict[str, dict[str, float]] = {}
    for c in numeric.columns:
        s = numeric[c].replace([np.inf, -np.inf], np.nan)
        if s.dropna().empty:
            out[c] = {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "coverage_percent": 0.0}
            continue
        out[c] = {
            "count": int(s.notna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std()) if s.notna().sum() > 1 else float("nan"),
            "min": float(s.min()),
            "max": float(s.max()),
            "coverage_percent": round(float(s.notna().mean()) * 100.0, 4),
        }
    return out


def temporal_report(df: pd.DataFrame) -> dict[str, Any]:
    if "week_start_utc" not in df.columns:
        return {"status": "skipped", "reason": "no week_start_utc column"}
    weeks = pd.to_datetime(df["week_start_utc"], errors="coerce", utc=True).dropna()
    if weeks.empty:
        return {"status": "failed", "reason": "no parseable weeks"}
    weeks_local = weeks.dt.tz_localize(None)
    unique_weeks = sorted(weeks_local.dt.to_period("W").unique())
    expected = pd.period_range(start=unique_weeks[0], end=unique_weeks[-1], freq="W")
    missing = sorted(set(expected) - set(unique_weeks))
    return {
        "status": "ok" if not missing else "gap",
        "first_week": str(unique_weeks[0]),
        "last_week": str(unique_weeks[-1]),
        "n_unique_weeks": int(len(unique_weeks)),
        "n_expected_weeks": int(len(expected)),
        "n_missing_weeks": int(len(missing)),
        "missing_weeks_sample": [str(p) for p in missing[:10]],
    }


def spatial_report(df: pd.DataFrame) -> dict[str, Any]:
    keys = [c for c in ["grid_cell_id", "week_start_utc"] if c in df.columns]
    if len(keys) < 2:
        return {"status": "skipped", "reason": f"missing keys (have {keys})"}
    dup_mask = df.duplicated(subset=keys, keep=False)
    return {
        "status": "ok" if not dup_mask.any() else "duplicates_found",
        "n_duplicate_rows": int(dup_mask.sum()),
        "n_unique_grid_cells": int(df["grid_cell_id"].nunique(dropna=True)),
        "n_grid_week_pairs": int(df.drop_duplicates(subset=keys).shape[0]),
    }


def correlation_sanity(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        return {"status": "skipped", "reason": "fewer than 2 numeric columns"}
    corr = numeric.corr(method="pearson")
    flagged: list[dict[str, Any]] = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.at[a, b]
            if pd.notna(v) and abs(v) >= HIGH_CORR_THRESHOLD:
                flagged.append({"feature_x": a, "feature_y": b, "pearson": round(float(v), 6)})
    flagged.sort(key=lambda d: -abs(d["pearson"]))
    return {
        "status": "ok" if not flagged else "high_correlation_flag",
        "threshold": HIGH_CORR_THRESHOLD,
        "n_flagged_pairs": len(flagged),
        "flagged_pairs": flagged[:50],
    }


def coverage_report(df: pd.DataFrame) -> dict[str, Any]:
    cov = ((1.0 - df.isna().mean()) * 100.0).round(4)
    cov_dict = {k: float(v) for k, v in cov.to_dict().items()}
    coastal = {k: cov_dict.get(k) for k in COASTAL_ONLY_FEATURES if k in cov_dict}
    return {
        "coverage_percent_per_column": cov_dict,
        "coastal_only_features": coastal,
        "note": "NDVI / land_response_index are expected to have low coverage (coastal/inland cells only).",
    }


def write_summary_md(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Final dataset validation summary")
    lines.append("")
    lines.append(f"- Dataset: `{report.get('dataset_path')}`")
    lines.append(f"- Rows × cols: **{report.get('shape')}**")
    lines.append("")

    miss = report.get("missing_values", {})
    high = miss.get("high_missing_columns", {})
    lines.append(f"## Missing values (>{miss.get('high_missing_threshold_percent')}% threshold)")
    if high:
        for k, v in sorted(high.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{k}`: {v:.2f}% missing")
    else:
        lines.append("- No columns above threshold.")
    lines.append("")

    inf_r = report.get("inf_values", {})
    lines.append("## Infinite values")
    lines.append(f"- Total inf cells: **{inf_r.get('total_inf_values', 0)}**")
    if inf_r.get("per_column"):
        for k, v in inf_r["per_column"].items():
            lines.append(f"  - `{k}`: {v}")
    lines.append("")

    temporal = report.get("temporal", {})
    lines.append("## Temporal consistency")
    lines.append(f"- Status: **{temporal.get('status')}**")
    lines.append(
        f"- Coverage: {temporal.get('first_week')} → {temporal.get('last_week')} "
        f"({temporal.get('n_unique_weeks')} of {temporal.get('n_expected_weeks')} weeks present, "
        f"{temporal.get('n_missing_weeks')} missing)"
    )
    if temporal.get("missing_weeks_sample"):
        lines.append(f"- First missing weeks (sample): {temporal['missing_weeks_sample']}")
    lines.append("")

    spatial = report.get("spatial", {})
    lines.append("## Spatial consistency")
    lines.append(f"- Status: **{spatial.get('status')}**")
    lines.append(
        f"- Duplicate grid-week rows: {spatial.get('n_duplicate_rows')} | "
        f"unique grid cells: {spatial.get('n_unique_grid_cells')} | "
        f"grid-week pairs: {spatial.get('n_grid_week_pairs')}"
    )
    lines.append("")

    corr = report.get("correlation_sanity", {})
    lines.append("## Correlation sanity (|corr| ≥ 0.98)")
    lines.append(f"- Status: **{corr.get('status')}**")
    lines.append(f"- Pairs flagged: {corr.get('n_flagged_pairs')}")
    for p in (corr.get("flagged_pairs") or [])[:20]:
        lines.append(f"  - `{p['feature_x']}` ↔ `{p['feature_y']}` (r={p['pearson']:+.4f})")
    lines.append("")

    cov = report.get("coverage", {})
    lines.append("## Coverage (per column, % non-null)")
    cov_map: dict[str, float] = cov.get("coverage_percent_per_column", {})
    if cov_map:
        sorted_cov = sorted(cov_map.items(), key=lambda kv: kv[1])
        lines.append("| column | coverage % |")
        lines.append("|---|---:|")
        for k, v in sorted_cov:
            lines.append(f"| `{k}` | {v:.2f} |")
    lines.append("")
    lines.append(cov.get("note", ""))
    lines.append("")

    warnings = report.get("warnings", [])
    lines.append("## Warnings")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- None.")
    lines.append("")

    path.write_text("\n".join(lines))


def run_validation(dataset_path: Path, out_dir: Path, logger: logging.Logger | None = None) -> dict[str, Any]:
    log = logger or LOGGER
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Loading dataset: %s", dataset_path)
    df = pd.read_parquet(dataset_path)
    log.info("Dataset shape: %s", df.shape)
    df = _coerce_numeric(df)

    warnings: list[str] = []

    miss = missing_value_report(df)
    if miss["high_missing_columns"]:
        warnings.append(
            f"{len(miss['high_missing_columns'])} columns exceed {HIGH_MISSING_THRESHOLD}% missing."
        )

    inf_r = inf_value_report(df)
    if inf_r["total_inf_values"] > 0:
        warnings.append(f"{inf_r['total_inf_values']} infinite values detected (will be treated as NaN downstream).")

    dist = distribution_report(df)
    temporal = temporal_report(df)
    if temporal.get("status") not in {"ok", "skipped"}:
        warnings.append(f"Temporal consistency: {temporal.get('status')} (missing {temporal.get('n_missing_weeks')} weeks).")
    spatial = spatial_report(df)
    if spatial.get("status") not in {"ok", "skipped"}:
        warnings.append(f"Spatial consistency: {spatial.get('status')} ({spatial.get('n_duplicate_rows')} duplicate grid-week rows).")
    corr = correlation_sanity(df)
    if corr.get("status") == "high_correlation_flag":
        warnings.append(f"{corr.get('n_flagged_pairs')} correlation pairs ≥{HIGH_CORR_THRESHOLD} (possible leakage / redundancy).")
    cov = coverage_report(df)

    report: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "missing_values": miss,
        "inf_values": inf_r,
        "distributions": dist,
        "temporal": temporal,
        "spatial": spatial,
        "correlation_sanity": corr,
        "coverage": cov,
        "warnings": warnings,
    }

    json_path = out_dir / "final_validation_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote %s", json_path)

    md_path = out_dir / "final_validation_summary.md"
    write_summary_md(md_path, report)
    log.info("Wrote %s", md_path)

    log.info("Validation warnings: %d", len(warnings))
    for w in warnings:
        log.info("  WARN: %s", w)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to ML-ready parquet")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for validation files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.dataset.exists():
        LOGGER.error("Dataset not found: %s", args.dataset)
        return 2
    run_validation(args.dataset, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
