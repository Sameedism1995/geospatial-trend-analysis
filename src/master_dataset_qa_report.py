"""
Generate data quality and validation metrics for the master dataset (no modeling).

Writes JSON to data/validation/qa_report.json by default.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Canonical spectral features (mean aggregates; thesis-facing names)
SPECTRAL_FEATURES: dict[str, str] = {
    "NDVI": "sentinel_ndvi_mean",
    "NDWI": "sentinel_ndwi_mean",
    "EVI": "sentinel_evi_mean",
    "NDTI": "sentinel_ndti_mean",
}

# Inclusive canonical ranges for sanity checks (literature-style bounds).
# EVI is allowed a slightly wider band because the implemented index can exceed |1|
# at bright/dark extremes (still flagged if absurd).
CANONICAL_RANGES: dict[str, tuple[float, float]] = {
    "NDVI": (-1.0, 1.0),
    "NDWI": (-1.0, 1.0),
    "EVI": (-2.0, 2.0),
    "NDTI": (-1.0, 1.0),
}

OBS_COUNT_COL = "sentinel_observation_count"
WEEK_COL = "week_start_utc"
GRID_COL = "grid_cell_id"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        return obj
    if isinstance(obj, (pd.Timestamp, datetime)):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _pct_non_null(series: pd.Series) -> float:
    n = len(series)
    if n == 0:
        return 0.0
    return float(series.notna().sum() / n * 100.0)


def _iqr_outlier_mask(values: np.ndarray) -> tuple[int, float, float]:
    """Tukey fences; returns (n_outliers, lower, upper)."""
    v = values[np.isfinite(values)]
    if v.size < 4:
        return 0, float("nan"), float("nan")
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    mask = (v < lo) | (v > hi)
    return int(mask.sum()), float(lo), float(hi)


def run_qa(
    df: pd.DataFrame,
    *,
    expected_weeks: int | None = None,
    low_density_threshold: float = 0.35,
    chronic_missing_threshold: float = 0.05,
) -> dict[str, Any]:
    df = df.copy()
    df[WEEK_COL] = pd.to_datetime(df[WEEK_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[WEEK_COL, GRID_COL])

    n_weeks = int(df[WEEK_COL].nunique())
    n_grids = int(df[GRID_COL].nunique())
    n_rows = len(df)

    weeks_sorted = sorted(df[WEEK_COL].dropna().unique())
    grids_sorted = sorted(df[GRID_COL].astype(str).unique())

    if expected_weeks is None:
        expected_weeks = n_weeks

    # --- Coverage: % non-null per feature, per week and per grid ---
    coverage: dict[str, Any] = {"spectral_features": {}, "overall": {}}

    for label, col in SPECTRAL_FEATURES.items():
        if col not in df.columns:
            coverage["spectral_features"][label] = {"error": f"missing column {col}"}
            continue
        s = df[col]
        coverage["overall"][label] = {
            "column": col,
            "n_non_null": int(s.notna().sum()),
            "n_total": n_rows,
            "pct_non_null": round(_pct_non_null(s), 4),
        }

        by_week: list[dict[str, Any]] = []
        for w in weeks_sorted:
            sub = df.loc[df[WEEK_COL] == w, col]
            n_g = len(sub)
            nn = int(sub.notna().sum())
            by_week.append(
                {
                    "week_start_utc": w.isoformat() if hasattr(w, "isoformat") else str(w),
                    "n_cells": n_g,
                    "n_non_null": nn,
                    "pct_non_null": round(_pct_non_null(sub), 4),
                }
            )

        by_grid: list[dict[str, Any]] = []
        for g in grids_sorted:
            sub = df.loc[df[GRID_COL].astype(str) == g, col]
            n_w = len(sub)
            nn = int(sub.notna().sum())
            by_grid.append(
                {
                    "grid_cell_id": g,
                    "n_weeks": n_w,
                    "n_non_null": nn,
                    "pct_non_null": round(_pct_non_null(sub), 4),
                }
            )

        coverage["spectral_features"][label] = {
            "column": col,
            "by_week": by_week,
            "by_grid": by_grid,
        }

    # --- Temporal: full week coverage per grid; weeks with zero observations ---
    rows_per_grid = df.groupby(GRID_COL, sort=False).size()
    rows_per_week = df.groupby(WEEK_COL, sort=False).size()

    expected_pairs = expected_weeks * n_grids
    temporal = {
        "expected_weeks": expected_weeks,
        "observed_weeks": n_weeks,
        "n_grids": n_grids,
        "n_rows": n_rows,
        "expected_rows_if_balanced": int(expected_weeks * n_grids),
        "balanced_panel_match": n_rows == expected_weeks * n_grids and n_weeks == expected_weeks,
        "grids_with_full_week_count": int((rows_per_grid == expected_weeks).sum()),
        "grids_with_incomplete_weeks": int((rows_per_grid != expected_weeks).sum()),
        "incomplete_grids": [
            {"grid_cell_id": str(g), "n_rows": int(c)}
            for g, c in rows_per_grid[rows_per_grid != expected_weeks].items()
        ],
        "weeks_with_row_count_mismatch": int((rows_per_week != n_grids).sum()),
        "week_row_counts": [
            {"week_start_utc": w.isoformat(), "n_rows": int(rows_per_week.loc[w])}
            for w in sorted(rows_per_week.index)
        ],
    }

    if OBS_COUNT_COL in df.columns:
        obs = pd.to_numeric(df[OBS_COUNT_COL], errors="coerce").fillna(0).astype(int)
        week_zero = (
            df.assign(_obs=obs)
            .groupby(WEEK_COL, sort=False)["_obs"]
            .sum()
            .reset_index()
        )
        zero_weeks = week_zero[week_zero["_obs"] == 0]
        temporal["weeks_with_zero_sentinel_observations_total"] = [
            {"week_start_utc": r[WEEK_COL].isoformat(), "total_observations": 0}
            for _, r in zero_weeks.iterrows()
        ]
        temporal["n_weeks_with_zero_sentinel_observations"] = len(zero_weeks)
    else:
        temporal["weeks_with_zero_sentinel_observations_total"] = []
        temporal["n_weeks_with_zero_sentinel_observations"] = None
        temporal["note"] = f"missing {OBS_COUNT_COL}"

    # --- Spatial: low density grids; chronically missing spectral ---
    spatial: dict[str, Any] = {
        "low_density_threshold_non_null_frac": low_density_threshold,
        "chronic_missing_threshold_non_null_frac": chronic_missing_threshold,
    }

    grid_ids = grids_sorted
    low_density: list[dict[str, Any]] = []
    chronic: list[dict[str, Any]] = []

    for g in grid_ids:
        sub = df.loc[df[GRID_COL].astype(str) == g]
        fracs: dict[str, float] = {}
        for label, col in SPECTRAL_FEATURES.items():
            if col not in sub.columns:
                fracs[label] = 0.0
            else:
                fracs[label] = float(sub[col].notna().sum() / max(len(sub), 1))
        min_frac = min(fracs.values()) if fracs else 0.0
        mean_frac = float(np.mean(list(fracs.values()))) if fracs else 0.0
        if min_frac < low_density_threshold:
            low_density.append(
                {
                    "grid_cell_id": g,
                    "min_feature_non_null_frac": round(min_frac, 4),
                    "mean_feature_non_null_frac": round(mean_frac, 4),
                    "per_feature_non_null_frac": {k: round(v, 4) for k, v in fracs.items()},
                }
            )
        if all(fracs.get(lbl, 0.0) < chronic_missing_threshold for lbl in SPECTRAL_FEATURES):
            chronic.append(
                {
                    "grid_cell_id": g,
                    "per_feature_non_null_frac": {k: round(v, 4) for k, v in fracs.items()},
                }
            )

    spatial["grids_low_spectral_density"] = sorted(
        low_density, key=lambda x: x["min_feature_non_null_frac"]
    )
    spatial["n_grids_low_spectral_density"] = len(low_density)
    spatial["grids_chronically_missing_spectral_features"] = chronic
    spatial["n_grids_chronically_missing_spectral_features"] = len(chronic)

    # --- Sanity: ranges + IQR outliers ---
    sanity: dict[str, Any] = {"canonical_ranges": CANONICAL_RANGES, "per_feature": {}}

    for label, col in SPECTRAL_FEATURES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        arr = valid.to_numpy(dtype=float)
        lo_r, hi_r = CANONICAL_RANGES[label]
        outside = int(((arr < lo_r) | (arr > hi_r)).sum()) if arr.size else 0
        n_out, fence_lo, fence_hi = _iqr_outlier_mask(arr)

        with np.errstate(all="ignore"):
            arr_min = float(np.nanmin(arr)) if arr.size else None
            arr_max = float(np.nanmax(arr)) if arr.size else None
            arr_mean = float(np.nanmean(arr)) if arr.size else None
            arr_std = float(np.nanstd(arr, ddof=0)) if arr.size else None

        sanity["per_feature"][label] = {
            "column": col,
            "n_finite": int(valid.shape[0]),
            "min": arr_min,
            "max": arr_max,
            "mean": arr_mean,
            "std": arr_std,
            "n_outside_canonical_range": outside,
            "pct_outside_canonical_range": round(outside / len(arr) * 100, 4) if arr.size else 0.0,
            "iqr_outlier_count": n_out,
            "iqr_fence_lower": None if np.isnan(fence_lo) else round(fence_lo, 6),
            "iqr_fence_upper": None if np.isnan(fence_hi) else round(fence_hi, 6),
            "pct_iqr_outliers": round(n_out / len(arr) * 100, 4) if arr.size else 0.0,
        }

    # --- Observation density ---
    obs_density: dict[str, Any] = {}
    if OBS_COUNT_COL in df.columns:
        oc = pd.to_numeric(df[OBS_COUNT_COL], errors="coerce").fillna(0).astype(int)
        arr = oc.to_numpy()
        obs_density = {
            "column": OBS_COUNT_COL,
            "n_zero": int((arr == 0).sum()),
            "pct_zero": round(float((arr == 0).mean() * 100), 4),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr, ddof=0)), 4),
            "percentiles": {
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            },
            "histogram": _histogram_counts(arr, bins=[0, 1, 2, 5, 10, 25, 50, 100, np.inf]),
        }
    else:
        obs_density = {"error": f"missing {OBS_COUNT_COL}"}

    # --- Thesis documentation summary (no modeling) ---
    doc_summary = {
        "panel": f"Balanced panel: {n_rows} rows, {n_weeks} weeks × {n_grids} grid cells.",
        "spectral_non_null_overall_pct": {
            lbl: coverage["overall"].get(lbl, {}).get("pct_non_null")
            for lbl in SPECTRAL_FEATURES
            if lbl in coverage.get("overall", {})
        },
        "temporal": (
            f"Expected {expected_weeks} weeks; observed {n_weeks}. "
            f"Grids with exactly {expected_weeks} rows: {temporal['grids_with_full_week_count']}/{n_grids}."
        ),
        "weeks_no_sentinel_obs_globally": temporal.get("n_weeks_with_zero_sentinel_observations"),
        "spatial": (
            f"Grids below {low_density_threshold:.0%} non-null (worst feature): "
            f"{spatial['n_grids_low_spectral_density']}. "
            f"Grids with all features under {chronic_missing_threshold:.0%} non-null: "
            f"{spatial['n_grids_chronically_missing_spectral_features']}."
        ),
        "sanity_outside_canonical_range": {
            lbl: sanity["per_feature"].get(lbl, {}).get("n_outside_canonical_range")
            for lbl in SPECTRAL_FEATURES
        },
        "observation_count_zeros_pct": obs_density.get("pct_zero"),
    }

    doc_summary["bullets"] = [
        doc_summary["panel"],
        doc_summary["temporal"],
        (
            "Spectral indices (NDVI, NDWI, EVI, NDTI): overall non-null "
            f"{doc_summary['spectral_non_null_overall_pct']}"
        ),
        (
            f"Weeks where summed Sentinel observation count is zero: "
            f"{doc_summary['weeks_no_sentinel_obs_globally']}"
        ),
        doc_summary["spatial"],
        (
            "Values outside canonical sanity bounds (see feature_sanity): "
            f"{doc_summary['sanity_outside_canonical_range']}"
        ),
        (
            f"Share of grid-weeks with zero Sentinel observations: "
            f"{doc_summary['observation_count_zeros_pct']}%"
        ),
    ]

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "methodology_notes": [
            "Quality report is descriptive validation only (no modeling or inference).",
            "Coverage uses sentinel_ndvi_mean, sentinel_ndwi_mean, sentinel_evi_mean, sentinel_ndti_mean.",
            "EVI sanity bounds are [-2, 2]; NDVI/NDWI/NDTI use [-1, 1].",
            "Outliers are Tukey IQR fences (1.5×IQR) on finite values per feature.",
            "Low-density grids: min per-feature non-null fraction across weeks < threshold (default 0.35).",
            "Chronic missing: every spectral feature non-null fraction < threshold (default 0.05).",
        ],
        "documentation_summary": doc_summary,
        "coverage": coverage,
        "temporal_validation": temporal,
        "spatial_validation": spatial,
        "feature_sanity": sanity,
        "sentinel_observation_density": obs_density,
    }


def _histogram_counts(arr: np.ndarray, bins: list[float]) -> list[dict[str, Any]]:
    arr = np.asarray(arr, dtype=float)
    edges = np.array(bins, dtype=float)
    out: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if np.isinf(hi):
            m = (arr >= lo) & (arr < np.inf)
            label = f"[{int(lo)}, inf)"
        else:
            m = (arr >= lo) & (arr < hi)
            label = f"[{int(lo)}, {int(hi)})"
        out.append({"bin": label, "count": int(m.sum())})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Master dataset QA report (JSON).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/master_dataset.parquet"),
        help="Path to master_dataset.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/qa_report.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--expected-weeks",
        type=int,
        default=None,
        help="Override expected week count (default: inferred from data)",
    )
    parser.add_argument(
        "--low-density-threshold",
        type=float,
        default=0.35,
        help="Flag grid if min feature non-null fraction falls below this (0-1).",
    )
    parser.add_argument(
        "--chronic-missing-threshold",
        type=float,
        default=0.05,
        help="Chronic missing if every spectral feature non-null frac is below this.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = args.input if args.input.is_absolute() else project_root / args.input
    output_path = args.output if args.output.is_absolute() else project_root / args.output

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    df = pd.read_parquet(input_path)
    report = run_qa(
        df,
        expected_weeks=args.expected_weeks,
        low_density_threshold=args.low_density_threshold,
        chronic_missing_threshold=args.chronic_missing_threshold,
    )
    report["input_path"] = str(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2, ensure_ascii=False)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
