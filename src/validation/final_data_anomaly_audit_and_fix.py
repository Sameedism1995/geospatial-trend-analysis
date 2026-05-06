"""Final data anomaly audit + fix.

Audits `final_run/processed/features_ml_ready.parquet` for:
  - missing-value patterns
  - unusable oil-slick features (100% missing or all-zero)
  - vessel-density temporal degeneracy (static per grid)
  - NO2 weekly outliers (MAD)
  - water-quality index out-of-range / extreme values
  - skewed positive features (creates log1p + robust z versions)
  - high-correlation alias clusters (|r| ≥ 0.98)

Produces a sibling deliverable `final_run_cleaned/` containing:
  - validation/                     diagnostics + recommended feature set
  - outputs/visualizations/         temporal-smoothing & cleaned-window plots
  - outputs/reports/                cleaned distance-decay CSVs
  - processed/features_cleaned_full.parquet     (raw + flags + transforms)
  - processed/features_ml_safe.parquet          (lean, alias-free, oil-free)
  - VALIDATION_AND_FIX_REPORT.md

Run:
    python3 src/validation/final_data_anomaly_audit_and_fix.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("anomaly_audit_fix")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "final_run" / "processed" / "features_ml_ready.parquet"
DEFAULT_OUTPUT_ROOT = ROOT / "final_run_cleaned"

OIL_FEATURES = ["oil_slick_probability_t", "detection_score", "oil_slick_count_t"]
WATER_INDICES_BOUNDED = ["ndwi_mean", "ndwi_median", "ndti_mean", "ndti_median", "ndci_mean", "ndci_median", "ndvi_mean", "ndvi_median"]
WATER_INDICES_UNBOUNDED = ["fai_mean", "fai_median", "b11_mean", "b11_median"]
SKEWED_POSITIVE = [
    "vessel_density_t",
    "vessel_density",
    "maritime_pressure_index",
    "port_exposure_score",
    "coastal_exposure_score",
    "distance_to_port_km",
    "distance_to_nearest_high_vessel_density_cell",
    "no2_std_t",
    "ndwi_std",
    "ndti_std",
    "ndvi_std",
]
ALIAS_GROUPS = {
    "vessel_density": ["vessel_density_t", "vessel_density"],
    "no2": ["no2_mean_t", "NO2_mean"],
    "ndvi_land": ["ndvi_mean", "land_response_index"],
}
HIGH_CORR_THRESHOLD = 0.98


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def configure_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def make_skeleton(base: Path) -> dict[str, Path]:
    paths = {
        "base": base,
        "processed": base / "processed",
        "validation": base / "validation",
        "reports": base / "outputs" / "reports",
        "viz_smoothing": base / "outputs" / "visualizations" / "temporal_smoothing",
        "viz_cleaned_windows": base / "outputs" / "visualizations" / "sliding_window_cleaned",
        "reports_cleaned_windows": base / "outputs" / "reports" / "sliding_window_cleaned",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Step 2 — Missingness audit
# ---------------------------------------------------------------------------

def missingness_audit(df: pd.DataFrame, val_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in df.columns:
        s = df[c]
        is_numeric = pd.api.types.is_numeric_dtype(s)
        zero_pct: float | None = None
        if is_numeric:
            with np.errstate(invalid="ignore"):
                zero_pct = round(float((s == 0).sum()) / max(1, len(s)) * 100.0, 4)
        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "non_null_count": int(s.notna().sum()),
                "missing_percent": round(float(s.isna().mean() * 100.0), 4),
                "zero_percent": zero_pct,
                "n_unique": int(s.nunique(dropna=True)),
                "mean": float(s.mean()) if is_numeric and s.notna().any() else None,
                "std": float(s.std()) if is_numeric and s.notna().sum() > 1 else None,
                "min": float(s.min()) if is_numeric and s.notna().any() else None,
                "max": float(s.max()) if is_numeric and s.notna().any() else None,
            }
        )
    out = pd.DataFrame(rows).sort_values("missing_percent", ascending=False)
    csv_path = val_dir / "missingness_audit.csv"
    out.to_csv(csv_path, index=False)
    LOGGER.info("Wrote %s", _rel(csv_path))

    md_path = val_dir / "missingness_summary.md"
    high = out[out["missing_percent"] > 30.0]
    lines = ["# Missingness summary", "", f"Total columns: **{len(out)}**", f"Columns with >30% missing: **{len(high)}**", ""]
    if not high.empty:
        lines.append("| column | missing % | non-null | zero % | n_unique |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in high.iterrows():
            zp = "n/a" if r["zero_percent"] is None else f"{r['zero_percent']:.2f}"
            lines.append(f"| `{r['column']}` | {r['missing_percent']:.2f} | {r['non_null_count']} | {zp} | {r['n_unique']} |")
    md_path.write_text("\n".join(lines))
    LOGGER.info("Wrote %s", _rel(md_path))
    return out


# ---------------------------------------------------------------------------
# Step 3a — Oil features
# ---------------------------------------------------------------------------

def audit_oil_features(df: pd.DataFrame) -> dict[str, Any]:
    info: dict[str, Any] = {"unusable": False, "details": {}}
    for c in OIL_FEATURES:
        if c not in df.columns:
            info["details"][c] = {"present": False}
            continue
        s = df[c]
        non_null = int(s.notna().sum())
        unique_non_null = int(s.dropna().nunique())
        info["details"][c] = {
            "present": True,
            "non_null_count": non_null,
            "missing_percent": round(float(s.isna().mean() * 100.0), 4),
            "unique_non_null_values": unique_non_null,
            "all_zero": bool(non_null > 0 and float(s.fillna(0).abs().sum()) == 0.0),
        }
    prob = info["details"].get("oil_slick_probability_t", {})
    score = info["details"].get("detection_score", {})
    count = info["details"].get("oil_slick_count_t", {})
    if prob.get("missing_percent") == 100.0 and score.get("missing_percent") == 100.0 and count.get("all_zero", False):
        info["unusable"] = True
        info["reason"] = "probability and detection_score are 100% missing; count is all-zero"
    LOGGER.info("Oil features unusable=%s reason=%s", info["unusable"], info.get("reason"))
    return info


# ---------------------------------------------------------------------------
# Step 3b — Vessel temporal diagnostic
# ---------------------------------------------------------------------------

def vessel_temporal_diagnostic(df: pd.DataFrame, val_dir: Path) -> dict[str, Any]:
    if "vessel_density_t" not in df.columns or "grid_cell_id" not in df.columns:
        return {"available": False}

    by_grid = df.groupby("grid_cell_id")["vessel_density_t"]
    unique_per_grid = by_grid.nunique(dropna=True)
    std_per_grid = by_grid.std(ddof=1)
    weeks_per_grid = df.groupby("grid_cell_id")["week_start_utc"].nunique() if "week_start_utc" in df.columns else None
    diag = pd.DataFrame(
        {
            "grid_cell_id": unique_per_grid.index,
            "unique_values_across_weeks": unique_per_grid.values,
            "std_across_weeks": std_per_grid.reindex(unique_per_grid.index).values,
            "weeks_observed": weeks_per_grid.reindex(unique_per_grid.index).values if weeks_per_grid is not None else np.nan,
        }
    )
    csv_path = val_dir / "vessel_temporal_diagnostic.csv"
    diag.to_csv(csv_path, index=False)
    LOGGER.info("Wrote %s", _rel(csv_path))

    n_grids = int(len(unique_per_grid))
    n_static = int((unique_per_grid <= 1).sum())
    static_share = round(100.0 * n_static / max(1, n_grids), 4)
    weekly_mean: list[float] = []
    weekly_std: list[float] = []
    if "week_start_utc" in df.columns:
        weekly = df.groupby("week_start_utc")["vessel_density_t"]
        weekly_mean = [float(v) for v in weekly.mean().tolist()]
        weekly_std = [float(v) for v in weekly.std(ddof=1).fillna(0).tolist()]
    flat_weekly = bool(np.nanstd(weekly_mean) < 1e-9) if weekly_mean else False
    is_static = static_share >= 90.0 or flat_weekly
    info = {
        "available": True,
        "n_grids": n_grids,
        "n_grids_static": n_static,
        "static_share_percent": static_share,
        "weekly_mean_std": float(np.nanstd(weekly_mean)) if weekly_mean else None,
        "is_static": bool(is_static),
        "interpretation": "vessel_density_t behaves as a per-grid SPATIAL pressure proxy, not weekly traffic" if is_static else "vessel_density_t shows temporal variation",
    }
    LOGGER.info("Vessel temporal: static_share=%.1f%% flat_weekly=%s → is_static=%s", static_share, flat_weekly, info["is_static"])
    return info


# ---------------------------------------------------------------------------
# Step 3c — NO2 outliers + smoothing
# ---------------------------------------------------------------------------

def no2_outliers_and_smooth(df: pd.DataFrame, val_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, Any] = {"present": False}
    if "no2_mean_t" not in df.columns or "week_start_utc" not in df.columns:
        return df, info

    weekly = df.groupby("week_start_utc")["no2_mean_t"].mean().sort_index()
    if weekly.empty:
        return df, info

    median = float(weekly.median())
    mad = float(np.median(np.abs(weekly - median)))
    threshold = 3.0 * 1.4826 * mad if mad > 0 else 0.0
    z = (weekly - median) / (1.4826 * mad) if mad > 0 else weekly * 0
    is_outlier = (np.abs(weekly - median) > threshold) if threshold > 0 else (weekly != weekly)

    out_df = pd.DataFrame(
        {
            "week_start_utc": weekly.index,
            "no2_weekly_mean": weekly.values,
            "robust_z": z.values,
            "is_outlier": is_outlier.values,
            "deviation_from_median": (weekly - median).values,
        }
    )
    csv_path = val_dir / "no2_weekly_outliers.csv"
    out_df.to_csv(csv_path, index=False)
    LOGGER.info("Wrote %s (outlier weeks: %d)", _rel(csv_path), int(is_outlier.sum()))

    # Add rolling smoothed columns at the row level
    df = df.sort_values(["grid_cell_id", "week_start_utc"]) if "grid_cell_id" in df.columns else df
    grouped = df.groupby("grid_cell_id")["no2_mean_t"] if "grid_cell_id" in df.columns else None
    if grouped is not None:
        df["no2_mean_t_rolling3"] = grouped.transform(lambda s: s.rolling(window=3, min_periods=1, center=True).mean())
        df["no2_mean_t_rolling5"] = grouped.transform(lambda s: s.rolling(window=5, min_periods=1, center=True).mean())
    else:
        df["no2_mean_t_rolling3"] = df["no2_mean_t"].rolling(window=3, min_periods=1, center=True).mean()
        df["no2_mean_t_rolling5"] = df["no2_mean_t"].rolling(window=5, min_periods=1, center=True).mean()

    info.update(
        {
            "present": True,
            "median": median,
            "mad": mad,
            "threshold": threshold,
            "n_weeks": int(len(weekly)),
            "n_outlier_weeks": int(is_outlier.sum()),
            "outlier_weeks": [str(w) for w in weekly.index[is_outlier.values].tolist()],
        }
    )
    return df, info


# ---------------------------------------------------------------------------
# Step 3d — water-quality validity + winsorization
# ---------------------------------------------------------------------------

def water_quality_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    extreme_flags = pd.Series(False, index=df.index)
    ndwi_extreme = pd.Series(False, index=df.index)
    ndti_extreme = pd.Series(False, index=df.index)

    for col in WATER_INDICES_BOUNDED + WATER_INDICES_UNBOUNDED:
        if col not in df.columns:
            continue
        s = df[col]
        out_of_range = pd.Series(False, index=df.index)
        if col in WATER_INDICES_BOUNDED:
            out_of_range = (s < -1.0001) | (s > 1.0001)
        # Extreme: outside p1/p99 winsor bounds
        valid = s.dropna()
        p1, p99 = (float(valid.quantile(0.01)), float(valid.quantile(0.99))) if not valid.empty else (np.nan, np.nan)
        extreme = (s <= p1) | (s >= p99) if valid.size else pd.Series(False, index=df.index)
        info[col] = {
            "non_null": int(s.notna().sum()),
            "min": float(valid.min()) if valid.size else None,
            "max": float(valid.max()) if valid.size else None,
            "p1": p1,
            "p99": p99,
            "n_out_of_range": int(out_of_range.sum()),
            "n_extreme_p1_p99": int(extreme.sum()),
        }
        # Winsorized version
        if valid.size:
            df[f"{col.replace('_median', '_mean')}_winsor"] = s.clip(lower=p1, upper=p99) if col in {"ndwi_mean", "ndti_mean", "ndci_mean", "fai_mean", "b11_mean", "ndvi_mean"} else s
        if col == "ndwi_mean":
            ndwi_extreme = ndwi_extreme | extreme | out_of_range
        if col == "ndti_mean":
            ndti_extreme = ndti_extreme | extreme | out_of_range
        extreme_flags = extreme_flags | extreme | out_of_range

    df["ndwi_extreme_flag"] = ndwi_extreme.fillna(False)
    df["ndti_extreme_flag"] = ndti_extreme.fillna(False)
    df["water_quality_extreme_flag"] = extreme_flags.fillna(False)
    LOGGER.info(
        "Water-quality flags: ndwi_extreme=%d ndti_extreme=%d any_water_extreme=%d",
        int(df["ndwi_extreme_flag"].sum()),
        int(df["ndti_extreme_flag"].sum()),
        int(df["water_quality_extreme_flag"].sum()),
    )
    return df, info


# ---------------------------------------------------------------------------
# Step 3e — skew transforms
# ---------------------------------------------------------------------------

def add_skew_transforms(df: pd.DataFrame) -> list[str]:
    added: list[str] = []
    for col in SKEWED_POSITIVE:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        # log1p safe only for non-negative values
        valid = s.dropna()
        if valid.size and float(valid.min()) >= 0:
            df[f"{col}_log1p"] = np.log1p(s)
            added.append(f"{col}_log1p")
        # robust z = (x - median) / IQR
        if valid.size:
            med = float(valid.median())
            q1 = float(valid.quantile(0.25))
            q3 = float(valid.quantile(0.75))
            iqr = q3 - q1
            df[f"{col}_robust_z"] = (s - med) / iqr if iqr > 0 else (s - med) * 0
            added.append(f"{col}_robust_z")
    LOGGER.info("Skew transforms added: %d columns", len(added))
    return added


# ---------------------------------------------------------------------------
# Step 3f — alias / high-correlation detection
# ---------------------------------------------------------------------------

def high_correlation_pairs(df: pd.DataFrame, val_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    numeric = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    cols = [c for c in numeric.columns if numeric[c].notna().sum() > 5 and numeric[c].nunique(dropna=True) > 1]
    corr = numeric[cols].corr(method="pearson")
    rows: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.at[a, b]
            if pd.notna(v) and abs(v) >= HIGH_CORR_THRESHOLD:
                key = frozenset({a, b})
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"feature_x": a, "feature_y": b, "pearson": round(float(v), 6)})
    rows.sort(key=lambda d: -abs(d["pearson"]))
    out = pd.DataFrame(rows)
    csv_path = val_dir / "high_correlation_pairs.csv"
    out.to_csv(csv_path, index=False)
    LOGGER.info("Wrote %s (%d pairs)", _rel(csv_path), len(out))

    # Build adjacency clusters across alias groups + |r|>=0.98 graph
    graph: dict[str, set[str]] = {}
    for _, r in out.iterrows():
        graph.setdefault(r["feature_x"], set()).add(r["feature_y"])
        graph.setdefault(r["feature_y"], set()).add(r["feature_x"])
    visited: set[str] = set()
    clusters: list[list[str]] = []
    for node in graph:
        if node in visited:
            continue
        stack, cluster = [node], []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            cluster.append(cur)
            stack.extend(graph.get(cur, set()))
        clusters.append(sorted(cluster))

    # Choose representatives
    preferred_order = [
        "vessel_density_t",
        "no2_mean_t",
        "ndvi_mean",
        "ndwi_mean",
        "ndti_mean",
        "ndci_mean",
        "fai_mean",
        "b11_mean",
        "distance_to_port_km",
        "coastal_exposure_score",
        "maritime_pressure_index",
        "atmospheric_transfer_index",
        "land_response_index",
    ]
    drop: set[str] = set()
    chosen_reps: dict[str, list[str]] = {}
    for cluster in clusters:
        rep = next((c for c in preferred_order if c in cluster), cluster[0])
        chosen_reps[rep] = cluster
        for member in cluster:
            if member != rep:
                drop.add(member)
    LOGGER.info("Alias clusters: %d (dropping %d alias members)", len(clusters), len(drop))
    return out, sorted(drop)


# ---------------------------------------------------------------------------
# Step 4 — temporal smoothing plots
# ---------------------------------------------------------------------------

def temporal_smoothing_plots(df: pd.DataFrame, viz_dir: Path) -> list[Path]:
    saved: list[Path] = []
    if "week_start_utc" not in df.columns:
        return saved
    targets = ["no2_mean_t", "ndwi_mean", "ndti_mean", "vessel_density_t"]
    for col in targets:
        if col not in df.columns:
            continue
        weekly = df.groupby("week_start_utc")[col].agg(["mean", "count"]).reset_index().sort_values("week_start_utc")
        weekly["roll3"] = weekly["mean"].rolling(window=3, min_periods=1, center=True).mean()
        weekly["roll5"] = weekly["mean"].rolling(window=5, min_periods=1, center=True).mean()

        fig, ax1 = plt.subplots(figsize=(11, 4.5))
        ax1.plot(weekly["week_start_utc"], weekly["mean"], lw=1.0, alpha=0.5, color="tab:grey", label="raw weekly mean")
        ax1.plot(weekly["week_start_utc"], weekly["roll3"], lw=1.6, color="tab:blue", label="rolling 3")
        ax1.plot(weekly["week_start_utc"], weekly["roll5"], lw=1.8, color="tab:red", label="rolling 5")
        ax1.set_xlabel("week")
        ax1.set_ylabel(col)
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.legend(loc="upper left", fontsize=8)
        ax2 = ax1.twinx()
        ax2.bar(weekly["week_start_utc"], weekly["count"], color="grey", alpha=0.18, width=5.0, label="n per week")
        ax2.set_ylabel("samples per week", color="grey")
        plt.title(f"Temporal smoothing — {col}")
        plt.tight_layout()
        path = viz_dir / f"smoothing_{col}.png"
        plt.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)
    LOGGER.info("Temporal smoothing plots: %d", len(saved))
    return saved


# ---------------------------------------------------------------------------
# Step 5 — cleaned distance-decay sliding windows
# ---------------------------------------------------------------------------

def sliding_window(df: pd.DataFrame, value_col: str, *, window_km: float = 50.0, step_km: float = 10.0, max_km: float = 1000.0) -> pd.DataFrame:
    if "distance_to_port_km" not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    valid = df[["distance_to_port_km", value_col]].dropna()
    if valid.empty:
        return pd.DataFrame()
    upper = min(max_km, float(valid["distance_to_port_km"].max()))
    starts = np.arange(0.0, max(upper - window_km, 0.0) + step_km, step_km)
    rows: list[dict[str, Any]] = []
    dist = valid["distance_to_port_km"].values
    vals = valid[value_col].values
    for s in starts:
        e = s + window_km
        m = (dist >= s) & (dist < e)
        n = int(m.sum())
        if n == 0:
            continue
        sub = vals[m]
        rows.append(
            {
                "window_start_km": s,
                "window_end_km": e,
                "window_mid_km": s + window_km / 2.0,
                "count": n,
                "mean": float(np.nanmean(sub)),
                "median": float(np.nanmedian(sub)),
                "std": float(np.nanstd(sub, ddof=1)) if n > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def cleaned_sliding_windows(df: pd.DataFrame, viz_dir: Path, csv_dir: Path, vessel_label: str) -> list[Path]:
    saved: list[Path] = []
    candidates = [
        (vessel_label, vessel_label),
        ("no2_mean_t_rolling3", "no2_mean_t_rolling3"),
        ("ndwi_mean_winsor", "ndwi_mean_winsor"),
        ("ndti_mean_winsor", "ndti_mean_winsor"),
        ("ndvi_mean_winsor", "ndvi_mean_winsor"),
    ]
    for label, col in candidates:
        if col not in df.columns:
            LOGGER.info("Cleaned window: skip (missing) %s", col)
            continue
        sw = sliding_window(df, col)
        if sw.empty:
            LOGGER.info("Cleaned window: empty %s", col)
            continue
        csv_path = csv_dir / f"sliding_window_{label}.csv"
        sw.to_csv(csv_path, index=False)
        fig, ax1 = plt.subplots(figsize=(9.5, 4.5))
        ax1.plot(sw["window_mid_km"], sw["mean"], lw=1.8, color="tab:blue", label="mean")
        ax1.plot(sw["window_mid_km"], sw["median"], lw=1.2, ls="--", color="tab:orange", label="median")
        ax1.fill_between(sw["window_mid_km"], sw["mean"] - sw["std"].fillna(0), sw["mean"] + sw["std"].fillna(0), color="tab:blue", alpha=0.12, label="±1 std")
        ax1.set_xlabel("distance to port [km]")
        ax1.set_ylabel(label)
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.legend(loc="upper left", fontsize=8)
        ax2 = ax1.twinx()
        ax2.bar(sw["window_mid_km"], sw["count"], width=9.0, color="grey", alpha=0.22, label="n per window")
        ax2.set_ylabel("samples per window", color="grey")
        plt.title(f"Cleaned sliding-window distance decay — {label}")
        plt.tight_layout()
        path = viz_dir / f"sliding_window_{label}.png"
        plt.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)
    LOGGER.info("Cleaned sliding-window plots: %d", len(saved))
    return saved


# ---------------------------------------------------------------------------
# Step 6 — cleaned datasets + recommended feature set
# ---------------------------------------------------------------------------

def build_cleaned_datasets(
    df: pd.DataFrame,
    *,
    drop_aliases: list[str],
    oil_unusable: bool,
    paths: dict[str, Path],
    missingness: pd.DataFrame,
    vessel_static: bool,
) -> tuple[Path, Path, list[str], list[str]]:
    full_path = paths["processed"] / "features_cleaned_full.parquet"
    df.to_parquet(full_path, index=False)
    LOGGER.info("Wrote %s (%d × %d)", _rel(full_path), len(df), len(df.columns))

    excluded: list[str] = []
    if oil_unusable:
        excluded.extend([c for c in OIL_FEATURES if c in df.columns])
    excluded.extend(drop_aliases)
    high_missing = missingness[(missingness["missing_percent"] > 80.0) & (~missingness["column"].isin({"ndvi_mean", "ndvi_median", "ndvi_std", "land_response_index", "ndvi_mean_winsor"}))]
    excluded.extend(high_missing["column"].tolist())
    excluded = sorted(set(excluded))
    keep = [c for c in df.columns if c not in excluded]

    # If vessel density is static, expose a clearly-named spatial proxy column
    if vessel_static and "vessel_density_t" in df.columns and "vessel_density_spatial_proxy" not in df.columns:
        df = df.copy()
        df["vessel_density_spatial_proxy"] = df["vessel_density_t"]
        if "vessel_density_t_log1p" in df.columns:
            df["vessel_density_spatial_proxy_log1p"] = df["vessel_density_t_log1p"]
        keep_extra = ["vessel_density_spatial_proxy"]
        if "vessel_density_t_log1p" in df.columns:
            keep_extra.append("vessel_density_spatial_proxy_log1p")
        keep = list(dict.fromkeys(keep + keep_extra))

    safe = df[keep].copy()
    safe_path = paths["processed"] / "features_ml_safe.parquet"
    safe.to_parquet(safe_path, index=False)
    LOGGER.info("Wrote %s (%d × %d) | excluded %d cols", _rel(safe_path), len(safe), len(safe.columns), len(excluded))

    rec_path = paths["validation"] / "recommended_feature_set.txt"
    rec_path.write_text("\n".join(safe.columns.tolist()))
    LOGGER.info("Wrote %s", _rel(rec_path))
    return full_path, safe_path, excluded, safe.columns.tolist()


# ---------------------------------------------------------------------------
# Step 7 — final report
# ---------------------------------------------------------------------------

def write_final_report(
    base: Path,
    *,
    input_path: Path,
    missingness: pd.DataFrame,
    oil_info: dict[str, Any],
    vessel_info: dict[str, Any],
    no2_info: dict[str, Any],
    water_info: dict[str, Any],
    skew_added: list[str],
    high_corr: pd.DataFrame,
    drop_aliases: list[str],
    excluded: list[str],
    safe_features: list[str],
    smoothing_plots: list[Path],
    cleaned_window_plots: list[Path],
    full_path: Path,
    safe_path: Path,
    anomalies_count: int,
) -> Path:
    rep = base / "VALIDATION_AND_FIX_REPORT.md"
    L: list[str] = []
    L.append("# Validation & fix report — final_run_cleaned/")
    L.append("")
    L.append(f"- Source dataset: `{_rel(input_path)}`")
    L.append(f"- Cleaned root: `{_rel(base)}`")
    L.append(f"- Anomalies found / addressed: **{anomalies_count}**")
    L.append("")

    L.append("## Anomalies found and how they were handled")
    L.append("")

    L.append("### A. Oil-slick features")
    if oil_info.get("unusable"):
        L.append("- **Status: UNUSABLE** — " + oil_info.get("reason", ""))
        L.append("- Action: excluded from `features_ml_safe.parquet`. Raw columns kept in `features_cleaned_full.parquet` for transparency.")
        L.append("- Thesis rule: **do not claim oil-slick findings from this run.**")
    else:
        L.append("- Status: usable.")
    for c, det in oil_info.get("details", {}).items():
        L.append(f"  - `{c}`: {det}")
    L.append("")

    L.append("### B. Vessel density temporal behaviour")
    if vessel_info.get("available"):
        L.append(f"- {vessel_info.get('static_share_percent')}% of grid cells have **only 1 unique value** of `vessel_density_t` across all weeks.")
        L.append(f"- Weekly-mean σ: {vessel_info.get('weekly_mean_std')}")
        L.append(f"- Verdict: **is_static = {vessel_info.get('is_static')}** — {vessel_info.get('interpretation')}.")
        if vessel_info.get("is_static"):
            L.append("- Action: cleaned dataset exposes `vessel_density_spatial_proxy` (and `*_log1p`) so downstream code can reference it without implying weekly traffic dynamics.")
            L.append("- Thesis rule: describe vessel_density_t as a **spatial pressure proxy**, not weekly vessel traffic.")
    L.append("")

    L.append("### C. NO2 weekly outliers (MAD)")
    if no2_info.get("present"):
        L.append(f"- Median weekly mean: {no2_info['median']:.3e} | MAD: {no2_info['mad']:.3e} | threshold (3·1.4826·MAD): {no2_info['threshold']:.3e}.")
        L.append(f"- Outlier weeks ({no2_info['n_outlier_weeks']} of {no2_info['n_weeks']}):")
        for w in no2_info.get("outlier_weeks", []):
            L.append(f"  - {w}")
        L.append("- Action: smoothed columns `no2_mean_t_rolling3`, `no2_mean_t_rolling5` added; raw `no2_mean_t` retained.")
        L.append("- Thesis rule: report rolling-mean curves; flag weeks with extreme negative values rather than deleting them.")
    L.append("")

    L.append("### D. Water-quality index validity")
    L.append("- Per-column audit (min/max/p1/p99/out-of-range/extreme):")
    for col, det in water_info.items():
        L.append(f"  - `{col}`: {det}")
    L.append("- Action: winsorized columns at [p1, p99] added (`*_winsor`); flag columns added: `ndwi_extreme_flag`, `ndti_extreme_flag`, `water_quality_extreme_flag`. Raw values preserved.")
    L.append("")

    L.append("### E. Skew transforms")
    L.append(f"- Added {len(skew_added)} columns (log1p / robust_z) for skewed positive features.")
    L.append("")

    L.append("### F. High-correlation alias clusters (|r| ≥ 0.98)")
    L.append(f"- Pairs flagged: {len(high_corr)}.")
    L.append(f"- Alias members dropped from ML-safe set: {len(drop_aliases)}.")
    L.append("- Drop list: " + ", ".join(f"`{c}`" for c in drop_aliases))
    L.append("")

    L.append("## Cleaned datasets")
    L.append(f"- Full: `{_rel(full_path)}` (raw + flags + transforms)")
    L.append(f"- ML-safe: `{_rel(safe_path)}` ({len(safe_features)} columns; excluded {len(excluded)})")
    L.append("- Excluded columns: " + (", ".join(f"`{c}`" for c in excluded) if excluded else "(none)"))
    L.append("")

    L.append("## Thesis-safe plots")
    for p in smoothing_plots:
        L.append(f"- `{_rel(p)}`")
    for p in cleaned_window_plots:
        L.append(f"- `{_rel(p)}`")
    L.append("")

    L.append("## Interpretation rules (DO / DON'T)")
    L.append("- **DO** present `vessel_density_t` as a spatial pressure proxy. **DON'T** claim weekly vessel-traffic dynamics from this column.")
    L.append("- **DO NOT** report any oil-slick-based result; the source data is unusable in this run.")
    L.append("- **DO** show NO2 as smoothed rolling means; **DON'T** delete outlier weeks.")
    L.append("- **DO** use winsorized water-quality columns for robust statistics; **DON'T** discard raw values.")
    L.append("- **DO** prefer the recommended feature list (no aliases) for any ML modelling.")
    L.append("")

    rep.write_text("\n".join(L))
    LOGGER.info("Wrote %s", _rel(rep))
    return rep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    paths = make_skeleton(args.output_root)
    configure_logging(paths["base"] / "logs.txt")
    LOGGER.info("=== Data anomaly audit + fix ===")
    LOGGER.info("Input: %s", _rel(args.input))
    LOGGER.info("Output: %s", _rel(args.output_root))

    if not args.input.exists():
        LOGGER.error("Input parquet not found: %s", args.input)
        return 2
    df = pd.read_parquet(args.input)
    LOGGER.info("Loaded shape: %s", df.shape)
    LOGGER.info("Columns: %s", df.columns.tolist())
    LOGGER.info(
        "Detected schema: time=%s grid=%s lat=%s lon=%s",
        "week_start_utc" if "week_start_utc" in df.columns else None,
        "grid_cell_id" if "grid_cell_id" in df.columns else None,
        "grid_centroid_lat" if "grid_centroid_lat" in df.columns else None,
        "grid_centroid_lon" if "grid_centroid_lon" in df.columns else None,
    )

    anomalies = 0

    # 2. missingness
    missingness = missingness_audit(df, paths["validation"])
    anomalies += int((missingness["missing_percent"] > 30.0).sum())

    # 3a. oil
    oil_info = audit_oil_features(df)
    if oil_info.get("unusable"):
        anomalies += len([c for c in OIL_FEATURES if c in df.columns])

    # 3b. vessel diagnostic
    vessel_info = vessel_temporal_diagnostic(df, paths["validation"])
    if vessel_info.get("available") and vessel_info.get("is_static"):
        anomalies += 1

    # 3c. NO2 outliers + smoothing (mutates df with rolling cols)
    df, no2_info = no2_outliers_and_smooth(df, paths["validation"])
    anomalies += int(no2_info.get("n_outlier_weeks") or 0)

    # 3d. water quality + winsorization
    df, water_info = water_quality_audit(df)
    anomalies += int(df["water_quality_extreme_flag"].sum())

    # 3e. skew transforms
    skew_added = add_skew_transforms(df)

    # 3f. correlation pairs / aliases
    high_corr, drop_aliases = high_correlation_pairs(df, paths["validation"])
    anomalies += len(drop_aliases)

    # 4. temporal smoothing plots
    smoothing_plots = temporal_smoothing_plots(df, paths["viz_smoothing"])

    # 5. cleaned sliding-window plots
    vessel_label = "vessel_density_t_log1p"
    if vessel_info.get("is_static") and "vessel_density_t" in df.columns:
        df["vessel_density_spatial_proxy"] = df["vessel_density_t"]
        if "vessel_density_t_log1p" in df.columns:
            df["vessel_density_spatial_proxy_log1p"] = df["vessel_density_t_log1p"]
            vessel_label = "vessel_density_spatial_proxy_log1p"
    cleaned_window_plots = cleaned_sliding_windows(df, paths["viz_cleaned_windows"], paths["reports_cleaned_windows"], vessel_label)

    # 6. cleaned datasets
    full_path, safe_path, excluded, safe_features = build_cleaned_datasets(
        df,
        drop_aliases=drop_aliases,
        oil_unusable=oil_info.get("unusable", False),
        paths=paths,
        missingness=missingness,
        vessel_static=bool(vessel_info.get("is_static")),
    )

    # 7. final report
    rep = write_final_report(
        paths["base"],
        input_path=args.input,
        missingness=missingness,
        oil_info=oil_info,
        vessel_info=vessel_info,
        no2_info=no2_info,
        water_info=water_info,
        skew_added=skew_added,
        high_corr=high_corr,
        drop_aliases=drop_aliases,
        excluded=excluded,
        safe_features=safe_features,
        smoothing_plots=smoothing_plots,
        cleaned_window_plots=cleaned_window_plots,
        full_path=full_path,
        safe_path=safe_path,
        anomalies_count=anomalies,
    )

    # final logs
    LOGGER.info("=== Done ===")
    LOGGER.info("Anomalies found / addressed: %d", anomalies)
    LOGGER.info("Features excluded from ml_safe: %d", len(excluded))
    LOGGER.info("Cleaned full dataset: %s", _rel(full_path))
    LOGGER.info("Cleaned ML-safe dataset: %s", _rel(safe_path))
    LOGGER.info("Validation report: %s", _rel(rep))
    LOGGER.info("Recommended thesis-safe plots:")
    for p in smoothing_plots + cleaned_window_plots:
        LOGGER.info("  %s", _rel(p))

    summary_json = paths["validation"] / "audit_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "anomalies_count": anomalies,
                "excluded_features": excluded,
                "safe_feature_count": len(safe_features),
                "oil_info": oil_info,
                "vessel_info": vessel_info,
                "no2_info": no2_info,
                "water_info": water_info,
                "drop_aliases": drop_aliases,
                "thesis_safe_plots": [_rel(p) for p in smoothing_plots + cleaned_window_plots],
                "outputs": {
                    "full": _rel(full_path),
                    "ml_safe": _rel(safe_path),
                    "report": _rel(rep),
                },
            },
            indent=2,
            default=str,
        )
    )
    LOGGER.info("Wrote %s", _rel(summary_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
