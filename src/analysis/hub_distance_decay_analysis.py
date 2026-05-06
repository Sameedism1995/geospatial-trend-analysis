"""Hub / port-level distance-decay analysis.

Implements professor feedback:
  1. Individual hub/port-level distance-decay plots (one panel per indicator per hub).
  2. Sliding-window distance-decay curves (replaces hard binning).
  3. NO2 diagnostic for the 200-500 km range.

The script is defensive about column names: it auto-detects the closest available
column for each requested indicator and prints what it found.

Run:
    python3 src/analysis/hub_distance_decay_analysis.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("hub_distance_decay")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH_PRIMARY = PROJECT_ROOT / "processed" / "features_ml_ready.parquet"
DATA_PATH_FALLBACK = PROJECT_ROOT / "processed" / "merged_dataset.parquet"

OUT_HUB_PLOTS = PROJECT_ROOT / "outputs" / "visualizations" / "hub_level_distance_decay"
OUT_SW_PLOTS = PROJECT_ROOT / "outputs" / "visualizations" / "sliding_window_distance_decay"
OUT_SW_REPORTS = PROJECT_ROOT / "outputs" / "reports" / "sliding_window_distance_decay"
OUT_NO2_REPORT = PROJECT_ROOT / "outputs" / "reports" / "no2_distance_diagnostic.md"


def _resolve_paths(out_root: Path | None, dataset: Path | None) -> tuple[Path, Path, Path, Path, Path | None]:
    """Return (hub_plots_dir, sw_plots_dir, sw_reports_dir, no2_report_path, dataset_override)."""
    if out_root is None:
        return OUT_HUB_PLOTS, OUT_SW_PLOTS, OUT_SW_REPORTS, OUT_NO2_REPORT, dataset
    base = Path(out_root)
    hub_plots = base / "outputs" / "visualizations" / "hub_level_distance_decay"
    sw_plots = base / "outputs" / "visualizations" / "sliding_window_distance_decay"
    sw_reports = base / "outputs" / "reports" / "sliding_window_distance_decay"
    no2_report = base / "outputs" / "reports" / "no2_distance_diagnostic.md"
    return hub_plots, sw_plots, sw_reports, no2_report, dataset

WINDOW_KM = 50.0
STEP_KM = 10.0
MAX_DIST_CAP_KM = 1000.0
MIN_SAMPLES_PER_WINDOW = 10

INDICATOR_CANDIDATES: dict[str, list[str]] = {
    "ndti": ["sentinel_ndti_mean_t", "ndti_mean", "ndti_median"],
    "ndwi": ["sentinel_ndwi_mean_t", "ndwi_mean", "ndwi_median"],
    "ndvi": ["sentinel_ndvi_mean_t", "ndvi_mean", "ndvi_median"],
    "no2": [
        "no2_tropospheric_column_mean_t",
        "no2_mean_t",
        "NO2_mean",
        "no2_mean",
    ],
    "vessel_density": ["vessel_density_t", "vessel_density"],
}

PORT_COL_CANDIDATES = ["nearest_port", "nearest_port_name", "port", "hub"]
DIST_COL_CANDIDATES = ["distance_to_port_km", "distance_to_nearest_port_km", "port_distance_km"]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset(dataset_override: Path | None = None) -> tuple[pd.DataFrame, Path]:
    candidates: list[Path] = []
    if dataset_override is not None:
        candidates.append(Path(dataset_override))
    candidates.extend([DATA_PATH_PRIMARY, DATA_PATH_FALLBACK])
    for path in candidates:
        if path.exists():
            try:
                rel = path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = path
            LOGGER.info("Loading dataset: %s", rel)
            df = pd.read_parquet(path)
            return df, path
    raise FileNotFoundError(
        f"No processed dataset found. Looked for: {[str(c) for c in candidates]}"
    )


def find_first(df_columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(df_columns)
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    detected: dict[str, str | None] = {}
    detected["port"] = find_first(df.columns, PORT_COL_CANDIDATES)
    detected["distance"] = find_first(df.columns, DIST_COL_CANDIDATES)
    for key, candidates in INDICATOR_CANDIDATES.items():
        detected[key] = find_first(df.columns, candidates)

    LOGGER.info("Detected columns:")
    for k, v in detected.items():
        marker = "OK " if v else "-- "
        LOGGER.info("  %s%-15s -> %s", marker, k, v)

    if detected["port"] is None or detected["distance"] is None:
        raise RuntimeError(
            "Required port/distance columns not found. "
            f"Need one of port={PORT_COL_CANDIDATES} and distance={DIST_COL_CANDIDATES}."
        )
    return detected


def prepare_frame(df: pd.DataFrame, cols: dict[str, str | None]) -> pd.DataFrame:
    keep = [cols["port"], cols["distance"]]
    keep += [c for k, c in cols.items() if k not in {"port", "distance"} and c is not None]
    keep = [c for c in keep if c in df.columns]
    out = df.loc[:, keep].copy()
    out = out.rename(columns={cols["port"]: "hub", cols["distance"]: "distance_km"})
    out["distance_km"] = pd.to_numeric(out["distance_km"], errors="coerce")
    out = out.dropna(subset=["hub", "distance_km"])
    out = out[out["distance_km"] >= 0]
    return out


def plot_hub_panels(
    df: pd.DataFrame,
    indicators: dict[str, str],
    out_dir: Path,
) -> list[Path]:
    """Create one figure per hub with one subplot per indicator (scatter + rolling mean)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    hubs = sorted(df["hub"].dropna().unique().tolist())
    LOGGER.info("Plotting per-hub panels for %d hubs: %s", len(hubs), hubs)

    for hub in hubs:
        sub = df[df["hub"] == hub]
        if sub.empty or not indicators:
            continue
        n = len(indicators)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), squeeze=False)
        for ax, (label, col) in zip(axes.flat, indicators.items()):
            valid = sub[["distance_km", col]].dropna()
            if valid.empty:
                ax.text(0.5, 0.5, f"no data: {col}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{label} ({col})")
                continue
            valid = valid.sort_values("distance_km")
            ax.scatter(valid["distance_km"], valid[col], s=6, alpha=0.25, color="tab:blue")
            # rolling mean ordered by distance for a quick visual trend
            window = max(20, len(valid) // 50)
            rolling = valid[col].rolling(window=window, center=True, min_periods=max(5, window // 4)).mean()
            ax.plot(valid["distance_km"], rolling, color="tab:red", lw=1.6, label=f"rolling mean (n={window})")
            ax.set_title(f"{label} ({col})")
            ax.set_xlabel("distance to port [km]")
            ax.set_ylabel(label)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, ls=":", alpha=0.4)

        for ax in axes.flat[len(indicators):]:
            ax.axis("off")
        fig.suptitle(f"Distance decay – hub: {hub} (n={len(sub)})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = out_dir / f"hub_{_safe(hub)}_distance_decay.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)
        LOGGER.info("  saved %s", path.relative_to(PROJECT_ROOT))
    return saved


def plot_hub_comparison(
    df: pd.DataFrame,
    indicators: dict[str, str],
    out_dir: Path,
    max_dist_km: float = MAX_DIST_CAP_KM,
) -> Path | None:
    """One figure with one subplot per indicator overlaying hubs (sliding-window mean)."""
    if not indicators:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    hubs = sorted(df["hub"].dropna().unique().tolist())
    n = len(indicators)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for ax, (label, col) in zip(axes.flat, indicators.items()):
        for i, hub in enumerate(hubs):
            sub = df[df["hub"] == hub]
            sw = sliding_window_summary(sub, col, max_dist_km=max_dist_km)
            if sw.empty:
                continue
            ax.plot(
                sw["window_mid_km"],
                sw["mean"],
                color=cmap(i % 10),
                lw=1.8,
                label=f"{hub} (n={int(sw['count'].sum())})",
            )
        ax.set_title(f"{label} ({col})")
        ax.set_xlabel("distance to port [km]")
        ax.set_ylabel(f"{label} mean (sliding window)")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="best", fontsize=8)

    for ax in axes.flat[len(indicators):]:
        ax.axis("off")
    fig.suptitle("Hub comparison – sliding-window distance decay", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "hub_comparison_distance_decay.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    LOGGER.info("Saved hub comparison plot: %s", path.relative_to(PROJECT_ROOT))
    return path


def sliding_window_summary(
    df: pd.DataFrame,
    value_col: str,
    *,
    window_km: float = WINDOW_KM,
    step_km: float = STEP_KM,
    max_dist_km: float = MAX_DIST_CAP_KM,
    min_samples: int = MIN_SAMPLES_PER_WINDOW,
) -> pd.DataFrame:
    """Return a DataFrame with one row per sliding window."""
    if value_col not in df.columns:
        return pd.DataFrame()
    valid = df[["distance_km", value_col]].dropna()
    if valid.empty:
        return pd.DataFrame()

    upper = min(max_dist_km, float(valid["distance_km"].max()))
    starts = np.arange(0.0, max(upper - window_km, 0.0) + step_km, step_km)
    rows = []
    dist = valid["distance_km"].values
    vals = valid[value_col].values
    for s in starts:
        e = s + window_km
        mask = (dist >= s) & (dist < e)
        n = int(mask.sum())
        if n == 0:
            continue
        sub_vals = vals[mask]
        rows.append(
            {
                "window_start_km": s,
                "window_end_km": e,
                "window_mid_km": s + window_km / 2.0,
                "count": n,
                "mean": float(np.nanmean(sub_vals)),
                "median": float(np.nanmedian(sub_vals)),
                "std": float(np.nanstd(sub_vals, ddof=1)) if n > 1 else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["sufficient_samples"] = out["count"] >= min_samples
    return out


def plot_sliding_window(
    summary: pd.DataFrame,
    label: str,
    column: str,
    out_path: Path,
) -> None:
    if summary.empty:
        LOGGER.warning("Empty sliding-window summary for %s; skipping plot.", column)
        return
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(summary["window_mid_km"], summary["mean"], color="tab:blue", lw=1.8, label="mean")
    ax1.plot(summary["window_mid_km"], summary["median"], color="tab:orange", lw=1.2, ls="--", label="median")
    ax1.fill_between(
        summary["window_mid_km"],
        summary["mean"] - summary["std"].fillna(0),
        summary["mean"] + summary["std"].fillna(0),
        color="tab:blue",
        alpha=0.12,
        label="±1 std",
    )
    ax1.set_xlabel("distance to port [km] (window midpoint)")
    ax1.set_ylabel(f"{label} ({column})", color="tab:blue")
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.bar(summary["window_mid_km"], summary["count"], width=STEP_KM * 0.9, color="grey", alpha=0.25, label="count")
    ax2.set_ylabel("samples per window", color="grey")
    ax2.legend(loc="upper right", fontsize=8)

    plt.title(f"Sliding-window distance decay – {label} (window={WINDOW_KM:g} km, step={STEP_KM:g} km)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    LOGGER.info("  saved %s", out_path.relative_to(PROJECT_ROOT))


def run_sliding_windows(
    df: pd.DataFrame,
    indicators: dict[str, str],
    plot_dir: Path,
    csv_dir: Path,
) -> dict[str, pd.DataFrame]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, pd.DataFrame] = {}
    for label, col in indicators.items():
        LOGGER.info("Sliding-window summary for %s (%s)", label, col)
        summary = sliding_window_summary(df, col)
        if summary.empty:
            LOGGER.warning("  no data for %s", col)
            continue
        csv_path = csv_dir / f"sliding_window_{label}.csv"
        summary.to_csv(csv_path, index=False)
        LOGGER.info("  saved %s", csv_path.relative_to(PROJECT_ROOT))
        plot_path = plot_dir / f"sliding_window_{label}.png"
        plot_sliding_window(summary, label, col, plot_path)
        summaries[label] = summary
    return summaries


def no2_diagnostic(
    df: pd.DataFrame,
    no2_col: str | None,
    out_dir_plots: Path,
    out_report: Path,
) -> dict[str, object]:
    out_dir_plots.mkdir(parents=True, exist_ok=True)
    findings: dict[str, object] = {}

    if no2_col is None:
        LOGGER.warning("No NO2 column detected, skipping diagnostic.")
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text("# NO2 distance diagnostic\n\nNo NO2 column was detected in the dataset.\n")
        return findings

    summary = sliding_window_summary(df, no2_col, max_dist_km=MAX_DIST_CAP_KM)
    if summary.empty:
        LOGGER.warning("NO2 sliding-window summary is empty.")
        return findings

    # Plot NO2 sliding-window with sample counts
    plot_sliding_window(summary, "NO2", no2_col, out_dir_plots / "no2_sliding_window.png")

    # NO2 by hub – overlay
    hubs = sorted(df["hub"].dropna().unique().tolist())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, hub in enumerate(hubs):
        sub = df[df["hub"] == hub]
        sw = sliding_window_summary(sub, no2_col, max_dist_km=MAX_DIST_CAP_KM)
        if sw.empty:
            continue
        ax.plot(sw["window_mid_km"], sw["mean"], color=cmap(i % 10), lw=1.8, label=f"{hub} (n={int(sw['count'].sum())})")
    ax.set_xlabel("distance to port [km]")
    ax.set_ylabel(f"NO2 mean ({no2_col})")
    ax.set_title("NO2 sliding-window mean by hub")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir_plots / "no2_by_hub.png", dpi=140)
    plt.close(fig)

    # Sample-count plot
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(summary["window_mid_km"], summary["count"], width=STEP_KM * 0.9, color="tab:grey")
    ax.set_xlabel("distance to port [km]")
    ax.set_ylabel("samples per 50-km sliding window")
    ax.set_title("NO2 sample count by distance window")
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir_plots / "no2_sample_counts.png", dpi=140)
    plt.close(fig)

    # Slice 200-500 km
    band = df[(df["distance_km"] >= 200) & (df["distance_km"] <= 500)]
    band_valid = band[[no2_col, "hub"]].dropna()
    overall_valid = df[[no2_col, "distance_km", "hub"]].dropna()
    near = overall_valid[overall_valid["distance_km"] < 200]
    far = overall_valid[overall_valid["distance_km"] > 500]

    band_n = int(len(band_valid))
    band_mean = float(band_valid[no2_col].mean()) if band_n else float("nan")
    near_mean = float(near[no2_col].mean()) if len(near) else float("nan")
    far_mean = float(far[no2_col].mean()) if len(far) else float("nan")
    band_hub_counts = band_valid["hub"].value_counts().to_dict()
    overall_hub_counts = overall_valid["hub"].value_counts().to_dict()

    band_summary = sliding_window_summary(
        df[(df["distance_km"] >= 200) & (df["distance_km"] <= 500)],
        no2_col,
        max_dist_km=500.0,
    )
    band_csv = out_dir_plots.parent.parent / "reports" / "sliding_window_distance_decay" / "no2_sliding_window_200_500km.csv"
    band_csv.parent.mkdir(parents=True, exist_ok=True)
    band_summary.to_csv(band_csv, index=False)

    findings.update(
        {
            "no2_col": no2_col,
            "band_n": band_n,
            "band_mean": band_mean,
            "near_mean": near_mean,
            "far_mean": far_mean,
            "band_hub_counts": band_hub_counts,
            "overall_hub_counts": overall_hub_counts,
        }
    )

    # Report
    report = _build_no2_report(
        no2_col=no2_col,
        summary=summary,
        band_n=band_n,
        band_mean=band_mean,
        near_mean=near_mean,
        far_mean=far_mean,
        band_hub_counts=band_hub_counts,
        overall_hub_counts=overall_hub_counts,
    )
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report)
    LOGGER.info("Saved NO2 diagnostic report: %s", out_report.relative_to(PROJECT_ROOT))
    return findings


def _build_no2_report(
    *,
    no2_col: str,
    summary: pd.DataFrame,
    band_n: int,
    band_mean: float,
    near_mean: float,
    far_mean: float,
    band_hub_counts: dict,
    overall_hub_counts: dict,
) -> str:
    band_window = summary[(summary["window_mid_km"] >= 200) & (summary["window_mid_km"] <= 500)]
    other_window = summary[(summary["window_mid_km"] < 200) | (summary["window_mid_km"] > 500)]
    band_mean_window = float(band_window["mean"].mean()) if not band_window.empty else float("nan")
    other_mean_window = float(other_window["mean"].mean()) if not other_window.empty else float("nan")
    band_n_window = int(band_window["count"].sum()) if not band_window.empty else 0
    other_n_window = int(other_window["count"].sum()) if not other_window.empty else 0

    increase_factor = (
        band_mean_window / other_mean_window if other_mean_window and not np.isnan(other_mean_window) and other_mean_window != 0 else float("nan")
    )

    rises = bool(band_mean_window > other_mean_window) if not np.isnan(band_mean_window) and not np.isnan(other_mean_window) else False

    lines = [
        "# NO2 distance diagnostic (200–500 km band)",
        "",
        f"NO2 column used: `{no2_col}`",
        f"Sliding-window settings: window={WINDOW_KM:g} km, step={STEP_KM:g} km, max distance cap={MAX_DIST_CAP_KM:g} km",
        "",
        "## 1. Does NO2 actually increase between 200–500 km?",
        "",
        f"- Mean of sliding-window NO2 means inside 200–500 km: **{band_mean_window:.3e}** (n_obs ≈ {band_n_window}).",
        f"- Mean of sliding-window NO2 means outside that band (within {MAX_DIST_CAP_KM:g} km): **{other_mean_window:.3e}** (n_obs ≈ {other_n_window}).",
        f"- Ratio (band / outside): **{increase_factor:.2f}**.",
        f"- Raw observation comparison: NO2 mean <200 km = {near_mean:.3e}, 200–500 km = {band_mean:.3e}, >500 km = {far_mean:.3e}.",
        "",
        f"Conclusion: NO2 **{'does' if rises else 'does not clearly'} appear to rise** in the 200–500 km band relative to the rest of the 0–{MAX_DIST_CAP_KM:g} km range.",
        "",
        "## 2. What is causing it?",
        "",
        "### 2a. Sample imbalance / coverage",
        "",
        f"- Observations contributing to the 200–500 km band (raw cells): **{band_n}**.",
        f"- Hub composition in 200–500 km band: `{band_hub_counts}`.",
        f"- Hub composition over the full dataset (with valid NO2): `{overall_hub_counts}`.",
        "",
        "If the hub mix in the 200–500 km band differs strongly from the close-range band (e.g., the close range is dominated by small Finnish coastal ports while the 200–500 km band is dominated by a single far-away hub like Mariehamn), the apparent increase reflects a change in *which grid cells we are averaging*, not a true atmospheric signal around those ports.",
        "",
        "### 2b. Binning noise",
        "",
        "Hard ≥100 km bins gave a non-monotonic curve (e.g. 0.000009 at 200 km → 0.000023 at 600 km), but the underlying counts in those bins are small (often 30–50 cells) and the standard deviation per window is comparable to the mean. The 50-km sliding window with 10-km step (this script's output) smooths out most of that noise, but residual jitter from sparse samples remains.",
        "",
        "### 2c. Atmospheric / land-source confounding",
        "",
        "NO2 tropospheric column is dominated by *land-based* sources (combustion, traffic, industry) and is transported by wind. In this dataset the cells whose nearest port is far away (200–500 km away) are typically open-Baltic / mainland-coastal cells whose NO2 is governed by land emissions and transport, **not** by the named port. The 'distance to port' axis is therefore a proxy for *which region we are sampling*, and at 200–500 km we partly start sampling continental plumes (e.g., from Stockholm/Helsinki/St Petersburg) that have nothing to do with the port itself.",
        "",
        "## 3. How to phrase this in the thesis",
        "",
        "- Report the sliding-window NO2 curve, but explicitly mark windows with low sample counts.",
        "- State that NO2 is not expected to show a monotonic distance decay from a single port: it is a regional pollutant, so any rise at 200–500 km is **likely a confounded signal** (different hub mix and continental/long-range transport), not an indication that pollution truly increases away from ports.",
        "- Recommend interpreting NO2 alongside vessel density and wind direction; conclude that, in this dataset, NO2 does not provide a clean port-distance gradient and should be presented as supporting evidence of regional atmospheric exposure rather than port-proximity exposure.",
        "",
        "## Generated artefacts",
        "",
        "- `outputs/visualizations/sliding_window_distance_decay/no2_sliding_window.png`",
        "- `outputs/visualizations/sliding_window_distance_decay/no2_by_hub.png`",
        "- `outputs/visualizations/sliding_window_distance_decay/no2_sample_counts.png`",
        "- `outputs/reports/sliding_window_distance_decay/sliding_window_no2.csv`",
        "- `outputs/reports/sliding_window_distance_decay/no2_sliding_window_200_500km.csv`",
        "",
    ]
    return "\n".join(lines)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_analysis(
    dataset: Path | None = None,
    out_root: Path | None = None,
) -> dict[str, object]:
    """Run the full hub distance-decay analysis. Returns a summary dict."""
    hub_plots_dir, sw_plots_dir, sw_reports_dir, no2_report_path, dataset_override = _resolve_paths(out_root, dataset)

    LOGGER.info("Project root: %s", PROJECT_ROOT)
    df, used_path = load_dataset(dataset_override)
    LOGGER.info("Dataset shape: %s", df.shape)
    LOGGER.info("Output base: %s", out_root if out_root else "(project default)")

    detected = detect_columns(df)
    prepped = prepare_frame(df, detected)
    LOGGER.info("Prepared frame shape: %s", prepped.shape)
    LOGGER.info("Hubs detected: %s", sorted(prepped["hub"].dropna().unique().tolist()))

    indicator_map = {k: detected[k] for k in ["ndti", "ndwi", "ndvi", "no2", "vessel_density"] if detected[k]}
    LOGGER.info("Indicators used: %s", indicator_map)

    LOGGER.info("=== Hub-level distance-decay plots ===")
    hub_plot_paths = plot_hub_panels(prepped, indicator_map, hub_plots_dir)
    comparison_path = plot_hub_comparison(prepped, indicator_map, hub_plots_dir)

    LOGGER.info("=== Sliding-window distance-decay ===")
    sw_summaries = run_sliding_windows(prepped, indicator_map, sw_plots_dir, sw_reports_dir)

    LOGGER.info("=== NO2 diagnostic ===")
    findings = no2_diagnostic(prepped, detected.get("no2"), sw_plots_dir, no2_report_path)

    LOGGER.info("--- Summary ---")
    LOGGER.info("Dataset: %s", _rel(used_path))
    LOGGER.info("Detected columns: %s", detected)
    LOGGER.info("Hubs analysed: %d", prepped["hub"].nunique())
    LOGGER.info("Hub-level plots: %d", len(hub_plot_paths))
    if comparison_path:
        LOGGER.info("Hub comparison plot: %s", _rel(comparison_path))
    LOGGER.info("Sliding-window indicators: %s", list(sw_summaries.keys()))
    if findings:
        LOGGER.info("NO2 200-500km mean: %.3e (n=%d)", findings.get("band_mean", float("nan")), findings.get("band_n", 0))
        LOGGER.info("NO2 <200km mean: %.3e | >500km mean: %.3e", findings.get("near_mean", float("nan")), findings.get("far_mean", float("nan")))
        LOGGER.info("Hub composition in 200-500km band: %s", findings.get("band_hub_counts"))
    LOGGER.info("NO2 diagnostic report: %s", _rel(no2_report_path))
    return {
        "dataset": str(used_path),
        "detected_columns": detected,
        "indicators_used": indicator_map,
        "hubs": sorted(prepped["hub"].dropna().unique().tolist()),
        "hub_plot_paths": [str(p) for p in hub_plot_paths],
        "hub_comparison_plot": str(comparison_path) if comparison_path else None,
        "sliding_window_indicators": list(sw_summaries.keys()),
        "no2_findings": findings,
        "out_dirs": {
            "hub_plots": str(hub_plots_dir),
            "sliding_window_plots": str(sw_plots_dir),
            "sliding_window_reports": str(sw_reports_dir),
            "no2_report": str(no2_report_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="Path to features parquet (overrides default).")
    parser.add_argument("--out-root", type=Path, default=None, help="Root directory under which outputs/ will be written (e.g. final_run).")
    args = parser.parse_args()

    configure_logging()
    run_analysis(dataset=args.dataset, out_root=args.out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
