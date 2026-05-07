"""Compare NO2, NDTI, NDWI between highest- and lowest-traffic zones by vessel density.

Defines "high traffic" as the top 10% of rows by log1p(vessel_density) and
"low traffic" as the bottom 10% (non-overlapping middle 80% is excluded).

Outputs a CSV summary (means, tests, effect sizes) and a 3-panel figure.

Run:
    python3 src/analysis/traffic_top10_bottom10_compare.py
    python3 src/analysis/traffic_top10_bottom10_compare.py \\
        --input final_run_stockholm_fixed_20260505_1356/processed/features_ml_ready.parquet \\
        --out-root final_run_stockholm_fixed_20260505_1356
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

LOGGER = logging.getLogger("traffic_top10_bottom10")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VESSEL_CANDIDATES = ["vessel_density_t", "vessel_density"]
NO2_CANDIDATES = ["no2_mean_t", "no2_tropospheric_column_mean_t", "NO2_mean", "no2_mean"]
NDTI_CANDIDATES = ["sentinel_ndti_mean_t", "ndti_mean", "ndti_median"]
NDWI_CANDIDATES = ["sentinel_ndwi_mean_t", "ndwi_mean", "ndwi_median"]

RUN_TAG = "traffic_top10_vessel_compare"


def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1))
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((float(np.mean(a)) - float(np.mean(b))) / pooled)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top/bottom 10% vessel density vs NO2, NDTI, NDWI.")
    p.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "processed" / "features_ml_ready.parquet",
        help="Path to features_ml_ready.parquet",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Optional run root (e.g. final_run_stockholm_fixed_...). Default: project root.",
    )
    return p.parse_args()


def resolve_out_dirs(out_root: Path | None) -> tuple[Path, Path]:
    root = Path(out_root) if out_root else PROJECT_ROOT
    return root / "outputs" / "reports" / RUN_TAG, root / "outputs" / "visualizations" / RUN_TAG


def assign_traffic_extremes(df: pd.DataFrame, vessel_col: str) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    v = pd.to_numeric(out[vessel_col], errors="coerce")
    logv = np.log1p(v.clip(lower=0))
    q_hi = logv.quantile(0.90)
    q_lo = logv.quantile(0.10)
    out["_log_vessel"] = logv
    out["traffic_extreme"] = np.select(
        [logv >= q_hi, logv <= q_lo],
        ["high_top10", "low_bottom10"],
        default="mid",
    )
    meta = {"q90_log1p": float(q_hi), "q10_log1p": float(q_lo)}
    return out, meta


def compare_groups(
    df: pd.DataFrame,
    outcome_col: str,
    label: str,
) -> dict[str, Any]:
    sub = df[df["traffic_extreme"].isin(["high_top10", "low_bottom10"])].copy()
    high = pd.to_numeric(sub.loc[sub["traffic_extreme"] == "high_top10", outcome_col], errors="coerce")
    low = pd.to_numeric(sub.loc[sub["traffic_extreme"] == "low_bottom10", outcome_col], errors="coerce")
    high = high.dropna().to_numpy()
    low = low.dropna().to_numpy()

    row: dict[str, Any] = {
        "outcome": label,
        "column": outcome_col,
        "n_high": len(high),
        "n_low": len(low),
        "high_mean": float(np.mean(high)) if len(high) else float("nan"),
        "low_mean": float(np.mean(low)) if len(low) else float("nan"),
        "high_median": float(np.median(high)) if len(high) else float("nan"),
        "low_median": float(np.median(low)) if len(low) else float("nan"),
        "mean_diff_high_minus_low": float("nan"),
        "cohens_d_high_vs_low": float("nan"),
        "ttest_p": float("nan"),
        "mannwhitney_p": float("nan"),
    }

    if len(high) >= 3 and len(low) >= 3:
        row["mean_diff_high_minus_low"] = float(np.mean(high) - np.mean(low))
        row["cohens_d_high_vs_low"] = _cohens_d(high, low)
        row["ttest_p"] = float(stats.ttest_ind(high, low, equal_var=False, nan_policy="omit").pvalue)
        row["mannwhitney_p"] = float(stats.mannwhitneyu(high, low, alternative="two-sided").pvalue)

    return row


def plot_three_panel(
    df: pd.DataFrame,
    specs: list[tuple[str, str]],
    out_png: Path,
) -> None:
    plot_df = df[df["traffic_extreme"].isin(["high_top10", "low_bottom10"])].copy()
    rows = []
    for short, col in specs:
        if col is None:
            continue
        tmp = plot_df[["traffic_extreme", col]].copy()
        tmp = tmp.rename(columns={col: "value"})
        tmp["indicator"] = short
        tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
        rows.append(tmp.dropna(subset=["value"]))
    if not rows:
        LOGGER.warning("No data to plot.")
        return
    long_df = pd.concat(rows, ignore_index=True)
    long_df["Traffic"] = long_df["traffic_extreme"].map(
        {"high_top10": "Top 10% vessel", "low_bottom10": "Bottom 10% vessel"}
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    order = ["Top 10% vessel", "Bottom 10% vessel"]
    for ax, short in zip(axes, ["NO2", "NDTI", "NDWI"]):
        sub = long_df[long_df["indicator"] == short]
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=sub,
            x="Traffic",
            y="value",
            order=order,
            hue="Traffic",
            hue_order=order,
            ax=ax,
            palette="Set2",
            legend=False,
        )
        sns.stripplot(
            data=sub,
            x="Traffic",
            y="value",
            order=order,
            ax=ax,
            color="0.35",
            alpha=0.15,
            size=1.5,
            dodge=False,
        )
        ax.set_title(short)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=12)

    fig.suptitle("NO2, NDTI, NDWI: top 10% vs bottom 10% vessel-density zones", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved figure: %s", out_png)


def main() -> int:
    configure_logging()
    args = parse_args()
    data_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    if not data_path.is_file():
        LOGGER.error("Input not found: %s", data_path)
        return 1

    reports_dir, viz_dir = resolve_out_dirs(args.out_root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    LOGGER.info("Loaded %s rows from %s", len(df), data_path)

    vessel_col = _first_present(df, VESSEL_CANDIDATES)
    if not vessel_col:
        LOGGER.error("No vessel density column found (tried %s).", VESSEL_CANDIDATES)
        return 1

    no2_col = _first_present(df, NO2_CANDIDATES)
    ndti_col = _first_present(df, NDTI_CANDIDATES)
    ndwi_col = _first_present(df, NDWI_CANDIDATES)
    LOGGER.info(
        "Columns: vessel=%s | NO2=%s | NDTI=%s | NDWI=%s",
        vessel_col,
        no2_col,
        ndti_col,
        ndwi_col,
    )

    df, qmeta = assign_traffic_extremes(df, vessel_col)
    n_high = int((df["traffic_extreme"] == "high_top10").sum())
    n_low = int((df["traffic_extreme"] == "low_bottom10").sum())
    n_mid = int((df["traffic_extreme"] == "mid").sum())
    LOGGER.info(
        "Traffic groups (log1p %s): high_top10=%d low_bottom10=%d mid=%d | q90=%.6f q10=%.6f",
        vessel_col,
        n_high,
        n_low,
        n_mid,
        qmeta["q90_log1p"],
        qmeta["q10_log1p"],
    )

    summary_rows = []
    specs: list[tuple[str, str]] = []
    for label, col in [("NO2", no2_col), ("NDTI", ndti_col), ("NDWI", ndwi_col)]:
        if col is None:
            LOGGER.warning("Skipping %s — no column found.", label)
            continue
        summary_rows.append(compare_groups(df, col, label))
        specs.append((label, col))

    summary = pd.DataFrame(summary_rows)
    summary["vessel_column"] = vessel_col
    summary["vessel_rule"] = "high: log1p(vessel) >= q90; low: log1p(vessel) <= q10"
    summary["n_rows_vessel_nonnull"] = int(pd.to_numeric(df[vessel_col], errors="coerce").notna().sum())
    summary["n_strata_high_top10"] = n_high
    summary["n_strata_low_bottom10"] = n_low
    for k, v in qmeta.items():
        summary[f"meta_{k}"] = v

    csv_path = reports_dir / "high_vs_low_vessel_top10_bottom10_summary.csv"
    summary.to_csv(csv_path, index=False)
    LOGGER.info("Wrote %s", csv_path)

    out_png = viz_dir / "no2_ndti_ndwi_top10_vs_bottom10_vessel.png"
    plot_three_panel(df, specs, out_png)

    return 0


if __name__ == "__main__":
    sys.exit(main())
