#!/usr/bin/env python3
"""
NO₂ stratified by Sentinel-1 dark-slick *probability proxy* (oil_slick_probability_t).

Strata (grid-week rows with valid oil proxy):
  - oil_high: top 10% of log1p(oil_slick_probability_t)
  - oil_low:  bottom 10%

Compare no2_mean_t between strata. Higher NO₂ when oil proxy is high → consistent with a
**localized maritime / ocean-surface signal in the stack**, with heavy caveats (urban NO₂
confounding, proxy ≠ confirmed spill).

Outputs:
  outputs/reports/run_no2_oil_slick_combo/no2_oil_stratified_summary.csv
  outputs/figures/run_no2_oil_slick_combo/no2_by_oil_slick_extreme.png

Run:
  python3 src/analysis/no2_oil_slick_stratified.py \\
    --input final_run_stockholm_fixed_20260505_1356/processed/features_ml_ready.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUN = "run_no2_oil_slick_combo"
REPORTS = PROJECT_ROOT / "outputs" / "reports" / RUN
FIGURES = PROJECT_ROOT / "outputs" / "figures" / RUN

OIL_COL = "oil_slick_probability_t"
NO2_COL = "no2_mean_t"


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1))
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((float(np.mean(a)) - float(np.mean(b))) / pooled)


def main() -> int:
    p = argparse.ArgumentParser(description="NO2 when oil proxy high vs low")
    p.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "final_run_stockholm_fixed_20260505_1356" / "processed" / "features_ml_ready.parquet",
    )
    args = p.parse_args()
    inp = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    if not inp.is_file():
        print(f"[error] Missing {inp}", file=sys.stderr)
        return 1

    df = pd.read_parquet(inp)
    if OIL_COL not in df.columns or NO2_COL not in df.columns:
        print(f"[error] Need columns {OIL_COL}, {NO2_COL}", file=sys.stderr)
        return 1

    oil = pd.to_numeric(df[OIL_COL], errors="coerce")
    no2 = pd.to_numeric(df[NO2_COL], errors="coerce")
    sub = df.loc[oil.notna()].copy()
    sub["_loil"] = np.log1p(oil.loc[sub.index].clip(lower=0))
    sub["_no2"] = no2.loc[sub.index]

    q90 = sub["_loil"].quantile(0.90)
    q10 = sub["_loil"].quantile(0.10)
    sub["oil_extreme"] = np.select(
        [sub["_loil"] >= q90, sub["_loil"] <= q10],
        ["oil_high", "oil_low"],
        default="mid",
    )

    hi = sub.loc[sub["oil_extreme"] == "oil_high", "_no2"].dropna().to_numpy()
    lo = sub.loc[sub["oil_extreme"] == "oil_low", "_no2"].dropna().to_numpy()

    row = {
        "stratification": "top10_bottom10_log1p_oil_slick_probability_t",
        "oil_q90_log1p": q90,
        "oil_q10_log1p": q10,
        "n_oil_high_no2": len(hi),
        "n_oil_low_no2": len(lo),
        "NO2_mean_when_oil_high": float(np.mean(hi)) if len(hi) else float("nan"),
        "NO2_mean_when_oil_low": float(np.mean(lo)) if len(lo) else float("nan"),
        "NO2_median_when_oil_high": float(np.median(hi)) if len(hi) else float("nan"),
        "NO2_median_when_oil_low": float(np.median(lo)) if len(lo) else float("nan"),
        "mean_diff_high_minus_low": float(np.mean(hi) - np.mean(lo)) if len(hi) and len(lo) else float("nan"),
        "cohens_d_high_vs_low": _cohens_d(hi, lo),
        "welch_t_p": float(stats.ttest_ind(hi, lo, equal_var=False).pvalue) if len(hi) >= 3 and len(lo) >= 3 else float("nan"),
        "mann_whitney_p": float(stats.mannwhitneyu(hi, lo, alternative="two-sided").pvalue)
        if len(hi) >= 3 and len(lo) >= 3
        else float("nan"),
        "no2_higher_when_oil_high": bool(np.mean(hi) > np.mean(lo)) if len(hi) and len(lo) else False,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS / "no2_oil_stratified_summary.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)

    plot_df = sub.loc[sub["oil_extreme"].isin(["oil_high", "oil_low"]), ["oil_extreme", "_no2"]].dropna()
    plot_df = plot_df.rename(columns={"_no2": NO2_COL, "oil_extreme": "Oil slick proxy (Sentinel-1)"})
    plot_df["Oil slick proxy (Sentinel-1)"] = plot_df["Oil slick proxy (Sentinel-1)"].map(
        {"oil_high": "High (top 10%)", "oil_low": "Low (bottom 10%)"}
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    order = ["High (top 10%)", "Low (bottom 10%)"]
    sns.boxplot(
        data=plot_df,
        x="Oil slick proxy (Sentinel-1)",
        y=NO2_COL,
        order=order,
        hue="Oil slick proxy (Sentinel-1)",
        hue_order=order,
        palette="Set2",
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df.sample(min(4000, len(plot_df))),
        x="Oil slick proxy (Sentinel-1)",
        y=NO2_COL,
        order=order,
        color="0.35",
        alpha=0.08,
        size=1.5,
        ax=ax,
    )
    ax.set_title("NO₂ tropospheric column when oil-slick *probability* proxy is high vs low")
    ax.set_xlabel("")
    fig.tight_layout()
    out_png = FIGURES / "no2_by_oil_slick_extreme.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print("=== NO₂ vs oil-slick probability proxy (Sentinel-1) ===")
    print(f"Input rows: {len(df)} | rows with oil proxy: {oil.notna().sum()}")
    print(f"NO2_mean when oil_high: {row['NO2_mean_when_oil_high']:.6e}  (n={row['n_oil_high_no2']})")
    print(f"NO2_mean when oil_low:  {row['NO2_mean_when_oil_low']:.6e}  (n={row['n_oil_low_no2']})")
    print(f"Difference (high − low): {row['mean_diff_high_minus_low']:.6e}")
    print(f"Cohen d: {row['cohens_d_high_vs_low']:.4f} | Welch p: {row['welch_t_p']:.4e} | MW p: {row['mann_whitney_p']:.4e}")
    if row["no2_higher_when_oil_high"]:
        print("Interpretation: NO₂ is **higher** when the oil-slick proxy is high → *consistent* with a stacked maritime-associated signal (not causal proof).")
    else:
        print("Interpretation: NO₂ is **not** higher when oil proxy is high in this split → weak / mixed maritime coupling in the available columns.")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
