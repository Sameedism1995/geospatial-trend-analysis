#!/usr/bin/env python3
"""Generate correlation heatmap, environmental distributions, and temporal trend figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "processed" / "features_ml_ready.parquet"
OUT = ROOT / "outputs" / "thesis" / "figures"
DPI = 300
MIN_COVERAGE = 0.08


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


def correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    exclude = {
        "grid_cell_id",
        "week_start_utc",
        "nearest_port",
        "coastal_exposure_band",
    }
    numeric = []
    n = len(df)
    for c in df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cov = df[c].notna().sum() / n if n else 0
        if cov >= MIN_COVERAGE:
            numeric.append(c)

    sub = df[numeric].copy()
    corr = sub.corr(method="pearson", min_periods=30)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        square=False,
        linewidths=0.35,
        linecolor="#ffffff",
        cbar_kws={"shrink": 0.65, "label": "Pearson r"},
        vmin=-1,
        vmax=1,
    )
    plt.title(
        "Pearson correlation — numeric ML-ready panel features\n(pairwise complete; columns ≥ {:.0f}% coverage)".format(
            MIN_COVERAGE * 100
        ),
        fontsize=12,
        pad=14,
    )
    plt.xticks(rotation=55, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def environmental_distributions(df: pd.DataFrame, path: Path) -> None:
    env_cols = [
        "ndwi_mean",
        "ndti_mean",
        "ndci_mean",
        "fai_mean",
        "b11_mean",
        "ndvi_mean",
        "detection_score",
    ]
    cols = [c for c in env_cols if c in df.columns]

    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for ax, col in zip(axes, cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 30:
            ax.text(0.5, 0.5, f"{col}\n(insufficient data)", ha="center", va="center")
            ax.axis("off")
            continue
        sns.histplot(s, kde=True, ax=ax, color="#2e86ab", edgecolor="white", linewidth=0.4)
        ax.set_title(col.replace("_", " "), fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    for j in range(len(cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Environmental / optical indicators (distribution of weekly grid-level values)\nfeatures_ml_ready.parquet — rows with valid measurements only per subplot",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def temporal_trends(df: pd.DataFrame, path: Path) -> None:
    df = df.copy()
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")

    trend_cols = [
        ("ndti_mean", "NDTI mean"),
        ("ndwi_mean", "NDWI mean"),
        ("NO2_mean", "NO2 mean"),
        ("vessel_density", "Vessel density"),
        ("maritime_pressure_index", "Maritime pressure index"),
        ("coastal_exposure_score", "Coastal exposure score"),
    ]

    weekly_rows = []
    for col, label in trend_cols:
        if col not in df.columns:
            continue
        g = df.groupby("week_start_utc", observed=True)[col].agg(["mean", "median", "count"]).reset_index()
        g["variable"] = label
        g["column"] = col
        weekly_rows.append(g.rename(columns={"mean": "mean_val", "median": "median_val", "count": "n_cells"}))

    if not weekly_rows:
        return

    long_df = pd.concat(weekly_rows, ignore_index=True)

    fig, axes = plt.subplots(len(trend_cols), 1, figsize=(11, 2.8 * len(trend_cols)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (col, label) in zip(axes, trend_cols):
        if col not in df.columns:
            ax.axis("off")
            continue
        sub = long_df[long_df["column"] == col].sort_values("week_start_utc")
        if sub.empty:
            ax.text(0.5, 0.5, f"{label}: no data", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.plot(sub["week_start_utc"], sub["mean_val"], color="#c0392b", lw=1.8, marker="o", ms=3, label="Spatial mean")
        ax.plot(
            sub["week_start_utc"],
            sub["median_val"],
            color="#2980b9",
            lw=1.2,
            ls="--",
            alpha=0.85,
            label="Spatial median",
        )
        ax.fill_between(
            sub["week_start_utc"],
            sub["mean_val"],
            sub["median_val"],
            alpha=0.12,
            color="grey",
        )
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Week start (UTC)", fontsize=10)
    plt.xticks(rotation=35, ha="right")
    fig.suptitle(
        "Temporal trends — weekly aggregation across grid cells\n(mean vs median highlight skew when optical coverage is sparse)",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = load_panel()

    correlation_heatmap(df, OUT / "correlation_heatmap_ml_ready.png")
    environmental_distributions(df, OUT / "environmental_distributions.png")
    temporal_trends(df, OUT / "temporal_trends_weekly_means.png")

    for p in sorted(OUT.glob("*.png")):
        if p.name.startswith(("correlation_", "environmental_", "temporal_")):
            print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
