#!/usr/bin/env python3
"""
Standalone figures from existing run_land_pollution_drivers_wind CSV outputs +
linked parquet joins (NO2, oil, vessel density).

Writes PNGs under outputs/figures/run_land_pollution_drivers_wind/

Usage:
  python3 src/analysis/plot_wind_pollution_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

RUN = "run_land_pollution_drivers_wind"
REPORTS = _ROOT / "outputs" / "reports" / RUN
FIGURES = _ROOT / "outputs" / "figures" / RUN
LINKED = _ROOT / "outputs/reports/run_nearest_land_ndvi_linkage/nearest_land_ndvi_linked_dataset.parquet"


def _norm_week(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.normalize()


def load_merged_panel() -> pd.DataFrame:
    wa = REPORTS / "wind_alignment_features.csv"
    no2_path = REPORTS / "no2_excess_features.csv"
    if not wa.is_file():
        raise FileNotFoundError(wa)

    ali = pd.read_csv(wa)
    ali["week_start_utc"] = _norm_week(ali["week_start_utc"])
    ali["grid_cell_id"] = ali["grid_cell_id"].astype(str)

    if no2_path.is_file():
        n2 = pd.read_csv(no2_path)
        n2["week_start_utc"] = _norm_week(n2["week_start_utc"])
        n2["grid_cell_id"] = n2["grid_cell_id"].astype(str)
        extra = [c for c in n2.columns if c not in {"grid_cell_id", "week_start_utc"}]
        ali = ali.merge(n2[["grid_cell_id", "week_start_utc", *extra]], on=["grid_cell_id", "week_start_utc"], how="left")

    if LINKED.is_file():
        cols = ["grid_cell_id", "week_start_utc", "oil_slick_probability_t", "vessel_density_t"]
        try:
            pan = pd.read_parquet(LINKED, columns=cols)
        except Exception:
            pan = pd.read_parquet(LINKED)
            cols_exist = [c for c in cols if c in pan.columns]
            pan = pan[cols_exist]
        pan["week_start_utc"] = _norm_week(pan["week_start_utc"])
        pan["grid_cell_id"] = pan["grid_cell_id"].astype(str)
        ali = ali.merge(pan, on=["grid_cell_id", "week_start_utc"], how="left")

    num_cols = [
        "wind_speed_mean",
        "wind_direction_to_degrees",
        "wind_alignment_score",
        "wind_aligned_to_land",
        "local_no2_excess",
        "weekly_no2_anomaly",
        "oil_slick_probability_t",
        "vessel_density_t",
        "bearing_lane_centroid_deg",
    ]
    for c in num_cols:
        if c in ali.columns:
            ali[c] = pd.to_numeric(ali[c], errors="coerce")

    return ali


def annotate_spearman(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    m = x.notna() & y.notna()
    if int(m.sum()) < 8:
        return
    r, p = stats.spearmanr(x.loc[m], y.loc[m])
    ax.text(
        0.02,
        0.98,
        f"Spearman ρ = {r:.3f}\np = {p:.3g}\nn = {int(m.sum())}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.85"),
    )


def plot_wind_speed(df: pd.DataFrame) -> None:
    w = df["wind_speed_mean"].dropna()
    if w.empty:
        return

    weekly = df.groupby("week_start_utc", observed=True)["wind_speed_mean"].agg(["mean", "std", "count"])

    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = pd.to_datetime(weekly.index)
    ax.plot(x, weekly["mean"], color="steelblue", lw=2, label="Mean across grids")
    if (weekly["std"].notna() & (weekly["count"] > 1)).any():
        ax.fill_between(
            x,
            weekly["mean"] - weekly["std"],
            weekly["mean"] + weekly["std"],
            color="steelblue",
            alpha=0.2,
            label="±1 std (within week)",
        )
    ax.set_ylabel("Wind speed mean (hourly avg, Open-Meteo units)")
    ax.set_xlabel("Week start (UTC)")
    ax.set_title("Weekly wind speed (coastal panel rows)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "wind_speed_weekly_timeseries.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(w, bins=40, kde=True, ax=ax, color="steelblue")
    ax.set_xlabel("wind_speed_mean")
    ax.set_title("Wind speed distribution (all panel × weeks)")
    fig.tight_layout()
    fig.savefig(FIGURES / "wind_speed_distribution.png", dpi=160)
    plt.close(fig)


def plot_wind_direction(df: pd.DataFrame) -> None:
    d = pd.to_numeric(df["wind_direction_to_degrees"], errors="coerce").dropna()
    if d.empty:
        return

    bins = np.linspace(0, 360, 17)
    counts, edges = np.histogram(d.to_numpy(), bins=bins)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    widths = np.diff(edges)
    theta = np.radians(edges[:-1] + widths / 2)
    bars = ax.bar(theta, counts, width=np.radians(widths[0]), bottom=0, alpha=0.85, color="teal", edgecolor="white", linewidth=0.5)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Wind direction toward (degrees clockwise from north)", pad=22)
    fig.tight_layout()
    fig.savefig(FIGURES / "wind_direction_rose.png", dpi=160)
    plt.close(fig)

    # Weekly resultant direction from mean u,v (avoids circular mean pitfalls on angles alone)
    u = pd.to_numeric(df["wind_u_mean"], errors="coerce")
    v = pd.to_numeric(df["wind_v_mean"], errors="coerce")
    sub = df.assign(_u=u, _v=v).dropna(subset=["_u", "_v", "week_start_utc"])
    if sub.empty:
        return
    agg = sub.groupby("week_start_utc", observed=True).agg(mu=("_u", "mean"), mv=("_v", "mean"))
    ang = (np.degrees(np.arctan2(agg["mu"].to_numpy(), agg["mv"].to_numpy())) + 360.0) % 360.0

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.scatter(
        pd.to_datetime(agg.index),
        ang,
        s=22,
        c="darkslateblue",
        alpha=0.8,
        zorder=3,
    )
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylim(-5, 365)
    ax.set_ylabel("Resultant ° from N (mean u,v)")
    ax.set_xlabel("Week start (UTC)")
    ax.set_title("Weekly mean wind vector direction (panel-wide mean u,v)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / "wind_direction_weekly_resultant.png", dpi=160)
    plt.close(fig)


def plot_alignment_pollution_flow(df: pd.DataFrame) -> None:
    s = pd.to_numeric(df["wind_alignment_score"], errors="coerce").dropna()
    if len(s):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(s, bins=40, kde=True, ax=ax, color="darkorange")
        ax.axvline(0.5, color="crimson", ls="--", lw=1, label="cos(60°)")
        ax.set_xlabel(r"Wind alignment score = cos($\Delta$ angle), lane seed → grid vs wind toward")
        ax.set_title("Wind alignment vs shipping-lane centroid bearing")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "wind_alignment_score_distribution.png", dpi=160)
        plt.close(fig)

    if "wind_alignment_category" in df.columns and len(df):
        vc = df["wind_alignment_category"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        vc.plot(kind="bar", ax=ax, color=["seagreen", "goldenrod", "indianred"][: len(vc)])
        ax.set_ylabel("Count (grid × week)")
        ax.set_title("Wind alignment category")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(FIGURES / "wind_alignment_category_counts.png", dpi=160)
        plt.close(fig)

    vd = pd.to_numeric(df["vessel_density_t"], errors="coerce") if "vessel_density_t" in df.columns else pd.Series(dtype=float)
    al = pd.to_numeric(df["wind_alignment_score"], errors="coerce")
    m = vd.notna() & al.notna()
    if int(m.sum()) > 50:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        hb = ax.hexbin(vd.loc[m], al.loc[m], gridsize=28, cmap="YlGnBu", mincnt=1)
        plt.colorbar(hb, ax=ax, label="count")
        ax.set_xlabel("Vessel density (t) - maritime activity proxy")
        ax.set_ylabel("Wind alignment score")
        ax.set_title("Wind alignment vs vessel density")
        annotate_spearman(ax, vd.loc[m], al.loc[m])
        fig.tight_layout()
        fig.savefig(FIGURES / "wind_alignment_vs_vessel_density.png", dpi=160)
        plt.close(fig)

    bd = pd.to_numeric(df["bearing_lane_centroid_deg"], errors="coerce")
    wt = pd.to_numeric(df["wind_direction_to_degrees"], errors="coerce")
    m2 = bd.notna() & wt.notna()
    if int(m2.sum()) > 50:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(bd.loc[m2], wt.loc[m2], s=8, alpha=0.35, c="steelblue")
        ax.plot([0, 360], [0, 360], ls=":", color="gray", lw=1, label="1:1")
        ax.set_xlabel("Bearing lane centroid to grid (deg from N)")
        ax.set_ylabel("Wind toward direction (deg from N)")
        ax.set_title("Pollution-path bearing vs wind direction")
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "bearing_vs_wind_direction_scatter.png", dpi=160)
        plt.close(fig)


def plot_no2_wind(df: pd.DataFrame) -> None:
    ys = []
    if "local_no2_excess" in df.columns:
        ys.append("local_no2_excess")
    if "weekly_no2_anomaly" in df.columns:
        ys.append("weekly_no2_anomaly")

    for ycol in ys:
        y = pd.to_numeric(df[ycol], errors="coerce")
        ws = pd.to_numeric(df["wind_speed_mean"], errors="coerce")
        al = pd.to_numeric(df["wind_alignment_score"], errors="coerce")

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        m1 = ws.notna() & y.notna()
        if int(m1.sum()) > 30:
            hb = axes[0].hexbin(ws.loc[m1], y.loc[m1], gridsize=30, cmap="viridis", mincnt=1)
            plt.colorbar(hb, ax=axes[0], label="count")
            annotate_spearman(axes[0], ws.loc[m1], y.loc[m1])
        axes[0].set_xlabel("Wind speed mean")
        axes[0].set_ylabel(ycol.replace("_", " "))
        axes[0].set_title("NO2 vs wind speed")

        m2 = al.notna() & y.notna()
        if int(m2.sum()) > 30:
            hb2 = axes[1].hexbin(al.loc[m2], y.loc[m2], gridsize=30, cmap="magma", mincnt=1)
            plt.colorbar(hb2, ax=axes[1], label="count")
            annotate_spearman(axes[1], al.loc[m2], y.loc[m2])
        axes[1].set_xlabel("Wind alignment score")
        axes[1].set_title("NO2 vs wind alignment")

        fig.suptitle(f"NO2-wind relationships - {ycol}", fontsize=12, y=1.02)
        fig.tight_layout()
        safe = ycol.replace(" ", "_")
        fig.savefig(FIGURES / f"no2_wind_hexbin_{safe}.png", dpi=160)
        plt.close(fig)


def plot_oil_wind(df: pd.DataFrame) -> None:
    if "oil_slick_probability_t" not in df.columns:
        return
    oil = pd.to_numeric(df["oil_slick_probability_t"], errors="coerce")
    ws = pd.to_numeric(df["wind_speed_mean"], errors="coerce")
    al = pd.to_numeric(df["wind_alignment_score"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    m1 = ws.notna() & oil.notna()
    if int(m1.sum()) > 30:
        hb = axes[0].hexbin(ws.loc[m1], oil.loc[m1], gridsize=30, cmap="YlOrRd", mincnt=1)
        plt.colorbar(hb, ax=axes[0], label="count")
        annotate_spearman(axes[0], ws.loc[m1], oil.loc[m1])
    axes[0].set_xlabel("Wind speed mean")
    axes[0].set_ylabel("Oil slick probability (transformed)")
    axes[0].set_title("Coastal oil proxy vs wind speed")

    m2 = al.notna() & oil.notna()
    if int(m2.sum()) > 30:
        hb2 = axes[1].hexbin(al.loc[m2], oil.loc[m2], gridsize=30, cmap="YlOrRd", mincnt=1)
        plt.colorbar(hb2, ax=axes[1], label="count")
        annotate_spearman(axes[1], al.loc[m2], oil.loc[m2])
    axes[1].set_xlabel("Wind alignment score")
    axes[1].set_title("Coastal risk context vs alignment")

    fig.suptitle("Oil slick proxy + wind (coastal panel)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "oil_slick_wind_hexbin_coastal_risk.png", dpi=160)
    plt.close(fig)

    if "wind_aligned_to_land" in df.columns:
        sub = df[df["wind_aligned_to_land"].notna()].copy()
        sub["_g"] = np.where(sub["wind_aligned_to_land"].eq(1.0), "Toward land ±45°", "Not aligned")
        sub["_oil"] = pd.to_numeric(sub["oil_slick_probability_t"], errors="coerce")
        sub = sub.dropna(subset=["_oil"])
        if len(sub) > 30:
            fig, ax = plt.subplots(figsize=(5.8, 4.2))
            sns.boxplot(data=sub, x="_g", y="_oil", hue="_g", palette="Set2", ax=ax, legend=False)
            ax.set_title("Oil slick proxy: wind-aligned coastal risk split")
            ax.set_xlabel("")
            fig.tight_layout()
            fig.savefig(FIGURES / "oil_slick_coastal_risk_by_wind_aligned.png", dpi=160)
            plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.95)

    try:
        df = load_merged_panel()
    except FileNotFoundError as e:
        print(f"[FATAL] {e}")
        return 1

    print(f"[INFO] Merged rows: {len(df)} columns: {df.shape[1]}")
    plot_wind_speed(df)
    plot_wind_direction(df)
    plot_alignment_pollution_flow(df)
    plot_no2_wind(df)
    plot_oil_wind(df)
    imgs = sorted(FIGURES.glob("*.png"))
    print(f"[OK] wrote {len(imgs)} PNGs → {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
