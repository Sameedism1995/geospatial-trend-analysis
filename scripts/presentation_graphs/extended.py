"""Exposure, wind, temporal persistence, ML, validation, anomalies, comparisons, summaries."""

from __future__ import annotations

import subprocess
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from shapely.geometry import box

from .common import (
    BALTIC_LAT,
    BALTIC_LON,
    PORT_COORDS,
    agg_cell_mean,
    load_features,
    load_modeling,
    load_ne_land,
    load_model_results,
    load_predictions,
    save_dual,
    trim_scale,
    ROOT,
)


def _features_cell_map(col: str, title: str, stem: str, category: str, cmap: str = "GnBu") -> None:
    fx = load_features()
    if col not in fx.columns:
        raise KeyError(col)
    g = agg_cell_mean(fx, col)
    lo, hi = trim_scale(g["v"])
    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    land = load_ne_land()
    if land is not None:
        b = box(BALTIC_LON[0], BALTIC_LAT[0], BALTIC_LON[1], BALTIC_LAT[1])
        land.clip(gpd.GeoSeries([b], crs="EPSG:4326")).plot(ax=ax, color="#f8fafc", edgecolor="#64748b", lw=0.25, zorder=0)
    sc = ax.scatter(g["lon"], g["lat"], c=g["v"], cmap=cmap, s=22, vmin=lo, vmax=hi, alpha=0.92, edgecolors="none")
    plt.colorbar(sc, ax=ax, shrink=0.56, label=col.replace("_", " "))
    ax.set_aspect("equal")
    ax.set_xlim(BALTIC_LON[0] - 0.12, BALTIC_LON[1] + 0.12)
    ax.set_ylim(BALTIC_LAT[0] - 0.08, BALTIC_LAT[1] + 0.08)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    save_dual(fig, category, stem)


def exposure_maps_and_decay() -> None:
    fx = load_features()
    _features_cell_map(
        "maritime_pressure_index",
        "Maritime exposure proxy (Maritime Pressure Index)",
        "mei_maritime_pressure_map",
        "exposure_analysis",
        cmap="YlGnBu",
    )
    _features_cell_map(
        "coastal_exposure_score",
        "Coastal exposure score",
        "esi_coastal_exposure_score_map",
        "exposure_analysis",
        cmap="GnBu",
    )
    _features_cell_map(
        "land_response_index",
        "Land response index",
        "land_response_index_map",
        "exposure_analysis",
        cmap="PuBuGn",
    )

    # Distance decay — port
    dfp = fx.dropna(subset=["distance_to_port_km", "maritime_pressure_index"]).copy()
    dfp["bin"] = pd.cut(dfp["distance_to_port_km"], bins=np.linspace(0, min(180, dfp["distance_to_port_km"].quantile(0.98)), 12))
    agg = dfp.groupby("bin", observed=False)["maritime_pressure_index"].median().reset_index(drop=True)
    mids = [iv.mid for iv in dfp.groupby("bin", observed=False).groups.keys() if hasattr(iv, "mid")]
    if len(mids) != len(agg):
        mids = range(len(agg))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    xcenters = []
    ys = []
    for iv, med in dfp.groupby("bin", observed=False)["maritime_pressure_index"].median().items():
        if pd.isna(iv) or not hasattr(iv, "mid"):
            continue
        xcenters.append(iv.mid)
        ys.append(med)
    ax.plot(xcenters, ys, color="#0369a1", lw=2.4, marker="o")
    ax.set_xlabel("Distance to nearest assigned port (km)")
    ax.set_ylabel("Median maritime pressure index")
    ax.set_title("Exposure decay from ports", fontweight="bold")
    save_dual(fig, "exposure_analysis", "distance_decay_port_mpi")

    # Shipping lane proxy decay
    col_d = "distance_to_nearest_high_vessel_density_cell"
    if col_d in fx.columns:
        dd = fx.dropna(subset=[col_d, "vessel_density"]).copy()
        dd["bin"] = pd.cut(dd[col_d], bins=12)
        xc = []
        ym = []
        for iv, ser in dd.groupby("bin", observed=False)["vessel_density"]:
            if hasattr(iv, "mid"):
                xc.append(iv.mid)
                ym.append(ser.median())
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(xc, ym, color="#b45309", lw=2.4, marker="s")
        ax.set_xlabel("Distance to high vessel-density cell (km)")
        ax.set_ylabel("Median vessel density")
        ax.set_title("Decay from dense shipping corridors", fontweight="bold")
        save_dual(fig, "exposure_analysis", "distance_decay_shipping_lane")

    # Coastal exposure bands
    if "coastal_exposure_band" in fx.columns:
        sub = fx.dropna(subset=["coastal_exposure_band", "maritime_pressure_index"])
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        order = sorted(sub["coastal_exposure_band"].astype(str).unique())
        data = [sub.loc[sub["coastal_exposure_band"].astype(str) == o, "maritime_pressure_index"] for o in order]
        bp = ax.boxplot(data, tick_labels=order, patch_artist=True)
        for p in bp["boxes"]:
            p.set_facecolor("#bfdbfe")
        ax.set_title("MPI by shoreline exposure band", fontweight="bold")
        ax.set_xlabel("Coastal exposure band")
        ax.set_ylabel("Maritime pressure index")
        save_dual(fig, "exposure_analysis", "coastal_band_mpi_boxplots")

    # Hotspots top decile MPI
    thr = fx["maritime_pressure_index"].quantile(0.9)
    hot = fx.loc[fx["maritime_pressure_index"] >= thr]
    hot_g = hot.groupby("grid_cell_id", sort=False).agg(lat=("grid_centroid_lat", "first"), lon=("grid_centroid_lon", "first"), mpi=("maritime_pressure_index", "mean"))
    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    land = load_ne_land()
    if land is not None:
        b = box(BALTIC_LON[0], BALTIC_LAT[0], BALTIC_LON[1], BALTIC_LAT[1])
        land.clip(gpd.GeoSeries([b], crs="EPSG:4326")).plot(ax=ax, color="#f1f5f9", edgecolor="#94a3b8", lw=0.25)
    ax.scatter(hot_g["lon"], hot_g["lat"], c=hot_g["mpi"], cmap="YlOrRd", s=36, edgecolors="#1e293b", linewidths=0.35)
    for _, pr in pd.DataFrame(PORT_COORDS.items(), columns=["name", "coord"]).iterrows():
        lat, lon = pr["coord"]
        ax.scatter(lon, lat, s=70, marker="*", color="#0f172a", zorder=6)
    ax.set_aspect("equal")
    ax.set_xlim(BALTIC_LON[0] - 0.12, BALTIC_LON[1] + 0.12)
    ax.set_ylim(BALTIC_LAT[0] - 0.08, BALTIC_LAT[1] + 0.08)
    ax.set_title("Hotspots — top-decile maritime pressure", fontweight="bold")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    save_dual(fig, "exposure_analysis", "exposure_hotspots_top_decile_mpi")

    # Composite scatter
    sub = fx.dropna(subset=["coastal_exposure_score", "maritime_pressure_index", "ndti_mean"])
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    lo, hi = trim_scale(sub["ndti_mean"])
    sc = ax.scatter(
        sub["coastal_exposure_score"],
        sub["maritime_pressure_index"],
        c=sub["ndti_mean"],
        cmap="GnBu",
        s=14,
        alpha=0.65,
        vmin=lo,
        vmax=hi,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="NDTI mean")
    ax.set_xlabel("Coastal exposure score")
    ax.set_ylabel("Maritime pressure index")
    ax.set_title("Composite coastal × maritime exposure", fontweight="bold")
    save_dual(fig, "exposure_analysis", "composite_coastal_maritime_scatter")


def integrated_exposure_visualization() -> None:
    """Multi-panel Baltic maps: maritime + coastal layers and an equal-weight integrated index."""
    fx = load_features()
    cols_req = {"maritime_pressure_index", "coastal_exposure_score", "grid_cell_id"}
    if not cols_req.issubset(fx.columns):
        raise KeyError("features_ml_ready missing columns for integrated exposure")

    mpi = agg_cell_mean(fx, "maritime_pressure_index").reset_index().rename(columns={"v": "mpi"})
    ces = agg_cell_mean(fx, "coastal_exposure_score").reset_index().rename(columns={"v": "ces"})
    merged = mpi.merge(ces[["grid_cell_id", "ces"]], on="grid_cell_id", how="inner")

    def _norm01(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.nanpercentile(arr, [5.0, 95.0])
        return np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    merged["integrated"] = (_norm01(merged["mpi"].astype(float).values) + _norm01(merged["ces"].astype(float).values)) / 2.0

    fig = plt.figure(figsize=(10.2, 8.6), constrained_layout=True)
    axd = fig.subplot_mosaic(
        [["mpi", "ces"], ["integ", "integ"]],
        gridspec_kw={"height_ratios": [1.0, 1.12]},
    )
    land = load_ne_land()
    b = box(BALTIC_LON[0], BALTIC_LAT[0], BALTIC_LON[1], BALTIC_LAT[1])
    geo_clip = gpd.GeoSeries([b], crs="EPSG:4326")

    def _one_map(ax, lon, lat, vals: np.ndarray, title: str, cmap: str, cbar_label: str) -> None:
        if land is not None:
            land.clip(geo_clip).plot(ax=ax, color="#f8fafc", edgecolor="#64748b", lw=0.28, zorder=0)
        lo, hi = trim_scale(pd.Series(vals))
        sc = ax.scatter(lon, lat, c=vals, cmap=cmap, s=24 if ax is not axd["integ"] else 28, vmin=lo, vmax=hi, alpha=0.9, edgecolors="none")
        plt.colorbar(sc, ax=ax, shrink=0.72, label=cbar_label)
        ax.set_aspect("equal")
        ax.set_xlim(BALTIC_LON[0] - 0.12, BALTIC_LON[1] + 0.12)
        ax.set_ylim(BALTIC_LAT[0] - 0.08, BALTIC_LAT[1] + 0.08)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        port_df = pd.DataFrame(PORT_COORDS.items(), columns=["name", "coord"])
        for _, pr in port_df.iterrows():
            plat, plon = pr["coord"]
            ax.scatter(plon, plat, s=68, marker="*", color="#0f172a", zorder=6, edgecolors="#fef08a", linewidths=0.35)

    _one_map(
        axd["mpi"],
        merged["lon"].values,
        merged["lat"].values,
        merged["mpi"].values,
        "Maritime pressure index",
        "YlGnBu",
        "MPI",
    )
    _one_map(
        axd["ces"],
        merged["lon"].values,
        merged["lat"].values,
        merged["ces"].values,
        "Coastal exposure score",
        "GnBu",
        "CES",
    )
    lo_i, hi_i = trim_scale(pd.Series(merged["integrated"]))
    if land is not None:
        land.clip(geo_clip).plot(ax=axd["integ"], color="#f8fafc", edgecolor="#64748b", lw=0.28, zorder=0)
    sc_i = axd["integ"].scatter(
        merged["lon"],
        merged["lat"],
        c=merged["integrated"],
        cmap="PuBuGn",
        s=30,
        vmin=lo_i,
        vmax=hi_i,
        alpha=0.9,
        edgecolors="none",
    )
    plt.colorbar(sc_i, ax=axd["integ"], shrink=0.65, label="Integrated index (0-1)")
    axd["integ"].set_aspect("equal")
    axd["integ"].set_xlim(BALTIC_LON[0] - 0.12, BALTIC_LON[1] + 0.12)
    axd["integ"].set_ylim(BALTIC_LAT[0] - 0.08, BALTIC_LAT[1] + 0.08)
    axd["integ"].set_title(
        "Integrated exposure — equal-weight normalized maritime + coastal",
        fontweight="bold",
    )
    axd["integ"].set_xlabel("Longitude (deg)")
    axd["integ"].set_ylabel("Latitude (deg)")
    for _, pr in pd.DataFrame(PORT_COORDS.items(), columns=["name", "coord"]).iterrows():
        plat, plon = pr["coord"]
        axd["integ"].scatter(plon, plat, s=72, marker="*", color="#0f172a", zorder=6, edgecolors="#fef08a", linewidths=0.35)

    fig.suptitle("Integrated maritime–coastal exposure (study lattice)", fontsize=13.5, fontweight="bold", y=0.98)
    save_dual(fig, "exposure_analysis", "integrated_exposure_visualization")


def wind_analysis_plots() -> None:
    fx = load_features()
    wp = ROOT / "outputs/reports/run_land_pollution_drivers_wind/wind_alignment_features.csv"
    if wp.is_file():
        wind = pd.read_csv(wp, parse_dates=["week_start_utc"])
        merge_cols = [
            "grid_cell_id",
            "week_start_utc",
            "grid_centroid_lat",
            "grid_centroid_lon",
            "NO2_mean",
        ]
        sub = fx[merge_cols].merge(
            wind[
                ["grid_cell_id", "week_start_utc", "wind_aligned_to_land", "wind_alignment_score"]
            ],
            on=["grid_cell_id", "week_start_utc"],
            how="inner",
        )
        sub["aligned"] = sub["wind_aligned_to_land"].astype(float)
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        data = [
            sub.loc[sub["aligned"] < 0.5, "NO2_mean"].dropna(),
            sub.loc[sub["aligned"] >= 0.5, "NO2_mean"].dropna(),
        ]
        bp = ax.boxplot(data, tick_labels=["Less shore-aligned", "More shore-aligned"], patch_artist=True)
        for p in bp["boxes"]:
            p.set_facecolor("#e0f2fe")
        ax.set_title("NO₂ vs shoreward wind alignment", fontweight="bold")
        ax.set_ylabel("NO₂")
        save_dual(fig, "wind_regime", "shoreward_vs_other_no2_boxplot")

        vc = wind["wind_alignment_category"].astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar(range(len(vc)), vc.values, color="#0284c7", edgecolor="none")
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(list(vc.index), rotation=25, ha="right")
        ax.set_ylabel("Row count")
        ax.set_title("Wind alignment category frequency", fontweight="bold")
        save_dual(fig, "wind_regime", "wind_alignment_category_counts")

        # Direction score map (cell mean)
        g = agg_cell_mean(sub, "wind_alignment_score")
        fig, ax = plt.subplots(figsize=(7.6, 6.8))
        land = load_ne_land()
        if land is not None:
            b = box(BALTIC_LON[0], BALTIC_LAT[0], BALTIC_LON[1], BALTIC_LAT[1])
            land.clip(gpd.GeoSeries([b], crs="EPSG:4326")).plot(ax=ax, color="#f8fafc", edgecolor="#94a3b8", lw=0.25)
        lo, hi = trim_scale(g["v"])
        sc = ax.scatter(g["lon"], g["lat"], c=g["v"], cmap="RdYlGn", s=22, vmin=lo, vmax=hi, edgecolors="none")
        plt.colorbar(sc, ax=ax, shrink=0.55, label="Mean alignment score")
        ax.set_aspect("equal")
        ax.set_title("Wind-alignment score (cell mean)", fontweight="bold")
        ax.set_xlim(BALTIC_LON[0] - 0.12, BALTIC_LON[1] + 0.12)
        ax.set_ylim(BALTIC_LAT[0] - 0.08, BALTIC_LAT[1] + 0.08)
        save_dual(fig, "wind_regime", "wind_alignment_score_map")

    if "atmospheric_transfer_index" in fx.columns:
        _features_cell_map(
            "atmospheric_transfer_index",
            "Atmospheric transfer index",
            "atmospheric_transfer_map",
            "wind_regime",
            cmap="Purples",
        )

    sub = fx.dropna(subset=["NO2_mean", "maritime_pressure_index"])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lo, hi = trim_scale(sub["maritime_pressure_index"])
    sc = ax.scatter(sub["NO2_mean"], sub["maritime_pressure_index"], c=sub["ndti_mean"], cmap="GnBu", s=12, alpha=0.55, vmin=trim_scale(sub["ndti_mean"])[0], vmax=trim_scale(sub["ndti_mean"])[1], edgecolors="none")
    plt.colorbar(sc, ax=ax, label="NDTI mean")
    ax.set_xlabel("NO₂ mean")
    ax.set_ylabel("Maritime pressure index")
    ax.set_title("Maritime exposure vs NO₂", fontweight="bold")
    save_dual(fig, "wind_regime", "wind_exposure_no2_scatter")


def temporal_lag_persistence() -> None:
    lag_csv = ROOT / "outputs/reports/temporal_lag_statistics.csv"
    if lag_csv.is_file():
        t = pd.read_csv(lag_csv)
        sub = t.loc[t["segment"].eq("lag_autocorr_spearman") & t["rho"].notna()]
        if not sub.empty:
            piv = sub.pivot_table(index="friendly", columns="lag_weeks", values="rho")
            fig, ax = plt.subplots(figsize=(6.8, 5.4))
            im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([str(int(c)) if pd.notna(c) else "" for c in piv.columns])
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels(list(piv.index), fontsize=9)
            plt.colorbar(im, ax=ax, label="Spearman ρ")
            ax.set_title("Lag autocorrelation (weekly pooled series)", fontweight="bold")
            ax.set_xlabel("Lag (weeks)")
            save_dual(fig, "temporal_lag_persistence", "lag_autocorr_heatmap")

    dm = load_modeling().copy()
    dm["week_start_utc"] = pd.to_datetime(dm["week_start_utc"], utc=True)
    w = dm.groupby(["week_start_utc", "grid_cell_id"])["sentinel_ndti_mean_t"].median().reset_index()
    piv = w.pivot(index="week_start_utc", columns="grid_cell_id", values="sentinel_ndti_mean_t")
    med_series = piv.median(axis=1).dropna()
    if len(med_series) > 5:
        fig, ax = plt.subplots(figsize=(8.6, 3.9))
        ax.plot(med_series.index, med_series.values, color="#0369a1", lw=2.2)
        ax.set_title("Weekly median NDTI (spatial median across cells)", fontweight="bold")
        fig.autofmt_xdate()
        save_dual(fig, "temporal_lag_persistence", "weekly_ndti_median_timeline")

    dm2 = load_modeling().copy()
    dm2["week_start_utc"] = pd.to_datetime(dm2["week_start_utc"], utc=True)
    cols_use = ["sentinel_ndti_mean_t", "sentinel_ndwi_mean_t", "vessel_density_t"]
    cols_use = [c for c in cols_use if c in dm2.columns]
    if cols_use:
        wm = dm2.groupby("week_start_utc")[cols_use].median().dropna(how="all")
        if wm.shape[0] > 5 and wm.shape[1] > 1:
            cr = wm.corr()
            fig, ax = plt.subplots(figsize=(5.8, 5.0))
            im = ax.imshow(cr.values, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(len(cr.columns)))
            ax.set_yticks(range(len(cr.columns)))
            short = [c.replace("sentinel_", "").replace("_t", "") for c in cr.columns]
            ax.set_xticklabels(short, rotation=35, ha="right")
            ax.set_yticklabels(short)
            plt.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
            ax.set_title("Weekly median driver persistence (corr across weeks)", fontweight="bold")
            save_dual(fig, "temporal_lag_persistence", "persistence_corr_weekly_medians")

    # t vs lag correlation scatter on panel medians
    dm = load_modeling().copy()
    dm["week_start_utc"] = pd.to_datetime(dm["week_start_utc"], utc=True)
    dm = dm.sort_values(["grid_cell_id", "week_start_utc"])
    dm["ndti_l1"] = dm.groupby("grid_cell_id")["sentinel_ndti_mean_t"].shift(1)
    dm["ndti_l2"] = dm.groupby("grid_cell_id")["sentinel_ndti_mean_t"].shift(2)
    sub = dm.dropna(subset=["sentinel_ndti_mean_t", "ndti_l1", "ndti_l2"])
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(sub["ndti_l1"], sub["sentinel_ndti_mean_t"], s=8, alpha=0.35, c="#0369a1", edgecolors="none")
    ax.set_xlabel("NDTI at t-1")
    ax.set_ylabel("NDTI at t")
    ax.set_title("Temporal persistence — lag-1 vs current", fontweight="bold")
    save_dual(fig, "temporal_lag_persistence", "scatter_ndti_lag1_vs_t")


def machine_learning_plots() -> None:
    mr = load_model_results()
    if not mr:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))
    models = ["ridge_median_impute_scaled", "hist_gradient_boosting"]
    labels = ["Ridge", "HGB"]

    rmse_v = []
    r2_v = []
    for m in models:
        rmse_v.append(mr["metrics"][m]["test"]["rmse"])
        r2_v.append(mr["metrics"][m]["test"]["r2"])

    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w / 2, rmse_v, w, label="RMSE", color="#0369a1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Test RMSE — ΔNDTI (single split)", fontweight="bold")
    axes[0].set_ylabel("RMSE")

    axes[1].bar(x - w / 2, r2_v, w, label="R²", color="#047857")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].axhline(0, color="0.35", lw=0.9)
    axes[1].set_title("Test R² — ΔNDTI", fontweight="bold")
    axes[1].set_ylabel("R2")
    save_dual(fig, "machine_learning", "baseline_rmse_r2_delta_ndti_split")

    mae_v = [mr["metrics"][m]["test"].get("mae", np.nan) for m in models]
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.bar(labels, mae_v, color=["#0369a1", "#0891b2"], edgecolor="white")
    ax.set_title("Test MAE - delta NDTI (single chronological split)", fontweight="bold")
    ax.set_ylabel("MAE")
    save_dual(fig, "machine_learning", "baseline_mae_delta_ndti_split")

    # ndti_next metrics if present
    if "metrics_ndti_next" in mr:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))
        rmse_n = [mr["metrics_ndti_next"][m]["test"]["rmse"] for m in models]
        r2_n = [mr["metrics_ndti_next"][m]["test"]["r2"] for m in models]
        axes[0].bar(x, rmse_n, color="#7c3aed")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_title("Test RMSE — ndti_next", fontweight="bold")
        axes[1].bar(x, r2_n, color="#db2777")
        axes[1].axhline(0, color="0.35", lw=0.9)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_title("Test R² — ndti_next", fontweight="bold")
        save_dual(fig, "machine_learning", "baseline_rmse_r2_ndti_next_split")

    pred = load_predictions()
    if pred is not None:
        te = pred.loc[pred["split"].astype(str).str.lower().eq("test")]
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.6))
        pairs = [
            ("pred_ridge_delta_ndti", "y_true_delta_ndti", "Ridge ΔNDTI"),
            ("pred_hist_gradient_boosting_delta_ndti", "y_true_delta_ndti", "HGB ΔNDTI"),
            ("pred_ridge_ndti_next", "y_true_ndti_next", "Ridge ndti_next"),
            ("pred_hist_gradient_boosting_ndti_next", "y_true_ndti_next", "HGB ndti_next"),
        ]
        for ax, (pc, yc, ttl) in zip(axes.ravel(), pairs):
            if pc not in te.columns:
                ax.axis("off")
                continue
            s = te.dropna(subset=[pc, yc])
            ax.scatter(s[yc], s[pc], s=10, alpha=0.35, c="#0369a1", edgecolors="none")
            lim = np.nanpercentile(np.r_[s[yc].values, s[pc].values], [2, 98])
            ax.plot(lim, lim, color="#94a3b8", lw=1.2, linestyle="--")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(ttl, fontsize=10, fontweight="bold")
        fig.suptitle("Predicted vs actual — withheld test weeks", fontweight="bold", y=1.01)
        save_dual(fig, "machine_learning", "predicted_vs_actual_four_panel")

        te = pred.loc[pred["split"].astype(str).str.lower().eq("test")]
        res = te["y_true_delta_ndti"] - te["pred_hist_gradient_boosting_delta_ndti"]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(res.dropna(), bins=45, color="#bae6fd", edgecolor="white")
        ax.axvline(0, color="#334155", lw=1.1)
        ax.set_title("Residuals — HGB ΔNDTI (test)", fontweight="bold")
        ax.set_xlabel("Actual - predicted")
        save_dual(fig, "machine_learning", "residual_hist_hgb_delta_ndti")

    imp = mr.get("feature_importance_hist_gradient_boosting") or mr.get("metrics", {}).get("hist_gradient_boosting", {})
    if isinstance(mr.get("feature_importance_hist_gradient_boosting"), list):
        rows = mr["feature_importance_hist_gradient_boosting"][:18]
        names = [r["feature"] for r in rows]
        vals = [r["importance_mean"] for r in rows]
        fig, ax = plt.subplots(figsize=(7.8, 6.4))
        ax.barh(names[::-1], vals[::-1], color="#0ea5e9")
        ax.set_title("Feature importance — HistGradientBoosting (ΔNDTI)", fontweight="bold")
        ax.set_xlabel("Mean permutation ΔRMSE")
        save_dual(fig, "machine_learning", "feature_importance_hgb_delta_ndti")


def validation_and_rolling() -> None:
    mr = load_model_results() or {}
    roll_csv = ROOT / "outputs/ml_cv_results/rolling_window_metrics.csv"
    if not roll_csv.is_file():
        subprocess.run(
            [sys.executable, str(ROOT / "src/ml/run_rolling_window_cv.py"), "--input", str(ROOT / "data/modeling_dataset.parquet")],
            cwd=str(ROOT),
            check=False,
        )

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.3, 1.45), 4.2, 1.2, boxstyle="round,pad=0.06", facecolor="#dbeafe", edgecolor="#1e3a8a", lw=1.5))
    ax.text(2.4, 2.55, "Train timeline\n(first 75% of weeks)", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(2.4, 1.85, "Fit scaler / models once", ha="center", va="center", fontsize=9)
    ax.add_patch(FancyBboxPatch((5.2, 1.45), 4.2, 1.2, boxstyle="round,pad=0.06", facecolor="#ffe4e6", edgecolor="#9f1239", lw=1.5))
    ax.text(7.3, 2.55, "Single hold-out tail\n(last ~25% weeks)", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(7.3, 1.85, "One-shot evaluation", ha="center", va="center", fontsize=9)
    ax.set_title("Standard chronological split", fontweight="bold", fontsize=13)

    ax.annotate("", xy=(9.8, 0.85), xytext=(5.5, 1.35), arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#334155"))
    ax.text(7.6, 0.95, "Risk: regime shift concentrates\nall test stress in one era", fontsize=9)
    save_dual(fig, "validation", "diagram_standard_temporal_split")

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    folds = [(1.2, "#bae6fd"), (3.1, "#93c5fd"), (5.0, "#60a5fa"), (6.9, "#3b82f6")]
    for i, (x0, c) in enumerate(folds):
        ax.add_patch(FancyBboxPatch((x0, 1.35), 1.45, 1.35, boxstyle="round,pad=0.05", facecolor=c, edgecolor="#1e40af"))
        ax.text(x0 + 0.72, 2.4, f"Fold {i+1}", ha="center", fontsize=9, fontweight="bold")
        ax.text(x0 + 0.72, 1.85, "Train grows ->", ha="center", fontsize=8)
        ax.text(x0 + 0.72, 1.55, "short test block", ha="center", fontsize=7.5)
    ax.text(5.0, 2.95, "Rolling expanding windows — fresh validation eras", ha="center", fontsize=12, fontweight="bold")
    save_dual(fig, "validation", "diagram_rolling_expanding_windows")

    if roll_csv.is_file():
        rw = pd.read_csv(roll_csv)
        delta = rw.loc[rw["target"].eq("delta_ndti")]
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        for model, sty in zip(["ridge_median_impute_scaled", "hist_gradient_boosting"], ["o-", "s--"]):
            sub = delta.loc[delta["model"].eq(model)].sort_values("fold_id")
            ax.plot(sub["fold_id"], sub["test_rmse"], sty, label=model.replace("_", " ")[:22], lw=2)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Test RMSE")
        ax.set_title("Rolling CV — ΔNDTI test RMSE by fold", fontweight="bold")
        ax.legend(fontsize=8)
        save_dual(fig, "validation", "rolling_rmse_trend_delta_ndti")

        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        for model, sty in zip(["ridge_median_impute_scaled", "hist_gradient_boosting"], ["o-", "s--"]):
            sub = delta.loc[delta["model"].eq(model)].sort_values("fold_id")
            ax.plot(sub["fold_id"], sub["test_r2"], sty, label=model.replace("_", " ")[:22], lw=2)
        ax.axhline(0, color="0.45", lw=0.9)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Test R²")
        ax.set_title("Rolling CV — ΔNDTI test R² by fold", fontweight="bold")
        ax.legend(fontsize=8)
        save_dual(fig, "validation", "rolling_r2_trend_delta_ndti")

    # Comparison graphic standard vs rolling (conceptual RMSE bands)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    categories = ["Single split\ntest RMSE", "Rolling CV\nmean test RMSE"]
    ridge_split = mr.get("metrics", {}).get("ridge_median_impute_scaled", {}).get("test", {}).get("rmse", np.nan)
    hgb_split = mr.get("metrics", {}).get("hist_gradient_boosting", {}).get("test", {}).get("rmse", np.nan)
    r_roll = np.nan
    h_roll = np.nan
    if roll_csv.is_file():
        rw = pd.read_csv(roll_csv)
        du = rw.loc[rw["target"].eq("delta_ndti")]
        r_roll = du.loc[du["model"].eq("ridge_median_impute_scaled"), "test_rmse"].mean()
        h_roll = du.loc[du["model"].eq("hist_gradient_boosting"), "test_rmse"].mean()
    ax.bar([0 - 0.2, 0 + 0.2], [ridge_split, hgb_split], width=0.38, label="Baseline split", color=["#0284c7", "#0369a1"])
    ax.bar([1 - 0.2, 1 + 0.2], [r_roll, h_roll], width=0.38, label="Rolling mean", color=["#f97316", "#ea580c"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(categories)
    ax.set_ylabel("RMSE (ΔNDTI)")
    ax.set_title("Standard split vs rolling-window CV — ΔNDTI RMSE", fontweight="bold")
    ax.legend(fontsize=9)
    save_dual(fig, "validation", "comparison_standard_vs_rolling_rmse_delta_ndti")


def anomaly_detection_plots() -> None:
    fx = load_features().copy()
    fx["week_start_utc"] = pd.to_datetime(fx["week_start_utc"], utc=True)
    thr = fx["ndti_mean"].quantile(0.90)
    cnt = fx.assign(exc=(fx["ndti_mean"] > thr).astype(int)).groupby("week_start_utc")["exc"].sum()
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.bar(cnt.index, cnt.values, width=5, color="#b91c1c", alpha=0.82)
    ax.set_title("High-NDTI exceedances per week (above global P90)", fontweight="bold")
    ax.set_ylabel("# cells")
    fig.autofmt_xdate()
    save_dual(fig, "anomaly_detection", "weekly_ndti_exceedance_counts")

    med = fx.groupby("week_start_utc")["ndti_mean"].median()
    z = (med - med.mean()) / med.std(ddof=0)
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.plot(z.index, z.values, color="#be123c", lw=2)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.axhline(2, color="#94a3b8", linestyle="--", lw=1)
    ax.axhline(-2, color="#94a3b8", linestyle="--", lw=1)
    ax.set_title("Rolling-context anomaly — median NDTI z-score by week", fontweight="bold")
    fig.autofmt_xdate()
    save_dual(fig, "anomaly_detection", "median_ndti_zscore_timeline")


def comparison_cross_port() -> None:
    fx = load_features().copy()
    ports = ["Turku", "Mariehamn", "Stockholm"]

    def tag_port(name: str) -> str | None:
        s = str(name).lower()
        for p in ports:
            if p.lower() in s:
                return p
        return None

    fx["_tag"] = fx["nearest_port"].map(tag_port)
    sub = fx.dropna(subset=["_tag"])
    if sub.empty:
        return
    metric_cols = [
        c for c in ["maritime_pressure_index", "NO2_mean", "ndti_mean", "coastal_exposure_score"] if c in sub.columns
    ]
    if not metric_cols:
        return
    agg = sub.groupby("_tag")[metric_cols].median().reindex([p for p in ports if p in sub["_tag"].unique()])
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(agg.index))
    w = 0.18
    for i, col in enumerate(metric_cols):
        vals = agg[col].astype(float).values
        vals = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-9)
        ax.bar(x + i * w, vals, w, label=col.replace("_", " "))
    ax.set_xticks(x + w * (len(metric_cols) - 1) / 2)
    ax.set_xticklabels(list(agg.index))
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_title("Cross-port comparison (z-scored medians)", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylabel("Standardised median")
    save_dual(fig, "comparison_analysis", "cross_port_zscore_medians")


def summary_five_panel() -> None:
    fig = plt.figure(figsize=(11.2, 7.8))
    texts = [
        "Framework\nSources -> fused panel -> exposure -> temporal ML",
        "Key finding\nCoastal gradients;\nlimited ΔNDTI extrapolation skill",
        "Exposure summary\nPort hotspots;\ncoastal decay",
        "Integrated signals\nMaritime + coastal +\natmospheric layers",
        "Contribution\nLeakage-aware splits;\nrolling validation",
    ]
    for i, txt in enumerate(texts):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.set_facecolor("#f8fafc")
        ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(1.4)
            s.set_edgecolor("#334155")
    fig.suptitle("Presentation summary tiles", fontsize=13.5, fontweight="bold")
    save_dual(fig, "summary_maps", "infographic_executive_summary_five_tile")


def summary_maps_mini() -> None:
    """Reuse lattice footprint — coastal-friendly colouring."""
    dm = load_modeling()
    g = agg_cell_mean(dm, "sentinel_ndti_mean_t")
    fig, ax = plt.subplots(figsize=(8.6, 7.8))
    land = load_ne_land()
    if land is not None:
        b = box(BALTIC_LON[0], BALTIC_LAT[0], BALTIC_LON[1], BALTIC_LAT[1])
        land.clip(gpd.GeoSeries([b], crs="EPSG:4326")).plot(ax=ax, color="#ecfdf5", edgecolor="#64748b", lw=0.28)
    lo, hi = trim_scale(g["v"])
    sc = ax.scatter(g["lon"], g["lat"], c=g["v"], cmap="GnBu", s=26, vmin=lo, vmax=hi, edgecolors="none")
    plt.colorbar(sc, ax=ax, shrink=0.55, label="Mean NDTI")
    ax.set_aspect("equal")
    ax.set_title("Environmental exposure summary — turbidity footprint", fontweight="bold")
    ax.set_xlim(BALTIC_LON[0] - 0.12, BALTIC_LON[1] + 0.12)
    ax.set_ylim(BALTIC_LAT[0] - 0.08, BALTIC_LAT[1] + 0.08)
    save_dual(fig, "summary_maps", "summary_environmental_exposure_ndti")


ALL_EXTENDED_JOBS = [
    exposure_maps_and_decay,
    integrated_exposure_visualization,
    wind_analysis_plots,
    temporal_lag_persistence,
    machine_learning_plots,
    validation_and_rolling,
    anomaly_detection_plots,
    comparison_cross_port,
    summary_maps_mini,
    summary_five_panel,
]
