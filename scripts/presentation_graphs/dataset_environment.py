"""Dataset overview + environmental indicator figures."""

from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import PolyCollection
from shapely.geometry import box

from .common import (
    BALTIC_LAT,
    BALTIC_LON,
    PORT_COORDS,
    agg_cell_mean,
    load_features,
    load_modeling,
    load_ne_land,
    save_dual,
    trim_scale,
    parse_res_deg,
)


def _clip_baltic(df: pd.DataFrame) -> pd.DataFrame:
    lo1, lo2 = BALTIC_LON
    la1, la2 = BALTIC_LAT
    return df.loc[
        df["grid_centroid_lon"].between(lo1, lo2) & df["grid_centroid_lat"].between(la1, la2)
    ]


def fig_baltic_study_extent_grid() -> None:
    dm = load_modeling()
    dm = _clip_baltic(dm)
    cells = dm.groupby("grid_cell_id", sort=False).agg(
        lon=("grid_centroid_lon", "first"),
        lat=("grid_centroid_lat", "first"),
        gid=("grid_cell_id", "first"),
    ).reset_index(drop=True)
    land = load_ne_land()
    fig, ax = plt.subplots(figsize=(7.8, 7.8))
    ax.set_facecolor("#cfe8fb")
    if land is not None:
        b = box(BALTIC_LON[0], BALTIC_LAT[0], BALTIC_LON[1], BALTIC_LAT[1])
        clip = land.clip(gpd.GeoSeries([b], crs="EPSG:4326"))
        clip.plot(ax=ax, facecolor="#f0f2f5", edgecolor="#5c6478", lw=0.35, zorder=1)

    verts = []
    for _, r in cells.iterrows():
        h = parse_res_deg(r["gid"]) / 2
        x, y = float(r["lon"]), float(r["lat"])
        verts.append(
            [
                (x - h, y - h),
                (x + h, y - h),
                (x + h, y + h),
                (x - h, y + h),
            ],
        )
    pc = PolyCollection(
        verts,
        facecolors="#dce6f8",
        edgecolors="#253045",
        linewidths=0.2,
        alpha=0.9,
        zorder=3,
    )
    ax.add_collection(pc)
    ax.set_aspect("equal")
    ax.set_xlim(BALTIC_LON[0] - 0.25, BALTIC_LON[1] + 0.25)
    ax.set_ylim(BALTIC_LAT[0] - 0.15, BALTIC_LAT[1] + 0.15)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Baltic study area & modelling grid", fontweight="bold")
    save_dual(fig, "dataset_overview", "01_baltic_study_area_grid")


def fig_port_locations() -> None:
    land = load_ne_land()
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.set_facecolor("#e8f4fc")
    if land is not None:
        b = box(BALTIC_LON[0], BALTIC_LAT[0], BALTIC_LON[1], BALTIC_LAT[1])
        land.clip(gpd.GeoSeries([b], crs="EPSG:4326")).plot(
            ax=ax, facecolor="#f6f7f9", edgecolor="#3d4a5c", lw=0.45, zorder=1
        )
    for name, (lat, lon) in PORT_COORDS.items():
        ax.scatter(lon, lat, s=120, marker="*", color="#0f172a", zorder=5, edgecolors="white", linewidths=0.4)
        ax.annotate(name, (lon + 0.25, lat + 0.1), fontsize=9, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(BALTIC_LON[0] - 0.2, BALTIC_LON[1] + 0.6)
    ax.set_ylim(BALTIC_LAT[0] - 0.12, BALTIC_LAT[1] + 0.5)
    ax.set_title("Study ports", fontweight="bold")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    save_dual(fig, "dataset_overview", "02_port_locations")


def fig_data_integration_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.5, 3.8, "Sentinel-2\n(NDTI, NDWI, …)", "#bfdbfe"),
        (3.5, 3.8, "EMODnet\nvessel density", "#a5f3fc"),
        (6.5, 3.8, "Auxiliary\n(NO2, wind)", "#c4b5fd"),
        (2.0, 1.4, "Master panel\n(master_dataset)", "#fde68a"),
        (5.5, 1.4, "Modelling panel\n(modeling_dataset)", "#fecdd3"),
        (3.75, 0.25, "ML & validation\n(run_delta_ndti_models)", "#bbf7d0"),
    ]
    for x, y, txt, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.3, 1.0, boxstyle="round,pad=0.05", facecolor=c, edgecolor="#334155", linewidth=1.1))
        ax.text(x + 1.15, y + 0.5, txt, ha="center", va="center", fontsize=9.5, fontweight="bold")

    arrows = [
        ((1.75, 3.8), (2.8, 2.9)),
        ((4.75, 3.8), (3.9, 2.9)),
        ((7.75, 3.8), (6.5, 2.9)),
        ((3.9, 1.4), (4.6, 1.05)),
        ((6.9, 1.4), (4.9, 0.85)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 1.15, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color="#334155",
            )
        )
    ax.set_title("Dataset integration pipeline (conceptual)", fontweight="bold", fontsize=13, y=1.02)
    save_dual(fig, "dataset_overview", "03_integration_architecture_diagram")


def fig_temporal_coverage() -> None:
    dm = load_modeling()
    dm["week_start_utc"] = pd.to_datetime(dm["week_start_utc"], utc=True)
    c = dm.groupby(dm["week_start_utc"].dt.strftime("%Y-%m-%d")).size().sort_index()
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.bar(range(len(c)), c.values, color="#0284c7", edgecolor="none", alpha=0.88, width=0.85)
    ax.set_xticks(range(0, len(c), max(1, len(c) // 12)))
    ax.set_xticklabels([c.index[i] for i in range(0, len(c), max(1, len(c) // 12))], rotation=45, ha="right")
    ax.set_ylabel("Row count")
    ax.set_title("Temporal coverage (observations per week)", fontweight="bold")
    ax.set_xlabel("Week start (UTC)")
    save_dual(fig, "dataset_overview", "04_temporal_coverage_bars")


def fig_weekly_observation_timeline() -> None:
    dm = load_modeling()
    dm["week_start_utc"] = pd.to_datetime(dm["week_start_utc"], utc=True)
    obs = dm.groupby("week_start_utc")["sentinel_observation_count_t"].sum().sort_index()
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.plot(obs.index, obs.values, color="#0369a1", lw=2.1, marker="o", markersize=3)
    ax.set_title("Weekly Sentinel observation count (sum over cells)", fontweight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Total count")
    fig.autofmt_xdate()
    save_dual(fig, "dataset_overview", "05_weekly_observation_timeline")


def fig_missingness_heatmap() -> None:
    dm = load_modeling().copy()
    dm["week_start_utc"] = pd.to_datetime(dm["week_start_utc"], utc=True)
    cols = [c for c in dm.columns if c not in ("grid_cell_id", "week_start_utc", "grid_res_deg")]
    miss = dm.groupby(dm["week_start_utc"].dt.strftime("%Y-%m-%d"))[cols].apply(lambda g: g.isna().mean())
    miss = miss.T
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    im = ax.imshow(miss.values, aspect="auto", cmap="Blues_r", vmin=0, vmax=1)
    ax.set_yticks(range(len(miss.index)))
    ax.set_yticklabels(miss.index, fontsize=7)
    step = max(1, miss.shape[1] // 14)
    ax.set_xticks(range(0, miss.shape[1], step))
    ax.set_xticklabels(miss.columns[::step], rotation=45, ha="right", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Missing rate")
    ax.set_title("Missingness heatmap by week", fontweight="bold")
    ax.set_xlabel("Week start")
    save_dual(fig, "dataset_overview", "06_missingness_heatmap_weeks")


def fig_spatial_coverage_density() -> None:
    dm = load_modeling()
    dm = _clip_baltic(dm)
    n = dm.groupby("grid_cell_id").size()
    xy = dm.groupby("grid_cell_id").agg(lat=("grid_centroid_lat", "first"), lon=("grid_centroid_lon", "first"))
    xy = xy.join(n.rename("cnt"), how="inner")
    fig, ax = plt.subplots(figsize=(7.8, 7.4))
    land = load_ne_land()
    if land is not None:
        b = box(BALTIC_LON[0], BALTIC_LAT[0], BALTIC_LON[1], BALTIC_LAT[1])
        land.clip(gpd.GeoSeries([b], crs="EPSG:4326")).plot(ax=ax, color="#eef1f6", edgecolor="#64748b", lw=0.3, zorder=0)
    sc = ax.scatter(
        xy["lon"],
        xy["lat"],
        c=xy["cnt"],
        cmap="YlGnBu",
        s=18,
        alpha=0.85,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Weeks observed (n)", shrink=0.55)
    ax.set_aspect("equal")
    ax.set_xlim(BALTIC_LON[0] - 0.15, BALTIC_LON[1] + 0.15)
    ax.set_ylim(BALTIC_LAT[0] - 0.1, BALTIC_LAT[1] + 0.1)
    ax.set_title("Spatial coverage density (# weekly records per cell)", fontweight="bold")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    save_dual(fig, "dataset_overview", "07_spatial_coverage_density")


def _cell_map(dm: pd.DataFrame, col: str, title: str, stem: str, cmap_name: str = "Blues") -> None:
    g = agg_cell_mean(dm, col)
    lo, hi = trim_scale(g["v"])
    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    land = load_ne_land()
    if land is not None:
        b = box(BALTIC_LON[0], BALTIC_LAT[0], BALTIC_LON[1], BALTIC_LAT[1])
        land.clip(gpd.GeoSeries([b], crs="EPSG:4326")).plot(ax=ax, color="#f4f6f8", edgecolor="#94a3b8", lw=0.25, zorder=0)
    sc = ax.scatter(g["lon"], g["lat"], c=g["v"], cmap=cmap_name, s=22, vmin=lo, vmax=hi, alpha=0.92, edgecolors="none")
    plt.colorbar(sc, ax=ax, shrink=0.58, label=col.replace("_", " "))
    ax.set_aspect("equal")
    ax.set_xlim(BALTIC_LON[0] - 0.12, BALTIC_LON[1] + 0.12)
    ax.set_ylim(BALTIC_LAT[0] - 0.08, BALTIC_LAT[1] + 0.08)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    save_dual(fig, "environmental_indicators", stem)


def env_maps_modeling() -> None:
    dm = load_modeling()
    pairs = [
        ("sentinel_ndwi_mean_t", "NDWI mean (week t)", "map_ndwi_mean"),
        ("sentinel_ndti_mean_t", "NDTI mean (week t)", "map_ndti_mean"),
        ("sentinel_ndvi_mean_t", "NDVI mean (week t)", "map_ndvi_mean"),
        ("vessel_density_t", "Vessel density (week t)", "map_vessel_density", "YlOrRd"),
    ]
    for item in pairs:
        col, title, stem = item[0], item[1], item[2]
        cmap = item[3] if len(item) > 3 else "GnBu"
        _cell_map(dm, col, title, stem, cmap)


def env_maps_features() -> None:
    fx = load_features()
    if "ndci_mean" not in fx.columns:
        return
    for col, title, stem, cmap in [
        ("ndci_mean", "NDCI (cell–week mean)", "map_ndci_mean", "PuBuGn"),
        ("NO2_mean", "NO₂ (cell–week mean)", "map_no2_mean", "OrRd"),
    ]:
        _cell_map(fx, col, title, stem, cmap)


def fig_environmental_distributions() -> None:
    dm = load_modeling()
    cols = [
        "sentinel_ndwi_mean_t",
        "sentinel_ndti_mean_t",
        "sentinel_ndvi_mean_t",
        "vessel_density_t",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.8))
    for ax, c in zip(axes.ravel(), cols):
        x = pd.to_numeric(dm[c], errors="coerce").dropna()
        x = x[x.between(x.quantile(0.01), x.quantile(0.99))]
        ax.hist(x, bins=40, color="#0ea5e9", edgecolor="white", alpha=0.88)
        ax.set_title(c.replace("sentinel_", "").replace("_t", ""), fontsize=10, fontweight="bold")
        ax.set_ylabel("Frequency")
    fig.suptitle("Environmental indicator distributions (winsorized 1–99%)", fontweight="bold", y=1.02)
    save_dual(fig, "environmental_indicators", "env_distributions_grid")


def fig_weekly_environmental_variability() -> None:
    dm = load_modeling().copy()
    dm["week_start_utc"] = pd.to_datetime(dm["week_start_utc"], utc=True)
    med = dm.groupby("week_start_utc")[["sentinel_ndti_mean_t", "sentinel_ndwi_mean_t"]].median()
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.plot(med.index, med["sentinel_ndti_mean_t"], label="Median NDTI", color="#0369a1", lw=2)
    ax.plot(med.index, med["sentinel_ndwi_mean_t"], label="Median NDWI", color="#047857", lw=2)
    ax.legend(loc="upper right")
    ax.set_title("Weekly panel medians — NDTI & NDWI", fontweight="bold")
    ax.set_ylabel("Index value")
    fig.autofmt_xdate()
    save_dual(fig, "environmental_indicators", "weekly_median_ndti_ndwi")


def fig_indicator_boxplots() -> None:
    dm = load_modeling()
    cols = ["sentinel_ndwi_mean_t", "sentinel_ndti_mean_t", "sentinel_ndvi_mean_t", "sentinel_evi_mean_t"]
    data = [pd.to_numeric(dm[c], errors="coerce").dropna() for c in cols]
    data = [d[(d >= d.quantile(0.02)) & (d <= d.quantile(0.98))] for d in data]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bp = ax.boxplot(data, tick_labels=[c.replace("sentinel_", "").replace("_t", "") for c in cols], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#bae6fd")
        patch.set_alpha(0.85)
    ax.set_title("Spectral indicators (trimmed extremes)", fontweight="bold")
    ax.set_ylabel("Value")
    save_dual(fig, "environmental_indicators", "boxplot_major_spectral")


def fig_correlation_heatmap() -> None:
    dm = load_modeling()
    cols = [
        c
        for c in dm.columns
        if c.startswith("sentinel_") and ("mean" in c or c.endswith("_t"))
        and pd.api.types.is_numeric_dtype(dm[c])
    ]
    sub = dm[cols].apply(pd.to_numeric, errors="coerce")
    r = sub.corr()
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(r.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(r.columns)))
    ax.set_yticks(range(len(r.columns)))
    ax.set_xticklabels([c.replace("sentinel_", "") for c in r.columns], rotation=65, ha="right", fontsize=7)
    ax.set_yticklabels([c.replace("sentinel_", "") for c in r.columns], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Feature correlation heatmap (modelling columns)", fontweight="bold")
    save_dual(fig, "environmental_indicators", "correlation_heatmap_spectral")
