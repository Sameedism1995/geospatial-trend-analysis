"""
Spatio-temporal human-activity impact layer: distance features, optional NO₂ / oil joins,
activity regimes, decay summaries, and thesis-ready plots — no change to model training.

Reads: data/modeling_dataset.parquet
Optional: --no2-path (default data/aux/no2_grid_week.parquet if present),
          --oil-path (default: data/aux/sentinel1_oil_slicks.parquet, else oil_slicks.parquet).
Writes: data/modeling_dataset_human_impact.parquet, data/run_summary.json, CSV summaries,
        figures including oil_shipping_relationship.png, data/visualizations/sanity/*.png,
        merges distance_decay_analysis / no2_analysis / oil_analysis into data/model_results.json.

NO₂ and oil are observational; missing values stay NaN (no imputation for signals).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# --- geo / Natural Earth ----------------------------------------------------

NE_COASTLINE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_coastline.geojson"
)
NE_LAND_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_land.geojson"
)


def haversine_nearest_distance_km(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    src = np.deg2rad(np.c_[source_lat, source_lon])
    tgt = np.deg2rad(np.c_[target_lat, target_lon])
    tree = BallTree(tgt, metric="haversine")
    dist_rad, _ = tree.query(src, k=1)
    return dist_rad[:, 0] * 6371.0088


def try_download(url: str, dest: Path) -> bool:
    """HTTPS download using certifi CA bundle (avoids macOS SSL verify failures)."""
    try:
        import ssl
        import urllib.request

        import certifi

        dest.parent.mkdir(parents=True, exist_ok=True)
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:  # noqa: S310
            dest.write_bytes(resp.read())
        return dest.exists()
    except Exception:  # noqa: BLE001
        return False


def load_land_boundary_points(cache_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Fallback: sample Natural Earth land polygon boundaries (distance ~ distance to coast)."""
    try:
        import geopandas as gpd
        from shapely.ops import unary_union
    except ImportError:
        return None
    path = cache_dir / "ne_110m_land.geojson"
    if not path.exists():
        if not try_download(NE_LAND_URL, path):
            return None
    try:
        land = gpd.read_file(path)
        if hasattr(land.geometry, "union_all"):
            geom = land.geometry.union_all()
        else:
            geom = unary_union(land.geometry)
        b = geom.boundary
    except Exception:  # noqa: BLE001
        return None
    lat_list: list[float] = []
    lon_list: list[float] = []
    lines = []
    if b.geom_type == "LineString":
        lines = [b]
    elif b.geom_type == "MultiLineString":
        lines = list(b.geoms)
    elif b.geom_type == "GeometryCollection":
        for g in b.geoms:
            if g.geom_type == "LineString":
                lines.append(g)
            elif g.geom_type == "MultiLineString":
                lines.extend(list(g.geoms))
    else:
        return None
    for line in lines:
        length = float(line.length)
        if length <= 0:
            continue
        n = max(3, min(800, int(length / 0.02)))
        for i in range(n + 1):
            p = line.interpolate(i / n, normalized=True)
            lat_list.append(float(p.y))
            lon_list.append(float(p.x))
    if not lat_list:
        return None
    return np.asarray(lat_list, dtype=float), np.asarray(lon_list, dtype=float)


def load_coastline_points(cache_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import geopandas as gpd
    except ImportError:
        return None
    path = cache_dir / "ne_110m_coastline.geojson"
    if not path.exists():
        if not try_download(NE_COASTLINE_URL, path):
            return None
    try:
        gdf = gpd.read_file(path)
    except Exception:  # noqa: BLE001
        return None
    lat_list: list[float] = []
    lon_list: list[float] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = []
        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue
        for line in lines:
            length = float(line.length)
            if length <= 0:
                continue
            n = max(3, min(500, int(length / 0.03)))
            for i in range(n + 1):
                p = line.interpolate(i / n, normalized=True)
                lat_list.append(float(p.y))
                lon_list.append(float(p.x))
    if not lat_list:
        return None
    return np.asarray(lat_list, dtype=float), np.asarray(lon_list, dtype=float)


def distance_to_coast_km_for_grids(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    coast_lat: np.ndarray,
    coast_lon: np.ndarray,
) -> np.ndarray:
    return haversine_nearest_distance_km(grid_lat, grid_lon, coast_lat, coast_lon)


def distance_bins_series(dist_km: pd.Series, prefix: str) -> pd.Series:
    """Bins: 0-10, 10-50, 50-100, 100+ km."""

    def _one(d: float) -> str | None:
        if pd.isna(d):
            return None
        if d < 10:
            return f"{prefix}_0_10_km"
        if d < 50:
            return f"{prefix}_10_50_km"
        if d < 100:
            return f"{prefix}_50_100_km"
        return f"{prefix}_100+_km"

    return dist_km.map(_one)


# --- shipping hubs (top 10% of grid-level vessel density) ---------------------


def shipping_hub_locations(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    g = (
        df.groupby("grid_cell_id", as_index=False)
        .agg(
            vessel_med=("vessel_density_t", "median"),
            grid_centroid_lat=("grid_centroid_lat", "first"),
            grid_centroid_lon=("grid_centroid_lon", "first"),
        )
        .dropna(subset=["vessel_med", "grid_centroid_lat", "grid_centroid_lon"])
    )
    if g.empty:
        return np.array([]), np.array([])
    thr = g["vessel_med"].quantile(0.90)
    hubs = g[g["vessel_med"] >= thr]
    return hubs["grid_centroid_lat"].to_numpy(), hubs["grid_centroid_lon"].to_numpy()


def distance_to_shipping_km_for_grids(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    hub_lat: np.ndarray,
    hub_lon: np.ndarray,
) -> np.ndarray:
    if hub_lat.size == 0:
        return np.full(grid_lat.shape[0], np.nan)
    return haversine_nearest_distance_km(grid_lat, grid_lon, hub_lat, hub_lon)


# --- optional NO2 -------------------------------------------------------------


def try_load_no2_parquet(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        n = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None
    n.columns = [str(c).strip() for c in n.columns]
    low = {c.lower(): c for c in n.columns}
    gcol = low.get("grid_cell_id") or low.get("grid_id")
    wcol = low.get("week_start_utc") or low.get("week") or low.get("time")
    vcol = None
    for key in ("no2_mean_t", "no2", "no2_value", "no₂"):
        if key in low:
            vcol = low[key]
            break
    if not gcol or not wcol or not vcol:
        return None
    out = n[[gcol, wcol, vcol]].copy()
    out.columns = ["grid_cell_id", "week_start_utc", "no2_mean_t"]
    out["week_start_utc"] = pd.to_datetime(out["week_start_utc"], utc=True, errors="coerce")
    out["no2_mean_t"] = pd.to_numeric(out["no2_mean_t"], errors="coerce")
    out = out.dropna(subset=["grid_cell_id", "week_start_utc"])
    # One value per grid-week (observational mean if duplicates)
    out = (
        out.groupby(["grid_cell_id", "week_start_utc"], as_index=False)["no2_mean_t"].mean()
    )
    return out


# --- optional oil (Sentinel-1 / proxy) ----------------------------------------


def try_load_oil_parquet(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        o = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None
    o.columns = [str(c).strip() for c in o.columns]
    low = {c.lower(): c for c in o.columns}
    gcol = low.get("grid_cell_id") or low.get("grid_id")
    wcol = low.get("week_start_utc") or low.get("week") or low.get("time")
    pcol = None
    pcol_key: str | None = None
    for key in ("oil_probability", "oil_slick_probability", "oil_prob", "oil_mask"):
        if key in low:
            pcol = low[key]
            pcol_key = key
            break
    ccol = None
    for key in ("oil_count", "oil_slick_count", "n_oil", "oil_events"):
        if key in low:
            ccol = low[key]
            break
    if not gcol or not wcol:
        return None
    if not pcol and not ccol:
        return None
    if pcol and ccol:
        out = o[[gcol, wcol, pcol, ccol]].copy()
        out.columns = [
            "grid_cell_id",
            "week_start_utc",
            "oil_slick_probability_t",
            "oil_slick_count_t",
        ]
    elif pcol:
        out = o[[gcol, wcol, pcol]].copy()
        out.columns = ["grid_cell_id", "week_start_utc", "oil_slick_probability_t"]
        out["oil_slick_count_t"] = 0.0
    else:
        out = o[[gcol, wcol, ccol]].copy()
        out.columns = ["grid_cell_id", "week_start_utc", "oil_slick_count_t"]
        out["oil_slick_probability_t"] = np.nan
    out["week_start_utc"] = pd.to_datetime(out["week_start_utc"], utc=True, errors="coerce")
    out["oil_slick_probability_t"] = pd.to_numeric(out["oil_slick_probability_t"], errors="coerce")
    out["oil_slick_count_t"] = pd.to_numeric(out["oil_slick_count_t"], errors="coerce").fillna(0.0)
    if pcol_key == "oil_mask":
        out["oil_slick_probability_t"] = out["oil_slick_probability_t"].clip(0, 1)
    out = out.dropna(subset=["grid_cell_id", "week_start_utc"])
    out = (
        out.groupby(["grid_cell_id", "week_start_utc"], as_index=False)
        .agg(
            oil_slick_probability_t=("oil_slick_probability_t", "mean"),
            oil_slick_count_t=("oil_slick_count_t", "sum"),
        )
    )
    return out


def add_no2_anomaly(df: pd.DataFrame, window: int = 12, min_periods: int = 4) -> pd.DataFrame:
    out = df.sort_values(["grid_cell_id", "week_start_utc"]).copy()
    out["no2_baseline_t"] = (
        out.groupby("grid_cell_id", group_keys=False)["no2_mean_t"]
        .transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    )
    out["no2_anomaly_t"] = out["no2_mean_t"] - out["no2_baseline_t"]
    return out


# --- activity regimes ---------------------------------------------------------


def assign_activity_regime(df: pd.DataFrame, use_no2: bool) -> pd.DataFrame:
    if use_no2:
        wk = df.groupby("week_start_utc", as_index=False).agg(
            mean_vessel=("vessel_density_t", "mean"),
            mean_ndti=("sentinel_ndti_mean_t", "mean"),
            mean_no2=("no2_mean_t", "mean"),
        )
    else:
        wk = df.groupby("week_start_utc", as_index=False).agg(
            mean_vessel=("vessel_density_t", "mean"),
            mean_ndti=("sentinel_ndti_mean_t", "mean"),
        )
    mv = wk["mean_vessel"].astype(float)
    m_ndti = wk["mean_ndti"].astype(float)
    if use_no2 and "mean_no2" in wk.columns and wk["mean_no2"].notna().sum() >= max(
        3, len(wk) // 4
    ):
        mn = pd.to_numeric(wk["mean_no2"], errors="coerce").fillna(
            pd.to_numeric(wk["mean_no2"], errors="coerce").median()
        )
        score = (
            0.35 * mv.rank(pct=True)
            + 0.35 * m_ndti.rank(pct=True)
            + 0.3 * mn.rank(pct=True)
        )
    elif mv.nunique(dropna=True) > 1:
        score = 0.5 * mv.rank(pct=True) + 0.5 * m_ndti.rank(pct=True)
    else:
        # Vessel layer is time-flat in this pipeline (e.g. normalized); use global mean NDTI as proxy.
        score = m_ndti
    if len(score) == 0 or float(score.nunique(dropna=True)) <= 1.0:
        wk["activity_regime"] = "MID_PERIOD"
    else:
        q25 = float(score.quantile(0.25))
        q75 = float(score.quantile(0.75))
        wk["activity_regime"] = np.where(
            score <= q25,
            "LOW_ACTIVITY_PERIOD",
            np.where(score >= q75, "HIGH_ACTIVITY_PERIOD", "MID_PERIOD"),
        )
    return wk[["week_start_utc", "activity_regime"]]


# --- plotting -----------------------------------------------------------------


def plot_decay_binned(
    dist: pd.Series,
    y1: pd.Series,
    y2: pd.Series,
    *,
    title: str,
    xlab: str,
    path: Path,
    label1: str = "mean NDTI",
    label2: str = "mean NDVI",
    n_bins: int = 24,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame({"d": dist, "y1": y1, "y2": y2}).dropna(subset=["d"])
    if m.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data for decay curve", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    try:
        m["bin"] = pd.qcut(m["d"], q=min(n_bins, max(4, m["d"].nunique())), duplicates="drop")
    except Exception:  # noqa: BLE001
        m["bin"] = pd.cut(m["d"], bins=min(n_bins, 12))
    g = m.groupby("bin", observed=False).agg(y1=("y1", "mean"), y2=("y2", "mean"), dmid=("d", "mean"))
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(g["dmid"], g["y1"], color="#2ca02c", marker="o", label=label1)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(label1, color="#2ca02c")
    ax2 = ax1.twinx()
    ax2.plot(g["dmid"], g["y2"], color="#1f77b4", marker="s", alpha=0.85, label=label2)
    ax2.set_ylabel(label2, color="#1f77b4")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_no2_decay(
    dist: pd.Series,
    no2: pd.Series,
    *,
    title: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame({"d": dist, "no2": no2}).dropna()
    if m.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.text(
            0.5,
            0.55,
            "NO₂ not available",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.38,
            "(no joined NO₂ values — optional --no2-path)",
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    try:
        m["bin"] = pd.qcut(m["d"], q=min(20, max(4, m["d"].nunique())), duplicates="drop")
    except Exception:  # noqa: BLE001
        m["bin"] = pd.cut(m["d"], bins=12)
    g = m.groupby("bin", observed=False).agg(no2=("no2", "mean"), dmid=("d", "mean"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(g["dmid"], g["no2"], color="#d62728", marker="o")
    ax.set_xlabel("Distance to coast (km)")
    ax.set_ylabel("mean NO₂ (joined column)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_regime_comparison(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    g = df.groupby("activity_regime", observed=False).agg(
        mean_ndti=("sentinel_ndti_mean_t", "mean"),
        mean_ndvi=("sentinel_ndvi_mean_t", "mean"),
        mean_vessel=("vessel_density_t", "mean"),
        mean_no2_mean_t=("no2_mean_t", "mean"),
        mean_oil_slick_probability_t=("oil_slick_probability_t", "mean"),
    )
    order = ["LOW_ACTIVITY_PERIOD", "MID_PERIOD", "HIGH_ACTIVITY_PERIOD"]
    g = g.reindex([x for x in order if x in g.index])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(g))
    w = 0.15
    ax.bar(x - 2 * w, g["mean_ndti"], width=w, label="mean NDTI")
    ax.bar(x - 1 * w, g["mean_ndvi"], width=w, label="mean NDVI")
    ax.bar(x + 0 * w, g["mean_vessel"], width=w, label="mean vessel density")
    off = 1
    if g["mean_no2_mean_t"].notna().any():
        ax.bar(x + off * w, g["mean_no2_mean_t"], width=w, label="mean NO₂")
        off += 1
    if g["mean_oil_slick_probability_t"].notna().any():
        ax.bar(x + off * w, g["mean_oil_slick_probability_t"], width=w, label="mean oil prob.")
    ax.set_xticks(x)
    ax.set_xticklabels(list(g.index), rotation=12)
    ax.set_ylabel("Value (mixed scales — interpret relatively)")
    ax.set_title("Activity regime comparison (observational layers; no causal claim)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_multi_layer_decay(
    df: pd.DataFrame,
    path: Path,
    *,
    has_no2: bool,
    has_oil: bool,
) -> None:
    """Binned means: NDTI (coast & shipping), NO₂ & oil vs coast bin (observational)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cats = ["0_10_km", "10_50_km", "50_100_km", "100+_km"]
    labels = ["0–10 km", "10–50 km", "50–100 km", "100+ km"]

    def bin_label(s: str | None) -> str | None:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return None
        s = str(s)
        for c in cats:
            if c in s:
                return c
        return None

    dfc = df.copy()
    dfc["cb"] = dfc["coast_distance_bin"].map(bin_label)
    dfc["sb"] = dfc["shipping_distance_bin"].map(bin_label)

    ndti_coast = dfc.groupby("cb")["sentinel_ndti_mean_t"].mean().reindex(cats)
    ndti_ship = dfc.groupby("sb")["sentinel_ndti_mean_t"].mean().reindex(cats)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(cats))
    ax.plot(x, ndti_coast.values, marker="o", color="#1f77b4", label="NDTI vs coast bin")
    ax.plot(x, ndti_ship.values, marker="s", color="#2ca02c", label="NDTI vs shipping bin")
    ax.set_ylabel("mean NDTI", color="#333")
    ax.tick_params(axis="y", labelcolor="#333")

    show_no2 = bool(has_no2 and dfc["no2_mean_t"].notna().any())
    show_oil = bool(has_oil and dfc["oil_slick_probability_t"].notna().any())
    if show_no2:
        no2_coast = dfc.groupby("cb")["no2_mean_t"].mean().reindex(cats)
        ax2 = ax.twinx()
        if show_oil:
            ax2.spines["right"].set_position(("axes", 1.0))
        ax2.plot(x, no2_coast.values, color="#d62728", marker="^", alpha=0.95, label="NO₂ vs coast bin")
        ax2.set_ylabel("mean NO₂", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.legend(loc="upper right", bbox_to_anchor=(1.33, 1.0), fontsize=8)
    if show_oil:
        oil_coast = dfc.groupby("cb")["oil_slick_probability_t"].mean().reindex(cats)
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.12 if show_no2 else 1.0))
        ax3.plot(
            x,
            oil_coast.values,
            color="#ff7f0e",
            marker="d",
            alpha=0.95,
            label="Oil prob. vs coast",
        )
        ax3.set_ylabel("mean oil probability", color="#ff7f0e")
        ax3.tick_params(axis="y", labelcolor="#ff7f0e")
        ax3.legend(loc="lower right", bbox_to_anchor=(1.33, 0.15), fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Distance bin (coast / shipping alignment)")
    ax.set_title("Multi-layer decay comparison (observational; NaNs preserved in aggregates)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_oil_shipping(df: pd.DataFrame, path: Path) -> None:
    """vessel_density_t vs oil_slick_probability_t (where both defined)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(
        {
            "v": pd.to_numeric(df["vessel_density_t"], errors="coerce"),
            "p": pd.to_numeric(df["oil_slick_probability_t"], errors="coerce"),
        }
    ).dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    if m.empty:
        ax.text(
            0.5,
            0.55,
            "Oil vs shipping: no overlapping data",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.38,
            "(need --oil-path with oil probability + vessel density)",
            ha="center",
            va="center",
            fontsize=9,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        if len(m) > 12_000:
            m = m.sample(n=12_000, random_state=42)
        ax.scatter(m["v"], m["p"], alpha=0.15, s=5, c="#444")
        ax.set_xlabel("vessel_density_t")
        ax.set_ylabel("oil_slick_probability_t")
        ax.set_title("Oil probability vs vessel density (observational)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sanity_checks(df: pd.DataFrame, sanity_dir: Path) -> dict[str, str]:
    """Histograms and scatter checks for distance features vs NDTI."""
    sanity_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    dc = pd.to_numeric(df["distance_to_coast_km"], errors="coerce")
    ds = pd.to_numeric(df["distance_to_shipping_km"], errors="coerce")
    ndti = pd.to_numeric(df["sentinel_ndti_mean_t"], errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 4))
    dc.dropna().hist(bins=48, ax=ax, color="#1f77b4", edgecolor="white", alpha=0.85)
    ax.set_xlabel("distance_to_coast_km")
    ax.set_ylabel("count")
    ax.set_title("Sanity: distance to coast (km)")
    fig.tight_layout()
    p = sanity_dir / "hist_distance_to_coast_km.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["hist_distance_to_coast_km"] = str(p.resolve())

    fig, ax = plt.subplots(figsize=(8, 4))
    ds.dropna().hist(bins=48, ax=ax, color="#2ca02c", edgecolor="white", alpha=0.85)
    ax.set_xlabel("distance_to_shipping_km")
    ax.set_ylabel("count")
    ax.set_title("Sanity: distance to shipping hubs (km)")
    fig.tight_layout()
    p = sanity_dir / "hist_distance_to_shipping_km.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths["hist_distance_to_shipping_km"] = str(p.resolve())

    def _scatter(x: pd.Series, y: pd.Series, xlab: str, fname: str, key: str) -> None:
        m = pd.DataFrame({"x": x, "y": y}).dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        if len(m) > 12_000:
            m = m.sample(n=12_000, random_state=42)
        ax.scatter(m["x"], m["y"], alpha=0.12, s=4, c="#444")
        ax.set_xlabel(xlab)
        ax.set_ylabel("sentinel_ndti_mean_t")
        ax.set_title(f"Sanity: {xlab} vs NDTI (subsample if large)")
        fig.tight_layout()
        outp = sanity_dir / fname
        fig.savefig(outp, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths[key] = str(outp.resolve())

    _scatter(dc, ndti, "distance_to_coast_km", "scatter_coast_vs_ndti.png", "scatter_coast_vs_ndti")
    _scatter(
        ds, ndti, "distance_to_shipping_km", "scatter_shipping_vs_ndti.png", "scatter_shipping_vs_ndti"
    )

    return paths


def build_run_summary(df: pd.DataFrame) -> dict[str, Any]:
    rows_total = int(len(df))
    rw_no2 = int(df["no2_mean_t"].notna().sum())
    cov_no2 = 100.0 * rw_no2 / rows_total if rows_total else 0.0
    miss_no2 = 1.0 - (rw_no2 / rows_total if rows_total else 0.0)
    oil_obs = df["oil_slick_probability_t"].notna() | (
        pd.to_numeric(df["oil_slick_count_t"], errors="coerce").fillna(0) > 0
    )
    rw_oil = int(oil_obs.sum())
    cov_oil = 100.0 * rw_oil / rows_total if rows_total else 0.0
    oil_ph = df["oil_slick_probability_t"].isna() & (
        pd.to_numeric(df["oil_slick_count_t"], errors="coerce").fillna(0) == 0
    )
    return {
        "rows_total": rows_total,
        "rows_with_distance_to_coast": int(df["distance_to_coast_km"].notna().sum()),
        "rows_with_shipping_distance": int(df["distance_to_shipping_km"].notna().sum()),
        "rows_with_no2": rw_no2,
        "no2_coverage_percent": round(cov_no2, 4),
        "no2_missing_ratio": round(miss_no2, 6),
        "rows_with_oil_data": rw_oil,
        "oil_coverage_percent": round(cov_oil, 4),
        "rows_with_oil_placeholder": int(oil_ph.sum()),
        "number_of_grids": int(df["grid_cell_id"].nunique()),
        "number_of_weeks": int(df["week_start_utc"].nunique()),
    }


def format_run_summary_text(run_summary: dict[str, Any]) -> str:
    lines = [
        "RUN SUMMARY",
        "-----------",
        f"rows_total: {run_summary['rows_total']}",
        f"rows_with_distance_to_coast: {run_summary['rows_with_distance_to_coast']}",
        f"rows_with_shipping_distance: {run_summary['rows_with_shipping_distance']}",
        f"rows_with_no2: {run_summary['rows_with_no2']}",
        f"no2_coverage_percent: {run_summary['no2_coverage_percent']}",
        f"no2_missing_ratio: {run_summary['no2_missing_ratio']}",
        f"rows_with_oil_data: {run_summary['rows_with_oil_data']}",
        f"oil_coverage_percent: {run_summary['oil_coverage_percent']}",
        f"rows_with_oil_placeholder: {run_summary['rows_with_oil_placeholder']}",
        f"number_of_grids: {run_summary['number_of_grids']}",
        f"number_of_weeks: {run_summary['number_of_weeks']}",
    ]
    return "\n".join(lines)


# --- main pipeline ------------------------------------------------------------


def run(
    input_parquet: Path,
    out_data_dir: Path,
    viz_dir: Path,
    model_results_path: Path,
    cache_dir: Path,
    no2_path: Path | None,
    oil_path: Path | None,
    enriched_parquet: Path | None,
    skip_model_results: bool,
) -> dict[str, Any]:
    df = pd.read_parquet(input_parquet)
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    for c in (
        "grid_centroid_lat",
        "grid_centroid_lon",
        "vessel_density_t",
        "sentinel_ndti_mean_t",
        "sentinel_ndvi_mean_t",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    grids = (
        df.groupby("grid_cell_id", as_index=False)
        .agg(
            grid_centroid_lat=("grid_centroid_lat", "first"),
            grid_centroid_lon=("grid_centroid_lon", "first"),
        )
        .dropna(subset=["grid_centroid_lat", "grid_centroid_lon"])
    )

    coast_pts = load_coastline_points(cache_dir)
    coast_source = "ne_110m_coastline"
    if coast_pts is None:
        coast_pts = load_land_boundary_points(cache_dir)
        coast_source = "ne_110m_land_boundary_fallback"
    if coast_pts is not None:
        clat, clon = coast_pts
        grids["distance_to_coast_km"] = distance_to_coast_km_for_grids(
            grids["grid_centroid_lat"].to_numpy(),
            grids["grid_centroid_lon"].to_numpy(),
            clat,
            clon,
        )
    else:
        grids["distance_to_coast_km"] = np.nan
        coast_source = "none"

    hub_lat, hub_lon = shipping_hub_locations(df)
    grids["distance_to_shipping_km"] = distance_to_shipping_km_for_grids(
        grids["grid_centroid_lat"].to_numpy(),
        grids["grid_centroid_lon"].to_numpy(),
        hub_lat,
        hub_lon,
    )

    grids["distance_to_urban_proxy_km"] = np.nan  # optional / not available

    gmap = grids.set_index("grid_cell_id")[
        [
            "distance_to_coast_km",
            "distance_to_shipping_km",
            "distance_to_urban_proxy_km",
        ]
    ]
    df = df.join(gmap, on="grid_cell_id")

    for _c in ("no2_mean_t", "no2_baseline_t", "no2_anomaly_t"):
        if _c in df.columns:
            df = df.drop(columns=[_c])
    for _c in ("oil_slick_probability_t", "oil_slick_count_t"):
        if _c in df.columns:
            df = df.drop(columns=[_c])

    no2_df = try_load_no2_parquet(no2_path)
    has_no2_file = no2_df is not None
    if has_no2_file:
        df = df.merge(no2_df, on=["grid_cell_id", "week_start_utc"], how="left")
    if "no2_mean_t" not in df.columns:
        df["no2_mean_t"] = np.nan
    df = add_no2_anomaly(df)
    no2_usable = bool(df["no2_mean_t"].notna().sum() >= 10)

    oil_df = try_load_oil_parquet(oil_path)
    has_oil_file = oil_df is not None
    if has_oil_file:
        df = df.merge(oil_df, on=["grid_cell_id", "week_start_utc"], how="left")
        df["oil_slick_count_t"] = pd.to_numeric(df["oil_slick_count_t"], errors="coerce").fillna(0.0)
    else:
        df["oil_slick_probability_t"] = np.nan
        df["oil_slick_count_t"] = 0.0
    _oil_cnt = pd.to_numeric(df["oil_slick_count_t"], errors="coerce").fillna(0)
    oil_usable = bool(df["oil_slick_probability_t"].notna().any() or (_oil_cnt > 0).any())

    use_no2_regime = no2_usable
    regimes = assign_activity_regime(df, use_no2=use_no2_regime)
    df = df.merge(regimes, on="week_start_utc", how="left")

    df["coast_distance_bin"] = distance_bins_series(df["distance_to_coast_km"], "coast")
    df["shipping_distance_bin"] = distance_bins_series(df["distance_to_shipping_km"], "ship")

    out_data_dir.mkdir(parents=True, exist_ok=True)

    decay_tbl = (
        df.groupby(["coast_distance_bin", "shipping_distance_bin"], observed=False)
        .agg(
            mean_ndti=("sentinel_ndti_mean_t", "mean"),
            mean_ndvi=("sentinel_ndvi_mean_t", "mean"),
            mean_no2_mean_t=("no2_mean_t", "mean"),
            mean_no2_anomaly_t=("no2_anomaly_t", "mean"),
            mean_oil_slick_probability_t=("oil_slick_probability_t", "mean"),
            mean_vessel=("vessel_density_t", "mean"),
            n=("grid_cell_id", "count"),
        )
        .reset_index()
        .sort_values(["coast_distance_bin", "shipping_distance_bin"])
    )
    decay_path = out_data_dir / "distance_decay_summary.csv"
    decay_tbl.to_csv(decay_path, index=False)

    event_tbl = (
        df.groupby("activity_regime", observed=False)
        .agg(
            mean_ndvi=("sentinel_ndvi_mean_t", "mean"),
            mean_ndti=("sentinel_ndti_mean_t", "mean"),
            mean_no2_mean_t=("no2_mean_t", "mean"),
            mean_no2_anomaly_t=("no2_anomaly_t", "mean"),
            mean_oil_slick_probability_t=("oil_slick_probability_t", "mean"),
            mean_oil_slick_count_t=("oil_slick_count_t", "mean"),
            mean_vessel_density=("vessel_density_t", "mean"),
            n_rows=("grid_cell_id", "count"),
        )
        .reset_index()
    )
    event_path = out_data_dir / "event_regime_summary.csv"
    event_tbl.to_csv(event_path, index=False)

    # Figures
    plot_decay_binned(
        df["distance_to_coast_km"],
        df["sentinel_ndti_mean_t"],
        df["sentinel_ndvi_mean_t"],
        title="Coast distance decay (binned means)",
        xlab="Geodesic distance to nearest coastline (km)",
        path=viz_dir / "coast_decay_curve.png",
    )
    plot_decay_binned(
        df["distance_to_shipping_km"],
        df["sentinel_ndti_mean_t"],
        df["sentinel_ndvi_mean_t"],
        title="Shipping-hub distance decay (binned means)",
        xlab="Distance to top-10% vessel-density hubs (km)",
        path=viz_dir / "shipping_decay_curve.png",
    )
    plot_no2_decay(
        df["distance_to_coast_km"],
        df["no2_mean_t"],
        title="NO₂ vs distance to coast (joined NO₂; may be empty)",
        path=viz_dir / "no2_decay_curve.png",
    )
    plot_oil_shipping(df, viz_dir / "oil_shipping_relationship.png")
    plot_regime_comparison(df, viz_dir / "activity_regime_comparison.png")
    plot_multi_layer_decay(
        df,
        viz_dir / "multi_layer_decay_comparison.png",
        has_no2=no2_usable,
        has_oil=oil_usable,
    )

    sanity_dir = viz_dir / "sanity"
    sanity_figures = plot_sanity_checks(df, sanity_dir)

    run_summary = build_run_summary(df)
    run_summary_path = out_data_dir / "run_summary.json"
    with run_summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print(format_run_summary_text(run_summary))
    print(f"Saved: {run_summary_path.resolve()}")

    if enriched_parquet is not None:
        enriched_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(enriched_parquet, index=False)

    summary: dict[str, Any] = {
        "run_summary": run_summary,
        "sanity_figures": sanity_figures,
        "inputs": {
            "modeling_parquet": str(input_parquet.resolve()),
            "natural_earth_cache": str(cache_dir.resolve()),
            "coast_distance_source": coast_source,
            "coastline_loaded": coast_pts is not None,
            "n_shipping_hubs": int(hub_lat.size),
            "no2_join_path": str(no2_path.resolve()) if no2_path and no2_path.exists() else None,
            "oil_join_path": str(oil_path.resolve()) if oil_path and oil_path.exists() else None,
            "no2_rows_joined_non_null": int(df["no2_mean_t"].notna().sum()),
            "no2_usable_for_regime": use_no2_regime,
        },
        "outputs": {
            "distance_decay_summary_csv": str(decay_path.resolve()),
            "event_regime_summary_csv": str(event_path.resolve()),
            "figures": {
                "coast_decay_curve_png": str((viz_dir / "coast_decay_curve.png").resolve()),
                "shipping_decay_curve_png": str((viz_dir / "shipping_decay_curve.png").resolve()),
                "no2_decay_curve_png": str((viz_dir / "no2_decay_curve.png").resolve()),
                "activity_regime_comparison_png": str(
                    (viz_dir / "activity_regime_comparison.png").resolve()
                ),
                "multi_layer_decay_comparison_png": str(
                    (viz_dir / "multi_layer_decay_comparison.png").resolve()
                ),
                "oil_shipping_relationship_png": str(
                    (viz_dir / "oil_shipping_relationship.png").resolve()
                ),
            },
            "enriched_parquet": str(enriched_parquet.resolve()) if enriched_parquet else None,
            "run_summary_json": str(run_summary_path.resolve()),
            "sanity_directory": str(sanity_dir.resolve()),
        },
        "activity_regime_note": (
            "Global mean vessel_density_t is constant over weeks in this parquet; "
            "regimes use weekly mean NDTI as the primary score (LOW/HIGH = bottom/top 25% weeks). "
            "When NO₂ is joined, ranks combine vessel, NDTI, and NO₂. "
            "NO₂ and oil are observational layers only; missing values are not imputed."
        ),
        "columns_added": [
            "distance_to_coast_km",
            "distance_to_shipping_km",
            "distance_to_urban_proxy_km",
            "no2_mean_t",
            "no2_baseline_t",
            "no2_anomaly_t",
            "oil_slick_probability_t",
            "oil_slick_count_t",
            "activity_regime",
            "coast_distance_bin",
            "shipping_distance_bin",
        ],
    }

    if not skip_model_results and model_results_path.exists():
        with model_results_path.open(encoding="utf-8") as f:
            mr = json.load(f)
        mr["distance_decay_analysis"] = summary
        mr["no2_analysis"] = {
            "no2_source_available": has_no2_file,
            "no2_non_null_values": int(df["no2_mean_t"].notna().sum()),
            "no2_coverage_percent": run_summary["no2_coverage_percent"],
            "no2_missing_ratio": run_summary["no2_missing_ratio"],
            "rolling_baseline_window_weeks": 12,
            "rolling_grouping": "per grid_cell_id, sorted by week_start_utc",
            "anomaly_column": "no2_anomaly_t",
            "note": "Observational only; NaNs preserved. Without --no2-path data, columns are NaN.",
        }
        mr["oil_analysis"] = {
            "oil_source_available": has_oil_file,
            "rows_with_oil_data": run_summary["rows_with_oil_data"],
            "oil_coverage_percent": run_summary["oil_coverage_percent"],
            "note": "Observational only; oil_slick_probability_t may be NaN; count uses 0 when absent.",
        }
        with model_results_path.open("w", encoding="utf-8") as f:
            json.dump(mr, f, indent=2)

    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Human impact + distance + event analysis layer")
    p.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "modeling_dataset.parquet",
    )
    p.add_argument("--out-data-dir", type=Path, default=root / "data")
    p.add_argument("--viz-dir", type=Path, default=root / "data" / "visualizations")
    p.add_argument("--model-results", type=Path, default=root / "data" / "model_results.json")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "data" / "downloads" / "natural_earth",
    )
    p.add_argument(
        "--no2-path",
        type=Path,
        default=root / "data" / "aux" / "no2_grid_week.parquet",
        help="Optional parquet: grid_cell_id, week_start_utc, no2 / no2_mean_t / no2_value",
    )
    p.add_argument(
        "--oil-path",
        type=Path,
        default=root / "data" / "aux" / "sentinel1_oil_slicks.parquet",
        help="Optional parquet (fallback: oil_slicks.parquet if default missing): grid_cell_id, week_start_utc, oil_*",
    )
    p.add_argument(
        "--enriched-parquet",
        type=Path,
        default=None,
        help="Write full enriched table (default: data/modeling_dataset_human_impact.parquet)",
    )
    p.add_argument("--skip-model-results", action="store_true")
    args = p.parse_args()
    ep = args.enriched_parquet
    if ep is None:
        ep = root / "data" / "modeling_dataset_human_impact.parquet"
    elif not ep.is_absolute():
        ep = root / ep
    inp = args.input if args.input.is_absolute() else root / args.input
    no2_p = args.no2_path if args.no2_path.is_absolute() else root / args.no2_path
    oil_p = args.oil_path if args.oil_path.is_absolute() else root / args.oil_path
    alt_oil = root / "data" / "aux" / "oil_slicks.parquet"
    if not oil_p.exists() and alt_oil.exists():
        oil_p = alt_oil
    run(
        input_parquet=inp,
        out_data_dir=args.out_data_dir if args.out_data_dir.is_absolute() else root / args.out_data_dir,
        viz_dir=args.viz_dir if args.viz_dir.is_absolute() else root / args.viz_dir,
        model_results_path=(
            args.model_results if args.model_results.is_absolute() else root / args.model_results
        ),
        cache_dir=args.cache_dir if args.cache_dir.is_absolute() else root / args.cache_dir,
        no2_path=no2_p,
        oil_path=oil_p,
        enriched_parquet=ep,
        skip_model_results=args.skip_model_results,
    )
    print("Human impact distance analysis complete.")


if __name__ == "__main__":
    main()
