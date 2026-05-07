#!/usr/bin/env python3
"""
Nearest-land NDVI linkage: attach each maritime coastal grid to the closest land NDVI cell.

Writes ONLY under outputs/{reports,figures,visualizations}/run_nearest_land_ndvi_linkage/.

Run:
  python3 src/analysis/run_nearest_land_ndvi_linkage.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.neighbors import BallTree

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from human_impact_distance_analysis import (  # noqa: E402
    distance_to_coast_km_for_grids,
    load_coastline_points,
    load_land_boundary_points,
)

RUN = "run_nearest_land_ndvi_linkage"
REPORTS = _ROOT / "outputs" / "reports" / RUN
FIGURES = _ROOT / "outputs" / "figures" / RUN
VIZ = _ROOT / "outputs" / "visualizations" / RUN

LAND_NDVI_COAST_KM = 30.0
MARITIME_COAST_KM = 50.0
BAND_LABELS = ["0-3 km", "3-7 km", "7-15 km", "15-30 km"]
EARTH_R_KM = 6371.0088

REQ_COLS = [
    "grid_cell_id",
    "week_start_utc",
    "grid_centroid_lat",
    "grid_centroid_lon",
    "vessel_density_t",
    "distance_to_nearest_high_vessel_density_cell",
    "oil_slick_probability_t",
    "detection_score",
    "no2_mean_t",
    "ndvi_mean",
]


def ensure_coast_distance(df: pd.DataFrame, cache: Path) -> pd.DataFrame:
    out = df.copy()
    dc_ok = (
        "distance_to_coast_km" in out.columns
        and float(out["distance_to_coast_km"].notna().mean()) > 0.99
    )
    if dc_ok:
        return out
    uniq = (
        out[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]]
        .drop_duplicates("grid_cell_id")
        .dropna(subset=["grid_centroid_lat", "grid_centroid_lon"])
    )
    pts = load_coastline_points(cache) or load_land_boundary_points(cache)
    if pts is None:
        out["distance_to_coast_km"] = np.nan
        return out
    lat0, lon0 = pts
    dc = distance_to_coast_km_for_grids(
        uniq["grid_centroid_lat"].to_numpy(dtype=float),
        uniq["grid_centroid_lon"].to_numpy(dtype=float),
        lat0,
        lon0,
    )
    m = pd.Series(dc, index=uniq["grid_cell_id"].astype(str))
    out["distance_to_coast_km"] = out["grid_cell_id"].astype(str).map(m)
    return out


def assign_shipping_band(dship: pd.Series) -> pd.Series:
    x = pd.to_numeric(dship, errors="coerce")
    out = pd.Series(np.nan, index=dship.index, dtype=object)
    out.loc[(x >= 0) & (x < 3)] = BAND_LABELS[0]
    out.loc[(x >= 3) & (x < 7)] = BAND_LABELS[1]
    out.loc[(x >= 7) & (x < 15)] = BAND_LABELS[2]
    out.loc[(x >= 15) & (x < 30)] = BAND_LABELS[3]
    out.loc[x >= 30] = np.nan
    return out


def haversine_nearest_land_indices(
    q_lat: np.ndarray,
    q_lon: np.ndarray,
    land_lat: np.ndarray,
    land_lon: np.ndarray,
    land_ids: np.ndarray,
    q_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query point, distance (km) and index into land_* arrays of nearest land cell with grid_id != q_id."""
    tgt = np.deg2rad(np.c_[land_lat.astype(float), land_lon.astype(float)])
    src = np.deg2rad(np.c_[q_lat.astype(float), q_lon.astype(float)])
    tree = BallTree(tgt, metric="haversine")
    k = min(max(8, 1), len(land_lat))
    dist_rad, ind = tree.query(src, k=k)
    dist_km = dist_rad * EARTH_R_KM
    chosen_d = np.full(len(src), np.nan)
    chosen_j = np.full(len(src), -1, dtype=np.int64)
    for i in range(len(src)):
        qid = str(q_ids[i])
        for j in range(dist_km.shape[1]):
            jj = int(ind[i, j])
            if str(land_ids[jj]) != qid:
                chosen_d[i] = float(dist_km[i, j])
                chosen_j[i] = jj
                break
        if chosen_j[i] < 0 and len(land_lat):
            jj = int(ind[i, 0])
            chosen_d[i] = float(dist_km[i, 0])
            chosen_j[i] = jj
    return chosen_d, chosen_j


def _cohens_d(hi: np.ndarray, lo: np.ndarray) -> float:
    hi, lo = hi[np.isfinite(hi)], lo[np.isfinite(lo)]
    if len(hi) < 2 or len(lo) < 2:
        return float("nan")
    v1, v2 = float(np.var(hi, ddof=1)), float(np.var(lo, ddof=1))
    n1, n2 = len(hi), len(lo)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1))
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((float(np.mean(hi)) - float(np.mean(lo))) / pooled)


def high_low_compare(
    df: pd.DataFrame, col: str, labels: pd.Series
) -> dict[str, Any]:
    sub = df.loc[labels.isin(["high", "low"])].copy()
    sub["_vx"] = labels.loc[sub.index]
    x = pd.to_numeric(sub[col], errors="coerce")
    hi = x.loc[sub["_vx"] == "high"].dropna().to_numpy(dtype=float)
    lo = x.loc[sub["_vx"] == "low"].dropna().to_numpy(dtype=float)
    hn, ln = len(hi), len(lo)
    r: dict[str, Any] = {
        "indicator": col,
        "n_high": hn,
        "n_low": ln,
        "high_mean": float(np.mean(hi)) if hn else np.nan,
        "low_mean": float(np.mean(lo)) if ln else np.nan,
        "mean_difference": float(np.mean(hi) - np.mean(lo)) if hn and ln else np.nan,
        "cohens_d": _cohens_d(hi, lo),
    }
    if hn >= 3 and ln >= 3:
        r["welch_p"] = float(stats.ttest_ind(hi, lo, equal_var=False).pvalue)
        r["mann_whitney_p"] = float(stats.mannwhitneyu(hi, lo, alternative="two-sided").pvalue)
    else:
        r["welch_p"], r["mann_whitney_p"] = np.nan, np.nan
    return r


def vessel_high_low_labels(v: pd.Series) -> pd.Series:
    lg = pd.Series(np.nan, index=v.index)
    m = pd.to_numeric(v, errors="coerce")
    lg = np.log1p(m.clip(lower=0))
    lab = pd.Series("exclude", index=v.index)
    finite = lg.notna()
    if int(finite.sum()) < 30:
        return lab
    q90, q10 = lg.loc[finite].quantile(0.90), lg.loc[finite].quantile(0.10)
    hi, lo = lg >= q90, lg <= q10
    lab.loc[finite & hi] = "high"
    lab.loc[finite & lo] = "low"
    lab.loc[finite & ~(hi | lo)] = "mid"
    return lab


def spearman_row(a: pd.Series, b: pd.Series, name: str) -> dict[str, Any]:
    m = a.notna() & b.notna()
    aa = pd.to_numeric(a.loc[m], errors="coerce")
    bb = pd.to_numeric(b.loc[m], errors="coerce")
    m2 = aa.notna() & bb.notna()
    aa, bb = aa[m2], bb[m2]
    if len(aa) < 8:
        return {"pair": name, "spearman_r": np.nan, "p_value": np.nan, "n": len(aa)}
    rho, p = stats.spearmanr(aa, bb)
    return {"pair": name, "spearman_r": float(rho), "p_value": float(p), "n": len(aa)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "final_run_stockholm_fixed_20260505_1356" / "processed" / "features_ml_ready.parquet",
    )
    ap.add_argument("--ne-cache", type=Path, default=_ROOT / "data" / "aux" / "natural_earth_coast_cache")
    args = ap.parse_args()
    inp = args.input if args.input.is_absolute() else _ROOT / args.input

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)

    if not inp.is_file():
        print(f"[FATAL] missing {inp}")
        return 1

    df = pd.read_parquet(inp)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        print(f"[FATAL] missing columns: {missing}")
        return 1

    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")

    df = ensure_coast_distance(df, Path(args.ne_cache))

    n_total = len(df)
    dc_all = pd.to_numeric(df["distance_to_coast_km"], errors="coerce")
    ndvi = pd.to_numeric(df["ndvi_mean"], errors="coerce")

    # STEP 2 — land NDVI candidates
    land_mask = ndvi.notna() & (dc_all <= LAND_NDVI_COAST_KM)
    land_rows = df.loc[land_mask, ["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon", "ndvi_mean"]].copy()
    land_rows["ndvi_mean"] = pd.to_numeric(land_rows["ndvi_mean"], errors="coerce")

    agg = (
        land_rows.groupby("grid_cell_id", as_index=False)
        .agg(
            mean_ndvi=("ndvi_mean", "mean"),
            median_ndvi=("ndvi_mean", "median"),
            valid_ndvi_count=("ndvi_mean", "count"),
            lat=("grid_centroid_lat", "first"),
            lon=("grid_centroid_lon", "first"),
        )
    )
    agg.to_csv(REPORTS / "land_ndvi_candidate_cells.csv", index=False)
    n_land_cells = len(agg)

    # STEP 3 — maritime grids
    vd = pd.to_numeric(df["vessel_density_t"], errors="coerce")
    oil = pd.to_numeric(df["oil_slick_probability_t"], errors="coerce")
    maritime_grid_mask = dc_all.notna() & (dc_all <= MARITIME_COAST_KM) & (vd.notna() | oil.notna())
    grids_mar = (
        df.loc[maritime_grid_mask, ["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]]
        .drop_duplicates("grid_cell_id")
        .dropna(subset=["grid_centroid_lat", "grid_centroid_lon"])
    )

    linkage_rows: list[dict[str, Any]] = []
    if n_land_cells == 0 or len(grids_mar) == 0:
        link_df = pd.DataFrame(
            columns=[
                "maritime_grid_cell_id",
                "grid_centroid_lat",
                "grid_centroid_lon",
                "nearest_land_grid_cell_id",
                "distance_to_nearest_land_ndvi_km",
                "nearest_land_ndvi_mean",
                "nearest_land_ndvi_median",
                "nearest_land_ndvi_valid_count",
            ]
        )
    else:
        land_lat = agg["lat"].to_numpy(dtype=float)
        land_lon = agg["lon"].to_numpy(dtype=float)
        land_mean = agg["mean_ndvi"].to_numpy(dtype=float)
        land_median = agg["median_ndvi"].to_numpy(dtype=float)
        land_cnt = agg["valid_ndvi_count"].to_numpy(dtype=int)
        land_gids = agg["grid_cell_id"].astype(str).to_numpy()

        q_lat = grids_mar["grid_centroid_lat"].to_numpy(dtype=float)
        q_lon = grids_mar["grid_centroid_lon"].to_numpy(dtype=float)
        q_ids = grids_mar["grid_cell_id"].astype(str).to_numpy()

        d_km, j_idx = haversine_nearest_land_indices(q_lat, q_lon, land_lat, land_lon, land_gids, q_ids)

        for i, gid in enumerate(grids_mar["grid_cell_id"].astype(str)):
            jj = int(j_idx[i])
            linkage_rows.append(
                {
                    "maritime_grid_cell_id": gid,
                    "grid_centroid_lat": float(q_lat[i]),
                    "grid_centroid_lon": float(q_lon[i]),
                    "nearest_land_grid_cell_id": str(agg.iloc[jj]["grid_cell_id"]),
                    "distance_to_nearest_land_ndvi_km": float(d_km[i]) if np.isfinite(d_km[i]) else np.nan,
                    "nearest_land_ndvi_mean": float(land_mean[jj]),
                    "nearest_land_ndvi_median": float(land_median[jj]),
                    "nearest_land_ndvi_valid_count": int(land_cnt[jj]),
                }
            )
        link_df = pd.DataFrame(linkage_rows)
    link_df.to_csv(REPORTS / "maritime_to_nearest_land_linkage.csv", index=False)
    n_mar_linked = len(link_df)

    # STEP 4 — merge to weekly rows
    link_map = link_df.rename(
        columns={
            "maritime_grid_cell_id": "grid_cell_id",
        }
    )
    if not link_map.empty:
        link_map = link_map.drop(columns=[c for c in ["grid_centroid_lat", "grid_centroid_lon"] if c in link_map.columns])
        link_map["grid_cell_id"] = link_map["grid_cell_id"].astype(str)
    df_out = df.copy()
    df_out["grid_cell_id"] = df_out["grid_cell_id"].astype(str)
    if not link_map.empty:
        df_out = df_out.merge(link_map, on="grid_cell_id", how="left")
    else:
        for c in [
            "nearest_land_grid_cell_id",
            "distance_to_nearest_land_ndvi_km",
            "nearest_land_ndvi_mean",
            "nearest_land_ndvi_median",
            "nearest_land_ndvi_valid_count",
        ]:
            df_out[c] = np.nan

    df_out["shipping_distance_band_refined"] = assign_shipping_band(
        pd.to_numeric(df_out["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    )

    out_pq = REPORTS / "nearest_land_ndvi_linked_dataset.parquet"
    df_out.to_parquet(out_pq, index=False)

    analysis = df_out[df_out["nearest_land_grid_cell_id"].notna()].copy()

    # STEP 6A — decay
    decay_inds = [
        "nearest_land_ndvi_mean",
        "nearest_land_ndvi_median",
        "no2_mean_t",
        "oil_slick_probability_t",
    ]
    decay_out_fixed: list[dict[str, Any]] = []
    for bl in BAND_LABELS:
        slab = analysis.loc[analysis["shipping_distance_band_refined"] == bl]
        ug = int(slab["grid_cell_id"].nunique()) if len(slab) else 0
        for col in decay_inds:
            if col not in slab.columns:
                continue
            x = pd.to_numeric(slab[col], errors="coerce").dropna()
            decay_out_fixed.append(
                {
                    "shipping_distance_band_refined": bl,
                    "variable": col,
                    "mean": float(x.mean()) if len(x) else np.nan,
                    "median": float(x.median()) if len(x) else np.nan,
                    "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                    "valid_sample_count": int(len(x)),
                    "unique_grid_count": ug,
                }
            )
    pd.DataFrame(decay_out_fixed).to_csv(REPORTS / "nearest_land_distance_decay.csv", index=False)

    # STEP 6B — coastal high vs low vessel (≤50 km maritime + linked)
    coastal_linked = df_out[
        df_out["nearest_land_grid_cell_id"].notna()
        & pd.to_numeric(df_out["distance_to_coast_km"], errors="coerce").notna()
        & (pd.to_numeric(df_out["distance_to_coast_km"], errors="coerce") <= MARITIME_COAST_KM)
    ].copy()
    vx = vessel_high_low_labels(coastal_linked["vessel_density_t"])
    hl_inds = ["nearest_land_ndvi_mean", "nearest_land_ndvi_median", "no2_mean_t", "oil_slick_probability_t"]
    hl_rows = [high_low_compare(coastal_linked, c, vx) for c in hl_inds if c in coastal_linked.columns]
    pd.DataFrame(hl_rows).to_csv(REPORTS / "nearest_land_high_vs_low_vessel.csv", index=False)

    # STEP 7 — correlations
    cor_rows = []
    if len(analysis):
        tgt = pd.to_numeric(analysis["nearest_land_ndvi_mean"], errors="coerce")
        pairs = [
            ("oil_slick_probability_t_vs_nearest_land_ndvi_mean", "oil_slick_probability_t"),
            ("detection_score_vs_nearest_land_ndvi_mean", "detection_score"),
            ("no2_mean_t_vs_nearest_land_ndvi_mean", "no2_mean_t"),
            ("vessel_density_t_vs_nearest_land_ndvi_mean", "vessel_density_t"),
            ("distance_to_nearest_high_vessel_density_cell_vs_nearest_land_ndvi_mean", "distance_to_nearest_high_vessel_density_cell"),
        ]
        for name, col in pairs:
            if col in analysis.columns:
                cor_rows.append(spearman_row(pd.to_numeric(analysis[col], errors="coerce"), tgt, name))
    pd.DataFrame(cor_rows).to_csv(REPORTS / "sea_land_correlation_check.csv", index=False)

    # Metrics for terminal
    med_link = float(np.nanmedian(link_df["distance_to_nearest_land_ndvi_km"])) if n_mar_linked and link_df["distance_to_nearest_land_ndvi_km"].notna().any() else float("nan")

    strongest_d, strongest_name = 0.0, ""
    for r in hl_rows:
        d = abs(float(r.get("cohens_d", np.nan)))
        if np.isfinite(d) and d >= strongest_d:
            strongest_d, strongest_name = d, str(r.get("indicator", ""))

    land_detected = "NO"
    if not hl_rows:
        land_detected = "INCONCLUSIVE"
    else:
        ok_test = False
        small = True
        for r in hl_rows:
            nh, nl = r.get("n_high", 0), r.get("n_low", 0)
            if nh >= 3 and nl >= 3:
                small = False
            d = r.get("cohens_d", np.nan)
            pw = r.get("welch_p", np.nan)
            pm = r.get("mann_whitney_p", np.nan)
            sig = (np.isfinite(pw) and pw < 0.05) or (np.isfinite(pm) and pm < 0.05)
            if np.isfinite(d) and abs(d) > 0.05 and sig:
                ok_test = True
        if small and not ok_test:
            land_detected = "INCONCLUSIVE"
        elif ok_test:
            land_detected = "YES"

    # STEP 8 — plots
    ddec = pd.DataFrame(decay_out_fixed)
    if not ddec.empty:
        for var, fname, ylab in [
            ("nearest_land_ndvi_mean", "nearest_land_ndvi_vs_shipping_band.png", "Nearest land NDVI (mean aggregate)"),
            ("oil_slick_probability_t", "oil_slick_vs_shipping_band.png", "oil_slick_probability_t"),
            ("no2_mean_t", "no2_vs_shipping_band.png", "no2_mean_t"),
        ]:
            sub = ddec[ddec["variable"] == var].copy()
            if sub.empty or sub["mean"].isna().all():
                continue
            sub = sub.set_index("shipping_distance_band_refined").reindex(BAND_LABELS).reset_index()
            fig, ax = plt.subplots(figsize=(7.5, 3.8))
            x = np.arange(len(sub))
            ax.bar(x, sub["mean"], yerr=sub["std"], capsize=3, color="steelblue", alpha=0.85, ecolor="0.35")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["shipping_distance_band_refined"], rotation=15)
            ax.set_title(f"{ylab} by shipping distance band (linked maritime rows)")
            ax.set_ylabel(ylab)
            fig.tight_layout()
            fig.savefig(FIGURES / fname, dpi=160)
            plt.close(fig)

    if len(coastal_linked) and vx.isin(["high", "low"]).any():
        sub_b = coastal_linked.loc[vx.isin(["high", "low"])].copy()
        sub_b["_vx"] = vx.loc[sub_b.index].map(
            {"high": "High vessel (top 10%)", "low": "Low vessel (bottom 10%)"}
        )
        sub_b["nearest_land_ndvi_mean"] = pd.to_numeric(sub_b["nearest_land_ndvi_mean"], errors="coerce")
        sub_b = sub_b.dropna(subset=["nearest_land_ndvi_mean"])
        if not sub_b.empty:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            order = ["High vessel (top 10%)", "Low vessel (bottom 10%)"]
            sns.boxplot(data=sub_b, x="_vx", y="nearest_land_ndvi_mean", order=order, ax=ax, hue="_vx", palette="Set2", legend=False)
            sns.stripplot(data=sub_b.sample(min(2000, len(sub_b))), x="_vx", y="nearest_land_ndvi_mean", order=order, ax=ax, color="0.35", alpha=0.35, size=2)
            ax.set_title("Nearest land NDVI (mean) vs vessel exposure")
            ax.set_xlabel("")
            fig.tight_layout()
            fig.savefig(FIGURES / "nearest_land_ndvi_vs_vessel_boxplot.png", dpi=160)
            plt.close(fig)

    # Map: matplotlib sample + folium interactive
    if n_land_cells and n_mar_linked and not link_df.empty:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.scatter(agg["lon"], agg["lat"], s=42, c="forestgreen", alpha=0.7, label="Land NDVI cells", zorder=2)
        ax.scatter(link_df["grid_centroid_lon"], link_df["grid_centroid_lat"], s=35, c="steelblue", alpha=0.65, label="Maritime cells", zorder=2)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(link_df), size=min(60, len(link_df)), replace=False)
        land_lon_map = agg.set_index(agg["grid_cell_id"].astype(str))["lon"].to_dict()
        land_lat_map = agg.set_index(agg["grid_cell_id"].astype(str))["lat"].to_dict()
        for i in sample_idx:
            row = link_df.iloc[i]
            nid = str(row["nearest_land_grid_cell_id"])
            mlon, mlat = float(row["grid_centroid_lon"]), float(row["grid_centroid_lat"])
            llon = float(land_lon_map.get(nid, np.nan))
            llat = float(land_lat_map.get(nid, np.nan))
            if np.isfinite(llon):
                ax.plot([mlon, llon], [mlat, llat], "k-", alpha=0.22, lw=0.7, zorder=1)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")
        ax.set_title("Maritime grids → nearest land NDVI cell (sample of links)")
        fig.tight_layout()
        fig.savefig(FIGURES / "linkage_map_sample.png", dpi=160)
        plt.close(fig)

        try:
            import folium
            from folium.plugins import MarkerCluster

            center_lat = float(grids_mar["grid_centroid_lat"].mean())
            center_lon = float(grids_mar["grid_centroid_lon"].mean())
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="CartoDB positron")
            lc = MarkerCluster(name="Land NDVI").add_to(m)
            for _, r in agg.iterrows():
                folium.CircleMarker(
                    location=[float(r["lat"]), float(r["lon"])],
                    radius=4,
                    color="green",
                    fill=True,
                    fill_opacity=0.7,
                    popup=f'land grid {r["grid_cell_id"]}',
                ).add_to(lc)
            mc = MarkerCluster(name="Maritime linked").add_to(m)
            for _, r in link_df.iterrows():
                folium.CircleMarker(
                    location=[float(r["grid_centroid_lat"]), float(r["grid_centroid_lon"])],
                    radius=4,
                    color="blue",
                    fill=True,
                    fill_opacity=0.7,
                    popup=f'maritime {r["maritime_grid_cell_id"]} → land {r["nearest_land_grid_cell_id"]}',
                ).add_to(mc)
            samp = link_df.sample(n=min(40, len(link_df)), random_state=7)
            for _, row in samp.iterrows():
                nid = str(row["nearest_land_grid_cell_id"])
                if nid not in land_lat_map:
                    continue
                folium.PolyLine(
                    locations=[
                        [float(row["grid_centroid_lat"]), float(row["grid_centroid_lon"])],
                        [float(land_lat_map[nid]), float(land_lon_map[nid])],
                    ],
                    color="gray",
                    weight=1,
                    opacity=0.5,
                ).add_to(m)
            folium.LayerControl(collapsed=False).add_to(m)
            m.save(str(VIZ / "linkage_map.html"))
            (VIZ / "linkage_map_note.txt").unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            (VIZ / "linkage_map_note.txt").write_text(f"Folium map skipped: {exc}\n", encoding="utf-8")

    imgs = sorted(FIGURES.glob("*.png"))
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Nearest land NDVI linkage</title></head><body>",
        "<h2>Figures</h2><ul>",
    ]
    for im in imgs:
        parts.append(f"<li><p>{im.name}</p><img src='../figures/{RUN}/{im.name}' style='max-width:900px'></li>")
    parts.append("</ul><p><a href='linkage_map.html'>Interactive map</a></p></body></html>")
    (VIZ / "figures_gallery.html").write_text("\n".join(parts), encoding="utf-8")

    # STEP 9 — interpretation
    interp = f"""# Nearest-land NDVI linkage — interpretation

## Why nearest-land linkage was used  
Many maritime/coastal-water grid cells rarely carry valid **pixel NDVI**. To still ask whether **vegetation signals** relate to **maritime exposure**, each eligible maritime centroid is paired with the **closest land-associated grid** that has aggregated NDVI (coast ≤{LAND_NDVI_COAST_KM:.0f} km).

## Candidate land cells  
Land NDVI candidates (non-null NDVI & coast ≤{LAND_NDVI_COAST_KM:.0f} km): **{n_land_cells}** distinct `grid_cell_id` cells (see `land_ndvi_candidate_cells.csv`).

## Maritime linkage  
Maritime/coastal grids (coast ≤{MARITIME_COAST_KM:.0f} km and vessel density or oil proxy present): **{n_mar_linked}** linked to a nearest land cell (`maritime_to_nearest_land_linkage.csv`).  

Median **distance** from maritime centroid to paired land centroid: **{med_link:.2f} km** (nan if empty).

## Shipping-distance structure  
Bands use `distance_to_nearest_high_vessel_density_cell` (0–3, 3–7, 7–15, 15–30 km). Inspect `nearest_land_distance_decay.csv` for whether linked **nearest_land_ndvi_mean** / **nearest_land_ndvi_median** shift across bands versus **NO₂** and **oil slick** proxy.

## High vs low vessel exposure  
Among linked rows with coast ≤{MARITIME_COAST_KM:.0f} km, **top vs bottom decile on log1p(vessel_density_t)** (see `nearest_land_high_vs_low_vessel.csv`).

Strongest explored **Cohen's d** (among those comparisons): **{strongest_name or "n/a"}** (|d|≈**{strongest_d:.4f}**).

**Automated land-impact flag:** **{land_detected}** — YES if any comparison shows |d|>0.05 with Welch or Mann–Whitney p<0.05; INCONCLUSIVE if groups too small / no tests; NO otherwise.

## Sea → land correlations  
Spearman checks on linked weekly rows: `sea_land_correlation_check.csv` (nearest-land NDVI mean vs oil, detection, NO₂, vessel density, lane distance).

## Limitations  
- Nearest land NDVI is a **spatial proxy**, not hydrological or dispersion modeling.  
- NDVI is **seasonal**, **cloud-affected**, and aggregated at coarse grid/week resolution.  
- Results are **associations only**, not causality.  
- A stronger approach is **dedicated Sentinel-2 land-mask extraction along oriented buffers** oriented from shore outward.

Main research question: *Can nearby land vegetation response be linked to maritime exposure by connecting coastal marine cells to the nearest valid land NDVI observations?*  

Answer here is framed by the magnitudes/signs in CSVs/plots—not a causal claim.

Generated artifact: **`nearest_land_ndvi_linked_dataset.parquet`** merges linkage columns onto the full weekly table by `grid_cell_id`.
"""

    (REPORTS / "interpretation_summary.md").write_text(interp, encoding="utf-8")

    print("=== run_nearest_land_ndvi_linkage ===")
    print(f"Full dataset rows: {n_total:,}")
    print(f"Land NDVI candidate grid cells (unique): {n_land_cells:,}")
    print(f"Maritime/coastal grids linked to nearest land: {n_mar_linked:,}")
    print(f"Median distance to nearest land NDVI centroid (km): {med_link:.4f}" if np.isfinite(med_link) else "Median distance to nearest land NDVI centroid (km): n/a")
    print(f"Strongest land-related |Cohen's d| ({strongest_name}): {strongest_d:.4f}" if strongest_name else "Strongest land-related effect: n/a")
    print(f"Land impact (heuristic): {land_detected}")
    print()
    print(
        "Main research question: Can nearby land vegetation response be linked to maritime exposure by "
        "connecting coastal marine cells to the nearest valid land NDVI observations?"
    )
    print("\nOutputs:")
    for p in sorted(
        set((REPORTS).glob("*")) | set(FIGURES.glob("*.png")) | set(VIZ.glob("*"))
    ):
        if p.is_file():
            print(" ", p.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
