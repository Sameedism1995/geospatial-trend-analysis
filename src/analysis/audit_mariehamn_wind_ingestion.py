#!/usr/bin/env python3
"""
Mariehamn-specific wind ingestion audit (no fabricated values).

Documents:
  - Raw ERA5 via Open-Meteo archive API at Mariehamn point (same helper as pipeline)
  - Spatial/temporal extent of coastal_wind_alignment_features.csv
  - For each radius: parquet cell-weeks near Mariehamn vs wind CSV join (where wind is dropped)

  python3 src/analysis/audit_mariehamn_wind_ingestion.py

Optional: --skip-live-api (no HTTP to Open-Meteo).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.run_coastal_exposure_analysis import merge_wind_vectors, prepare_panel  # noqa: E402
from analysis.run_land_pollution_drivers_wind import fetch_open_meteo_cluster_wind  # noqa: E402
from analysis.run_portwise_coastal_exposure import haversine_km_vec  # noqa: E402

MARIE_LAT = 60.0973
MARIE_LON = 19.9348
REPORTS = _ROOT / "outputs" / "reports"
OUT_CSV = REPORTS / "mariehamn_wind_ingestion_audit.csv"
DEFAULT_PARQUET = _ROOT / "outputs" / "processed" / "features_ml_ready_coastal_wind.parquet"
WIND_CSV = REPORTS / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv"
COAST_KM = 30.0

CACHED_PARQUETS = [
    _ROOT / "outputs" / "processed" / "features_ml_ready_coastal_wind.parquet",
    _ROOT / "outputs" / "reports" / "run_nearest_land_ndvi_linkage" / "nearest_land_ndvi_linked_dataset.parquet",
]


def _row(
    stage: str,
    radius_km: float,
    row_count: int,
    nn_u: int,
    nn_v: int,
    nn_spd: int,
    nn_dir: int,
    min_d: str,
    max_d: str,
    reason: str,
) -> dict:
    return {
        "stage": stage,
        "radius_km": radius_km,
        "row_count": row_count,
        "non_null_u10": nn_u,
        "non_null_v10": nn_v,
        "non_null_wind_speed": nn_spd,
        "non_null_wind_direction": nn_dir,
        "min_date": min_d,
        "max_date": max_d,
        "reason_if_zero": reason,
    }


def panel_m_mask(df: pd.DataFrame) -> pd.Series:
    dc = pd.to_numeric(df["distance_to_coast_km"], errors="coerce")
    ds = pd.to_numeric(df["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    return dc.notna() & (dc <= COAST_KM) & ds.notna() & (ds >= 0) & (ds < 30)


def merge_wind_csv_full(df: pd.DataFrame, wind_csv: Path | None) -> pd.DataFrame:
    """Left-merge all columns from coastal export, overlaying any duplicate names so CSV wins."""
    if wind_csv is None or not Path(wind_csv).is_file():
        return df
    wf = pd.read_csv(wind_csv)
    if not {"grid_cell_id", "week_start_utc"}.issubset(wf.columns):
        return df
    wf = wf.copy()
    wf["grid_cell_id"] = wf["grid_cell_id"].astype(str)
    wf["_wk_merge"] = pd.to_datetime(wf["week_start_utc"], utc=True).dt.normalize()
    out = df.copy()
    overlap = [c for c in wf.columns if c in out.columns and c not in ("grid_cell_id", "week_start_utc")]
    out = out.drop(columns=overlap, errors="ignore")
    out["_wk_merge"] = pd.to_datetime(out["week_start_utc"], utc=True).dt.normalize()
    wf2 = wf.drop(columns=["week_start_utc"], errors="ignore")
    merged = out.merge(wf2, on=["grid_cell_id", "_wk_merge"], how="left")
    return merged.drop(columns=["_wk_merge"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--wind-csv", type=Path, default=WIND_CSV)
    ap.add_argument("--ne-cache", type=Path, default=_ROOT / "data" / "aux" / "natural_earth_coast_cache")
    ap.add_argument("--skip-live-api", action="store_true")
    args = ap.parse_args()

    rows: list[dict] = []

    if not args.parquet.is_file():
        print(f"[FATAL] missing {args.parquet}")
        return 1

    df_base = pd.read_parquet(args.parquet)
    df_base["grid_cell_id"] = df_base["grid_cell_id"].astype(str)
    tmin = pd.to_datetime(df_base["week_start_utc"], utc=True).min()
    tmax = pd.to_datetime(df_base["week_start_utc"], utc=True).max()
    start_d = tmin.date().isoformat()
    end_d = tmax.date().isoformat()

    lat0 = pd.to_numeric(df_base["grid_centroid_lat"], errors="coerce").to_numpy()
    lon0 = pd.to_numeric(df_base["grid_centroid_lon"], errors="coerce").to_numpy()
    d_marie = haversine_km_vec(lat0, lon0, MARIE_LAT, MARIE_LON)

    rows.append(
        _row(
            "parquet_panel_all_rows_after_read",
            math.nan,
            len(df_base),
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            str(tmin),
            str(tmax),
            f"CRS=WGS84 implicit; centroid_lat [{float(np.nanmin(lat0)):.4f},{float(np.nanmax(lat0)):.4f}] "
            f"centroid_lon [{float(np.nanmin(lon0)):.4f},{float(np.nanmax(lon0)):.4f}]",
        ),
    )

    wind_df = pd.DataFrame()
    if args.wind_csv.is_file():
        wind_df = pd.read_csv(args.wind_csv)
        wind_df["grid_cell_id"] = wind_df["grid_cell_id"].astype(str)
        wk_w = pd.to_datetime(wind_df["week_start_utc"], utc=True)
        if {"grid_centroid_lat", "grid_centroid_lon"}.issubset(wind_df.columns):
            la = pd.to_numeric(wind_df["grid_centroid_lat"], errors="coerce")
            lo = pd.to_numeric(wind_df["grid_centroid_lon"], errors="coerce")
            lat_s, lon_s = f"{float(la.min()):.4f}/{float(la.max()):.4f}", f"{float(lo.min()):.4f}/{float(lo.max()):.4f}"
        else:
            meta = df_base.drop_duplicates("grid_cell_id")[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]]
            wg = wind_df.merge(meta, on="grid_cell_id", how="left")
            la = pd.to_numeric(wg["grid_centroid_lat"], errors="coerce")
            lo = pd.to_numeric(wg["grid_centroid_lon"], errors="coerce")
            lat_s = f"{float(la.min()):.4f}/{float(la.max()):.4f}"
            lon_s = f"{float(lo.min()):.4f}/{float(lo.max()):.4f}"
        ucols = wind_df["grid_cell_id"].nunique()
        rows.append(
            _row(
                "coastal_wind_alignment_features_csv",
                math.nan,
                len(wind_df),
                int(pd.to_numeric(wind_df.get("wind_u_mean"), errors="coerce").notna().sum()),
                int(pd.to_numeric(wind_df.get("wind_v_mean"), errors="coerce").notna().sum()),
                int(pd.to_numeric(wind_df.get("wind_speed_mean"), errors="coerce").notna().sum()),
                int(pd.to_numeric(wind_df.get("wind_direction_to_degrees"), errors="coerce").notna().sum()),
                str(wk_w.min()),
                str(wk_w.max()),
                f"unique_grid_cells_in_export={ucols}; joined_centroid_lat {lat_s} lon {lon_s}; "
                "subset of panel_m from run_coastal_wind_transport only",
            ),
        )
    else:
        rows.append(
            _row(
                "coastal_wind_alignment_features_csv",
                math.nan,
                0,
                0,
                0,
                0,
                0,
                "",
                "",
                "file_missing",
            ),
        )

    cache_rows: list[str] = []
    for pq in CACHED_PARQUETS:
        if not pq.is_file():
            cache_rows.append(f"{pq.name}: missing")
            continue
        tdf = pd.read_parquet(pq)
        if "grid_centroid_lat" in tdf.columns:
            alat = pd.to_numeric(tdf["grid_centroid_lat"], errors="coerce")
            alon = pd.to_numeric(tdf["grid_centroid_lon"], errors="coerce")
            ts = pd.to_datetime(tdf["week_start_utc"], utc=True, errors="coerce")
            cache_rows.append(
                f"{pq.name}: rows={len(tdf)} lat[{float(alat.min()):.4f},{float(alat.max()):.4f}] "
                f"lon[{float(alon.min()):.4f},{float(alon.max()):.4f}] "
                f"week[{ts.min()}..{ts.max()}] CRS=WGS84_via_centroids",
            )
        else:
            cache_rows.append(f"{pq.name}: rows={len(tdf)} (no centroid cols)")

    rows.append(
        _row(
            "cache_parquet_inventory",
            math.nan,
            sum(1 for p in CACHED_PARQUETS if p.is_file()),
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            "",
            "",
            " | ".join(cache_rows),
        ),
    )

    if not args.skip_live_api:
        try:
            raw_wk = fetch_open_meteo_cluster_wind(
                np.array([MARIE_LAT], dtype=float),
                np.array([MARIE_LON], dtype=float),
                start_d,
                end_d,
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                _row(
                    "open_meteo_era5_single_point_mariehamn_raw",
                    math.nan,
                    0,
                    0,
                    0,
                    0,
                    0,
                    "",
                    "",
                    f"fetch_failed:{exc}",
                ),
            )
        else:
            if raw_wk.empty:
                rows.append(
                    _row(
                        "open_meteo_era5_single_point_mariehamn_raw",
                        math.nan,
                        0,
                        0,
                        0,
                        0,
                        0,
                        "",
                        "",
                        "empty_concat",
                    ),
                )
            else:
                ts = pd.to_datetime(raw_wk["week_start_utc"], utc=True, errors="coerce")
                uu = pd.to_numeric(raw_wk["wind_u_mean"], errors="coerce")
                vv = pd.to_numeric(raw_wk["wind_v_mean"], errors="coerce")
                ws = pd.to_numeric(raw_wk["wind_speed_mean"], errors="coerce")
                wdir = pd.to_numeric(raw_wk["wind_direction_to_degrees"], errors="coerce")
                nuw = int(raw_wk["week_start_utc"].nunique()) if "week_start_utc" in raw_wk.columns else 0
                rows.append(
                    _row(
                        "open_meteo_era5_single_point_mariehamn_raw",
                        math.nan,
                        len(raw_wk),
                        int(uu.notna().sum()),
                        int(vv.notna().sum()),
                        int(ws.notna().sum()),
                        int(wdir.notna().sum()),
                        str(ts.min()),
                        str(ts.max()),
                        f"weekly_rows_from_hourly_archive_api; unique_weeks={nuw}; u/v derived from Open-Meteo hourly speed/dir",
                    ),
                )
    else:
        rows.append(
            _row(
                "open_meteo_era5_single_point_mariehamn_raw",
                math.nan,
                0,
                0,
                0,
                0,
                0,
                "",
                "",
                "skipped",
            ),
        )

    meta_cells = df_base.drop_duplicates("grid_cell_id")[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]]
    meta_cells["_km_marie"] = haversine_km_vec(
        pd.to_numeric(meta_cells["grid_centroid_lat"], errors="coerce").to_numpy(),
        pd.to_numeric(meta_cells["grid_centroid_lon"], errors="coerce").to_numpy(),
        MARIE_LAT,
        MARIE_LON,
    )

    for rad in (3, 7, 15, 30, 50):
        gids = set(meta_cells.loc[meta_cells["_km_marie"] <= float(rad), "grid_cell_id"].astype(str))
        sub_p = df_base[df_base["grid_cell_id"].isin(gids)]
        wk_sub = pd.to_datetime(sub_p["week_start_utc"], utc=True)
        nuw = int(wk_sub.nunique())
        reason = f"unique_grid_cells={len(gids)}; unique_weeks={nuw}"
        if len(sub_p) == 0:
            reason = "no_parquet_centroids_within_radius"

        if not wind_df.empty and gids:
            wsub = wind_df[wind_df["grid_cell_id"].isin(gids)]
            uu = pd.to_numeric(wsub.get("wind_u_mean"), errors="coerce")
            vv = pd.to_numeric(wsub.get("wind_v_mean"), errors="coerce")
            ws = pd.to_numeric(wsub.get("wind_speed_mean"), errors="coerce")
            wd = pd.to_numeric(wsub.get("wind_direction_to_degrees"), errors="coerce")
            wk2 = pd.to_datetime(wsub["week_start_utc"], utc=True)
            if len(wsub) == 0:
                zreason = reason + "; wind_csv_rows=0 (Mariehamn-area cells not in panel_m export)"
            else:
                zreason = reason + f"; wind_csv_non_null_alignment={int(pd.to_numeric(wsub.get('coastal_wind_alignment_score'), errors='coerce').notna().sum())}"
            rows.append(
                _row(
                    f"wind_csv_rows_cells_within_{rad}km_of_mariehamn",
                    float(rad),
                    int(len(wsub)),
                    int(uu.notna().sum()),
                    int(vv.notna().sum()),
                    int(ws.notna().sum()),
                    int(wd.notna().sum()),
                    str(wk2.min()) if len(wsub) else "",
                    str(wk2.max()) if len(wsub) else "",
                    zreason,
                ),
            )
        else:
            nz = "no_centroids_in_radius" if not gids else "wind_csv_missing"
            rows.append(
                _row(
                    f"wind_csv_rows_cells_within_{rad}km_of_mariehamn",
                    float(rad),
                    0,
                    0,
                    0,
                    0,
                    0,
                    "",
                    "",
                    nz,
                ),
            )

        rows.append(
            _row(
                f"parquet_rows_centroids_within_{rad}km_before_panel_filters",
                float(rad),
                int(len(sub_p)),
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                str(wk_sub.min()) if len(sub_p) else "",
                str(wk_sub.max()) if len(sub_p) else "",
                reason,
            ),
        )

    df_merged = merge_wind_csv_full(df_base, args.wind_csv if args.wind_csv.is_file() else None)
    df_merged["_km_marie"] = haversine_km_vec(
        pd.to_numeric(df_merged["grid_centroid_lat"], errors="coerce").to_numpy(),
        pd.to_numeric(df_merged["grid_centroid_lon"], errors="coerce").to_numpy(),
        MARIE_LAT,
        MARIE_LON,
    )
    df_merged = prepare_panel(df_merged, Path(args.ne_cache))
    pm = panel_m_mask(df_merged)

    u_all = pd.to_numeric(df_merged.get("wind_u_mean"), errors="coerce")
    v_all = pd.to_numeric(df_merged.get("wind_v_mean"), errors="coerce")
    rows.append(
        _row(
            "after_full_wind_csv_merge_on_all_parquet_rows",
            math.nan,
            len(df_merged),
            int(u_all.notna().sum()),
            int(v_all.notna().sum()),
            int(pd.to_numeric(df_merged.get("wind_speed_mean"), errors="coerce").notna().sum()),
            int(pd.to_numeric(df_merged.get("wind_direction_to_degrees"), errors="coerce").notna().sum()),
            str(tmin),
            str(tmax),
            "full column merge from coastal_wind_alignment_features.csv where keys match",
        ),
    )
    rows.append(
        _row(
            "after_spatial_clip_panel_m_same_as_transport_script",
            math.nan,
            int(pm.sum()),
            int((pm & u_all.notna()).sum()),
            int((pm & v_all.notna()).sum()),
            int((pm & pd.to_numeric(df_merged.get("wind_speed_mean"), errors="coerce").notna()).sum()),
            int((pm & pd.to_numeric(df_merged.get("wind_direction_to_degrees"), errors="coerce").notna()).sum()),
            str(tmin),
            str(tmax),
            "panel_m=coastal<=30km and vessel_distance in [0,30) km",
        ),
    )

    ca = pd.to_numeric(df_merged.get("coastal_wind_alignment_score"), errors="coerce")

    for rad in (3, 7, 15, 30, 50):
        ring = df_merged["_km_marie"] <= float(rad)
        sub = df_merged.loc[ring]
        uu = pd.to_numeric(sub.get("wind_u_mean"), errors="coerce")
        vv = pd.to_numeric(sub.get("wind_v_mean"), errors="coerce")
        ws = pd.to_numeric(sub.get("wind_speed_mean"), errors="coerce")
        wd = pd.to_numeric(sub.get("wind_direction_to_degrees"), errors="coerce")
        cc = ca.reindex(sub.index)
        wk3 = pd.to_datetime(sub["week_start_utc"], utc=True)
        nuw = int(wk3.nunique())
        nz = ""
        if len(sub) == 0:
            nz = "no_rows"
        elif uu.notna().sum() == 0:
            nz = f"no_matching_wind_csv_keys; unique_weeks={nuw}"
        else:
            nz = f"non_null_coastal_alignment={int(cc.notna().sum())}; unique_weeks={nuw}"
        rows.append(
            _row(
                f"merged_prepare_panel_rows_within_{rad}km",
                float(rad),
                int(len(sub)),
                int(uu.notna().sum()),
                int(vv.notna().sum()),
                int(ws.notna().sum()),
                int(wd.notna().sum()),
                str(wk3.min()) if len(sub) else "",
                str(wk3.max()) if len(sub) else "",
                nz,
            ),
        )

    df_merge_only = merge_wind_vectors(df_base.copy(), args.wind_csv if args.wind_csv.is_file() else None)
    df_merge_only = prepare_panel(df_merge_only, Path(args.ne_cache))
    m_only_u = pd.to_numeric(df_merge_only.get("wind_u_mean"), errors="coerce")
    ring50 = haversine_km_vec(
        pd.to_numeric(df_merge_only["grid_centroid_lat"], errors="coerce").to_numpy(),
        pd.to_numeric(df_merge_only["grid_centroid_lon"], errors="coerce").to_numpy(),
        MARIE_LAT,
        MARIE_LON,
    ) <= 50.0
    rows.append(
        _row(
            "after_coastal_exposure_merge_wind_vectors_only_u_v_within_50km",
            50.0,
            int(ring50.sum()),
            int((ring50 & m_only_u.notna()).sum()),
            int((ring50 & pd.to_numeric(df_merge_only.get("wind_v_mean"), errors="coerce").notna()).sum()),
            math.nan,
            math.nan,
            str(tmin),
            str(tmax),
            "port_exposure_analysis_uses_merge_wind_vectors (u,v only from CSV)",
        ),
    )

    rows.append(
        _row(
            "note_ingestion_bounds_user_request",
            math.nan,
            0,
            0,
            0,
            0,
            0,
            "",
            "",
            "If Mariehamn-area cells lack rows in coastal_wind_alignment_features.csv, re-run "
            "run_coastal_wind_transport.py with --union-port-buffers-km 50 --ports-csv data/aux/baltic_ports.csv "
            "on the same base panel; lat 59.5–60.7 lon 19.0–20.8 is informative but API fetch is per cluster centroid.",
        ),
    )

    out = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
