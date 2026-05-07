#!/usr/bin/env python3
"""
Wind / coastal-alignment coverage audit for port-centric exposure debugging.

Traces where coastal_wind_alignment_features.csv overlaps (or not) the study parquet
by focal port annuli (WGS84 haversine). Produces wind_coverage_audit.csv and
Stockholm-focused debug maps (buffers, coastline, cell match status).

  python3 src/analysis/audit_wind_coverage.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.run_portwise_coastal_exposure import haversine_km_vec  # noqa: E402
from human_impact_distance_analysis import load_coastline_points  # noqa: E402

REPORTS = _ROOT / "outputs" / "reports"
FIGURES = _ROOT / "outputs" / "figures" / "wind_coverage_audit"
WIND_CSV = REPORTS / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv"
DEFAULT_PARQUET = _ROOT / "outputs" / "processed" / "features_ml_ready_coastal_wind.parquet"

PORTS_FOCUS = {
    "Stockholm": (59.3293, 18.0686),
    "Turku": (60.435, 22.225),
    "Mariehamn": (60.097, 19.934),
}
BANDS = [(0.0, 3.0), (3.0, 7.0), (7.0, 15.0), (15.0, 30.0)]


def band_label(lo: float, hi: float) -> str:
    return f"{lo:.0f}-{hi:.0f} km"


def load_parquet_unique_cells(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        path,
        columns=["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon", "week_start_utc"],
    )
    df["grid_cell_id"] = df["grid_cell_id"].astype(str)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--wind-csv", type=Path, default=WIND_CSV)
    ap.add_argument("--ne-cache", type=Path, default=_ROOT / "data" / "aux" / "natural_earth_coast_cache")
    ap.add_argument("--focus-map-port", type=str, default="Stockholm", choices=list(PORTS_FOCUS.keys()))
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    if not args.parquet.is_file():
        print(f"[FATAL] missing {args.parquet}")
        return 1

    df = load_parquet_unique_cells(args.parquet)
    lat0 = pd.to_numeric(df["grid_centroid_lat"], errors="coerce").to_numpy()
    lon0 = pd.to_numeric(df["grid_centroid_lon"], errors="coerce").to_numpy()
    base_uniq = df.drop_duplicates("grid_cell_id")[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]].dropna()

    wind_exists = args.wind_csv.is_file()
    wf = pd.DataFrame()
    wind_cell_set: set[str] = set()
    if wind_exists:
        wf = pd.read_csv(
            args.wind_csv,
            usecols=["grid_cell_id", "week_start_utc", "wind_u_mean", "wind_v_mean", "coastal_wind_alignment_score"],
        )
        wf["grid_cell_id"] = wf["grid_cell_id"].astype(str)
        wind_cell_set = set(wf["grid_cell_id"].unique())

    n_parquet_rows = len(df)
    n_parquet_cellweeks = len(df)
    n_parquet_unique_cells = base_uniq["grid_cell_id"].nunique()
    n_wind_rows = len(wf) if wind_exists else 0
    n_wind_unique_cellweeks = len(wf) if wind_exists else 0
    n_wind_unique_cells = len(wind_cell_set)

    rows.append(
        {
            "audit_type": "global",
            "port": "all",
            "distance_band": "all",
            "metric": "parquet_rows",
            "n_value": n_parquet_rows,
            "detail": "Full modeling parquet grid × week observations",
            "exclusion_or_note": "",
        }
    )
    rows.append(
        {
            "audit_type": "global",
            "port": "all",
            "distance_band": "all",
            "metric": "wind_csv_rows",
            "n_value": n_wind_rows,
            "detail": "Rows written by run_coastal_wind_transport (coastal×shipping panel export)",
            "exclusion_or_note": "" if wind_exists else "wind CSV missing",
        },
    )
    rows.append(
        {
            "audit_type": "global",
            "port": "all",
            "distance_band": "all",
            "metric": "wind_csv_unique_grid_cells",
            "n_value": n_wind_unique_cells,
            "detail": "Distinct grid_cell_id keys in wind CSV",
            "exclusion_or_note": "Wind CSV only contains cells in panel_m used for that pipeline run",
        },
    )

    # Temporal overlap: week keys
    p_wk = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce").dt.normalize()
    parquet_weeks = set(p_wk.dropna().unique())
    if wind_exists:
        w_wk = pd.to_datetime(wf["week_start_utc"], utc=True, errors="coerce").dt.normalize()
        wind_weeks = set(w_wk.dropna().unique())
        inter = parquet_weeks & wind_weeks
        rows.append(
            {
                "audit_type": "temporal",
                "port": "all",
                "distance_band": "all",
                "metric": "unique_weeks_parquet",
                "n_value": len(parquet_weeks),
                "detail": "Normalized UTC week starts in parquet",
                "exclusion_or_note": "",
            },
        )
        rows.append(
            {
                "audit_type": "temporal",
                "port": "all",
                "distance_band": "all",
                "metric": "unique_weeks_wind_csv",
                "n_value": len(wind_weeks),
                "detail": "",
                "exclusion_or_note": "",
            },
        )
        rows.append(
            {
                "audit_type": "temporal",
                "port": "all",
                "distance_band": "all",
                "metric": "week_overlap_count",
                "n_value": len(inter),
                "detail": "",
                "exclusion_or_note": "If << parquet weeks, merge drops wind for those weeks",
            },
        )

    for pname, (plat, plon) in PORTS_FOCUS.items():
        d_km = haversine_km_vec(
            pd.to_numeric(base_uniq["grid_centroid_lat"], errors="coerce").to_numpy(),
            pd.to_numeric(base_uniq["grid_centroid_lon"], errors="coerce").to_numpy(),
            plat,
            plon,
        )
        uc = base_uniq.copy()
        uc["_d_km"] = d_km

        for lo, hi in BANDS:
            if lo == 0:
                band = (uc["_d_km"] >= 0) & (uc["_d_km"] <= hi)
            else:
                band = (uc["_d_km"] > lo) & (uc["_d_km"] <= hi)
            cells_band = set(uc.loc[band, "grid_cell_id"].astype(str))
            in_wind = cells_band & wind_cell_set
            ovl_pct = 100.0 * len(in_wind) / len(cells_band) if len(cells_band) else float("nan")

            rows.append(
                {
                    "audit_type": "port_annulus_cells",
                    "port": pname,
                    "distance_band": band_label(lo, hi),
                    "metric": "unique_parquet_cells_in_band",
                    "n_value": len(cells_band),
                    "detail": "Distinct centroids within distance of port",
                    "exclusion_or_note": "",
                },
            )
            rows.append(
                {
                    "audit_type": "port_annulus_cells",
                    "port": pname,
                    "distance_band": band_label(lo, hi),
                    "metric": "cells_also_in_wind_csv",
                    "n_value": len(in_wind),
                    "detail": "Intersection with wind CSV grid_cell_id set",
                    "exclusion_or_note": f"coverage_vs_band={ovl_pct:.1f}%",
                },
            )
            if len(cells_band) and len(in_wind) == 0:
                rows.append(
                    {
                        "audit_type": "merge_diagnosis",
                        "port": pname,
                        "distance_band": band_label(lo, hi),
                        "metric": "reason_no_wind_overlap",
                        "n_value": 0,
                        "detail": "",
                        "exclusion_or_note": "Wind CSV built from run_coastal_wind_transport panel_m only; "
                        "cells near this port are not in that 30-cell subset unless pipeline extended "
                        "(see --union-port-buffers-km on run_coastal_wind_transport.py). Not a CRS/week bug if "
                        "week counts align globally.",
                    },
                )

    # Merge simulation: left merge wind onto parquet for Stockholm area
    plat, plon = PORTS_FOCUS["Stockholm"]
    d_full = haversine_km_vec(lat0, lon0, plat, plon)
    m30 = d_full <= 30.0
    sub = df.loc[m30].copy()
    sub["_wk"] = pd.to_datetime(sub["week_start_utc"], utc=True, errors="coerce").dt.normalize()
    if wind_exists:
        wj = wf[["grid_cell_id", "week_start_utc", "coastal_wind_alignment_score", "wind_u_mean"]].copy()
        wj["grid_cell_id"] = wj["grid_cell_id"].astype(str)
        wj["_wk"] = pd.to_datetime(wj["week_start_utc"], utc=True, errors="coerce").dt.normalize()
        merged = sub.merge(
            wj.drop(columns=["week_start_utc"]),
            on=["grid_cell_id", "_wk"],
            how="left",
        )
        n_sub = len(merged)
        n_matched = int(pd.to_numeric(merged["wind_u_mean"], errors="coerce").notna().sum())
        rows.append(
            {
                "audit_type": "merge_simulation_Stockholm_30km",
                "port": "Stockholm",
                "distance_band": "0-30 km cell-weeks",
                "metric": "parquet_cell_weeks",
                "n_value": n_sub,
                "detail": "",
                "exclusion_or_note": "",
            },
        )
        rows.append(
            {
                "audit_type": "merge_simulation_Stockholm_30km",
                "port": "Stockholm",
                "distance_band": "0-30 km cell-weeks",
                "metric": "matched_wind_u_after_left_merge",
                "n_value": n_matched,
                "detail": "",
                "exclusion_or_note": f"dropped_unmatched={n_sub - n_matched}; keys absent from wind CSV",
            },
        )

    out = pd.DataFrame(rows)
    outp = REPORTS / "wind_coverage_audit.csv"
    out.to_csv(outp, index=False)
    print(f"[OK] wrote {outp}")

    # --- Stockholm debug map (focus port) ---
    pname = args.focus_map_port
    plat, plon = PORTS_FOCUS[pname]
    d_all = haversine_km_vec(
        pd.to_numeric(base_uniq["grid_centroid_lat"], errors="coerce").to_numpy(),
        pd.to_numeric(base_uniq["grid_centroid_lon"], errors="coerce").to_numpy(),
        plat,
        plon,
    )
    u2 = base_uniq.copy()
    u2["_d"] = d_all
    near = u2[u2["_d"] <= 35.0].copy()
    near["in_wind_csv"] = near["grid_cell_id"].astype(str).isin(wind_cell_set)

    cp = load_coastline_points(Path(args.ne_cache))
    coast_lat = coast_lon = np.array([])
    if cp is not None:
        coast_lat, coast_lon = cp

    dlon, dlat = 1.6, 1.0
    lon0, lon1 = plon - dlon, plon + dlon
    lat0b, lat1b = plat - dlat, plat + dlat

    fig, ax = plt.subplots(figsize=(9.5, 8.8))
    if len(coast_lon):
        m = (
            (coast_lon >= lon0 - 0.8)
            & (coast_lon <= lon1 + 0.8)
            & (coast_lat >= lat0b - 0.5)
            & (coast_lat <= lat1b + 0.5)
        )
        ax.plot(coast_lon[m], coast_lat[m], color="#0a3d2e", lw=1.0, alpha=0.8, label="Coastline (NE)")

    sc_no = ax.scatter(
        near.loc[~near["in_wind_csv"], "grid_centroid_lon"],
        near.loc[~near["in_wind_csv"], "grid_centroid_lat"],
        c="orangered",
        s=28,
        alpha=0.75,
        label="Cell centroid: NOT in wind CSV grid set",
        zorder=4,
    )
    sc_ok = ax.scatter(
        near.loc[near["in_wind_csv"], "grid_centroid_lon"],
        near.loc[near["in_wind_csv"], "grid_centroid_lat"],
        c="teal",
        s=38,
        alpha=0.85,
        label="Cell centroid: in wind CSV",
        zorder=5,
    )

    for r_km, ls in ((3, ":"), (7, "--"), (15, "-."), (30, "-")):
        circ = plt.Circle((plon, plat), r_km / 111.0, fill=False, linestyle=ls, color="#444", linewidth=1.05)
        ax.add_patch(circ)
    ax.plot(plon, plat, "r*", markersize=16, label=f"{pname} focal", zorder=6)

    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0b, lat1b)
    ax.set_aspect(1.0 / math.cos(math.radians(plat)))
    ax.set_xlabel("Longitude °E")
    ax.set_ylabel("Latitude °N")
    ax.set_title(
        f"Wind table coverage vs study grid — {pname}\n"
        "orange = no row in coastal_wind_alignment_features.csv for that grid_cell_id; teal = present",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"debug_grid_vs_wind_csv_{pname.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Optional: sample wind vectors from CSV if any teal cells exist
    if wind_exists and near["in_wind_csv"].any() and "wind_u_mean" in wf.columns:
        gids = near.loc[near["in_wind_csv"], "grid_cell_id"].astype(str).head(200)
        wsub = wf[wf["grid_cell_id"].isin(gids)].drop_duplicates("grid_cell_id").head(80)
        if len(wsub):
            wc = wsub.merge(
                near[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]].drop_duplicates("grid_cell_id"),
                on="grid_cell_id",
                how="left",
            )
            _f2, ax2 = plt.subplots(figsize=(9.5, 8.8))
            if len(coast_lon):
                m_coast = (
                    (coast_lon >= lon0 - 0.8)
                    & (coast_lon <= lon1 + 0.8)
                    & (coast_lat >= lat0b - 0.5)
                    & (coast_lat <= lat1b + 0.5)
                )
                ax2.plot(coast_lon[m_coast], coast_lat[m_coast], color="#0a3d2e", lw=0.9, alpha=0.7)
            ax2.scatter(wc["grid_centroid_lon"], wc["grid_centroid_lat"], c="gray", s=25, alpha=0.5)
            ax2.quiver(
                wc["grid_centroid_lon"],
                wc["grid_centroid_lat"],
                pd.to_numeric(wc["wind_u_mean"], errors="coerce"),
                pd.to_numeric(wc["wind_v_mean"], errors="coerce"),
                scale=90,
                width=0.004,
                color="navy",
                alpha=0.6,
            )
            ax2.plot(plon, plat, "r*", markersize=14)
            ax2.set_xlim(lon0, lon1)
            ax2.set_ylim(lat0b, lat1b)
            ax2.set_aspect(1.0 / math.cos(math.radians(plat)))
            ax2.set_title(f"Sample wind vectors from CSV (subset) — {pname}")
            plt.tight_layout()
            plt.savefig(FIGURES / f"debug_wind_vectors_subset_{pname.lower()}.png", dpi=200, bbox_inches="tight")
            plt.close()

    print(f"[OK] figures -> {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
