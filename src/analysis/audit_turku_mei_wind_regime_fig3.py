#!/usr/bin/env python3
"""
Trace why maritime exposure index (MEI) can be missing from the wind-regime contrast figure (Fig3)
while present elsewhere: row-level Turku annulus + decay-table checks.

Writes: outputs/reports/turku_mei_missing_debug.csv

  python3 src/analysis/audit_turku_mei_wind_regime_fig3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC))

from analysis.run_coastal_exposure_analysis import (  # noqa: E402
    build_indices,
    merge_wind_vectors,
    prepare_panel,
)
from analysis.run_portwise_coastal_exposure import (  # noqa: E402
    FOCAL_PORTS,
    FIGURE_PORTS,
    WIND_REGIME_ZONE_PRIORITY,
    aggregate_long_table,
    attach_focal_port_distances,
    first_zone_for_wind_regime_figure,
    metric_definitions,
    zone_masks,
    wind_subsets,
    _safe_slug,
)

REPORT = _ROOT / "outputs" / "reports" / "turku_mei_missing_debug.csv"
DEFAULT_PARQUET = _ROOT / "outputs" / "processed" / "features_ml_ready_coastal_wind.parquet"
WIND_CSV = _ROOT / "outputs" / "reports" / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv"
NE = _ROOT / "data" / "aux" / "natural_earth_coast_cache"


def _port_distance_bin(dist_km: float) -> str:
    if not np.isfinite(dist_km):
        return "unknown"
    if dist_km <= 3.0:
        return "0-3 km"
    if dist_km <= 7.0:
        return "3-7 km"
    if dist_km <= 15.0:
        return "7-15 km"
    if dist_km <= 30.0:
        return "15-30 km"
    return ">30 km"


def _shoreward_label(sb: float | None) -> str:
    if sb is None or not np.isfinite(sb):
        return "missing"
    if sb == 1.0:
        return "shoreward"
    if sb == 0.0:
        return "non_shoreward"
    return "other"


def _legacy_fig3_exclusion_reason(row: pd.Series) -> str:
    """Explains why a row would not contribute to the *old* Fig3 (fixed 0–3 km) MEI bars."""
    reasons: list[str] = []
    d = float(row["distance_km"]) if pd.notna(row["distance_km"]) else float("nan")
    if not np.isfinite(d) or d > 3.0:
        reasons.append("outside_legacy_Fig3_annulus_dist_gt_3km")
    m = row["MEI"]
    if pd.isna(m) or not np.isfinite(float(m)):
        reasons.append("MEI_nan_intra_week_rank_inputs")
    sb = row["shoreward_binary"]
    if pd.isna(sb) or float(sb) not in (0.0, 1.0):
        reasons.append("shoreward_binary_not_0_or_1")
    return "; ".join(reasons) if reasons else "eligible_for_legacy_Fig3_MEI_stratum"


def main() -> int:
    print("=== 1) Rebuild panel (same as run_portwise_coastal_exposure) ===")
    df = pd.read_parquet(DEFAULT_PARQUET)
    df = merge_wind_vectors(df, WIND_CSV)
    df = attach_focal_port_distances(df)
    df = prepare_panel(df, NE)
    df = build_indices(df)

    dc = "dist_turku_km"
    dser = pd.to_numeric(df[dc], errors="coerce")
    mei = pd.to_numeric(df["maritime_exposure_index"], errors="coerce")
    vd = pd.to_numeric(df["vessel_density_t"], errors="coerce")
    ca = pd.to_numeric(df["coastal_wind_alignment_score"], errors="coerce")
    sh = pd.to_numeric(df["shoreward_binary"], errors="coerce")

    print("Full panel rows:", len(df))
    print("Turku columns present:", dc in df.columns)

    ring03 = dser <= 3.0
    print("\n=== Turku ≤3 km (legacy Fig3 annulus) ===")
    print("rows in annulus:", int(ring03.sum()))
    print("MEI non-null in annulus:", int(mei[ring03].notna().sum()))
    print("vessel_density non-null:", int(vd[ring03].notna().sum()))
    print("coastal_wind_alignment non-null:", int(ca[ring03].notna().sum()))
    d2c = pd.to_numeric(df["distance_to_coast_km"], errors="coerce")
    print("distance_to_coast_km non-null (≤3km ring):", int(d2c[ring03].notna().sum()))
    print("shoreward_binary 0/1:", int(sh[ring03].eq(0).sum()), int(sh[ring03].eq(1).sum()), "other/missing:", int((~sh[ring03].isin([0.0, 1.0])).sum()))

    coastal = df["coastal_panel"].fillna(False)
    masks = zone_masks(dser, coastal)
    zm = masks["0-3 km"]
    wsubs = wind_subsets(df["shoreward_binary"] if "shoreward_binary" in df.columns else pd.Series(np.nan, index=df.index))
    for wl, wm in wsubs.items():
        base = zm & wm & dser.notna()
        sub_mei = pd.to_numeric(df.loc[base, "maritime_exposure_index"], errors="coerce")
        print(f"  zone 0-3 km × {wl}: n_rows={int(base.sum())} MEI_non_null={int(sub_mei.notna().sum())}")

    print("\n=== Decay table (aggregate_long_table) MEI Turku ===")
    metrics = metric_definitions(df)
    rows_t = aggregate_long_table(df, "Turku", "dist_turku_km", metrics)
    decay_t = pd.DataFrame(rows_t)
    mei_rows = decay_t.loc[decay_t["metric"].eq("maritime_exposure_index")]
    print(mei_rows[["distance_zone", "wind_regime", "n", "mean"]].to_string(index=False))

    print("\n=== Fig3 zone selection (both ports, both strata) ===")
    print("Priority list:", WIND_REGIME_ZONE_PRIORITY)

    ports = list(FIGURE_PORTS.keys())
    rows_all = []
    for pname in FOCAL_PORTS:
        rows_all.extend(aggregate_long_table(df, pname, f"dist_{_safe_slug(pname)}_km", metrics))
    decay_full = pd.DataFrame(rows_all)
    for m in ["maritime_exposure_index", "atmospheric_coastal_exposure_index", "environmental_stress_index"]:
        z = first_zone_for_wind_regime_figure(decay_full, m, ports)
        print(f"  {m}: {z}")

    print("\n=== Dashboard vs Fig3 (decay lookup) ===")
    print("Dashboard MEI uses decay_tbl wind_regime=='all' for ALL zones in ZONE_ORDER — NaNs where no row.")
    print("Fig3 (old) used ONLY distance_zone=='0-3 km' + shoreward/non — no row for MEI ⇒ NaN bars.")

    print("\n=== Annulus definition (zone_masks) ===")
    print(" '0-3 km' => dist_km <= 3.0  (includes 0; not coastal-core-only).")
    print(" coastal-core is separate key '0-3 km (coastal core)' => d<=3 & coastal_panel.")

    # Row-level CSV: Turku centroid rows within 30 km
    sub = df.loc[dser <= 30.0].copy()
    sub["distance_km"] = pd.to_numeric(sub[dc], errors="coerce")
    sub["distance_bin"] = sub["distance_km"].map(_port_distance_bin)
    sub["shoreward_class"] = sub["shoreward_binary"].map(lambda x: _shoreward_label(float(x) if pd.notna(x) else np.nan))
    sub["MEI"] = pd.to_numeric(sub["maritime_exposure_index"], errors="coerce")

    def _incl(row: pd.Series) -> bool:
        dd = float(row["distance_km"])
        m = float(row["MEI"]) if pd.notna(row["MEI"]) else float("nan")
        sbb = row.get("shoreward_binary")
        ok_d = np.isfinite(dd) and dd <= 3.0
        ok_m = np.isfinite(m)
        ok_w = pd.notna(sbb) and float(sbb) in (0.0, 1.0)
        return bool(ok_d and ok_m and ok_w)

    sub["included_in_plot"] = sub.apply(_incl, axis=1)
    sub["exclusion_reason"] = sub.apply(_legacy_fig3_exclusion_reason, axis=1)

    out = sub[
        [
            "grid_cell_id",
            "week_start_utc",
            "distance_km",
            "distance_bin",
            "shoreward_class",
            "MEI",
            "included_in_plot",
            "exclusion_reason",
        ]
    ].copy()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT, index=False)
    print(f"\n[OK] wrote {REPORT} rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
