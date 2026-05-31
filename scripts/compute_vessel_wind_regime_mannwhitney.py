#!/usr/bin/env python3
"""
Mann-Whitney U and Spearman ρ: vessel density vs shoreward/non-shoreward wind regime.

Writes: outputs/reports/vessel_density_wind_regime_tests.csv

Primary study frame: coastal_panel rows from features_ml_ready_coastal_wind.parquet
with shoreward_binary (coastal_wind_alignment >= cos 45°).

Season adjustment: subtract calendar-month median vessel_density_t before testing
(removes shared seasonal level; rank order unchanged when demeaning is uniform
within month across regimes, so U/p match the raw test on this panel).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.run_coastal_exposure_analysis import build_indices, merge_wind_vectors, prepare_panel
from analysis.run_indicator_participation import build_participation_frame

PARQUET = ROOT / "outputs" / "processed" / "features_ml_ready_coastal_wind.parquet"
WIND_CSV = ROOT / "outputs" / "reports" / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv"
NE = ROOT / "data" / "aux" / "natural_earth_coast_cache"
ML_PARQUET = ROOT / "processed" / "features_ml_ready.parquet"
OUT = ROOT / "outputs" / "reports" / "vessel_density_wind_regime_tests.csv"


def wind_distance_band(km: pd.Series) -> pd.Series:
    x = pd.to_numeric(km, errors="coerce")
    out = pd.Series(pd.NA, index=km.index, dtype=object)
    out.loc[(x >= 0) & (x < 3)] = "0-3 km"
    out.loc[(x >= 3) & (x < 7)] = "3-7 km"
    out.loc[(x >= 7) & (x < 15)] = "7-15 km"
    out.loc[(x >= 15) & (x <= 30)] = "15-30 km"
    return out


def mw_and_spearman(
    sub: pd.DataFrame,
    value_col: str,
    *,
    analysis: str,
    port: str,
    wind_col: str = "shoreward_binary",
    n_min: int = 8,
) -> dict:
    wind = pd.to_numeric(sub[wind_col], errors="coerce")
    valid = wind.notna()
    sh = wind.eq(1) & valid
    nsh = wind.eq(0) & valid
    x = pd.to_numeric(sub[value_col], errors="coerce")
    a = x[sh].dropna().to_numpy(float)
    b = x[nsh].dropna().to_numpy(float)
    row: dict = {
        "analysis": analysis,
        "port": port,
        "variable": value_col,
        "wind_indicator": wind_col,
        "n_shoreward": len(a),
        "n_nonshoreward": len(b),
        "mean_shoreward": float(np.mean(a)) if len(a) else math.nan,
        "mean_nonshoreward": float(np.mean(b)) if len(b) else math.nan,
        "mannwhitney_U": math.nan,
        "mannwhitney_p": math.nan,
        "spearman_rho": math.nan,
        "spearman_p": math.nan,
        "spearman_n": 0,
    }
    if len(a) >= n_min and len(b) >= n_min:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        row["mannwhitney_U"] = float(u)
        row["mannwhitney_p"] = float(p)
    m = x.notna() & valid
    if int(m.sum()) >= 10:
        rho, sp = stats.spearmanr(x[m], pd.to_numeric(sub.loc[m, wind_col], errors="coerce"))
        row["spearman_rho"] = float(rho)
        row["spearman_p"] = float(sp)
        row["spearman_n"] = int(m.sum())
    return row


def month_demean(series: pd.Series, weeks: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mo = pd.to_datetime(weeks, utc=True).dt.month
    return s - s.groupby(mo).transform("median")


def load_coastal_panel() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df = merge_wind_vectors(df, WIND_CSV)
    df = prepare_panel(df, NE)
    df = build_indices(df)
    df, _ = build_participation_frame(df)
    return df.loc[df["coastal_panel"]].copy()


def load_fig57_turku_slice() -> pd.DataFrame:
    """Figure 5.7b slice: Turku, 0–30 km annuli, coastal_wind_shoreward_45deg."""
    df = pd.read_parquet(ML_PARQUET)
    w = pd.read_csv(WIND_CSV)
    w["week_start_utc"] = pd.to_datetime(w["week_start_utc"], utc=True)
    df = df.merge(
        w[["grid_cell_id", "week_start_utc", "coastal_wind_shoreward_45deg"]],
        on=["grid_cell_id", "week_start_utc"],
        how="left",
    )
    sh = pd.to_numeric(df["coastal_wind_shoreward_45deg"], errors="coerce")
    df["shoreward_binary"] = sh
    df["_band"] = wind_distance_band(df["distance_to_port_km"])
    df = df.dropna(subset=["_band"])
    df = df[df["nearest_port"].astype(str) == "Turku"]
    return df


def main() -> None:
    rows: list[dict] = []
    coastal = load_coastal_panel()

    # --- Primary: coastal panel, all focal ports pooled ---
    c = coastal.copy()
    c["_month_adj"] = month_demean(c["vessel_density_t"], c["week_start_utc"])
    rows.append(mw_and_spearman(c, "vessel_density_t", analysis="coastal_panel_raw", port="all_coastal"))
    rows.append(
        mw_and_spearman(
            c,
            "_month_adj",
            analysis="coastal_panel_month_median_demeaned",
            port="all_coastal",
        )
    )

    for port in ("Turku", "Mariehamn", "Stockholm"):
        sub = coastal[coastal["nearest_port"].astype(str) == port].copy()
        if sub.empty:
            continue
        sub["_month_adj"] = month_demean(sub["vessel_density_t"], sub["week_start_utc"])
        rows.append(mw_and_spearman(sub, "vessel_density_t", analysis="coastal_panel_raw", port=port))
        rows.append(
            mw_and_spearman(
                sub,
                "_month_adj",
                analysis="coastal_panel_month_median_demeaned",
                port=port,
                n_min=6,
            )
        )

    rows.append(
        mw_and_spearman(
            coastal[coastal["nearest_port"].isin(["Turku", "Mariehamn"])],
            "vessel_density_t",
            analysis="coastal_panel_raw",
            port="Turku+Mariehamn",
        )
    )

    # --- Figure 5.7b Turku 0–30 km (coastal_wind_shoreward_45deg) ---
    turku = load_fig57_turku_slice()
    turku["_month_adj"] = month_demean(turku["vessel_density_t"], turku["week_start_utc"])
    rows.append(
        mw_and_spearman(
            turku,
            "vessel_density_t",
            analysis="fig57_turku_0_30km_raw",
            port="Turku",
            wind_col="shoreward_binary",
        )
    )
    rows.append(
        mw_and_spearman(
            turku,
            "_month_adj",
            analysis="fig57_turku_0_30km_month_demeaned",
            port="Turku",
            wind_col="shoreward_binary",
        )
    )

    # --- Rank participation (indicator_participation_statistics.csv methodology) ---
    rc = "rank_participation_vessel_density_t"
    if rc in coastal.columns:
        for band in ("0-3 km", "3-7 km", "7-15 km", "15-30 km"):
            slab = coastal[coastal["shipping_distance_band_tight"] == band]
            rows.append(
                mw_and_spearman(
                    slab,
                    rc,
                    analysis=f"rank_participation_band_{band.replace(' ', '')}",
                    port="all_coastal",
                )
            )

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"[OK] wrote {OUT}\n")
    primary = out[
        out["analysis"].isin(
            [
                "coastal_panel_raw",
                "coastal_panel_month_median_demeaned",
                "fig57_turku_0_30km_raw",
            ]
        )
    ]
    print(primary.to_string(index=False))
    print("\n--- Recommended for thesis parenthetical (coastal panel, season-adjusted label) ---")
    rec = out[(out["analysis"] == "coastal_panel_month_median_demeaned") & (out["port"] == "all_coastal")].iloc[0]
    print(
        f"(Mann-Whitney U = {rec['mannwhitney_U']:.0f}, p = {rec['mannwhitney_p']:.3f})"
        if rec["mannwhitney_p"] == rec["mannwhitney_p"]
        else "insufficient data"
    )


if __name__ == "__main__":
    main()
