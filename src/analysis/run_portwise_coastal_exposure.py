#!/usr/bin/env python3
"""
Port-wise coastal exposure structuring (association / exposure lens — not causal attribution).

Reads ML-ready parquet, attaches focal-port distances (Stockholm, Turku, Mariehamn),
builds coastal panel + indices, aggregates by port-centred distance zones and wind regime,
writes thesis tables and figures.

  python3 src/analysis/run_portwise_coastal_exposure.py
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.run_coastal_exposure_analysis import (  # noqa: E402
    boot_mean_ci,
    build_indices,
    cliffs_delta,
    merge_wind_vectors,
    prepare_panel,
)
from human_impact_distance_analysis import load_coastline_points  # noqa: E402

REPORTS = _ROOT / "outputs" / "reports"
FIGURES = _ROOT / "outputs" / "figures" / "portwise_exposure"
DEFAULT_INPUT = _ROOT / "outputs" / "processed" / "features_ml_ready_coastal_wind.parquet"


# Focal ports for distance-based statistics (all columns in decay CSV / diagnostics).
FOCAL_PORTS: dict[str, tuple[float, float]] = {
    "Stockholm": (59.3293, 18.0686),
    "Turku": (60.435, 22.225),
    "Mariehamn": (60.0973, 19.9348),
}

# Thesis figures compare Turku vs Mariehamn only (Stockholm wind/maritime audit is separate).
FIGURE_PORTS: dict[str, tuple[float, float]] = {
    "Turku": (60.435, 22.225),
    "Mariehamn": (60.0973, 19.9348),
}

ZONE_ORDER = ["0-3 km (coastal core)", "0-3 km", "3-7 km", "7-15 km", "15-30 km"]

# Raw / directional indicators (exclude composite rank indices) for Turku–Mariehamn comparison figures.
INDICATOR_ONLY_SPECS: list[tuple[str, str]] = [
    ("vessel_density_t", "Vessel density"),
    ("local_no2_excess", "Local NO2 excess"),
    ("coastal_wind_alignment_score", "Coastal wind alignment"),
    ("pollution_transport_wind_alignment_score", "Pollution-transport wind alignment"),
    ("oil_slick_proxy", "Oil slick probability"),
    ("ndti_mean", "NDTI (turbidity)"),
    ("ndwi_mean", "NDWI"),
    ("ndvi_mean", "NDVI"),
    ("focal_port_exposure_score", "Focal port exposure score"),
]

# Pooled “0–30 km” summary in Fig6: simple mean of zone means over standard annuli (matches ranking bands).
POOLED_BAND_ZONES = ["0-3 km", "3-7 km", "7-15 km", "15-30 km"]

# Wind-regime bar chart: scan these annuli in order (shared-valid-annulus mode only).
WIND_REGIME_ZONE_PRIORITY = ["0-3 km", "3-7 km", "7-15 km", "15-30 km"]

# Display titles for composite indices (thesis-quality figure labels).
COMPOSITE_METRIC_DISPLAY: dict[str, str] = {
    "maritime_exposure_index": "Maritime exposure index",
    "atmospheric_coastal_exposure_index": "Atmospheric coastal exposure index",
    "environmental_stress_index": "Environmental stress index",
}


@dataclass
class SharedAnnulusSelection:
    """Diagnostics for shared-valid-annulus cross-port wind-regime comparison (both ports, both strata, n ≥ 1)."""

    metric: str
    selected_annulus: str
    turku_n_shoreward: int
    turku_n_nonshoreward: int
    mariehamn_n_shoreward: int
    mariehamn_n_nonshoreward: int
    reason_selected: str


def _stratum_n(decay_tbl: pd.DataFrame, port: str, metric: str, zone: str, wind_regime: str) -> int:
    sub = decay_tbl[
        (decay_tbl["port"] == port)
        & (decay_tbl["metric"] == metric)
        & (decay_tbl["distance_zone"] == zone)
        & (decay_tbl["wind_regime"] == wind_regime)
    ]
    return int(sub["n"].iloc[0]) if len(sub) else 0


def select_shared_valid_annulus(
    decay_tbl: pd.DataFrame,
    metric: str,
    ports: list[str],
) -> SharedAnnulusSelection | None:
    """
    Nearest annulus (fixed priority order) where Turku and Mariehamn (or given ports)
    each have shoreward and non_shoreward strata with n >= 1 in decay_tbl.
    """
    for zone in WIND_REGIME_ZONE_PRIORITY:
        ok = True
        n_map: dict[tuple[str, str], int] = {}
        for p in ports:
            for wr in ("shoreward", "non_shoreward"):
                n = _stratum_n(decay_tbl, p, metric, zone, wr)
                n_map[(p, wr)] = n
                if n < 1:
                    ok = False
        if ok:
            return SharedAnnulusSelection(
                metric=metric,
                selected_annulus=zone,
                turku_n_shoreward=n_map.get(("Turku", "shoreward"), 0),
                turku_n_nonshoreward=n_map.get(("Turku", "non_shoreward"), 0),
                mariehamn_n_shoreward=n_map.get(("Mariehamn", "shoreward"), 0),
                mariehamn_n_nonshoreward=n_map.get(("Mariehamn", "non_shoreward"), 0),
                reason_selected=f"nearest_annulus_priority_order_all_ports_both_wind_regimes_n_ge_1:{zone}",
            )
    return None


def first_zone_for_wind_regime_figure(
    decay_tbl: pd.DataFrame,
    metric: str,
    ports: list[str],
) -> str | None:
    """Backward-compatible: return selected zone label only."""
    sel = select_shared_valid_annulus(decay_tbl, metric, ports)
    return sel.selected_annulus if sel else None


def _safe_slug(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name.strip().lower())


def haversine_km_vec(lat: np.ndarray, lon: np.ndarray, plat: float, plon: float) -> np.ndarray:
    """Pairwise-ish: (lat, lon) vectors vs single port (degrees)."""
    rlat1 = np.radians(lat.astype(float))
    rlon1 = np.radians(lon.astype(float))
    rlat2 = math.radians(plat)
    rlon2 = math.radians(plon)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * math.cos(rlat2) * (np.sin(dlon / 2.0) ** 2)
    h = np.clip(h, 0.0, 1.0)
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(h))


def attach_focal_port_distances(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lat = pd.to_numeric(out["grid_centroid_lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(out["grid_centroid_lon"], errors="coerce").to_numpy(dtype=float)
    for pname, (plat, plon) in FOCAL_PORTS.items():
        col = f"dist_{_safe_slug(pname)}_km"
        out[col] = haversine_km_vec(lat, lon, plat, plon)
    return out


def focal_port_exposure_score(vessel: pd.Series, dist_km: pd.Series) -> pd.Series:
    v = pd.to_numeric(vessel, errors="coerce")
    d = pd.to_numeric(dist_km, errors="coerce")
    return v / (1.0 + d)


def zone_masks(dist_km: pd.Series, coastal_panel: pd.Series) -> dict[str, pd.Series]:
    d = pd.to_numeric(dist_km, errors="coerce")
    cp = coastal_panel.fillna(False).astype(bool)
    return {
        "0-3 km (coastal core)": (d <= 3.0) & cp,
        "0-3 km": d <= 3.0,
        "3-7 km": (d > 3.0) & (d <= 7.0),
        "7-15 km": (d > 7.0) & (d <= 15.0),
        "15-30 km": (d > 15.0) & (d <= 30.0),
    }


def resolve_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def metric_definitions(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(output_key, resolved_column) for numeric aggregation."""
    rows: list[tuple[str, str]] = [
        ("vessel_density_t", "vessel_density_t"),
        ("local_no2_excess", "local_no2_excess"),
        ("coastal_wind_alignment_score", "coastal_wind_alignment_score"),
        ("pollution_transport_wind_alignment_score", "pollution_transport_wind_alignment_score"),
        ("maritime_exposure_index", "maritime_exposure_index"),
        ("atmospheric_coastal_exposure_index", "atmospheric_coastal_exposure_index"),
        ("environmental_stress_index", "environmental_stress_index"),
    ]
    oil_c = resolve_col(df, ("oil_slick_probability_t", "oil_slick_proxy"))
    if oil_c:
        rows.append(("oil_slick_proxy", oil_c))
    for key, cands in (
        ("ndti_mean", ("ndti_mean_t", "ndti_mean", "sentinel_ndti_mean_t")),
        ("ndwi_mean", ("ndwi_mean_t", "ndwi_mean", "sentinel_ndwi_mean_t")),
        ("ndvi_mean", ("ndvi_mean_t", "ndvi_mean", "sentinel_ndvi_mean_t")),
    ):
        c = resolve_col(df, cands)
        if c:
            rows.append((key, c))
    if "port_exposure_score" in df.columns:
        rows.append(("port_exposure_score", "port_exposure_score"))
    return [(k, c) for k, c in rows if c in df.columns]


def wind_subsets(sh: pd.Series) -> dict[str, pd.Series]:
    shn = pd.to_numeric(sh, errors="coerce")
    return {
        "all": pd.Series(True, index=sh.index),
        "shoreward": shn.eq(1.0),
        "non_shoreward": shn.eq(0.0),
    }


def aggregate_long_table(
    df: pd.DataFrame,
    port_name: str,
    dist_col: str,
    metrics: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    coastal = df["coastal_panel"].fillna(False) if "coastal_panel" in df.columns else pd.Series(False, index=df.index)
    d = df[dist_col]
    masks = zone_masks(d, coastal)
    sh = df["shoreward_binary"] if "shoreward_binary" in df.columns else pd.Series(np.nan, index=df.index)
    wsubs = wind_subsets(sh)
    vessel_c = "vessel_density_t" if "vessel_density_t" in df.columns else None

    rows: list[dict[str, Any]] = []
    v_focal = focal_port_exposure_score(df[vessel_c], d) if vessel_c else None

    for zone, zm in masks.items():
        for wind_lbl, wm in wsubs.items():
            base = zm & wm & d.notna()
            if wind_lbl == "all":
                sub_all = df.loc[zm & d.notna()]
                sh_freq = (
                    float(pd.to_numeric(sub_all["shoreward_binary"], errors="coerce").eq(1).mean())
                    if "shoreward_binary" in sub_all.columns and len(sub_all) > 0
                    else float("nan")
                )
            else:
                sh_freq = float("nan")

            sub = df.loc[base]
            n = int(len(sub))
            if n == 0:
                continue

            # Focal recomputed exposure (association with port proximity geometry)
            if v_focal is not None:
                vf = pd.to_numeric(v_focal.loc[base], errors="coerce").dropna().to_numpy(float)
                mf, lo, hi = boot_mean_ci(vf)
                rows.append(
                    {
                        "port": port_name,
                        "distance_zone": zone,
                        "wind_regime": wind_lbl,
                        "metric": "focal_port_exposure_score",
                        "n": len(vf),
                        "mean": mf,
                        "median": float(np.median(vf)) if len(vf) else np.nan,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "spearman_dist_vs_metric": np.nan,
                        "shoreward_frequency": sh_freq,
                    },
                )

            for mkey, mcol in metrics:
                v = pd.to_numeric(sub[mcol], errors="coerce").dropna().to_numpy(float)
                if len(v) == 0:
                    continue
                mf, lo, hi = boot_mean_ci(v)
                rho_d = np.nan
                vd = pd.to_numeric(d.loc[base], errors="coerce")
                vv = pd.to_numeric(sub[mcol], errors="coerce")
                mcoef = vv.notna() & vd.notna()
                if int(mcoef.sum()) >= 15:
                    xs = vd.loc[mcoef].to_numpy()
                    ys = vv.loc[mcoef].to_numpy()
                    if np.unique(xs).size > 1 and np.unique(ys).size > 1:
                        rho_d, _ = stats.spearmanr(xs, ys)

                rows.append(
                    {
                        "port": port_name,
                        "distance_zone": zone,
                        "wind_regime": wind_lbl,
                        "metric": mkey,
                        "n": len(v),
                        "mean": mf,
                        "median": float(np.median(v)),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "spearman_dist_vs_metric": float(rho_d) if rho_d == rho_d else np.nan,
                        "shoreward_frequency": sh_freq,
                    },
                )

    return rows


def port_pairwise_mannwhitney(
    df: pd.DataFrame,
    port_a: str,
    port_b: str,
    dist_col_a: str,
    dist_col_b: str,
    zone: str,
    metric_col: str,
    wind_pred: pd.Series | None,
) -> dict[str, Any] | None:
    coastal = df["coastal_panel"].fillna(False)
    masks = zone_masks(df[dist_col_a], coastal)
    masks_b = zone_masks(df[dist_col_b], coastal)
    if zone not in masks:
        return None
    ma = masks[zone] & df[dist_col_a].notna()
    mb = masks_b[zone] & df[dist_col_b].notna()
    if wind_pred is not None:
        ma = ma & wind_pred.reindex(ma.index).fillna(False)
        mb = mb & wind_pred.reindex(mb.index).fillna(False)
    a = pd.to_numeric(df.loc[ma, metric_col], errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(df.loc[mb, metric_col], errors="coerce").dropna().to_numpy(float)
    if len(a) < 6 or len(b) < 6:
        return None
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "zone": zone,
        "metric": metric_col,
        "port_a": port_a,
        "port_b": port_b,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff_a_minus_b": float(np.mean(a) - np.mean(b)),
        "mannwhitney_p": float(u.pvalue),
        "cliffs_delta_a_vs_b": cliffs_delta(a, b),
    }


def build_rankings(decay_tbl: pd.DataFrame, df: pd.DataFrame, ports_filter: list[str]) -> pd.DataFrame:
    """Rank ports on pooled 0–30 km means and regime contrast / heuristics (filtered port list)."""
    t = decay_tbl.loc[
        (decay_tbl["wind_regime"] == "all") & (decay_tbl["port"].isin(ports_filter))
    ].copy()

    def _port_mean(metric: str, zones: list[str]) -> pd.Series:
        sub = t[(t["metric"] == metric) & (t["distance_zone"].isin(zones))]
        return sub.groupby("port")["mean"].mean()

    def _coverage_cells(port: str, col: str) -> int:
        dc = f"dist_{_safe_slug(port)}_km"
        if dc not in df.columns or col not in df.columns:
            return 0
        m = pd.to_numeric(df[dc], errors="coerce") <= 30.0
        return int(pd.to_numeric(df.loc[m, col], errors="coerce").notna().sum())

    pooled_z = ["0-3 km", "3-7 km", "7-15 km", "15-30 km"]
    Maritime = _port_mean("maritime_exposure_index", pooled_z)
    Atmo = _port_mean("atmospheric_coastal_exposure_index", pooled_z)
    Stress = _port_mean("environmental_stress_index", pooled_z)

    def _rank_series(s: pd.Series) -> pd.Series:
        return s.rank(ascending=False, method="average", na_option="keep")

    r_m = _rank_series(Maritime)
    r_a = _rank_series(Atmo)
    r_s = _rank_series(Stress)

    decay_m = []
    wind_sep = []
    ports = list(ports_filter)
    for p in ports:
        sub_m = t[(t["port"] == p) & (t["metric"] == "maritime_exposure_index") & (t["wind_regime"] == "all")]
        m03 = (
            float(sub_m.loc[sub_m["distance_zone"].eq("0-3 km"), "mean"].iloc[0])
            if len(sub_m[sub_m["distance_zone"].eq("0-3 km")])
            else np.nan
        )
        m1530 = (
            float(sub_m.loc[sub_m["distance_zone"].eq("15-30 km"), "mean"].iloc[0])
            if len(sub_m[sub_m["distance_zone"].eq("15-30 km")])
            else np.nan
        )
        decay_m.append(m03 - m1530 if (m03 == m03 and m1530 == m1530) else np.nan)

        ws = decay_tbl.loc[
            (decay_tbl["port"] == p)
            & (decay_tbl["metric"] == "environmental_stress_index")
            & (decay_tbl["distance_zone"].isin(pooled_z))
        ]
        sh = ws.loc[ws["wind_regime"].eq("shoreward"), ["distance_zone", "mean"]].set_index("distance_zone")
        ns = ws.loc[ws["wind_regime"].eq("non_shoreward"), ["distance_zone", "mean"]].set_index("distance_zone")
        common = sh.index.intersection(ns.index)
        if len(common):
            wind_sep.append(float(np.nanmean(np.abs(sh.loc[common, "mean"] - ns.loc[common, "mean"]))))
        else:
            wind_sep.append(np.nan)

    decay_ser = pd.Series(decay_m, index=ports)
    wind_ser = pd.Series(wind_sep, index=ports)
    r_decay = decay_ser.rank(ascending=False, method="average", na_option="keep")
    r_wind = wind_ser.rank(ascending=False, method="average", na_option="keep")

    out = pd.DataFrame(
        {
            "port": ports,
            "cells_30km_nonnull_vessel_density": [_coverage_cells(p, "vessel_density_t") for p in ports],
            "cells_30km_nonnull_coastal_wind_alignment": [_coverage_cells(p, "coastal_wind_alignment_score") for p in ports],
            "cells_30km_nonnull_maritime_index": [_coverage_cells(p, "maritime_exposure_index") for p in ports],
            "mean_maritime_index_pooled_bands": [Maritime.get(p, np.nan) for p in ports],
            "rank_maritime_exposure": [r_m.get(p, np.nan) for p in ports],
            "mean_atmospheric_index_pooled_bands": [Atmo.get(p, np.nan) for p in ports],
            "rank_atmospheric_structuring": [r_a.get(p, np.nan) for p in ports],
            "mean_stress_index_pooled_bands": [Stress.get(p, np.nan) for p in ports],
            "rank_environmental_stress": [r_s.get(p, np.nan) for p in ports],
            "inland_decay_maritime_mean_0_3_minus_15_30": list(decay_ser),
            "rank_inland_decay_maritime": list(r_decay),
            "wind_regime_separation_stress_index": list(wind_ser),
            "rank_wind_regime_dependence_stress": list(r_wind),
        },
    )
    out["mean_rank_across_criteria"] = out[
        ["rank_maritime_exposure", "rank_atmospheric_structuring", "rank_environmental_stress"]
    ].mean(axis=1, skipna=True)
    out["rank_composite_exposure_signal"] = out["mean_rank_across_criteria"].rank(ascending=True, method="average", na_option="keep")
    return out


def plot_index_ranking(rank_df: pd.DataFrame, save_path: Path | None = None) -> None:
    ports = rank_df["port"].tolist()
    x = np.arange(len(ports))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    series = [
        ("mean_maritime_index_pooled_bands", "Maritime exposure index", "#1f77b4", -w),
        ("mean_atmospheric_index_pooled_bands", "Atmospheric coastal exposure", "#ff7f0e", 0.0),
        ("mean_stress_index_pooled_bands", "Environmental stress index", "#2ca02c", w),
    ]
    legend_used: set[str] = set()
    for col, lab, color, off in series:
        for j, p in enumerate(ports):
            row = rank_df.loc[rank_df["port"] == p]
            h = float(row[col].iloc[0]) if len(row) else float("nan")
            leg = lab if lab not in legend_used else "_nolegend_"
            if np.isfinite(h):
                ax.bar(x[j] + off, h, width=w, color=color, label=leg, edgecolor="white", linewidth=0.5)
                legend_used.add(lab)
            else:
                ax.text(
                    x[j] + off,
                    0.02,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#555",
                    rotation=0,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(ports, rotation=12)
    ax.set_ylabel("Mean index (0–1 rank scale; pooled port-centred bands)")
    ax.set_title("Port comparison (Turku vs Mariehamn): composite exposure indices (association / structuring lens)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    cov_bits: list[str] = []
    if "cells_30km_nonnull_vessel_density" in rank_df.columns:
        for _, r in rank_df.iterrows():
            cov_bits.append(
                f"{r['port']}: vessel n={int(r['cells_30km_nonnull_vessel_density'])}, "
                f"wind n={int(r['cells_30km_nonnull_coastal_wind_alignment'])} "
                f"(≤30 km centroids)",
            )
    note = (
        "“n/a” = index undefined in this grid build: maritime and atmospheric composites need non-null "
        "**vessel density** and **coastal wind alignment** (and related fields) within the port window. "
        "That is missing coverage, not proof of zero maritime or atmospheric behaviour. "
        + (" Coverage: " + " | ".join(cov_bits) if cov_bits else "")
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=7.2, color="#333")
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(
        save_path or (FIGURES / "Fig1_port_exposure_index_ranking.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_decay_curves(
    decay_tbl: pd.DataFrame,
    *,
    save_dir: Path | None = None,
    filename_pattern: str = "Fig2_decay_curves_{slug}.png",
    suptitle_template: str | None = None,
) -> None:
    """
    Fixed-band distance-decay mode: all standard annuli in ZONE_ORDER (gaps = missing / n/a).
    Shoreward vs non-shoreward lines with bootstrap CIs.
    """
    metrics = [
        ("maritime_exposure_index", "Maritime exposure index"),
        ("local_no2_excess", "Local NO2 excess"),
        ("environmental_stress_index", "Environmental stress index"),
    ]
    xi = np.arange(len(ZONE_ORDER))
    out_dir = save_dir or FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    for port in FIGURE_PORTS:
        fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.5), sharex=True)
        for ax, (mkey, mlab) in zip(axes, metrics):
            has_any = False
            for wind_lbl, color, mk in (
                ("shoreward", "#1b7837", "o"),
                ("non_shoreward", "#5f5f5f", "s"),
            ):
                sub = decay_tbl[
                    (decay_tbl["port"] == port)
                    & (decay_tbl["metric"] == mkey)
                    & (decay_tbl["wind_regime"] == wind_lbl)
                ]
                sub = sub.set_index("distance_zone").reindex(ZONE_ORDER)
                if sub["mean"].notna().any():
                    has_any = True
                yerr_lo = sub["mean"] - sub["ci95_low"]
                yerr_hi = sub["ci95_high"] - sub["mean"]
                ax.errorbar(
                    xi,
                    sub["mean"],
                    yerr=[yerr_lo, yerr_hi],
                    fmt=mk + "-",
                    color=color,
                    capsize=3,
                    label=wind_lbl.replace("_", " "),
                )
            if not has_any:
                ax.text(0.5, 0.5, "No joint coverage\nfor this index", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_ylabel(mlab)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")
        axes[-1].set_xticks(xi)
        axes[-1].set_xticklabels(ZONE_ORDER, rotation=18)
        axes[-1].set_xlabel("Distance from focal port (port-centred zones)")
        if suptitle_template is not None:
            fig.suptitle(suptitle_template.format(port=port), fontsize=11, y=1.01)
        else:
            fig.suptitle(
                f"{port}: fixed-band distance decay (all zones; gaps = missing; bootstrap 95% CI; not causal)",
                fontsize=11,
                y=1.01,
            )
        fig.tight_layout()
        fname = filename_pattern.format(slug=_safe_slug(port))
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)


def build_shared_annulus_selection_df(
    decay_tbl: pd.DataFrame,
    metrics: list[str],
    ports: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for m in metrics:
        sel = select_shared_valid_annulus(decay_tbl, m, ports)
        if sel is not None:
            rows.append(
                {
                    "indicator": m,
                    "selected_annulus": sel.selected_annulus,
                    "Turku_n_shoreward": sel.turku_n_shoreward,
                    "Turku_n_nonshoreward": sel.turku_n_nonshoreward,
                    "Mariehamn_n_shoreward": sel.mariehamn_n_shoreward,
                    "Mariehamn_n_nonshoreward": sel.mariehamn_n_nonshoreward,
                    "reason_selected": sel.reason_selected,
                },
            )
        else:
            warnings.warn(
                f"[shared-valid annulus] No annulus in {WIND_REGIME_ZONE_PRIORITY} satisfies "
                f"both ports × shoreward/non-shoreward with n≥1 for indicator={m} (no imputation).",
                UserWarning,
                stacklevel=2,
            )
            rows.append(
                {
                    "indicator": m,
                    "selected_annulus": "n/a",
                    "Turku_n_shoreward": 0,
                    "Turku_n_nonshoreward": 0,
                    "Mariehamn_n_shoreward": 0,
                    "Mariehamn_n_nonshoreward": 0,
                    "reason_selected": "no_band_qualifies_in_priority_list_0-3_3-7_7-15_15-30_km",
                },
            )
    return pd.DataFrame(rows)


def plot_wind_regime_bars(
    decay_tbl: pd.DataFrame,
    *,
    save_path: Path | None = None,
    diagnostics_path: Path | None = None,
    log_selection: bool = True,
) -> pd.DataFrame:
    """
    Shared-valid-annulus comparison mode: per metric, first annulus where BOTH ports have
    shoreward and non_shoreward strata with n≥1 in decay_tbl. Does not impute missing bands.
    """
    metrics = ["maritime_exposure_index", "atmospheric_coastal_exposure_index", "environmental_stress_index"]
    ports = list(FIGURE_PORTS.keys())
    sel_df = build_shared_annulus_selection_df(decay_tbl, metrics, ports)
    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        sel_df.to_csv(diagnostics_path, index=False)

    if log_selection:
        print("[shared-valid annulus] Wind-regime contrast selections:")
        print(sel_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5), sharey=False)
    for ax, m in zip(axes, metrics):
        sel = select_shared_valid_annulus(decay_tbl, m, ports)
        x = np.arange(len(ports))
        w = 0.35
        disp = COMPOSITE_METRIC_DISPLAY.get(m, m.replace("_", " "))
        if sel is None:
            ax.clear()
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.5,
                0.55,
                "No shared valid annulus",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.text(
                0.5,
                0.32,
                "(no single band: both ports,\nboth wind regimes, n≥1)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="#444",
            )
            ax.set_title(f"{disp}\n(n/a)", fontsize=9)
            continue

        zone = sel.selected_annulus
        sh: list[float] = []
        ns: list[float] = []
        for p in ports:
            r1 = decay_tbl[
                (decay_tbl["port"] == p)
                & (decay_tbl["distance_zone"] == zone)
                & (decay_tbl["wind_regime"] == "shoreward")
                & (decay_tbl["metric"] == m)
            ]
            r2 = decay_tbl[
                (decay_tbl["port"] == p)
                & (decay_tbl["distance_zone"] == zone)
                & (decay_tbl["wind_regime"] == "non_shoreward")
                & (decay_tbl["metric"] == m)
            ]
            sh.append(float(r1["mean"].iloc[0]) if len(r1) else float("nan"))
            ns.append(float(r2["mean"].iloc[0]) if len(r2) else float("nan"))

        ax.bar(x - w / 2, sh, width=w, label="Shoreward", color="#1b7837")
        ax.bar(x + w / 2, ns, width=w, label="Non-shoreward", color="#888888")
        ax.set_xticks(x)
        ax.set_xticklabels(ports, rotation=12)
        zhuman = zone.replace("-", "–")
        ax.set_title(f"{disp}\n({zhuman})", fontsize=9)
        ax.text(
            0.5,
            -0.38,
            "Shared-valid-annulus comparison; not causal",
            transform=ax.transAxes,
            ha="center",
            fontsize=7.5,
            color="#333",
        )
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.0, color="#bbb", lw=0.4, zorder=0)

    axes[0].set_ylabel("Mean (0–1 rank scale for indices)")
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7)
    out_path = save_path or (FIGURES / "Fig3_port_wind_regime_comparison_0_3km.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(
        "Shared-valid-annulus wind-regime comparison (composite indices)\n"
        "Annulus chosen per indicator (nearest band: both ports, shoreward & non-shoreward, n≥1). "
        "Fixed-band decay curves are separate figures.",
        fontsize=10,
        y=1.08,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return sel_df


def _pooled_indicator_mean(decay_tbl: pd.DataFrame, port: str, metric_key: str) -> float:
    sub = decay_tbl[
        (decay_tbl["port"] == port)
        & (decay_tbl["metric"] == metric_key)
        & (decay_tbl["wind_regime"] == "all")
        & (decay_tbl["distance_zone"].isin(POOLED_BAND_ZONES))
    ]
    if sub.empty:
        return float("nan")
    return float(pd.to_numeric(sub["mean"], errors="coerce").mean())


def plot_indicator_only_pooled_turku_mariehamn(decay_tbl: pd.DataFrame) -> None:
    """Small multiples: Turku vs Mariehamn bars per raw indicator (pooled annuli means; all-wind stratum)."""
    available = set(decay_tbl["metric"].unique())
    specs = [(k, lab) for k, lab in INDICATOR_ONLY_SPECS if k in available]
    if not specs:
        return
    ncol = 3
    nrow = int(math.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.2, 3.0 * nrow), squeeze=False)
    ports_cmp = ["Turku", "Mariehamn"]
    colors = {"Turku": "#1f77b4", "Mariehamn": "#2ca02c"}
    for i, (mkey, mlab) in enumerate(specs):
        r, c = divmod(i, ncol)
        ax = axes[r][c]
        vals = [_pooled_indicator_mean(decay_tbl, p, mkey) for p in ports_cmp]
        xpos = np.arange(2)
        ax.bar(xpos, vals, color=[colors[p] for p in ports_cmp], width=0.55, edgecolor="white", linewidth=0.5)
        ax.set_xticks(xpos)
        ax.set_xticklabels(ports_cmp, fontsize=8)
        ax.set_title(mlab, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.0, color="#999", lw=0.5, zorder=0)
    for j in range(len(specs), nrow * ncol):
        r, c = divmod(j, ncol)
        axes[r][c].set_visible(False)
    fig.suptitle(
        "Indicators only — Turku vs Mariehamn "
        "(mean of zone means over 0–3 … 15–30 km; all wind; each panel uses its own y scale / units).",
        fontsize=10.5,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig6_indicators_pooled_turku_mariehamn.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_indicator_only_zones_turku_mariehamn(decay_tbl: pd.DataFrame) -> None:
    """Small multiples: same indicators vs port-centred distance zone (Turku vs Mariehamn lines)."""
    available = set(decay_tbl["metric"].unique())
    specs = [(k, lab) for k, lab in INDICATOR_ONLY_SPECS if k in available]
    if not specs:
        return
    xi = np.arange(len(ZONE_ORDER))
    ncol = 3
    nrow = int(math.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.8, 3.1 * nrow), squeeze=False)
    colors = {"Turku": "#1f77b4", "Mariehamn": "#2ca02c"}
    for i, (mkey, mlab) in enumerate(specs):
        r, c = divmod(i, ncol)
        ax = axes[r][c]
        for port in ("Turku", "Mariehamn"):
            sub = decay_tbl[
                (decay_tbl["port"] == port)
                & (decay_tbl["metric"] == mkey)
                & (decay_tbl["wind_regime"] == "all")
            ]
            sub = sub.set_index("distance_zone").reindex(ZONE_ORDER)
            ax.plot(
                xi,
                pd.to_numeric(sub["mean"], errors="coerce"),
                "o-",
                color=colors[port],
                label=port,
                lw=1.1,
                ms=3.5,
            )
        ax.set_xticks(xi)
        ax.set_xticklabels(ZONE_ORDER, rotation=18, ha="right", fontsize=7)
        ax.set_title(mlab, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        ax.axhline(0.0, color="#bbb", lw=0.4, zorder=0)
    for j in range(len(specs), nrow * ncol):
        r, c = divmod(j, ncol)
        axes[r][c].set_visible(False)
    fig.suptitle(
        "Indicators only — distance structure (Turku vs Mariehamn; all wind; panels share metric, scales differ).",
        fontsize=10.5,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig7_indicators_by_zone_turku_mariehamn.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_port_map(
    df: pd.DataFrame,
    port_name: str,
    dist_col: str,
    coast_lat: np.ndarray,
    coast_lon: np.ndarray,
    plat: float,
    plon: float,
) -> None:
    d = pd.to_numeric(df[dist_col], errors="coerce")
    win = d <= 35.0
    sub = df.loc[win].copy()
    if len(sub) < 5:
        return

    spec: dict[str, Any] = {
        "mei": ("maritime_exposure_index", "mean"),
        "lat": ("grid_centroid_lat", "first"),
        "lon": ("grid_centroid_lon", "first"),
        "no2": ("local_no2_excess", "mean"),
        "vd": ("vessel_density_t", "mean"),
    }
    if "wind_u_mean" in sub.columns:
        spec["wu"] = ("wind_u_mean", "mean")
        spec["wv"] = ("wind_v_mean", "mean")
    agg = sub.groupby("grid_cell_id", as_index=False).agg(**spec)

    dlon, dlat = 1.35, 0.85
    lon0, lon1 = plon - dlon, plon + dlon
    lat0, lat1 = plat - dlat, plat + dlat

    if len(coast_lat) and len(coast_lon):
        cmask = (
            (coast_lon >= lon0 - 0.5)
            & (coast_lon <= lon1 + 0.5)
            & (coast_lat >= lat0 - 0.5)
            & (coast_lat <= lat1 + 0.5)
        )
        clat, clon = coast_lat[cmask], coast_lon[cmask]
    else:
        clat = clon = np.array([])

    fig, ax = plt.subplots(figsize=(8.8, 8.0))
    if len(clon):
        ax.plot(clon, clat, color="#0c3b2e", lw=1.0, alpha=0.75, label="Coastline (Natural Earth)")
    sc = ax.scatter(
        agg["lon"],
        agg["lat"],
        c=agg["mei"],
        cmap="magma",
        s=np.clip(pd.to_numeric(agg["vd"], errors="coerce").fillna(0) * 120.0 + 25.0, 30.0, 220.0),
        edgecolors="k",
        linewidths=0.25,
        alpha=0.88,
        zorder=4,
    )
    plt.colorbar(sc, ax=ax, shrink=0.55, label="Maritime exposure index (mean)")
    hi = pd.to_numeric(agg["no2"], errors="coerce").quantile(0.9)
    hot = pd.to_numeric(agg["no2"], errors="coerce") >= hi
    if hot.any():
        ax.scatter(
            agg.loc[hot, "lon"],
            agg.loc[hot, "lat"],
            s=55,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.1,
            label="NO2 excess >= p90",
            zorder=5,
        )
    if "wu" in agg.columns and pd.to_numeric(agg["wu"], errors="coerce").notna().any():
        ax.quiver(
            agg["lon"],
            agg["lat"],
            pd.to_numeric(agg["wu"], errors="coerce"),
            pd.to_numeric(agg["wv"], errors="coerce"),
            scale=80,
            width=0.004,
            color="navy",
            alpha=0.45,
            zorder=3,
        )

    for r_km, sty in ((3, "--"), (7, ":"), (15, "-."), (30, "-")):
        circ = plt.Circle((plon, plat), r_km / 111.0, fill=False, linestyle=sty, color="#555", linewidth=1.0, zorder=2)
        ax.add_patch(circ)

    ax.plot(plon, plat, "r*", markersize=14, label="Port", zorder=6)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / math.cos(math.radians(plat)))
    ax.set_xlabel("Longitude °E")
    ax.set_ylabel("Latitude °N")
    ax.set_title(f"{port_name}: exposure intensity & context (≤35 km; WGS84; association map)")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(FIGURES / f"Fig4_port_map_{_safe_slug(port_name)}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(decay_tbl: pd.DataFrame, rank_df: pd.DataFrame, port_name: str) -> None:
    fig = plt.figure(figsize=(10.5, 7.0))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    # Mini ranking snippet
    rrow = rank_df.loc[rank_df["port"] == port_name].iloc[0]
    ax0.axis("off")
    ax0.set_title("Port summary ranks (1 = highest among three ports)")
    lines = [
        f"Maritime exposure rank: {rrow['rank_maritime_exposure']:.0f}",
        f"Atmospheric structuring rank: {rrow['rank_atmospheric_structuring']:.0f}",
        f"Stress index rank: {rrow['rank_environmental_stress']:.0f}",
        f"Inland decay (Δ 0–3 vs 15–30 km, MEI): {rrow['inland_decay_maritime_mean_0_3_minus_15_30']:.3f}",
        f"Wind separation (stress): {rrow['wind_regime_separation_stress_index']:.3f}",
    ]
    ax0.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace")

    metrics = ["maritime_exposure_index", "environmental_stress_index", "focal_port_exposure_score"]
    xi = np.arange(len(ZONE_ORDER))
    for m, color in zip(metrics, ["#1f77b4", "#2ca02c", "#9467bd"]):
        sub = decay_tbl[
            (decay_tbl["port"] == port_name) & (decay_tbl["metric"] == m) & (decay_tbl["wind_regime"] == "all")
        ]
        sub = sub.set_index("distance_zone").reindex(ZONE_ORDER)
        ax1.plot(xi, sub["mean"], "o-", color=color, label=m.replace("_", " ")[:32], lw=1.2)
    ax1.set_xticks(xi)
    ax1.set_xticklabels(ZONE_ORDER, rotation=15, fontsize=8)
    ax1.set_ylabel("Mean")
    ax1.set_title("Distance structure (all wind)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    for wind_lbl, color in ("shoreward", "#1b7837"), ("non_shoreward", "#666"):
        sub = decay_tbl[
            (decay_tbl["port"] == port_name)
            & (decay_tbl["metric"] == "coastal_wind_alignment_score")
            & (decay_tbl["wind_regime"] == wind_lbl)
        ]
        sub = sub.set_index("distance_zone").reindex(ZONE_ORDER)
        ax2.plot(xi, sub["mean"], "s-", color=color, label=wind_lbl, lw=1.1)
    ax2.set_xticks(xi)
    ax2.set_xticklabels(ZONE_ORDER, rotation=15, fontsize=8)
    ax2.set_ylabel("Coastal wind alignment")
    ax2.set_title("Coastal wind alignment vs port-centred distance (directional coupling proxy)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(f"{port_name} — integrated port dashboard (non-causal exposure lens)", fontsize=12)
    fig.savefig(FIGURES / f"Fig5_dashboard_{_safe_slug(port_name)}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def validation_summary(df: pd.DataFrame) -> str:
    lines = ["## Nearest-port field vs focal distance (coverage check)", ""]
    if "nearest_port" not in df.columns:
        lines.append("`nearest_port` not in dataset.")
        return "\n".join(lines)
    for pname in FOCAL_PORTS:
        dc = f"dist_{_safe_slug(pname)}_km"
        nearby = pd.to_numeric(df[dc], errors="coerce") <= 30
        vc = df.loc[nearby, "nearest_port"].value_counts(dropna=False).head(6)
        lines.append(f"### {pname}: cells ≤30 km from focal coordinate: {int(nearby.sum())}")
        lines.append(vc.to_string())
        lines.append("")
    return "\n".join(lines)


def write_summary_md(
    rank_df: pd.DataFrame,
    decay_tbl: pd.DataFrame,
    pairwise: pd.DataFrame,
    validation_block: str,
) -> None:
    lines = [
        "# Port-wise coastal exposure analysis",
        "",
        "**Framing:** association-based coastal and maritime **exposure structuring** around focal Baltic ports. ",
        "Distances are computed to **canonical port coordinates** (WGS84). Comparison **figures** are limited to ",
        "**Turku vs Mariehamn**; **Stockholm** remains in the long CSV tables for diagnostics and in ",
        "`outputs/reports/wind_coverage_audit.csv` (wind merge root cause).",
        "",
        "## Zones",
        "- Port-centred annuli: **0–3 km**, **3–7 km**, **7–15 km**, **15–30 km**.",
        "- **0–3 km (coastal core):** intersection of ≤3 km from port and **`coastal_panel`** (marine-adjacent study mask). ",
        "Highlights urban–port/coastal-interface behaviour.",
        "",
        "## Outputs",
        "- `port_exposure_ranking.csv` — pooled-band means and heuristic ranks.",
        "- `port_distance_decay_statistics.csv` — long-format means/medians/bootstrap CI per port × zone × wind × metric.",
        "- **`outputs/reports/wind_coverage_audit.csv`** — why wind alignment exists or not by port annulus; ",
        "run `python3 src/analysis/audit_wind_coverage.py` to refresh.",
        "",
        "### Statistical notes",
        "- **Mann–Whitney:** non-parametric comparison of pooled grid-week draws between ports in the same zone.",
        "- **Bootstrap CI:** 95% percentile intervals on zone means.",
        "- **Spearman** (in decay table): monotonic association of **focal distance** with each metric within the ",
        "zone/wind filter (descriptive gradient, not transport proof).",
        "",
        "## Data coverage caveat",
        "Composite indices (especially **maritime exposure**) require non-missing **vessel density** and **coastal wind ",
        "alignment** fields per grid-week. Cells near **Stockholm** in this build largely lack those inputs, so maritime ",
        "and some atmospheric summaries can be **sparse or empty** while stress and Sentinel means may still compute. ",
        "Use `cells_30km_nonnull_*` columns in `port_exposure_ranking.csv` and per-zone `n` in the decay table—**do not ",
        "interpret missing bars as zero physical exposure**.",
        "",
        "## Figure captions",
        "- **Fig1** — Bar comparison (**Turku vs Mariehamn**): mean maritime, atmospheric coastal, and environmental stress indices ",
        "(pooled over standard annuli).",
        "- **Fig2** — **Fixed-band** per-port decay: all standard zones (gaps = missing); indices and NO2 excess; shoreward vs non-shoreward with bootstrap CI.",
        "- **Fig3** — **Shared-valid-annulus** comparison (Turku vs Mariehamn): shoreward vs non-shoreward **per composite**; annulus = first of "
        "0–3 / 3–7 / 7–15 / 15–30 km where **both** ports have n≥1 in **both** wind strata (see `shared_annulus_selection.csv` in thesis bundle). "
        "Not the same as forcing a single band for every indicator in Fig2.",
        "- **Fig4** — Focused map around **Turku** or **Mariehamn** (each its own panel).",
        "- **Fig5** — Dashboard for **Turku** or **Mariehamn**.",
        "- **Fig6** — **Indicators only** (no composite indices): Turku vs Mariehamn **pooled** annulus means "
        "(0–3 … 15–30 km), one subplot per indicator.",
        "- **Fig7** — Same **raw indicators** vs **distance zone** with overlaid Turku / Mariehamn lines.",
        "",
        "## Interpretation hooks (allowed language)",
        "- Compare **spatial decay** of rank-scale indices outward from each hub.",
        "- Discuss **directional environmental association** via shoreward/non-shoreward splits.",
        "- Contrast **maritime-associated structuring** versus Sentinel or NO₂ behaviours where coverage allows.",
        "- Avoid causal transport, deposition, or source apportionment claims.",
        "",
    ]
    (REPORTS / "portwise_exposure_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--ne-cache", type=Path, default=_ROOT / "data" / "aux" / "natural_earth_coast_cache")
    ap.add_argument(
        "--wind-features-csv",
        type=Path,
        default=_ROOT / "outputs" / "reports" / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv",
    )
    args = ap.parse_args()

    if not Path(args.input).is_file():
        print(f"[FATAL] missing {args.input}")
        return 1

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    df = merge_wind_vectors(df, args.wind_features_csv)
    df = attach_focal_port_distances(df)
    df = prepare_panel(df, Path(args.ne_cache))
    df = build_indices(df)

    metrics = metric_definitions(df)

    all_rows: list[dict[str, Any]] = []
    for pname in FOCAL_PORTS:
        dc = f"dist_{_safe_slug(pname)}_km"
        all_rows.extend(aggregate_long_table(df, pname, dc, metrics))

    decay_tbl = pd.DataFrame(all_rows)
    decay_tbl.to_csv(REPORTS / "port_distance_decay_statistics.csv", index=False)

    rank_df = build_rankings(decay_tbl, df, list(FIGURE_PORTS.keys()))
    rank_df.to_csv(REPORTS / "port_exposure_ranking.csv", index=False)

    # Pairwise MW: shoreward pooled within zone — key metrics
    pairwise_rows: list[dict[str, Any]] = []
    port_list = list(FIGURE_PORTS.keys())
    targets = ["maritime_exposure_index", "local_no2_excess", "environmental_stress_index"]
    coastal_wind_ok = pd.to_numeric(df["shoreward_binary"], errors="coerce").eq(1)
    for i, pa in enumerate(port_list):
        for pb in port_list[i + 1 :]:
            dca = f"dist_{_safe_slug(pa)}_km"
            dcb = f"dist_{_safe_slug(pb)}_km"
            for zone in ZONE_ORDER:
                for met in targets:
                    r = port_pairwise_mannwhitney(df, pa, pb, dca, dcb, zone, met, None)
                    if r:
                        r = {**r, "shoreward_stratum": False}
                        pairwise_rows.append(r)
                    r2 = port_pairwise_mannwhitney(df, pa, pb, dca, dcb, zone, met, coastal_wind_ok)
                    if r2:
                        r2 = {**r2, "shoreward_stratum": True}
                        pairwise_rows.append(r2)
    pd.DataFrame(pairwise_rows).to_csv(REPORTS / "port_pairwise_mannwhitney.csv", index=False)

    val = validation_summary(df)
    write_summary_md(rank_df, decay_tbl, pd.DataFrame(pairwise_rows), val)

    cp = load_coastline_points(Path(args.ne_cache))
    if cp is None:
        coast_lat = np.array([])
        coast_lon = np.array([])
    else:
        coast_lat, coast_lon = cp

    sns.set_theme(style="whitegrid", font_scale=0.95)
    plot_index_ranking(rank_df)
    decay_fig = decay_tbl.loc[decay_tbl["port"].isin(FIGURE_PORTS.keys())].copy()
    plot_decay_curves(decay_fig)
    plot_wind_regime_bars(decay_fig)
    for pname, (plat, plon) in FIGURE_PORTS.items():
        plot_port_map(df, pname, f"dist_{_safe_slug(pname)}_km", coast_lat, coast_lon, plat, plon)
        plot_dashboard(decay_tbl, rank_df, pname)

    plot_indicator_only_pooled_turku_mariehamn(decay_tbl)
    plot_indicator_only_zones_turku_mariehamn(decay_tbl)

    print(f"[OK] {REPORTS / 'port_distance_decay_statistics.csv'}")
    print(f"[OK] {REPORTS / 'port_exposure_ranking.csv'}")
    print(f"[OK] {REPORTS / 'portwise_exposure_analysis_summary.md'}")
    print(f"[OK] figures -> {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
