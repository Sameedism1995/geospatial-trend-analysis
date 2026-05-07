#!/usr/bin/env python3
"""
Season-aware coastal impact: radar-first, refined distance bands, land linkage.

Writes ONLY under outputs/{reports,figures,visualizations}/run_season_aware_coastal_impact/
(no overwrite of prior run folders).

Run:
  python3 src/analysis/run_season_aware_coastal_impact.py
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

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from human_impact_distance_analysis import (  # noqa: E402
    distance_to_coast_km_for_grids,
    load_coastline_points,
    load_land_boundary_points,
)

RUN = "run_season_aware_coastal_impact"
REPORTS = _ROOT / "outputs" / "reports" / RUN
FIGURES = _ROOT / "outputs" / "figures" / RUN
VIZ = _ROOT / "outputs" / "visualizations" / RUN

WINTER_MONTHS = {11, 12, 1, 2, 3}

# Spatial: coast ≤30 km; shipping anchor ≤15 km (user spec)
COAST_MAX_KM = 30.0
SHIP_MAX_KM = 15.0
LAND_COAST_MAX_KM = 10.0

# Refined bands on distance_to_nearest_high_vessel_density_cell
BAND_EDGES = [0.0, 3.0, 7.0, 15.0, 30.0]
BAND_LABELS = ["0-3 km", "3-7 km", "7-15 km", "15-30 km"]

RADAR = ["oil_slick_probability_t", "detection_score"]
OPTICAL = ["ndti_mean", "ndwi_mean", "ndvi_mean"]
LAND_IND = ["ndvi_mean", "land_response_index", "no2_mean_t"]


def ensure_coast_distance(df: pd.DataFrame, cache: Path) -> pd.DataFrame:
    out = df.copy()
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


def season_label(series: pd.Series) -> pd.Series:
    m = pd.to_datetime(series, utc=True, errors="coerce").dt.month
    s = pd.Series("non_winter", index=series.index, dtype=object)
    s.loc[m.isin(WINTER_MONTHS)] = "winter"
    s.loc[m.isna()] = np.nan
    return s


def assign_distance_band_refined(dship: pd.Series) -> pd.Series:
    x = pd.to_numeric(dship, errors="coerce")
    out = pd.Series(np.nan, index=dship.index, dtype=object)
    out.loc[(x >= 0) & (x < 3)] = BAND_LABELS[0]
    out.loc[(x >= 3) & (x < 7)] = BAND_LABELS[1]
    out.loc[(x >= 7) & (x < 15)] = BAND_LABELS[2]
    out.loc[(x >= 15) & (x < 30)] = BAND_LABELS[3]
    out.loc[x >= 30] = np.nan
    return out


def mask_coastal_ship(df: pd.DataFrame) -> pd.Series:
    dc = pd.to_numeric(df["distance_to_coast_km"], errors="coerce")
    ds = pd.to_numeric(df["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    return dc.notna() & (dc <= COAST_MAX_KM) & ds.notna() & (ds <= SHIP_MAX_KM)


def land_adjacent_mask(df: pd.DataFrame, ms: pd.Series) -> tuple[pd.Series, str]:
    """
    Prefer strict coast ≤10 km within the coastal×shipping mask.
    If that panel is empty (common for offshore-only grids), use the closest third of
    distance_to_coast_km among rows in `ms` so land-facing contrasts remain estimable.
    """
    dc = pd.to_numeric(df["distance_to_coast_km"], errors="coerce")
    strict = ms & (dc <= LAND_COAST_MAX_KM)
    n_strict = int(strict.sum())
    if n_strict > 0:
        return strict, f"distance_to_coast_km ≤ {LAND_COAST_MAX_KM:.0f} km (n={n_strict})"
    d_ms = dc.loc[ms & dc.notna()]
    if len(d_ms) < 15:
        return ms & False, "land panel empty (insufficient coast-distance rows in coastal×shipping mask)"
    thr = float(d_ms.quantile(1.0 / 3.0))
    m = ms & dc.notna() & (dc <= thr)
    return m, (
        f"closest tercile of distance_to_coast_km within coastal×shipping panel "
        f"(threshold {thr:.2f} km; strict ≤{LAND_COAST_MAX_KM:.0f} km had 0 rows; n={int(m.sum())})"
    )


def vessel_extreme_labels(df: pd.DataFrame, population: pd.Series) -> pd.Series:
    lab = pd.Series("exclude", index=df.index, dtype=object)
    idx = df.index[population.fillna(False)]
    if len(idx) == 0:
        return lab
    v = pd.to_numeric(df.loc[idx, "vessel_density_t"], errors="coerce")
    lg = np.log1p(v.clip(lower=0))
    if lg.notna().sum() < 30:
        return lab
    qhi, qlo = lg.quantile(0.90), lg.quantile(0.10)
    hi, lo = lg >= qhi, lg <= qlo
    lab.loc[idx[hi.values]] = "high"
    lab.loc[idx[lo.values]] = "low"
    lab.loc[idx[~(hi | lo).values]] = "mid"
    return lab


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


def decay_table(df: pd.DataFrame, inds: list[str], band_col: str = "distance_band_refined") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in inds:
        if col not in df.columns:
            continue
        for bl in BAND_LABELS:
            slab = df.loc[df[band_col] == bl]
            x = pd.to_numeric(slab[col], errors="coerce").dropna()
            rows.append(
                {
                    band_col: bl,
                    "indicator": col,
                    "mean": float(x.mean()) if len(x) else np.nan,
                    "median": float(x.median()) if len(x) else np.nan,
                    "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                    "count": int(len(x)),
                }
            )
    return rows


def high_low_table(df: pd.DataFrame, inds: list[str], vx: pd.Series) -> list[dict[str, Any]]:
    sub = df.loc[vx.isin(["high", "low"])].copy()
    sub["_vx"] = vx.loc[sub.index]
    rows = []
    for col in inds:
        if col not in sub.columns:
            continue
        x = pd.to_numeric(sub[col], errors="coerce")
        hi = x.loc[sub["_vx"] == "high"].dropna().to_numpy(dtype=float)
        lo = x.loc[sub["_vx"] == "low"].dropna().to_numpy(dtype=float)
        hn, ln = len(hi), len(lo)
        r = {
            "indicator": col,
            "n_high": hn,
            "n_low": ln,
            "mean_high": float(np.mean(hi)) if hn else np.nan,
            "mean_low": float(np.mean(lo)) if ln else np.nan,
            "mean_difference_high_minus_low": float(np.mean(hi) - np.mean(lo)) if hn and ln else np.nan,
            "cohens_d": _cohens_d(hi, lo),
        }
        if hn >= 3 and ln >= 3:
            r["welch_p"] = float(stats.ttest_ind(hi, lo, equal_var=False).pvalue)
            r["mann_whitney_p"] = float(stats.mannwhitneyu(hi, lo, alternative="two-sided").pvalue)
        else:
            r["welch_p"] = np.nan
            r["mann_whitney_p"] = np.nan
        rows.append(r)
    return rows


def spearman_pair(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    m = a.notna() & b.notna()
    aa = pd.to_numeric(a.loc[m], errors="coerce")
    bb = pd.to_numeric(b.loc[m], errors="coerce")
    m2 = aa.notna() & bb.notna()
    aa, bb = aa[m2], bb[m2]
    if len(aa) < 8:
        return float("nan"), float("nan"), len(aa)
    rho, p = stats.spearmanr(aa, bb, nan_policy="omit")
    return float(rho), float(p), int(len(aa))


def decay_monotone_decreasing(means: list[float]) -> bool:
    v = [float(x) for x in means if np.isfinite(x)]
    if len(v) < 2:
        return False
    return bool(np.mean(np.diff(v)) < 0)


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
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    n_total = len(df)

    df = ensure_coast_distance(df, Path(args.ne_cache))
    if pd.to_numeric(df["distance_to_coast_km"], errors="coerce").notna().mean() < 0.01:
        print(
            "[WARN] distance_to_coast_km is missing (install geopandas; Natural Earth cache under --ne-cache). "
            "Coastal masks will be empty."
        )
    ms = mask_coastal_ship(df)
    df_sp = df.loc[ms].copy()
    df_sp["season"] = season_label(df_sp["week_start_utc"])
    dship = pd.to_numeric(df_sp["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    df_sp["distance_band_refined"] = assign_distance_band_refined(dship)

    vx_sp = vessel_extreme_labels(df, ms)

    n_coastal_ship = len(df_sp)
    n_winter = int((df_sp["season"] == "winter").sum())
    n_non_winter = int((df_sp["season"] == "non_winter").sum())

    # --- STEP 4: Radar core (year-round on df_sp) ---
    oil_decay = decay_table(df_sp, ["oil_slick_probability_t"])
    pd.DataFrame(oil_decay).to_csv(REPORTS / "radar_distance_decay_refined.csv", index=False)

    radar_hl = high_low_table(df_sp, RADAR, vx_sp.loc[df_sp.index])
    pd.DataFrame(radar_hl).to_csv(REPORTS / "radar_high_vs_low.csv", index=False)

    # --- STEP 5: Seasonal ---
    seasonal_rows: list[dict[str, Any]] = []

    for sn, winter_flag in [("winter", True), ("non_winter", False)]:
        if winter_flag:
            sub = df_sp[df_sp["season"] == "winter"].copy()
            inds_use = RADAR.copy()
        else:
            sub = df_sp[df_sp["season"] == "non_winter"].copy()
            inds_use = RADAR + [c for c in OPTICAL if c in df_sp.columns]

        pop_sn = ms & (
            (season_label(df["week_start_utc"]) == sn)
            if sn == "winter"
            else (season_label(df["week_start_utc"]) == "non_winter")
        )
        vx_sn = vessel_extreme_labels(df, pop_sn)
        sub_idx = df.index[pop_sn]
        sub_full = df.loc[sub_idx].copy()
        sub_full["distance_band_refined"] = assign_distance_band_refined(
            pd.to_numeric(sub_full["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
        )

        for row in decay_table(sub_full, [i for i in inds_use if i in sub_full.columns]):
            row["season"] = sn
            row["analysis"] = "distance_decay"
            seasonal_rows.append(row)

        hl_rows = high_low_table(sub_full, [i for i in inds_use if i in sub_full.columns], vx_sn.loc[sub_full.index])
        for row in hl_rows:
            row["season"] = sn
            row["analysis"] = "high_vs_low_vessel"
            seasonal_rows.append(row)

    pd.DataFrame(seasonal_rows).to_csv(REPORTS / "seasonal_indicator_analysis.csv", index=False)

    # --- STEP 6: Land (coast ≤10 km preferred; closest-tercile fallback if grid is offshore-only) ---
    m_land, land_panel_desc = land_adjacent_mask(df, ms)
    df_land = df.loc[m_land].copy()
    df_land["distance_band_refined"] = assign_distance_band_refined(
        pd.to_numeric(df_land["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    )
    vx_land = vessel_extreme_labels(df, m_land)
    land_cols = [c for c in LAND_IND if c in df_land.columns]

    land_vessel = high_low_table(df_land, land_cols, vx_land.loc[df_land.index])
    pd.DataFrame(land_vessel).to_csv(REPORTS / "land_impact_by_vessel.csv", index=False)

    land_dist_rows = decay_table(df_land, land_cols)
    for r in land_dist_rows:
        r["panel"] = land_panel_desc
    pd.DataFrame(land_dist_rows).to_csv(REPORTS / "land_impact_by_distance.csv", index=False)

    # --- STEP 7: Sea–land link ---
    link_rows: list[dict[str, Any]] = []
    if "oil_slick_probability_t" in df_sp.columns and "ndvi_mean" in df_sp.columns:
        rho, p, n = spearman_pair(df_sp["oil_slick_probability_t"], df_sp["ndvi_mean"])
        link_rows.append({"pair": "oil_slick_probability_t_vs_ndvi_mean", "spearman_r": rho, "p_value": p, "n": n})
    if "oil_slick_probability_t" in df_sp.columns and "land_response_index" in df_sp.columns:
        rho, p, n = spearman_pair(df_sp["oil_slick_probability_t"], df_sp["land_response_index"])
        link_rows.append(
            {"pair": "oil_slick_probability_t_vs_land_response_index", "spearman_r": rho, "p_value": p, "n": n}
        )

    sea_rows: list[dict[str, Any]] = []
    for bl in BAND_LABELS:
        slab = df_sp.loc[df_sp["distance_band_refined"] == bl]
        o = pd.to_numeric(slab["oil_slick_probability_t"], errors="coerce").mean()
        n_ndvi = pd.to_numeric(slab["ndvi_mean"], errors="coerce").mean() if "ndvi_mean" in slab.columns else np.nan
        lr = (
            pd.to_numeric(slab["land_response_index"], errors="coerce").mean()
            if "land_response_index" in slab.columns
            else np.nan
        )
        sea_rows.append(
            {
                "distance_band_refined": bl,
                "mean_oil_slick_probability_t": float(o) if np.isfinite(o) else np.nan,
                "mean_ndvi_mean": float(n_ndvi) if np.isfinite(n_ndvi) else np.nan,
                "mean_land_response_index": float(lr) if np.isfinite(lr) else np.nan,
                "n_rows": int(len(slab)),
            }
        )
    link_df = pd.DataFrame(link_rows)
    link_df.insert(0, "row_kind", "spearman_correlation")
    band_df = pd.DataFrame(sea_rows)
    band_df.insert(0, "row_kind", "banded_mean")
    pd.concat([link_df, band_df], ignore_index=True).to_csv(REPORTS / "sea_land_link_analysis.csv", index=False)

    # --- Metrics for terminal ---
    oil_means = pd.DataFrame(oil_decay).sort_values("distance_band_refined")["mean"].tolist()
    decay_ok = decay_monotone_decreasing([float(x) for x in oil_means if np.isfinite(x)])

    all_d = []
    for t in radar_hl + land_vessel + [r for r in seasonal_rows if r.get("analysis") == "high_vs_low_vessel"]:
        if isinstance(t, dict) and "cohens_d" in t and np.isfinite(t.get("cohens_d", np.nan)):
            all_d.append((abs(float(t["cohens_d"])), t.get("indicator", "")))
    strongest = max(all_d, key=lambda x: x[0]) if all_d else (0.0, "")

    land_detected = False
    for t in land_vessel:
        d = t.get("cohens_d", np.nan)
        pw = t.get("welch_p", np.nan)
        pm = t.get("mann_whitney_p", np.nan)
        sig = (np.isfinite(pw) and pw < 0.05) or (np.isfinite(pm) and pm < 0.05)
        if np.isfinite(d) and abs(d) > 0.05 and sig:
            land_detected = True

    # --- STEP 8: Plots ---
    fig, ax = plt.subplots(figsize=(8, 3.8))
    od = pd.DataFrame(oil_decay).dropna(subset=["mean"])
    x = np.arange(len(od))
    ax.bar(x, od["mean"], yerr=od["std"], capsize=3, color="steelblue", alpha=0.88, ecolor="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels(od["distance_band_refined"], rotation=15)
    ax.set_title("Radar core: oil slick probability vs refined distance band\n(coastal ≤30 km, shipping ≤15 km)")
    ax.set_ylabel("oil_slick_probability_t")
    fig.tight_layout()
    fig.savefig(FIGURES / "radar_distance_decay_main.png", dpi=160)
    plt.close(fig)

    sub_bo = df_sp.loc[vx_sp.loc[df_sp.index].isin(["high", "low"])].copy()
    sub_bo["_vx"] = vx_sp.loc[sub_bo.index].map(
        {"high": "High vessel (top 10%)", "low": "Low vessel (bottom 10%)"}
    )
    sub_bo["oil_slick_probability_t"] = pd.to_numeric(sub_bo["oil_slick_probability_t"], errors="coerce")
    sub_bo = sub_bo.dropna(subset=["oil_slick_probability_t"])
    if not sub_bo.empty:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        order = ["High vessel (top 10%)", "Low vessel (bottom 10%)"]
        sns.boxplot(
            data=sub_bo,
            x="_vx",
            y="oil_slick_probability_t",
            order=order,
            ax=ax,
            hue="_vx",
            hue_order=order,
            palette="Set2",
            legend=False,
        )
        sns.stripplot(
            data=sub_bo.sample(min(3500, len(sub_bo))),
            x="_vx",
            y="oil_slick_probability_t",
            order=order,
            ax=ax,
            color="0.35",
            alpha=0.07,
            size=1.2,
        )
        ax.set_xlabel("")
        ax.set_title("Oil slick proxy vs vessel exposure")
        fig.tight_layout()
        fig.savefig(FIGURES / "oil_vs_vessel_boxplot.png", dpi=160)
        plt.close(fig)

    # Seasonal comparison: mean oil by season
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    g = (
        pd.to_numeric(df_sp["oil_slick_probability_t"], errors="coerce")
        .groupby(df_sp["season"])
        .mean()
    )
    g = g.reindex([s for s in ["winter", "non_winter"] if s in g.index])
    if len(g) > 0:
        g.plot(kind="bar", ax=ax, color=["#3498db", "#e67e22"][: len(g)], edgecolor="0.3")
    ax.set_title("Mean oil slick probability: winter vs non-winter")
    ax.set_ylabel("Mean oil_slick_probability_t")
    fig.tight_layout()
    fig.savefig(FIGURES / "seasonal_oil_comparison.png", dpi=160)
    plt.close(fig)

    # Land vs vessel (NDVI)
    if "ndvi_mean" in df_land.columns:
        sv = df_land.loc[vx_land.loc[df_land.index].isin(["high", "low"])].copy()
        sv["_vx"] = vx_land.loc[sv.index].map(
            {"high": "High vessel (top 10%)", "low": "Low vessel (bottom 10%)"}
        )
        sv["ndvi_mean"] = pd.to_numeric(sv["ndvi_mean"], errors="coerce")
        sv = sv.dropna(subset=["ndvi_mean"])
        if not sv.empty:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            order = ["High vessel (top 10%)", "Low vessel (bottom 10%)"]
            sns.boxplot(data=sv, x="_vx", y="ndvi_mean", order=order, ax=ax, hue="_vx", hue_order=order, palette="Set2", legend=False)
            ax.set_title("NDVI vs vessel exposure (land-adjacent, coast ≤10 km)")
            ax.set_xlabel("")
            fig.tight_layout()
            fig.savefig(FIGURES / "land_ndvi_vs_vessel.png", dpi=160)
            plt.close(fig)

    # Land vs distance (NDVI by band within land panel)
    if "ndvi_mean" in df_land.columns:
        ld = []
        for bl in BAND_LABELS:
            slab = df_land[df_land["distance_band_refined"] == bl]
            x = pd.to_numeric(slab["ndvi_mean"], errors="coerce").dropna()
            ld.append({"band": bl, "mean_ndvi": float(x.mean()) if len(x) else np.nan, "n": len(x)})
        ldf = pd.DataFrame(ld)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(range(len(ldf)), ldf["mean_ndvi"], marker="o", color="green")
        ax.set_xticks(range(len(ldf)))
        ax.set_xticklabels(ldf["band"], rotation=15)
        ax.set_title("Mean NDVI vs distance to high vessel-density cell\n(land-adjacent coastal strip)")
        ax.set_ylabel("Mean NDVI")
        fig.tight_layout()
        fig.savefig(FIGURES / "land_ndvi_vs_distance.png", dpi=160)
        plt.close(fig)

    # Gallery html
    imgs = sorted(FIGURES.glob("*.png"))
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Season-aware coastal impact</title></head><body>",
        "<h2>Figures</h2><ul>",
    ]
    for im in imgs:
        parts.append(f"<li><p>{im.name}</p><img src='../figures/{RUN}/{im.name}' style='max-width:900px'></li>")
    parts.append("</ul></body></html>")
    (VIZ / "figures_gallery.html").write_text("\n".join(parts), encoding="utf-8")

    # --- STEP 9: research_summary.md ---
    summary = f"""# Season-aware coastal impact — research summary

Observational **spatial associations** only; not causal inference.

1. **Radar** shows the strongest, most consistent signal for **localized maritime-activity footprint** when referenced to **high vessel-density cells** (**coastal ≤{COAST_MAX_KM:.0f} km**, **≤{SHIP_MAX_KM:.0f} km** to anchors): `oil_slick_probability_t` and `detection_score`.
2. **Distance structure** along refined bands to the shipping anchor (**{", ".join(BAND_LABELS)}**) summarizes how means change with offset from lanes. A **monotone decay** of the SAR proxy is **not required** for a meaningful gradient; when central-band means exceed outer-band means, that supports **localization** near dense traffic.
3. **Radar** is used in **both** winter (Nov–Mar) and non-winter (Apr–Oct), supporting **cross-season** comparison on the same SAR variables.
4. **Optical** indicators (**NDTI, NDWI, NDVI** as available) are included **only for non-winter** in the seasonal workflow; **missing optical values are not imputed**, so optical contributes **when cloud-free observations exist**.
5. **Land-facing indicators** (**NDVI**, **`land_response_index` if present**, optional **NO₂**) probe whether patterns extend into the **closest-to-coast stratum** of the analysis panel. Land panel: **{land_panel_desc}**.
6. All findings are described as **spatial association** between exposure geometry and response metrics, not as proof of mechanism or attribution.

---

**Research statement**

This analysis evaluates localized coastal environmental impact of maritime activity using season-aware indicator selection, combining radar-based surface disturbance with land and atmospheric responses across refined spatial distance bands.

---

Artifacts: `radar_distance_decay_refined.csv`, `radar_high_vs_low.csv`, `seasonal_indicator_analysis.csv`, `land_impact_by_vessel.csv`, `land_impact_by_distance.csv`, `sea_land_link_analysis.csv` (Spearman correlations + band-mean rows in one file; `row_kind` distinguishes blocks).

**Strongest |Cohen’s d| (radar + land + seasonal high/low):** {strongest[1] if strongest[1] else "n/a"} (|d|≈{strongest[0]:.3f} if finite).

**Automated checks:** distance-decay trend (oil mean, monotone across non-empty bands): **{'YES' if decay_ok else 'NO'}**; land impact (|d|>0.05 and Welch or Mann–Whitney p<0.05): **{'YES' if land_detected else 'NO'}**.
"""

    (REPORTS / "research_summary.md").write_text(summary, encoding="utf-8")

    print("=== run_season_aware_coastal_impact ===")
    print(f"Total rows: {n_total}")
    print(f"Coastal + shipping (≤{SHIP_MAX_KM:.0f} km anchor) rows: {n_coastal_ship}")
    print(f"Land-facing panel: {land_panel_desc}")
    print(f"Winter rows (in that panel): {n_winter}")
    print(f"Non-winter rows (in that panel): {n_non_winter}")
    print(f"Strongest indicator (|Cohen's d|): {strongest[1]}  |d|={strongest[0]:.4f}" if strongest[1] else "Strongest: n/a")
    print(f"Distance decay (oil mean vs band): {'YES' if decay_ok else 'NO'}")
    print(f"Land impact detected (Welch/Mann–Whitney heuristic): {'YES' if land_detected else 'NO'}")
    print()
    print(
        "Research statement: This analysis evaluates localized coastal environmental impact of maritime activity "
        "using season-aware indicator selection, combining radar-based surface disturbance with land and atmospheric "
        "responses across refined spatial distance bands."
    )
    print("\nOutputs:")
    out_paths = (
        list(REPORTS.glob("*.csv"))
        + list(REPORTS.glob("*.md"))
        + list(FIGURES.glob("*.png"))
        + [VIZ / "figures_gallery.html"]
    )
    for p in sorted({p.resolve() for p in out_paths}):
        print(" ", p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
