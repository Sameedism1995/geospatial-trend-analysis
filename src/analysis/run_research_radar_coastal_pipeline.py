#!/usr/bin/env python3
"""
Research radar + coastal pipeline: localized association (not causality) between
maritime proximity / vessel exposure and radar, atmospheric, optical, and land indicators.

Writes ONLY under outputs/{reports,figures,visualizations}/run_research_radar_coastal_pipeline/

Run:
  python3 src/analysis/run_research_radar_coastal_pipeline.py
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

RUN = "run_research_radar_coastal_pipeline"
REPORTS = _ROOT / "outputs" / "reports" / RUN
FIGURES = _ROOT / "outputs" / "figures" / RUN
VIZ = _ROOT / "outputs" / "visualizations" / RUN

WINTER_MONTHS = {11, 12, 1, 2, 3}
DIST_BANDS = [(0, 5), (5, 10), (10, 20), (20, 30)]

RADAR_WINTER = ["oil_slick_probability_t", "detection_score"]
STEP2_INDICATORS = ["oil_slick_probability_t", "detection_score", "no2_mean_t", "ndti_mean", "ndwi_mean"]

GROUP_DEFS: dict[str, list[str]] = {
    "A_surface_disturbance": ["oil_slick_probability_t", "detection_score", "ndti_mean"],
    "B_atmospheric": ["no2_mean_t"],
    "C_land_coastal": ["ndvi_mean", "land_response_index"],
}


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


def mask_coastal_shipping(df: pd.DataFrame) -> pd.Series:
    dc = df["distance_to_coast_km"]
    ds = pd.to_numeric(df["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    return dc.notna() & (dc <= 30) & ds.notna() & (ds <= 20)


def mask_coastal(df: pd.DataFrame) -> pd.Series:
    dc = df["distance_to_coast_km"]
    return dc.notna() & (dc <= 30)


def mask_winter(df: pd.DataFrame) -> pd.Series:
    t = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    return t.dt.month.isin(WINTER_MONTHS)


def vessel_extreme_labels(df: pd.DataFrame, population: pd.Series) -> pd.Series:
    lab = pd.Series("exclude", index=df.index, dtype=object)
    idx = df.index[population.fillna(False)]
    if len(idx) == 0:
        return lab
    v = pd.to_numeric(df.loc[idx, "vessel_density_t"], errors="coerce")
    lg = np.log1p(v.clip(lower=0))
    if lg.notna().sum() < 50:
        return lab
    qhi, qlo = lg.quantile(0.90), lg.quantile(0.10)
    hi = lg >= qhi
    lo = lg <= qlo
    lab.loc[idx[hi.values]] = "high"
    lab.loc[idx[lo.values]] = "low"
    lab.loc[idx[~(hi | lo).values]] = "mid"
    return lab


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1))
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((float(np.mean(a)) - float(np.mean(b))) / pooled)


def high_low_comparison(
    df: pd.DataFrame,
    cols: list[str],
    *,
    vessel_col: pd.Series | None = None,
) -> pd.DataFrame:
    """vessel_col: boolean mask for population defining high/low deciles."""
    rows: list[dict[str, Any]] = []
    pop = vessel_col if vessel_col is not None else pd.Series(True, index=df.index)
    vx = vessel_extreme_labels(df, pop)
    sub = df.loc[vx.isin(["high", "low"])].copy()
    sub["_vx"] = vx.loc[sub.index]
    for c in cols:
        if c not in sub.columns:
            continue
        x = pd.to_numeric(sub[c], errors="coerce")
        hi = x.loc[sub["_vx"] == "high"].dropna().to_numpy(dtype=float)
        lo = x.loc[sub["_vx"] == "low"].dropna().to_numpy(dtype=float)
        hn, ln = len(hi), len(lo)
        r: dict[str, Any] = {
            "indicator": c,
            "n_high_vessel": hn,
            "n_low_vessel": ln,
            "mean_high_vessel": float(np.mean(hi)) if hn else float("nan"),
            "mean_low_vessel": float(np.mean(lo)) if ln else float("nan"),
            "mean_difference_high_minus_low": float(np.mean(hi) - np.mean(lo)) if hn and ln else float("nan"),
            "cohens_d": _cohens_d(hi, lo),
        }
        if hn >= 3 and ln >= 3:
            r["welch_t_p"] = float(stats.ttest_ind(hi, lo, equal_var=False).pvalue)
            r["mann_whitney_p"] = float(stats.mannwhitneyu(hi, lo, alternative="two-sided").pvalue)
        else:
            r["welch_t_p"] = float("nan")
            r["mann_whitney_p"] = float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def distance_decay_rows(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    dship = pd.to_numeric(df["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for lo, hi in DIST_BANDS:
        m = (dship >= lo) & (dship < hi)
        slab = df.loc[m]
        label = f"{lo}-{hi} km"
        for col in indicators:
            if col not in slab.columns:
                continue
            x = pd.to_numeric(slab[col], errors="coerce")
            v = x.dropna()
            rows.append(
                {
                    "distance_band": label,
                    "indicator": col,
                    "mean": float(v.mean()) if len(v) else float("nan"),
                    "median": float(v.median()) if len(v) else float("nan"),
                    "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                    "sample_count": int(len(v)),
                }
            )
    return pd.DataFrame(rows)


def decay_downward(monotone_vals: list[float]) -> bool:
    vv = np.asarray(monotone_vals, dtype=float)
    vv = vv[np.isfinite(vv)]
    if len(vv) < 2:
        return False
    return bool(np.nanmean(np.diff(vv)) < 0)


def spearman_corr(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    m = a.notna() & b.notna()
    aa = pd.to_numeric(a.loc[m], errors="coerce")
    bb = pd.to_numeric(b.loc[m], errors="coerce")
    m2 = aa.notna() & bb.notna()
    aa, bb = aa.loc[m2], bb.loc[m2]
    if len(aa) < 8:
        return float("nan"), float("nan"), int(len(aa))
    rho, p = stats.spearmanr(aa, bb, nan_policy="omit")
    return float(rho), float(p), int(len(aa))


def indicator_group_analysis(df_cs: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlations + composite note if all pairwise in cluster > 0.5."""
    rows: list[dict[str, Any]] = []
    for gname, cols in GROUP_DEFS.items():
        present = [c for c in cols if c in df_cs.columns]
        if len(present) < 2:
            for c in present:
                rows.append({"group": gname, "indicator_a": c, "indicator_b": "", "spearman_r": np.nan, "n": 0, "combine_eligible_pair": False})
            continue
        # pairwise
        corrs: list[float] = []
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                r, p, n = spearman_corr(df_cs[a], df_cs[b])
                elig = np.isfinite(r) and abs(r) > 0.5
                rows.append(
                    {
                        "group": gname,
                        "indicator_a": a,
                        "indicator_b": b,
                        "spearman_r": r,
                        "spearman_p": p,
                        "n": n,
                        "combine_eligible_pair": bool(elig),
                    }
                )
                if np.isfinite(r):
                    corrs.append(abs(r))

        exp_pairs = len(present) * (len(present) - 1) // 2
        triple_ok = bool(corrs and len(corrs) == exp_pairs and min(corrs) > 0.5)
        rows.append(
            {
                "group": gname,
                "indicator_a": "_SUMMARY",
                "indicator_b": "all_pairs_min_abs_r",
                "spearman_r": float(min(corrs)) if corrs else float("nan"),
                "spearman_p": np.nan,
                "n": int(len(df_cs)),
                "combine_eligible_pair": bool(triple_ok),
            }
        )
    return pd.DataFrame(rows)


def composite_z_columns(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    zs = []
    for c in cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        mu, sd = float(x.mean(skipna=True)), float(x.std(skipna=True))
        if not np.isfinite(sd) or sd == 0:
            zs.append(pd.Series(np.nan, index=df.index))
        else:
            zs.append((x - mu) / sd)
    if not zs:
        return pd.Series(np.nan, index=df.index)
    mat = pd.concat(zs, axis=1).mean(axis=1, skipna=True)
    return mat


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "final_run_stockholm_fixed_20260505_1356" / "processed" / "features_ml_ready.parquet",
    )
    p.add_argument("--ne-cache", type=Path, default=_ROOT / "data" / "aux" / "natural_earth_coast_cache")
    args = p.parse_args()
    inp = args.input if args.input.is_absolute() else _ROOT / args.input
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)

    if not inp.is_file():
        print(f"[FATAL] {inp} not found")
        return 1

    df = pd.read_parquet(inp)
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    n_total = len(df)

    df = ensure_coast_distance(df, Path(args.ne_cache))
    mc = mask_coastal_shipping(df)
    m_coast_only = mask_coastal(df)

    df_w_cs = df.loc[mask_winter(df) & mc].copy()
    df_cs_all = df.loc[mc].copy()
    df_coast_only_panel = df.loc[m_coast_only].copy()

    n_winter_cs = len(df_w_cs)
    n_coastal_ship = len(df_cs_all)
    n_coastal = int(m_coast_only.sum())

    # --- Step 1: winter radar ---
    winter_hilo = high_low_comparison(df, RADAR_WINTER, vessel_col=mask_winter(df) & mc)
    winter_hilo["panel"] = "winter_coastal_near_shipping"
    winter_hilo.to_csv(REPORTS / "winter_radar_high_vs_low.csv", index=False)

    w_decay = distance_decay_rows(df_w_cs, RADAR_WINTER)
    w_decay["panel"] = "winter_coastal_near_shipping"
    w_decay.to_csv(REPORTS / "winter_radar_distance_decay.csv", index=False)

    # --- Step 2: all indicators coastal+shipping full year ---
    s2_cols = [c for c in STEP2_INDICATORS if c in df.columns]
    s2_comp = high_low_comparison(df, s2_cols, vessel_col=mc)
    s2_comp["panel"] = "all_year_coastal_near_shipping"
    s2_comp.to_csv(REPORTS / "coastal_all_indicators_comparison.csv", index=False)

    s2_decay = distance_decay_rows(df_cs_all, s2_cols)
    s2_decay["panel"] = "all_year_coastal_near_shipping"
    s2_decay.to_csv(REPORTS / "coastal_distance_decay_all_indicators.csv", index=False)

    # --- Step 3: groups ---
    grp_df = indicator_group_analysis(df_cs_all)
    grp_df.to_csv(REPORTS / "indicator_group_analysis.csv", index=False)

    # --- Step 4: land impact (coastal only) ---
    land_cols = [c for c in ["ndvi_mean", "no2_mean_t", "oil_slick_probability_t"] if c in df.columns]
    land_comp = high_low_comparison(df, land_cols, vessel_col=m_coast_only)
    land_comp["row_type"] = "high_low_top_bottom_vessel"
    cor_parts: list[dict[str, Any]] = []
    if "ndvi_mean" in df_coast_only_panel.columns:
        if "oil_slick_probability_t" in df_coast_only_panel.columns:
            rho, pv, nn = spearman_corr(df_coast_only_panel["ndvi_mean"], df_coast_only_panel["oil_slick_probability_t"])
            cor_parts.append(
                {
                    "indicator": "SPEARMAN_ndvi_vs_oil_slick_probability_t",
                    "row_type": "spearman_within_coastal",
                    "spearman_r": rho,
                    "spearman_p": pv,
                    "n": nn,
                    "mean_high_vessel": np.nan,
                    "mean_low_vessel": np.nan,
                    "mean_difference_high_minus_low": np.nan,
                    "cohens_d": np.nan,
                    "welch_t_p": np.nan,
                    "mann_whitney_p": np.nan,
                    "n_high_vessel": np.nan,
                    "n_low_vessel": np.nan,
                }
            )
        if "no2_mean_t" in df_coast_only_panel.columns:
            rho, pv, nn = spearman_corr(df_coast_only_panel["ndvi_mean"], df_coast_only_panel["no2_mean_t"])
            cor_parts.append(
                {
                    "indicator": "SPEARMAN_ndvi_vs_no2_mean_t",
                    "row_type": "spearman_within_coastal",
                    "spearman_r": rho,
                    "spearman_p": pv,
                    "n": nn,
                    "mean_high_vessel": np.nan,
                    "mean_low_vessel": np.nan,
                    "mean_difference_high_minus_low": np.nan,
                    "cohens_d": np.nan,
                    "welch_t_p": np.nan,
                    "mann_whitney_p": np.nan,
                    "n_high_vessel": np.nan,
                    "n_low_vessel": np.nan,
                }
            )

    lp = pd.DataFrame(cor_parts) if cor_parts else pd.DataFrame()
    land_out = pd.concat([land_comp, lp], ignore_index=True)
    land_out.to_csv(REPORTS / "land_impact_analysis.csv", index=False)

    # --- diagnostics for terminal ---
    s2_sorted = s2_comp.dropna(subset=["cohens_d"]).copy()
    strongest_d_row = (
        s2_sorted.iloc[s2_sorted["cohens_d"].abs().argmax()].to_dict()
        if len(s2_sorted)
        else {}
    )
    small_p = s2_comp.copy()
    wm = pd.to_numeric(small_p["welch_t_p"], errors="coerce")
    mm = pd.to_numeric(small_p["mann_whitney_p"], errors="coerce")
    small_p["_p"] = np.fmin(wm.fillna(np.inf), mm.fillna(np.inf))
    small_p.loc[small_p["_p"] == np.inf, "_p"] = np.nan
    strongest_p_row = (
        small_p.loc[small_p["_p"].idxmin()].to_dict()
        if small_p["_p"].notna().any()
        else {}
    )

    oil_w = w_decay[w_decay["indicator"] == "oil_slick_probability_t"]["mean"].tolist()
    radar_decay_ok = decay_downward(oil_w) if len(oil_w) >= 2 else False

    ndti_b = (
        s2_decay[s2_decay["indicator"] == "no2_mean_t"]
        .set_index("distance_band")["mean"]
        .reindex([f"{a}-{b} km" for a, b in DIST_BANDS])
        .tolist()
    )
    no2_high = s2_comp[s2_comp["indicator"] == "no2_mean_t"]
    no2_decay_down = decay_downward([float(x) for x in ndti_b if np.isfinite(x)])
    no2_vessel_positive = False
    if len(no2_high):
        no2_vessel_positive = float(no2_high["mean_difference_high_minus_low"].iloc[0]) > 0

    no2_consistent = bool((no2_decay_down and no2_vessel_positive) or (not no2_decay_down and not no2_vessel_positive))
    no2_pattern_note = (
        "NO2 decay vs vessel-high pattern alignment (heuristic)"
        + (" — consistent sign story" if no2_consistent else " — mixed / opposite signs")
    )

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, ind, title in zip(axes, RADAR_WINTER, ["Oil slick probability (winter)", "Detection score (winter)"]):
        sl = w_decay[w_decay["indicator"] == ind]
        x = np.arange(len(sl))
        ax.bar(x, sl["mean"], yerr=sl["std"], capsize=3, color="steelblue", alpha=0.85, ecolor="0.35")
        ax.set_xticks(x)
        ax.set_xticklabels(sl["distance_band"], rotation=15)
        ax.set_title(title)
        ax.set_ylabel(ind)
    fig.suptitle("Winter radar — distance to high vessel-density cell (coastal ≤30 km)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "winter_radar_distance_decay.png", dpi=160)
    plt.close(fig)

    def box_vessel(y: str, fn: str, ttl: str, frame: pd.DataFrame, pop: pd.Series) -> None:
        vx = vessel_extreme_labels(df, pop)
        sub = frame.loc[vx.isin(["high", "low"])].copy()
        sub["exposure"] = vx.loc[sub.index].map({"high": "High vessel (top 10%)", "low": "Low vessel (bottom 10%)"})
        sub[y] = pd.to_numeric(sub[y], errors="coerce")
        sub = sub.dropna(subset=[y])
        if sub.empty:
            return
        fig, ax = plt.subplots(figsize=(5.4, 4))
        order = ["High vessel (top 10%)", "Low vessel (bottom 10%)"]
        sns.boxplot(data=sub, x="exposure", y=y, order=order, ax=ax, hue="exposure", hue_order=order, palette="Set2", legend=False)
        sns.stripplot(data=sub.sample(min(4000, len(sub))), x="exposure", y=y, order=order, ax=ax, color="0.35", alpha=0.06, size=1.2)
        ax.set_title(ttl)
        ax.set_xlabel("")
        fig.tight_layout()
        fig.savefig(FIGURES / fn, dpi=160)
        plt.close(fig)

    box_vessel("oil_slick_probability_t", "oil_slick_vs_vessel_boxplot.png", "Oil proxy vs vessel exposure (coastal, near shipping)", df_cs_all, mc)
    box_vessel("no2_mean_t", "no2_vs_vessel_boxplot.png", "NO₂ vs vessel exposure (coastal, near shipping)", df_cs_all, mc)

    # Coastal impact vs distance: mean z-composite of oil, detection, no2 per band
    zcols = [c for c in ["oil_slick_probability_t", "detection_score", "no2_mean_t"] if c in df_cs_all.columns]
    df_cs_all["_impact_z"] = composite_z_columns(df_cs_all, zcols)
    imp_decay = distance_decay_rows(df_cs_all.assign(**{}), ["_impact_z"])
    imp_decay = imp_decay.rename(columns={"indicator": "metric"})
    imp_decay["metric"] = "mean_z_composite_oil_detection_no2"
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sl = imp_decay
    x = np.arange(len(sl))
    ax.plot(x, sl["mean"], marker="o", color="darkred")
    ax.fill_between(x, sl["mean"] - sl["std"], sl["mean"] + sl["std"], alpha=0.2, color="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels(sl["distance_band"], rotation=15)
    ax.set_title("Coastal impact proxy (z-mean: oil + detection + NO₂) vs distance to shipping anchor")
    ax.set_ylabel("Composite z (exploratory)")
    fig.tight_layout()
    fig.savefig(FIGURES / "coastal_impact_vs_distance.png", dpi=160)
    plt.close(fig)

    # Group correlation heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for ax, (gname, cols) in zip(axes, GROUP_DEFS.items()):
        present = [c for c in cols if c in df_cs_all.columns]
        if len(present) < 2:
            ax.set_visible(False)
            continue
        M = df_cs_all[present].apply(pd.to_numeric, errors="coerce")
        C = M.corr(method="spearman", min_periods=20)
        sns.heatmap(C, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
        ax.set_title(gname.replace("_", " "))
    fig.suptitle("Indicator groups — Spearman correlation (coastal + near shipping)", y=1.05)
    fig.tight_layout()
    fig.savefig(FIGURES / "indicator_group_comparison.png", dpi=160)
    plt.close(fig)

    gallery_html = VIZ / "figures_gallery.html"
    imgs = sorted(FIGURES.glob("*.png"))
    body_parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Radar coastal pipeline figures</title></head>",
        "<body style='font-family:system-ui'><h2>Research radar coastal pipeline — figures</h2>",
        "<p>Open this file locally; paths are relative to this folder.</p><ul>",
    ]
    for im in imgs:
        rel = f"../figures/{RUN}/{im.name}"
        body_parts.append(f"<li><p><strong>{im.name}</strong></p><img src='{rel}' style='max-width:900px' alt=''></li>")
    body_parts.append("</ul></body></html>")
    gallery_html.write_text("\n".join(body_parts), encoding="utf-8")

    summary_path = REPORTS / "research_summary.md"
    lines = [
        "# Research summary — radar-first coastal pipeline",
        "",
        "Framing: **localized association** only; no causal claims.",
        "",
        "## 1. Radar (Sentinel-1–derived proxies)",
        "Winter months (Nov–Mar) under **coastal ≤30 km** and **≤20 km to high vessel-density cells** highlight **surface disturbance** metrics (`oil_slick_probability_t`, `detection_score`). "
        "These are **exploratory signals** (look-alikes, wind, speckle) — not confirmed slicks.",
        "",
        "## 2. Spatial footprint",
        "Restricting to the coastal strip and shipping-corridor distance emphasizes **short-range co-location** with chronic vessel-density structure rather than port-centric 100–1000 km axes.",
        "",
        "## 3. Surface disturbance decay",
        "Inspect `winter_radar_distance_decay.csv`: means often decline with distance band if the **surface disturbance** layer is lane-localized.",
        "",
        "## 4. Atmospheric influence",
        "`no2_mean_t` reflects **regional atmospheric** processes; expect **moderate** alignment with vessel extremes relative to SAR oil proxies.",
        "",
        "## 5. Optical water column",
        "NDTI / NDWI carry **cloud / season / ill-posedness** noise; treat as **weak / conditional** relative to winter radar.",
        "",
        "## 6. Land / coastal vegetation",
        "NDVI and `land_response_index` (if present) test **indirect** land-side response; effects are often **small and confounded** by mixed pixels.",
        "",
        "## Artifacts",
        f"- `{REPORTS.name}/` — CSV tables and this file",
        f"- `{FIGURES.name}/` — figures listed in run log",
        "",
        "### Empirical run snapshot (auto)",
        f"- Total rows: **{n_total}**",
        f"- Winter + coastal + near-shipping: **{n_winter_cs}**",
        f"- Coastal + near-shipping (all year): **{n_coastal_ship}**",
        f"- Strongest |Cohen's d| (step-2 indicators): **{strongest_d_row.get('indicator', 'n/a')}** (d={strongest_d_row.get('cohens_d', float('nan')):.4f})"
        if strongest_d_row
        else "- Strongest |Cohen's d|: n/a",
        f"- Smallest combined p (step-2): **{strongest_p_row.get('indicator', 'n/a')}** (p≈{strongest_p_row.get('_p', float('nan')):.2e})"
        if strongest_p_row
        else "- Smallest p: n/a",
        f"- Winter oil proxy mean trend vs distance (0→30 km bands): **{'downward (mean)' if radar_decay_ok else 'not clearly monotonic'}**",
        f"- {no2_pattern_note}",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    files = sorted(
        {str(p) for p in REPORTS.glob("*")}
        | {str(p) for p in FIGURES.glob("*")}
        | {str(p) for p in VIZ.glob("*")}
    )
    print("=== run_research_radar_coastal_pipeline ===")
    print(f"Total rows: {n_total}")
    print(f"Winter + coastal + near-shipping rows: {n_winter_cs}")
    print(f"Coastal (≤30 km coast) rows: {n_coastal}")
    print(f"Coastal + near-shipping (all year) rows: {n_coastal_ship}")
    if strongest_d_row:
        print(f"Strongest indicator by |Cohen's d| (step 2): {strongest_d_row['indicator']}  d={strongest_d_row['cohens_d']:.4f}")
    if strongest_p_row:
        print(f"Strongest by p-value (step 2): {strongest_p_row['indicator']}  p≈{strongest_p_row['_p']:.2e}")
    print(f"Winter radar (oil proxy) suggests distance decay: {'YES (mean declines with band)' if radar_decay_ok else 'UNCLEAR'}")
    print(f"NO₂ pattern heuristic: {'aligned' if no2_consistent else 'mixed'} — {no2_pattern_note}")
    print("\nGenerated files:")
    for f in sorted(files):
        print(" ", f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
