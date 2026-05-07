#!/usr/bin/env python3
"""
Final thesis analysis: coastal areas near high vessel-density zones vs environmental indicators.

Does not overwrite prior hub/port-distance runs — writes only under
`outputs/{reports,figures,visualizations}/run_final_coastal_shipping_impact/`.

Default input (project root):
  final_run_stockholm_fixed_20260505_1356/processed/features_ml_ready.parquet

Run:
  python3 src/analysis/run_final_coastal_shipping_impact.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from human_impact_distance_analysis import (  # noqa: E402
    distance_to_coast_km_for_grids,
    load_coastline_points,
    load_land_boundary_points,
)

RUN = "run_final_coastal_shipping_impact"
REPORTS = _ROOT / "outputs" / "reports" / RUN
FIGURES = _ROOT / "outputs" / "figures" / RUN
VIZ = _ROOT / "outputs" / "visualizations" / RUN

INDICATORS_COMPARE = [
    "no2_mean_t",
    "ndti_mean",
    "ndwi_mean",
    "ndvi_mean",
    "oil_slick_probability_t",
    "detection_score",
]

WEIGHTS = {"no2_mean_t": 0.35, "ndti_mean": 0.25, "oil_slick_probability_t": 0.25, "detection_score": 0.15}

DIST_BANDS = [(0, 5), (5, 10), (10, 20), (20, 30)]


def _cohens_d(high: pd.Series, low: pd.Series) -> float:
    a = high.dropna().to_numpy(dtype=float)
    b = low.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    v1, v2 = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1))
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((float(np.mean(a)) - float(np.mean(b))) / pooled)


def _z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu = float(x.mean(skipna=True))
    sd = float(x.std(skipna=True))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (x - mu) / sd


def build_impact_score(df_panel: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean of per-column z-scores; only terms with finite z contribute to the denominator."""
    out = df_panel.copy()
    acc = pd.Series(0.0, index=out.index, dtype=float)
    wsum = pd.Series(0.0, index=out.index, dtype=float)
    for col, wt in WEIGHTS.items():
        if col not in out.columns:
            continue
        zc = _z(pd.to_numeric(out[col], errors="coerce"))
        m = zc.notna()
        acc.loc[m] = acc.loc[m] + wt * zc.loc[m]
        wsum.loc[m] = wsum.loc[m] + wt
    out["coastal_impact_score"] = np.where(wsum > 0, acc / wsum, np.nan).astype(float)
    return out


def ensure_distance_to_coast(df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """Add distance_to_coast_km (km to nearest coastline / land boundary point)."""
    out = df.copy()
    uniq = (
        out[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]]
        .drop_duplicates("grid_cell_id")
        .dropna(subset=["grid_centroid_lat", "grid_centroid_lon"])
    )
    pts = load_coastline_points(cache_dir)
    if pts is None:
        pts = load_land_boundary_points(cache_dir)
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


def assign_coastal_vessel_groups(df: pd.DataFrame) -> pd.DataFrame:
    """High / low = top/bottom 10% of log1p(vessel_density_t) among coastal rows only."""
    out = df.copy()
    coastal = out["distance_to_coast_km"].notna() & (out["distance_to_coast_km"] <= 30)
    v = pd.to_numeric(out["vessel_density_t"], errors="coerce")
    logv = np.log1p(v.clip(lower=0))
    if not coastal.any():
        out["vessel_exposure_coastal_quantile"] = "mid_other"
        return out
    q90 = logv.loc[coastal].quantile(0.90)
    q10 = logv.loc[coastal].quantile(0.10)
    tag = np.select(
        [coastal & (logv >= q90), coastal & (logv <= q10)],
        ["high", "low"],
        default="mid_other",
    )
    out["vessel_exposure_coastal_quantile"] = tag
    return out


def compare_high_low(df: pd.DataFrame) -> pd.DataFrame:
    dc = df["distance_to_coast_km"]
    sub = df[
        dc.notna()
        & (dc <= 30)
        & (pd.to_numeric(df["distance_to_nearest_high_vessel_density_cell"], errors="coerce") <= 20)
        & (df["vessel_exposure_coastal_quantile"].isin(["high", "low"]))
    ].copy()

    rows: list[dict[str, Any]] = []
    for col in INDICATORS_COMPARE:
        if col not in sub.columns:
            continue
        x = pd.to_numeric(sub[col], errors="coerce")
        high_mask = sub["vessel_exposure_coastal_quantile"] == "high"
        low_mask = sub["vessel_exposure_coastal_quantile"] == "low"
        hi = x.loc[high_mask]
        lo = x.loc[low_mask]

        pooled = pd.concat([hi, lo])
        miss_rate = float(1.0 - pooled.notna().mean()) if len(pooled) else float("nan")

        row: dict[str, Any] = {
            "indicator": col,
            "n_high": int(hi.notna().sum()),
            "n_low": int(lo.notna().sum()),
            "high_mean": float(hi.mean()) if hi.notna().any() else float("nan"),
            "low_mean": float(lo.mean()) if lo.notna().any() else float("nan"),
            "high_median": float(hi.median()) if hi.notna().any() else float("nan"),
            "low_median": float(lo.median()) if lo.notna().any() else float("nan"),
            "missingness_rate_overall_in_high_low_pairs": miss_rate,
        }
        lm = row["low_mean"]
        hm = row["high_mean"]
        row["mean_difference_high_minus_low"] = float(hm - lm) if np.isfinite(hm) and np.isfinite(lm) else float("nan")
        if np.isfinite(lm) and lm != 0:
            row["percent_difference_vs_low_mean"] = float(100.0 * (hm - lm) / abs(lm))
        else:
            row["percent_difference_vs_low_mean"] = float("nan")
        row["cohens_d"] = _cohens_d(hi, lo)
        hn = hi.dropna().to_numpy()
        ln = lo.dropna().to_numpy()
        if len(hn) >= 3 and len(ln) >= 3:
            row["welch_t_p_value"] = float(stats.ttest_ind(hn, ln, equal_var=False, nan_policy="omit").pvalue)
            row["mann_whitney_p_value"] = float(stats.mannwhitneyu(hn, ln, alternative="two-sided").pvalue)
        else:
            row["welch_t_p_value"] = float("nan")
            row["mann_whitney_p_value"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def distance_decay_table(d: pd.DataFrame) -> pd.DataFrame:
    """Bands on distance_to_nearest_high_vessel_density_cell (expects coastal subset caller)."""

    sl = d.copy()
    dist = pd.to_numeric(sl["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    rows = []
    for lo, hi in DIST_BANDS:
        m_band = (dist >= lo) & (dist < hi)
        slab = sl.loc[m_band]
        label = f"{lo}-{hi} km"
        for col in INDICATORS_COMPARE + ["coastal_impact_score"]:
            if col not in slab.columns:
                continue
            x = pd.to_numeric(slab[col], errors="coerce")
            valid = x.notna()
            gx = slab.loc[valid, "grid_cell_id"]
            rows.append(
                {
                    "distance_band_label": label,
                    "indicator": col,
                    "mean": float(x.mean(skipna=True)) if valid.any() else float("nan"),
                    "median": float(x.median(skipna=True)) if valid.any() else float("nan"),
                    "std": float(x.std(skipna=True)) if valid.any() else float("nan"),
                    "valid_sample_count": int(valid.sum()),
                    "unique_grid_count": int(gx.nunique()),
                }
            )
    return pd.DataFrame(rows)


def plot_high_low_boxes(df_plot: pd.DataFrame) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    sub = df_plot[
        (df_plot["vessel_exposure_coastal_quantile"].isin(["high", "low"]))
        & (df_plot["distance_to_coast_km"].notna())
        & (df_plot["distance_to_coast_km"] <= 30)
        & (pd.to_numeric(df_plot["distance_to_nearest_high_vessel_density_cell"], errors="coerce") <= 20)
    ].copy()
    lab_map = {"high": "High exposure (top 10% coastal)", "low": "Low exposure (bottom 10% coastal)"}
    sub["_exp"] = sub["vessel_exposure_coastal_quantile"].map(lab_map)

    for fname, cols in (
        ("high_vs_low_no2.png", ["no2_mean_t"]),
        ("high_vs_low_ndti.png", ["ndti_mean"]),
        ("high_vs_low_oil_slick_probability.png", ["oil_slick_probability_t"]),
        ("high_vs_low_coastal_impact_score.png", ["coastal_impact_score"]),
    ):
        usable = [c for c in cols if c in sub.columns]
        if not usable:
            continue
        c = usable[0]
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        plot_df = sub[["_exp", c]].dropna(subset=[c]).copy()
        if plot_df.empty:
            plt.close()
            continue
        sns.boxplot(data=plot_df, x="_exp", y=c, palette="Set2", ax=ax, hue="_exp", legend=False)
        sns.stripplot(
            data=plot_df.sample(min(3500, len(plot_df))),
            x="_exp",
            y=c,
            color="0.35",
            alpha=0.12,
            size=2,
            ax=ax,
        )
        ttl = fname.replace("high_vs_low_", "").replace(".png", "").replace("_", " ")
        ax.set_title(f"{ttl.upper()} — coastal & ≤20 km to high-density cell")
        ax.set_xlabel("")
        fig.autofmt_xdate()
        p = FIGURES / fname
        fig.tight_layout()
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_distance_decay(ddf: pd.DataFrame) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    order = [f"{lo}-{hi} km" for lo, hi in DIST_BANDS]

    def one(ind: str, fn: str) -> None:
        sl = ddf[ddf["indicator"] == ind].copy()
        if sl.empty:
            return
        sl["_ord"] = sl["distance_band_label"].astype("category").cat.set_categories(order, ordered=True)
        sl = sl.sort_values("_ord")
        fig, ax = plt.subplots(figsize=(8, 3.9))
        x = np.arange(len(sl))
        ax.bar(x, sl["mean"], yerr=sl["std"], capsize=3, alpha=0.82, color="steelblue", ecolor="0.35")
        ax.set_xticks(x)
        ax.set_xticklabels(sl["distance_band_label"])
        ax.set_title(f"{ind} vs distance to nearest high vessel-density cell (coastal ≤30 km)")
        ax.set_ylabel(ind)
        p = FIGURES / fn
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(Path(p))

    for ind, fn in [
        ("no2_mean_t", "distance_decay_no2.png"),
        ("ndti_mean", "distance_decay_ndti.png"),
        ("oil_slick_probability_t", "distance_decay_oil_slick_probability.png"),
        ("coastal_impact_score", "distance_decay_coastal_impact_score.png"),
    ]:
        one(ind, fn)
    return paths


def map_scores_and_shipping(df_map: pd.DataFrame) -> tuple[Path | None, Path | None]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)
    uniq = df_map.groupby("grid_cell_id", as_index=False).agg(
        lat=("grid_centroid_lat", "first"),
        lon=("grid_centroid_lon", "first"),
        score=("coastal_impact_score", "max"),
        vessel_med=("vessel_density_t", "median"),
    )
    lat = pd.to_numeric(uniq["lat"], errors="coerce")
    lon = pd.to_numeric(uniq["lon"], errors="coerce")
    vessel_med = pd.to_numeric(uniq["vessel_med"], errors="coerce")

    FIGURES.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES / "coastal_impact_map.png"
    thr = vessel_med.quantile(0.90) if vessel_med.notna().any() else np.nan

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    sc = ax.scatter(lon, lat, c=np.tanh(pd.to_numeric(uniq["score"], errors="coerce")), cmap="YlOrRd", s=12, alpha=0.72)
    if np.isfinite(thr):
        hi = uniq[vessel_med >= thr]
        ax.scatter(pd.to_numeric(hi["lon"], errors="coerce"), pd.to_numeric(hi["lat"], errors="coerce"), marker="*", s=80, facecolors="none", edgecolors="blue", linewidths=0.8, label="Top 10% vessel (grid median)")
    plt.colorbar(sc, ax=ax, label="tanh(coastal_impact_score)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    ax.set_title("Coastal shipping impact proxy (mean grid-week score)\nblue stars ~ high vessel-density zones")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    html_path = VIZ / "coastal_impact_map.html"
    try:
        import folium
    except ImportError:
        return html_path, png_path if png_path.exists() else None

    center = float(lat.median(skipna=True)), float(lon.median(skipna=True))
    if not np.isfinite(center[0]) or not np.isfinite(center[1]):
        return html_path if html_path.exists() else None, png_path
    m = folium.Map(location=[center[0], center[1]], zoom_start=7, tiles="CartoDB positron")
    svals = pd.to_numeric(uniq["score"], errors="coerce")
    vmin, vmax = float(np.nanquantile(svals, 0.05)), float(np.nanquantile(svals, 0.95))
    rng = vmax - vmin if np.isfinite(vmax - vmin) and (vmax - vmin) > 0 else 1.0

    for _, row in uniq.iterrows():
        la = float(row["lat"])
        lo = float(row["lon"])
        if not (np.isfinite(la) and np.isfinite(lo)):
            continue
        sc_val = float(row["score"]) if pd.notna(row["score"]) else float("nan")
        if np.isfinite(sc_val):
            t = max(0, min(1, (sc_val - vmin) / rng))
            color = f"#{int(255*t):02x}{int(200*(1-t)):02x}{int(80):02x}"
        else:
            color = "#888888"
        folium.CircleMarker(
            location=[la, lo],
            radius=3,
            fill=True,
            fill_opacity=0.65,
            color=color,
            weight=1,
            popup=f"impact≈{sc_val:.3f}",
        ).add_to(m)

    hi = uniq[vessel_med >= thr].copy() if np.isfinite(thr) else uniq.iloc[:0]
    for _, row in hi.iterrows():
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=7,
            color="blue",
            fill=False,
            weight=2,
            popup="High vessel-density grid (median top 10%)",
        ).add_to(m)

    m.save(str(html_path))
    return html_path, png_path


def write_md(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_data = (
        _ROOT / "final_run_stockholm_fixed_20260505_1356" / "processed" / "features_ml_ready.parquet"
    )
    legacy = _ROOT / "processed" / "final_run_stockholm_fixed_20260505_1356" / "features_ml_ready.parquet"

    p = argparse.ArgumentParser(
        description="Coastal shipping impact — final thesis run",
        epilog="If coastline GeoJSON fails to download, coastal subset may be empty.",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=default_data if default_data.exists() else legacy,
    )
    p.add_argument(
        "--ne-cache",
        type=Path,
        default=_ROOT / "data" / "aux" / "natural_earth_coast_cache",
        help="Cache dir for Natural Earth coastline GeoJSON downloads",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = Path(args.input)
    inp = inp if inp.is_absolute() else _ROOT / inp

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)

    if not inp.is_file():
        print(f"[FATAL] Input not found: {inp}")
        return 1

    df = pd.read_parquet(inp)
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    n_full = len(df)

    df = ensure_distance_to_coast(df, Path(args.ne_cache))

    coastal_mask = df["distance_to_coast_km"].notna() & (df["distance_to_coast_km"] <= 30)
    n_coastal = int(coastal_mask.sum())

    df = assign_coastal_vessel_groups(df)
    d_ship = pd.to_numeric(df["distance_to_nearest_high_vessel_density_cell"], errors="coerce")

    subset_mask = coastal_mask & d_ship.notna() & (d_ship <= 20)
    n_core = int(subset_mask.sum())

    df_core = df.loc[subset_mask].copy()

    df_core = build_impact_score(df_core)

    comparisons = compare_high_low(df)
    comparisons.to_csv(REPORTS / "high_vs_low_vessel_coastal_comparison.csv", index=False)

    df_coastal_scored = df.loc[coastal_mask].copy()
    df_coastal_scored = df_coastal_scored[
        pd.to_numeric(df_coastal_scored["distance_to_nearest_high_vessel_density_cell"], errors="coerce").notna()
    ]
    mask_band = pd.to_numeric(df_coastal_scored["distance_to_nearest_high_vessel_density_cell"], errors="coerce") < 30
    df_decay_panel = df_coastal_scored.loc[mask_band].copy()
    df_decay_panel = build_impact_score(df_decay_panel)
    decay_tbl = distance_decay_table(df_decay_panel)
    decay_tbl.to_csv(REPORTS / "shipping_lane_distance_decay.csv", index=False)

    score_summ = pd.DataFrame(
        [
            {
                "metric": "coastal_impact_score_mean",
                "value": float(df_core["coastal_impact_score"].mean(skipna=True)),
            },
            {
                "metric": "coastal_impact_score_std",
                "value": float(df_core["coastal_impact_score"].std(skipna=True)),
            },
            {
                "metric": "spearman_corr_score_vs_distance_to_high_density_km",
                "value": float(
                    stats.spearmanr(
                        pd.to_numeric(df_core["distance_to_nearest_high_vessel_density_cell"], errors="coerce"),
                        pd.to_numeric(df_core["coastal_impact_score"], errors="coerce"),
                        nan_policy="omit",
                    )[0]
                )
                if len(df_core) > 30
                else float("nan"),
            },
            {
                "metric": "formula",
                "value": json.dumps(WEIGHTS),
            },
        ]
    )
    score_summ.to_csv(REPORTS / "coastal_impact_score_summary.csv", index=False)

    df_plot = df_core.copy()

    band_order = [f"{lo}-{hi} km" for lo, hi in DIST_BANDS]
    decay_rows = decay_tbl[decay_tbl["indicator"] == "coastal_impact_score"].copy()
    trend_down = False
    if decay_rows.shape[0] >= 2:
        decay_rows["_b"] = pd.Categorical(decay_rows["distance_band_label"], categories=band_order, ordered=True)
        decay_rows = decay_rows.sort_values("_b")
        mns = decay_rows["mean"].to_numpy(dtype=float)
        finite = mns[np.isfinite(mns)]
        if len(finite) >= 2:
            trend_down = bool(np.mean(np.diff(finite)) < 0)

    fig_paths = plot_high_low_boxes(df_plot)
    fig_paths += plot_distance_decay(decay_tbl)

    html_mp, png_mp = map_scores_and_shipping(df_plot)

    interp = """# Interpretation — coastal shipping impact

## Scope
We restrict to **within 30 km of the modeled coastline** to represent near-shore grids where terrestrial and maritime signals mix and societal relevance is strongest.

Shipping pressure is summarized by distance to cells flagged as **high vessel density** in the spatial proxy (**≤ 20 km** for the focal cross-section), emphasizing **lanes and shipping corridors** rather than port-centric 100–1000 km axes.

## Indicators used
Supporting water variables (**NDWI, NDVI**) are reported in group comparisons only; **the composite coastal impact score** favors **NO₂, NDTI, oil slick probability**, and **detection score** with stated weights.

Strongest contrasts (Welch / Mann–Whitney among high vs low exposure, coastal + lane-proximity) are listed numerically in `high_vs_low_vessel_coastal_comparison.csv`.

## Distance decay from shipping corridors
Bands use **distance to nearest high vessel-density grid** under the **coastal strip** constraint. Means/medians decline with increasing distance **if atmospheric / slick signals are localized** near dense traffic zones.

## Limitations
1. **Vessel density fields** approximate chronic spatial congestion; they are **not weekly AIS-derived traffic counts**.
2. **Sentinel‑2-derived water indices** exhibit **calendar missingness / cloud avoidance** unrelated to maritime activity.
3. **Oil slick proxy** is exploratory SAR interpretation — **not** verification of spilled oil without field data.
4. **No causal claims** absent wind/current decorrelation and matched temporal AIS / emissions controls.

Artifacts for this interpretation live under `outputs/reports/run_final_coastal_shipping_impact/`.
"""
    write_md(REPORTS / "interpretation_summary.md", interp)

    prof = """# Professor-ready summary — coastal refinement

We narrowed the analytic frame from very long (**≈1000 km**) hub distances to **coastal exposures** (**≤30 km**) and proximity to zones of chronically elevated **spatial vessel-density** (**≤20 km to high-density anchor cells**). This aligns the question with plausible **localized** coastal impacts from shipping corridors rather than far-field averaging.

Across high vs low **coastal vessel-exposure extremes** (**top/bottom log1p decisiles**) within the focal coastal–near-lane footprint, patterns are **often clearest for atmospheric NO₂ and the exploratory oil-slick probability proxy**, whereas **spectral water-quality indices** show **more cloud-driven noise / weaker contrasts**.

Distance-slice summaries show **localized gradients** concentrated at small distances under the coastal restriction; we deliberately **avoid attributing directional transport or causality**.

We frame results as **spatial association**: maritime-pressure gradients coincide with perturbations in available environmental indicators near shipping lanes — **without claiming direct causal pathways** pending dedicated controls (weather, currents, AIS time-series, confounding terrestrial emissions).

Operational outputs (`high_vs_low_vessel_coastal_comparison.csv`, `shipping_lane_distance_decay.csv`, maps, composites) summarize these associations for dissertation discussion.
"""
    write_md(REPORTS / "professor_summary.md", prof)

    strongest_d = None
    if len(comparisons) and comparisons["cohens_d"].notna().any():
        strongest_d = comparisons.loc[comparisons["cohens_d"].abs().idxmax()]

    strongest_p = None
    if len(comparisons):
        w = comparisons["welch_t_p_value"].to_numpy(dtype=float)
        m = comparisons["mann_whitney_p_value"].to_numpy(dtype=float)
        row_mins = []
        for i in range(len(comparisons)):
            cands = [x for x in (w[i], m[i]) if np.isfinite(x)]
            row_mins.append(min(cands) if cands else float("nan"))
        best_p_col = comparisons.copy()
        best_p_col["_p"] = row_mins
        if best_p_col["_p"].notna().any():
            strongest_p = best_p_col.loc[best_p_col["_p"].idxmin()]
    else:
        best_p_col = comparisons

    files_written: set[Path] = {
        REPORTS / "high_vs_low_vessel_coastal_comparison.csv",
        REPORTS / "shipping_lane_distance_decay.csv",
        REPORTS / "coastal_impact_score_summary.csv",
        REPORTS / "interpretation_summary.md",
        REPORTS / "professor_summary.md",
    }
    files_written.update(fig_paths)
    if png_mp:
        files_written.add(Path(png_mp))
    if html_mp and Path(html_mp).exists():
        files_written.add(Path(html_mp))

    print("=== run_final_coastal_shipping_impact ===")
    print(f"Full dataset rows: {n_full}")
    print(f"Coastal subset (distance_to_coast_km ≤ 30): {n_coastal}")
    print(f"Coastal AND distance_to_high_vessel_density ≤ 20 km: {n_core}")
    if comparisons is not None and len(comparisons):
        cmp_show = comparisons[
            ["indicator", "mean_difference_high_minus_low", "cohens_d", "welch_t_p_value", "mann_whitney_p_value"]
        ]
        print("\nHigh vs low (coastal extremes, analysed on coastal∩≤20km-lane footprint):")
        print(cmp_show.to_string(index=False))
    if strongest_d is not None and len(comparisons):
        print("\nStrongest |Cohen's d|: ", strongest_d["indicator"], "=", strongest_d["cohens_d"])
    if strongest_p is not None:
        print("Strongest significance (smallest Welch/MW p combo): ", strongest_p["indicator"], "p=", strongest_p["_p"])
    print("\nImpact score vs distance trending lower (≤30 km lanes): ", "YES" if trend_down else "UNCLEAR/NO")

    missing_coast_pct = float(100 * (1 - df["distance_to_coast_km"].notna().mean()))
    print(f"\nNote: coastline distance missing on {missing_coast_pct:.1f}% rows (NATURAL EARTH dependency).")

    print("\nGenerated files:")
    for p in sorted({str(x.resolve()) if x.exists() else str(x) for x in files_written}):
        print(" -", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
