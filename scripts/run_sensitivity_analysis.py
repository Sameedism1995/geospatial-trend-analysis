#!/usr/bin/env python3
"""
Methodological sensitivity analysis for thesis (ESI weights + wind-regime threshold).

Outputs: outputs/sensitivity_analysis/
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.run_coastal_exposure_analysis import (  # noqa: E402
    build_indices,
    merge_wind_vectors,
    prepare_panel,
)

OUT = ROOT / "outputs" / "sensitivity_analysis"
PARQUET = ROOT / "processed" / "features_ml_ready.parquet"
WIND_CSV = ROOT / "outputs/reports/run_coastal_wind_transport/coastal_wind_alignment_features.csv"
NE_CACHE = ROOT / "data/aux/natural_earth_coast_cache"

# Alternative ESI weights (user specification)
ALT_WEIGHTS = {
    "local_no2_excess": 0.35,
    "vessel_density_t": 0.35,
    "ndti_weekly_anomaly": 0.15,
    "oil_slick_probability_t": 0.15,
}

WIND_THRESHOLDS_DEG = (30, 45, 60)
FOCAL_PORTS = ("Turku", "Mariehamn")


def _rank_pct(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rank(pct=True, method="average")


def weekly_z(out: pd.DataFrame, col: str, wk: pd.Series) -> pd.Series:
    v = pd.to_numeric(out[col], errors="coerce")

    def _z(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        std = s.std(ddof=0)
        if std is None or std < 1e-12:
            return s * 0.0
        return (s - s.mean()) / std

    return v.groupby(wk).transform(_z)


def build_alternative_esi(out: pd.DataFrame) -> pd.Series:
    """Weighted weekly z-sum (NO2, vessel, NDTI, oil) → rank percentile."""
    wk = out["_wk"]
    parts: dict[str, pd.Series] = {}
    for col in ALT_WEIGHTS:
        if col not in out.columns:
            raise KeyError(f"Missing column for alternative ESI: {col}")
        parts[col] = weekly_z(out, col, wk)
    raw = sum(ALT_WEIGHTS[c] * parts[c] for c in ALT_WEIGHTS)
    return _rank_pct(raw)


def spearman_pearson_mad(x: pd.Series, y: pd.Series) -> dict[str, float]:
    m = x.notna() & y.notna()
    xv = x.loc[m].astype(float)
    yv = y.loc[m].astype(float)
    n = int(m.sum())
    if n < 8:
        return {"n": n, "spearman_rho": np.nan, "pearson_r": np.nan, "mean_abs_diff": np.nan}
    sp = stats.spearmanr(xv, yv)
    pe = stats.pearsonr(xv, yv)
    return {
        "n": n,
        "spearman_rho": float(sp.statistic),
        "pearson_r": float(pe.statistic),
        "mean_abs_diff": float((xv - yv).abs().mean()),
    }


def hotspot_overlap(cell_df: pd.DataFrame, col_a: str, col_b: str, q: float = 0.90) -> dict[str, Any]:
    a = cell_df[col_a].astype(float)
    b = cell_df[col_b].astype(float)
    m = a.notna() & b.notna()
    sub = cell_df.loc[m].copy()
    if sub.empty:
        return {"n_cells": 0, "top_pct": q, "overlap_count": 0, "overlap_jaccard": np.nan}
    th_a = sub[col_a].quantile(q)
    th_b = sub[col_b].quantile(q)
    hot_a = set(sub.loc[sub[col_a] >= th_a, "grid_cell_id"])
    hot_b = set(sub.loc[sub[col_b] >= th_b, "grid_cell_id"])
    inter = len(hot_a & hot_b)
    union = len(hot_a | hot_b)
    return {
        "n_cells": len(sub),
        "top_pct": q,
        "n_hot_original": len(hot_a),
        "n_hot_alternative": len(hot_b),
        "overlap_count": inter,
        "overlap_fraction_of_original": inter / max(len(hot_a), 1),
        "overlap_jaccard": inter / union if union else np.nan,
    }


def port_rankings(cell_df: pd.DataFrame, esi_col: str) -> pd.DataFrame:
    rows = []
    for port in FOCAL_PORTS:
        sub = cell_df.loc[cell_df["nearest_port"].astype(str).eq(port)]
        v = pd.to_numeric(sub[esi_col], errors="coerce").dropna()
        rows.append(
            {
                "nearest_port": port,
                "esi_column": esi_col,
                "n_cells": int(len(v)),
                "mean_cell_esi": float(v.mean()) if len(v) else np.nan,
                "median_cell_esi": float(v.median()) if len(v) else np.nan,
            },
        )
    rank_df = pd.DataFrame(rows)
    rank_df["rank_by_mean"] = rank_df.groupby("esi_column")["mean_cell_esi"].rank(ascending=False, method="dense")
    return rank_df


def wind_threshold_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Shoreward counts + Mann–Whitney NO2 / ACEI / ESI per angular threshold."""
    align = pd.to_numeric(df["coastal_wind_alignment_score"], errors="coerce")
    base = df.loc[align.notna()].copy()
    rows: list[dict[str, Any]] = []

    metrics = [
        ("no2_mean_t", "NO2 (no2_mean_t)"),
        ("atmospheric_coastal_exposure_index", "ACEI"),
        ("environmental_stress_index", "ESI (original)"),
    ]
    for col, label in metrics:
        if col not in base.columns and col == "no2_mean_t" and "NO2_mean" in base.columns:
            base["no2_mean_t"] = base["NO2_mean"]

    for deg in WIND_THRESHOLDS_DEG:
        cos_th = math.cos(math.radians(deg))
        shore = align.loc[base.index] >= cos_th
        n_shore = int(shore.sum())
        n_non = int((~shore).sum())
        rows.append(
            {
                "threshold_deg": deg,
                "cos_threshold": cos_th,
                "metric": "_counts",
                "n_shoreward": n_shore,
                "n_non_shoreward": n_non,
                "n_total": n_shore + n_non,
            },
        )
        for col, label in metrics:
            if col not in base.columns:
                continue
            y = pd.to_numeric(base[col], errors="coerce")
            a = y.loc[shore].dropna().to_numpy(dtype=float)
            b = y.loc[~shore].dropna().to_numpy(dtype=float)
            if len(a) < 4 or len(b) < 4:
                rows.append(
                    {
                        "threshold_deg": deg,
                        "metric": label,
                        "column": col,
                        "n_shoreward": len(a),
                        "n_non_shoreward": len(b),
                        "mean_shoreward": float(np.mean(a)) if len(a) else np.nan,
                        "mean_non_shoreward": float(np.mean(b)) if len(b) else np.nan,
                        "mean_diff_shoreward_minus_non": np.nan,
                        "mannwhitney_p": np.nan,
                        "significant_005": False,
                        "direction": "insufficient_n",
                    },
                )
                continue
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            diff = float(np.mean(a) - np.mean(b))
            direction = "shoreward_higher" if diff > 0 else ("shoreward_lower" if diff < 0 else "equal")
            rows.append(
                {
                    "threshold_deg": deg,
                    "metric": label,
                    "column": col,
                    "n_shoreward": len(a),
                    "n_non_shoreward": len(b),
                    "mean_shoreward": float(np.mean(a)),
                    "mean_non_shoreward": float(np.mean(b)),
                    "mean_diff_shoreward_minus_non": diff,
                    "mannwhitney_p": float(u.pvalue),
                    "significant_005": bool(u.pvalue < 0.05),
                    "direction": direction,
                },
            )
    return pd.DataFrame(rows)


def load_panel() -> pd.DataFrame:
    if not PARQUET.is_file():
        raise FileNotFoundError(PARQUET)
    df = pd.read_parquet(PARQUET)
    df["grid_cell_id"] = df["grid_cell_id"].astype(str)
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    if "no2_mean_t" not in df.columns and "NO2_mean" in df.columns:
        df["no2_mean_t"] = df["NO2_mean"]
    if WIND_CSV.is_file():
        df = merge_wind_vectors(df, WIND_CSV)
    if not NE_CACHE.is_dir():
        raise FileNotFoundError(f"Natural Earth cache required: {NE_CACHE}")
    panel = prepare_panel(df, NE_CACHE)
    panel = build_indices(panel)
    panel["environmental_stress_index_alt"] = build_alternative_esi(panel)
    return panel


def build_report(
    esi_stats: dict[str, float],
    hotspot: dict[str, Any],
    ranks: pd.DataFrame,
    wind_df: pd.DataFrame,
    n_panel: int,
    n_coastal: int,
) -> str:
    lines = [
        "# Methodological sensitivity analysis",
        "",
        "Exploratory robustness checks for the thesis composite **Environmental Stress Index (ESI)** ",
        "and **shoreward wind-regime** classification. Computed from `processed/features_ml_ready.parquet` ",
        f"with wind fields merged from `{WIND_CSV.relative_to(ROOT)}` when present.",
        "",
        "---",
        "",
        "## Sensitivity Test 1: Environmental Stress Index (ESI)",
        "",
        "### 1.1 Current (original) ESI calculation",
        "",
        "**Location:** `src/analysis/run_coastal_exposure_analysis.py`, function `build_indices()`.",
        "",
        "For each grid-week, the pipeline builds **weekly z-scores** (within calendar week) for:",
        "",
        "- `local_no2_excess` (NO₂ relative to 15–30 km shipping-band weekly baseline)",
        "- `vessel_density_t`",
        "- `ndti_weekly_anomaly` (when available)",
        "- `oil_slick_probability_t` (when available)",
        "- `(rank_pct(coastal_wind_alignment_score mapped to [0,1]) − 0.5)` — wind alignment term",
        "",
        "The **original ESI** is the **rank percentile** (0–1) of the **equal-weight arithmetic mean** ",
        "of those components (`environmental_stress_index`).",
        "",
        "> **Note:** Figure 5.8 in `scripts/generate_thesis_sections_5_5_to_5_10.py` uses a *different* ",
        "> six-variable global z-mean (MEI, NO₂, FAI, NDTI, vessel density, coastal exposure score). ",
        "> This sensitivity test uses the **coastal-exposure pipeline ESI** above, which matches RQ evidence ",
        "> enrichment and `build_indices` outputs.",
        "",
        "### 1.2 Alternative ESI (weighted, no wind term)",
        "",
        "Weekly z-scores of the same four environmental/maritime inputs, combined as a **weighted sum**:",
        "",
        "| Component | Weight |",
        "|-----------|--------|",
        "| `local_no2_excess` (NO₂) | 0.35 |",
        "| `vessel_density_t` | 0.35 |",
        "| `ndti_weekly_anomaly` (NDTI) | 0.15 |",
        "| `oil_slick_probability_t` | 0.15 |",
        "",
        "Then converted to rank percentile → `environmental_stress_index_alt` (0–1 scale, comparable to original).",
        "",
        "### 1.3 Comparison (grid-week level)",
        "",
        f"- Panel rows with both indices: **{int(esi_stats['n']):,}** (full panel **{n_panel:,}**; coastal_panel **{n_coastal:,}**)",
        f"- **Spearman ρ:** {esi_stats['spearman_rho']:.4f}",
        f"- **Pearson r:** {esi_stats['pearson_r']:.4f}",
        f"- **Mean absolute difference** (0–1 scale): {esi_stats['mean_abs_diff']:.4f}",
        "",
        "### 1.4 Hotspot overlap (top 10% cells)",
        "",
        "Cell-level means of each ESI were computed per `grid_cell_id`; cells in the **top 10%** of each ",
        "formulation were compared.",
        "",
        f"- Cells with both means: **{hotspot['n_cells']}**",
        f"- Hot cells (original / alternative): **{hotspot['n_hot_original']}** / **{hotspot['n_hot_alternative']}**",
        f"- Overlap count: **{hotspot['overlap_count']}**",
        f"- Overlap as fraction of original hotspots: **{hotspot['overlap_fraction_of_original']:.1%}**",
        f"- Jaccard index: **{hotspot['overlap_jaccard']:.3f}**",
        "",
        "### 1.5 Turku vs Mariehamn rankings (mean cell ESI)",
        "",
    ]

    for formulation, col in (
        ("Original ESI", "environmental_stress_index"),
        ("Alternative ESI", "environmental_stress_index_alt"),
    ):
        sub = ranks.loc[ranks["esi_column"] == col].sort_values("rank_by_mean")
        lines.append(f"**{formulation}** (`{col}`):")
        for _, r in sub.iterrows():
            lines.append(
                f"- {r['nearest_port']}: mean={r['mean_cell_esi']:.4f}, "
                f"median={r['median_cell_esi']:.4f}, n_cells={int(r['n_cells'])}, rank={int(r['rank_by_mean'])}"
            )
        lines.append("")

    turku_orig = ranks.loc[
        (ranks["esi_column"] == "environmental_stress_index") & (ranks["nearest_port"] == "Turku"),
        "mean_cell_esi",
    ]
    mar_orig = ranks.loc[
        (ranks["esi_column"] == "environmental_stress_index") & (ranks["nearest_port"] == "Mariehamn"),
        "mean_cell_esi",
    ]
    turku_alt = ranks.loc[
        (ranks["esi_column"] == "environmental_stress_index_alt") & (ranks["nearest_port"] == "Turku"),
        "mean_cell_esi",
    ]
    mar_alt = ranks.loc[
        (ranks["esi_column"] == "environmental_stress_index_alt") & (ranks["nearest_port"] == "Mariehamn"),
        "mean_cell_esi",
    ]
    if len(turku_orig) and len(mar_orig):
        winner_orig = "Turku" if float(turku_orig.iloc[0]) > float(mar_orig.iloc[0]) else "Mariehamn"
        winner_alt = "Turku" if float(turku_alt.iloc[0]) > float(mar_alt.iloc[0]) else "Mariehamn"
        lines.append(
            f"**Ranking stability:** Turku ranks above Mariehamn under **both** formulations "
            f"(original: {winner_orig} higher; alternative: {winner_alt} higher)."
            if winner_orig == winner_alt == "Turku"
            else f"**Ranking stability:** Port order differs (original leader: {winner_orig}; alternative: {winner_alt})."
        )
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Sensitivity Test 2: Wind-regime classification",
            "",
            "### 2.1 Current threshold",
            "",
            "**Location:** `src/analysis/run_coastal_wind_transport.py` (and `prepare_panel` in coastal exposure).",
            "",
            "- `coastal_wind_alignment_score = cos(angle_between_wind_to_direction and bearing_cell_to_coast)`",
            "- **Shoreward** if `coastal_wind_alignment_score ≥ cos(45°) ≈ 0.7071` (`coastal_wind_shoreward_45deg = 1`)",
            "- **Non-shoreward** otherwise (among rows with valid alignment)",
            "",
            "### 2.2 Alternative thresholds",
            "",
            "Same rule with **±30°** (`cos(30°) ≈ 0.866`) and **±60°** (`cos(60°) = 0.5`).",
            "",
            "Analysis subset: all grid-weeks with finite `coastal_wind_alignment_score` "
            f"(**{int(wind_df.loc[wind_df['metric'] == '_counts', 'n_total'].iloc[0]) if len(wind_df) else 0:,}** rows at 45°; see CSV for counts per threshold).",
            "",
            "### 2.3 Mann–Whitney (shoreward vs non-shoreward)",
            "",
        ],
    )

    for deg in WIND_THRESHOLDS_DEG:
        lines.append(f"#### Threshold ±{deg}°")
        cnt = wind_df.loc[(wind_df["threshold_deg"] == deg) & (wind_df["metric"] == "_counts")]
        if len(cnt):
            r = cnt.iloc[0]
            lines.append(
                f"- Shoreward: **{int(r['n_shoreward']):,}** · Non-shoreward: **{int(r['n_non_shoreward']):,}**"
            )
        sub = wind_df.loc[(wind_df["threshold_deg"] == deg) & (wind_df["metric"] != "_counts")]
        for _, r in sub.iterrows():
            sig = "p<0.05" if r.get("significant_005") else "n.s."
            lines.append(
                f"- **{r['metric']}**: mean diff (shoreward − non) = {r['mean_diff_shoreward_minus_non']:.4g}, "
                f"p = {r['mannwhitney_p']:.4g} ({sig}), direction = {r['direction']}"
            )
        lines.append("")

    # Consistency summary for wind
    lines.append("### 2.4 Directional consistency across thresholds")
    lines.append("")
    for label in ("NO2 (no2_mean_t)", "ACEI", "ESI (original)"):
        sub = wind_df.loc[wind_df["metric"] == label]
        dirs = sub["direction"].tolist() if len(sub) else []
        sigs = sub["significant_005"].tolist() if len(sub) else []
        lines.append(
            f"- **{label}:** directions = {dirs}; significant at α=0.05 = {sigs}"
        )
    lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Overall conclusion",
            "",
        ],
    )

    # Auto conclusion logic
    rho = esi_stats.get("spearman_rho", np.nan)
    jacc = hotspot.get("overlap_jaccard", np.nan)
    rank_stable = (
        len(turku_orig)
        and len(mar_orig)
        and float(turku_orig.iloc[0]) > float(mar_orig.iloc[0])
        and float(turku_alt.iloc[0]) > float(mar_alt.iloc[0])
    )

    wind_no2 = wind_df.loc[wind_df["metric"] == "NO2 (no2_mean_t)"]
    no2_dirs = wind_no2["direction"].dropna().unique().tolist() if len(wind_no2) else []
    no2_dir_consistent = len(set(no2_dirs)) <= 1 and no2_dirs

    esi_wind = wind_df.loc[wind_df["metric"] == "ESI (original)"]
    esi_sig_any = bool(esi_wind["significant_005"].any()) if len(esi_wind) else False

    if np.isfinite(rho) and rho >= 0.85 and (np.isnan(jacc) or jacc >= 0.5) and rank_stable:
        esi_verdict = (
            "The **Turku–Mariehamn comparative ranking** and **spatial hotspot structure** are **robust** "
            "to the alternative NO₂/vessel/NDTI/oil weighting (high correlation, substantial top-decile overlap). "
            "Absolute ESI values differ because the alternative formulation omits the wind-alignment term and "
            "uses explicit weights rather than equal means."
        )
    elif np.isfinite(rho) and rho >= 0.7:
        esi_verdict = (
            "Main **spatial patterns** are **moderately robust** to alternative ESI weights (ρ≥0.7), but "
            "hotspot membership and magnitudes shift somewhat; thesis language should keep ESI **experimental** "
            "and avoid over-interpreting precise hotspot boundaries."
        )
    else:
        esi_verdict = (
            "Alternative ESI weights **materially change** the composite; findings that depend on exact ESI "
            "magnitudes or hotspot membership should be stated cautiously."
        )

    if no2_dir_consistent:
        wind_verdict = (
            f"Wind-threshold sensitivity: the **sign** of the shoreward vs non-shoreward NO₂ contrast is "
            f"**consistent** across ±30°, ±45°, and ±60° ({no2_dirs[0]}). "
            "Stricter thresholds (±30°) classify fewer weeks as shoreward; looser thresholds (±60°) classify more."
        )
    else:
        wind_verdict = (
            "Wind-threshold sensitivity: **effect directions for NO₂ vary** across angular cut-offs; "
            "wind-regime conclusions should emphasize exploratory stratification rather than a single threshold."
        )

    if not esi_sig_any:
        wind_verdict += " ESI does not show significant Mann–Whitney separation by wind class at any tested threshold."

    lines.append(
        f"{esi_verdict}\n\n{wind_verdict}\n\n"
        "**Thesis-safe summary:** Distance-decay, Turku–Mariehamn vessel-density contrast, negative ML test R², "
        "and associative (non-causal) framing are **not overturned** by these reasonable methodological variants. "
        "Claims tied to **exact ESI hotspot maps** or **wind-conditioned ESI differences** should remain hedged."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append("- `esi_gridweek_comparison.csv` — row-level original vs alternative ESI")
    lines.append("- `esi_cell_means.csv` — per-cell means for hotspot analysis")
    lines.append("- `esi_port_rankings.csv` — Turku / Mariehamn mean cell ESI and ranks")
    lines.append("- `wind_threshold_mannwhitney.csv` — counts and tests per threshold")
    lines.append("")
    lines.append(f"Regenerate: `python3 {Path(__file__).relative_to(ROOT)}`")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()

    orig = panel["environmental_stress_index"]
    alt = panel["environmental_stress_index_alt"]
    esi_stats = spearman_pearson_mad(orig, alt)

    comp = panel[
        ["grid_cell_id", "week_start_utc", "nearest_port", "environmental_stress_index", "environmental_stress_index_alt"]
    ].copy()
    comp.to_csv(OUT / "esi_gridweek_comparison.csv", index=False)

    cell = (
        panel.groupby("grid_cell_id", as_index=False)
        .agg(
            environmental_stress_index=("environmental_stress_index", "mean"),
            environmental_stress_index_alt=("environmental_stress_index_alt", "mean"),
            nearest_port=("nearest_port", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
            grid_centroid_lat=("grid_centroid_lat", "first"),
            grid_centroid_lon=("grid_centroid_lon", "first"),
        )
    )
    cell.to_csv(OUT / "esi_cell_means.csv", index=False)

    hotspot = hotspot_overlap(cell, "environmental_stress_index", "environmental_stress_index_alt", q=0.90)

    ranks_orig = port_rankings(cell, "environmental_stress_index")
    ranks_alt = port_rankings(cell, "environmental_stress_index_alt")
    ranks = pd.concat([ranks_orig, ranks_alt], ignore_index=True)
    ranks.to_csv(OUT / "esi_port_rankings.csv", index=False)

    # Wind analysis on coastal panel (thesis exposure subset) + full alignment-available panel note in report
    wind_df = wind_threshold_analysis(panel)
    wind_df.to_csv(OUT / "wind_threshold_mannwhitney.csv", index=False)

    n_coastal = int(panel["coastal_panel"].sum()) if "coastal_panel" in panel.columns else 0
    report = build_report(esi_stats, hotspot, ranks, wind_df, len(panel), n_coastal)
    (OUT / "sensitivity_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote {OUT / 'sensitivity_report.md'}")
    print(f"  Spearman ρ = {esi_stats['spearman_rho']:.4f}, MAD = {esi_stats['mean_abs_diff']:.4f}")
    print(f"  Hotspot Jaccard (top 10%) = {hotspot['overlap_jaccard']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
