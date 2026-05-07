#!/usr/bin/env python3
"""
Indicator participation in coastal exposure (association / structuring only — not causal).

Quantifies normalized indicator behaviour by shipping-distance band × wind regime, with
figures and optional permutation-importance (relative predictive participation).

  python3 src/analysis/run_indicator_participation.py

Reads `features_ml_ready_coastal_wind.parquet` plus optional wind-vector merge CSV.
"""

from __future__ import annotations

import argparse
import math
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

from analysis.run_coastal_exposure_analysis import BANDS as DIST_BANDS
from analysis.run_coastal_exposure_analysis import boot_mean_ci, build_indices
from analysis.run_coastal_exposure_analysis import merge_wind_vectors, prepare_panel

REPORTS = _ROOT / "outputs" / "reports"
FIGURES = _ROOT / "outputs" / "figures" / "indicator_participation"
DEFAULT_INPUT = _ROOT / "outputs" / "processed" / "features_ml_ready_coastal_wind.parquet"


def resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# (output_key, display_label, pandas column candidates...)
INDICATOR_SPECS: list[tuple[str, str, tuple[str, ...]]] = [
    ("vessel_density_t", "Vessel density (t)", ("vessel_density_t",)),
    ("local_no2_excess", "Local NO2 excess", ("local_no2_excess",)),
    ("coastal_wind_alignment", "Coastal wind alignment", ("coastal_wind_alignment_score",)),
    ("pollution_transport_alignment", "Pollution transport alignment", ("pollution_transport_wind_alignment_score",)),
    ("oil_slick_proxy", "Oil slick proxy", ("oil_slick_probability_t",)),
    (
        "sentinel_ndti",
        "Sentinel NDTI (mean)",
        ("sentinel_ndti_mean_t", "ndti_mean_t", "ndti_mean"),
    ),
    (
        "sentinel_ndwi",
        "Sentinel NDWI (mean)",
        ("sentinel_ndwi_mean_t", "ndwi_mean_t", "ndwi_mean"),
    ),
    (
        "sentinel_ndvi",
        "Sentinel NDVI (mean)",
        ("sentinel_ndvi_mean_t", "ndvi_mean_t", "ndvi_mean"),
    ),
    ("maritime_exposure_index", "Maritime exposure index", ("maritime_exposure_index",)),
    ("atmospheric_coastal_exposure_index", "Atmospheric coastal exposure index", ("atmospheric_coastal_exposure_index",)),
    ("environmental_stress_index", "Environmental stress index", ("environmental_stress_index",)),
]

# Thesis-facing thematic buckets for stacked composition (mean rank sums within band, then row-normalised).
THEMATIC_COMPOSITION_GROUPS: list[tuple[str, str, tuple[str, ...]]] = [
    ("no2", r"NO$_2$ excess", ("local_no2_excess",)),
    ("vessel", "Vessel density", ("vessel_density_t",)),
    (
        "wind",
        "Wind alignment (coastal + transport)",
        ("coastal_wind_alignment", "pollution_transport_alignment"),
    ),
    ("oil", "Oil slick proxy", ("oil_slick_proxy",)),
    (
        "sentinel_ndx",
        "Sentinel NDTI / NDWI / NDVI",
        ("sentinel_ndti", "sentinel_ndwi", "sentinel_ndvi"),
    ),
    (
        "exposure_indices",
        "Maritime / atmospheric / stress indices",
        (
            "maritime_exposure_index",
            "atmospheric_coastal_exposure_index",
            "environmental_stress_index",
        ),
    ),
]

# FigE node colouring: maritime vs atmospheric coupling vs sentinel surface vs composite stress.
INDICATOR_FAMILY: dict[str, str] = {
    "vessel_density_t": "maritime",
    "oil_slick_proxy": "maritime",
    "maritime_exposure_index": "maritime",
    "local_no2_excess": "atmospheric",
    "coastal_wind_alignment": "atmospheric",
    "pollution_transport_alignment": "atmospheric",
    "atmospheric_coastal_exposure_index": "atmospheric",
    "sentinel_ndti": "sentinel_surface",
    "sentinel_ndwi": "sentinel_surface",
    "sentinel_ndvi": "sentinel_surface",
    "environmental_stress_index": "composite_stress",
}
FAMILY_FACE_COLOR: dict[str, str] = {
    "maritime": "#c6dbef",
    "atmospheric": "#fdd0a2",
    "sentinel_surface": "#c7e9c0",
    "composite_stress": "#dadaeb",
}
FAMILY_EDGE_COLOR: dict[str, str] = {
    "maritime": "#08519c",
    "atmospheric": "#d94801",
    "sentinel_surface": "#238b45",
    "composite_stress": "#6a51a3",
}


def robust_z(series: pd.Series) -> pd.Series:
    """Robust scale: (x-median)/(IQR/1.34896); resistant to outliers."""
    x = pd.to_numeric(series, errors="coerce")
    lo, hi = x.quantile(0.25), x.quantile(0.75)
    iqr = float(hi - lo)
    if not math.isfinite(iqr) or iqr < 1e-12:
        return x * np.nan
    scale = iqr / 1.34896
    z = (x - x.median()) / scale
    return z.clip(-3.5, 3.5)


def rank_pct_coastal(s: pd.Series, mask: pd.Series) -> pd.Series:
    """Percentile rank within coastal panel only (pooled weeks × grids)."""
    x = pd.to_numeric(s, errors="coerce")
    coastal_vals = x.loc[mask].dropna()
    if coastal_vals.empty:
        return pd.Series(np.nan, index=s.index)

    ranks = pd.Series(np.nan, index=s.index)
    order = coastal_vals.rank(pct=True, method="average")
    ranks.loc[order.index] = order
    return ranks


def cliffs_delta_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 4 or len(b) < 4:
        return float("nan")
    rng = np.random.default_rng(7)
    ma, mb = min(len(a), 400), min(len(b), 400)
    if len(a) > ma:
        a = rng.choice(a, ma, replace=False)
    if len(b) > mb:
        b = rng.choice(b, mb, replace=False)
    delta = float((a[:, None] > b[None, :]).mean() - (a[:, None] < b[None, :]).mean())
    return delta


def build_participation_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add rank_participation_* and rz_* columns; return coastal subset + key map."""
    coastal_m = df["coastal_panel"]
    resolved: dict[str, str] = {}
    df = df.copy()
    missing: list[str] = []
    for key, lab, cand in INDICATOR_SPECS:
        col = resolve_column(df, cand)
        if col is None:
            missing.append(key)
            df[f"rank_participation_{key}"] = np.nan
            df[f"rz_participation_{key}"] = np.nan
            resolved[key] = "__missing__"
            continue
        resolved[key] = col
        base = pd.to_numeric(df[col], errors="coerce")
        df[f"_tmp_base_{key}"] = base
        df[f"rank_participation_{key}"] = rank_pct_coastal(base, coastal_m)
        rz = robust_z(base.loc[coastal_m].dropna())
        rz_series = pd.Series(np.nan, index=df.index)
        rz_series.loc[rz.index] = rz.values
        df[f"rz_participation_{key}"] = rz_series
    return df, resolved


def regime_statistics(df: pd.DataFrame) -> pd.DataFrame:
    coastal = df.loc[df["coastal_panel"]].copy()
    rank_cols = [c for c in coastal.columns if c.startswith("rank_participation_")]
    strip = lambda s: s.replace("rank_participation_", "")

    rows: list[dict[str, Any]] = []
    for band in DIST_BANDS:
        for wind_lbl, pred in (
            ("shoreward", lambda d: pd.to_numeric(d["shoreward_binary"], errors="coerce").eq(1)),
            ("non_shoreward", lambda d: pd.to_numeric(d["shoreward_binary"], errors="coerce").eq(0)),
        ):
            slab = coastal.loc[pred(coastal)]
            slab = slab[slab["shipping_distance_band_tight"] == band]
            opp = coastal.loc[~pred(coastal)]
            opp = opp[opp["shipping_distance_band_tight"] == band]

            for rc in rank_cols:
                k = strip(rc)
                v = pd.to_numeric(slab[rc], errors="coerce").dropna().to_numpy(float)
                m, lo, hi = boot_mean_ci(v, n_boot=2000, seed=hash((band, wind_lbl, k)) % 2**31)
                med = float(np.median(v)) if len(v) else np.nan
                vr = float(np.var(v, ddof=1)) if len(v) > 1 else np.nan
                opp_v = pd.to_numeric(opp[rc], errors="coerce").dropna().to_numpy(float)
                cdelta = cliffs_delta_np(v, opp_v) if len(v) >= 8 and len(opp_v) >= 8 else float("nan")
                mw_p = np.nan
                if len(v) >= 8 and len(opp_v) >= 8:
                    mw_p = float(stats.mannwhitneyu(v, opp_v, alternative="two-sided").pvalue)
                rows.append(
                    {
                        "indicator_key": k,
                        "distance_band": band,
                        "wind_regime": wind_lbl,
                        "normalization": "rank_percentile_coastal_panel",
                        "n": int(len(v)),
                        "mean_rank_participation": m,
                        "median_rank_participation": med,
                        "variance_rank_participation": vr,
                        "ci95_boot_low": lo,
                        "ci95_boot_high": hi,
                        "cliffs_delta_vs_opposite_wind_same_band": cdelta,
                        "mannwhitney_p_vs_opposite_wind": mw_p,
                    },
                )

    rz_cols = [c for c in coastal.columns if c.startswith("rz_participation_")]
    strip_rz = lambda s: s.replace("rz_participation_", "")
    for band in DIST_BANDS:
        for wind_lbl, pred in (
            ("shoreward", lambda d: pd.to_numeric(d["shoreward_binary"], errors="coerce").eq(1)),
            ("non_shoreward", lambda d: pd.to_numeric(d["shoreward_binary"], errors="coerce").eq(0)),
        ):
            slab = coastal.loc[pred(coastal)]
            slab = slab[slab["shipping_distance_band_tight"] == band]
            for rz in rz_cols:
                k = strip_rz(rz)
                x = pd.to_numeric(slab[rz], errors="coerce").dropna().to_numpy(float)
                m2, lo2, hi2 = boot_mean_ci(x, n_boot=1500)
                rows.append(
                    {
                        "indicator_key": k,
                        "distance_band": band,
                        "wind_regime": wind_lbl,
                        "normalization": "robust_z_coastal_masked",
                        "n": len(x),
                        "mean_rank_participation": m2,
                        "median_rank_participation": float(np.median(x)) if len(x) else np.nan,
                        "variance_rank_participation": float(np.var(x, ddof=1)) if len(x) > 1 else np.nan,
                        "ci95_boot_low": lo2,
                        "ci95_boot_high": hi2,
                        "cliffs_delta_vs_opposite_wind_same_band": np.nan,
                        "mannwhitney_p_vs_opposite_wind": np.nan,
                    },
                )

    return pd.DataFrame(rows)


def participation_matrix(df: pd.DataFrame, shoreward: bool) -> pd.DataFrame:
    coastal = df.loc[df["coastal_panel"]]
    wx = coastal["shoreward_binary"].eq(1.0 if shoreward else 0.0)
    mats = []
    for band in DIST_BANDS:
        slab = coastal.loc[wx & (coastal["shipping_distance_band_tight"] == band)]
        row = {}
        for key, _, _ in INDICATOR_SPECS:
            c = f"rank_participation_{key}"
            if c not in slab.columns:
                row[key] = np.nan
            else:
                row[key] = float(pd.to_numeric(slab[c], errors="coerce").mean(skipna=True))
        mats.append(row)
    return pd.DataFrame(mats, index=DIST_BANDS)


def plot_heatmaps(df: pd.DataFrame) -> None:
    for shore, title, fname in (
        (True, "Shoreward wind (cos alignment >= cos 45 deg)", "FigA_participation_heatmap_shoreward.png"),
        (False, "Non-shoreward wind", "FigA_participation_heatmap_non_shoreward.png"),
    ):
        mat = participation_matrix(df, shoreward=shore).T
        labels_display = dict((k, lab) for k, lab, _ in INDICATOR_SPECS)
        mat.index = [labels_display.get(i, i) for i in mat.index]
        fig_w = max(10, len(DIST_BANDS) * 2.8)
        fig, ax = plt.subplots(figsize=(fig_w, 9))
        sns.heatmap(
            mat,
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Mean coastal rank percentile (participation scale)"},
        )
        ax.set_title(
            f"Indicator participation by distance regime — {title}\n(rank pooling; structuring lens, not attribution)",
            fontsize=11,
        )
        ax.set_xlabel("Shipping-distance band")
        ax.set_ylabel("Environmental indicators / exposure composites")
        fig.tight_layout()
        fig.savefig(FIGURES / fname, dpi=200)
        plt.close(fig)


def plot_stacked_composition(df: pd.DataFrame) -> None:
    for shore, title, fname in (
        (True, "shoreward", "FigB_stacked_composition_shoreward.png"),
        (False, "non_shoreward", "FigB_stacked_composition_non_shoreward.png"),
    ):
        mat = participation_matrix(df, shoreward=shore)
        prop = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        bottom = np.zeros(len(DIST_BANDS))
        idx = np.arange(len(DIST_BANDS))
        colors = plt.cm.tab20(np.linspace(0, 1, len(INDICATOR_SPECS)))
        for i, spec in enumerate(INDICATOR_SPECS):
            k, lab = spec[0], spec[1]
            vals = prop[k].fillna(0).to_numpy(float)
            ax.bar(idx, vals, bottom=bottom, label=lab[:42], width=0.72, color=colors[i], edgecolor="white", lw=0.4)
            bottom += vals
        ax.set_xticks(idx)
        ax.set_xticklabels(DIST_BANDS, rotation=18)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Exposure composition proxy by band — {title}\n(mean rank shares; participation not causal)",
            fontsize=10,
        )
        ax.set_ylabel("Proportional participation (sum within band)")
        ax.legend(ncol=2, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28))
        fig.tight_layout()
        fig.savefig(FIGURES / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_stacked_composition_thematic(df: pd.DataFrame) -> None:
    """Stacked bars by thesis theme: NO2, vessel, wind, oil, Sentinel triad, composite indices."""
    group_labels = {g[0]: g[1] for g in THEMATIC_COMPOSITION_GROUPS}
    group_keys = [g[0] for g in THEMATIC_COMPOSITION_GROUPS]
    cmap = plt.cm.Set2(np.linspace(0, 1, len(group_keys)))

    for shore, title, fname in (
        (True, "shoreward", "FigB_thematic_stacked_shoreward.png"),
        (False, "non_shoreward", "FigB_thematic_stacked_non_shoreward.png"),
    ):
        mat = participation_matrix(df, shoreward=shore)
        parts: dict[str, pd.Series] = {}
        for gid, _glab, keys in THEMATIC_COMPOSITION_GROUPS:
            sub = mat[[k for k in keys if k in mat.columns]]
            parts[gid] = sub.sum(axis=1, min_count=1) if not sub.empty else pd.Series(0.0, index=mat.index)
        them = pd.DataFrame(parts)
        prop = them.div(them.sum(axis=1).replace(0, np.nan), axis=0)

        fig, ax = plt.subplots(figsize=(10, 5.2))
        bottom = np.zeros(len(DIST_BANDS))
        idx = np.arange(len(DIST_BANDS))
        for i, gid in enumerate(group_keys):
            vals = prop[gid].fillna(0).to_numpy(float)
            ax.bar(
                idx,
                vals,
                bottom=bottom,
                label=group_labels[gid],
                width=0.72,
                color=cmap[i],
                edgecolor="white",
                lw=0.55,
            )
            bottom += vals
        ax.set_xticks(idx)
        ax.set_xticklabels(DIST_BANDS, rotation=18)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"Thematic exposure composition — {title}\n"
            "(shares of summed mean rank participation by band; association / structuring)",
            fontsize=10,
        )
        ax.set_ylabel("Proportional participation (within band)")
        ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        fig.tight_layout()
        fig.savefig(FIGURES / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_wind_comparison_violins(df: pd.DataFrame, top_keys: list[str]) -> None:
    coastal = df.loc[df["coastal_panel"]].copy()
    coastal["_wind"] = np.where(
        coastal["shoreward_binary"].eq(1),
        "shoreward",
        "non-shoreward",
    )
    long_rows = []
    lab_map = dict((spec[0], spec[1]) for spec in INDICATOR_SPECS)
    for k in top_keys:
        c = f"rank_participation_{k}"
        if c not in coastal.columns:
            continue
        sub = coastal[["shipping_distance_band_tight", "_wind", c]].dropna().copy()
        sub["_label"] = lab_map.get(k, k)
        sub.rename(columns={c: "rank_part"}, inplace=True)
        long_rows.append(sub)
    if not long_rows:
        return
    long_df = pd.concat(long_rows, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7), sharey=True)
    for ax, wind in zip(axes, ["shoreward", "non-shoreward"]):
        sl = long_df[long_df["_wind"] == wind]
        sns.violinplot(
            data=sl,
            x="shipping_distance_band_tight",
            y="rank_part",
            hue="_label",
            order=DIST_BANDS,
            ax=ax,
            cut=0,
            inner="quart",
            linewidth=0.6,
        )
        ax.set_title(f"{wind} wind")
        ax.set_xlabel("Distance band")
        ax.set_ylabel("Rank participation [0,1]")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, title="Indicator")
    fig.suptitle("FigC. Wind-regime participation distributions (selected indicators)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "FigC_wind_regime_violin_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distance_curves(df: pd.DataFrame) -> None:
    coastal = df.loc[df["coastal_panel"]]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    xi = np.arange(len(DIST_BANDS))
    for axi, ttl in enumerate(["Shoreward", "Non-shoreward"]):
        ax = axes[axi]
        for spec in INDICATOR_SPECS:
            k = spec[0]
            lab = spec[1][:34]
            c = f"rank_participation_{k}"
            if c not in coastal.columns:
                continue
            line_vals: list[float] = []
            for band in DIST_BANDS:
                pred_sh = coastal["shipping_distance_band_tight"].eq(band) & (
                    coastal["shoreward_binary"].eq(1)
                    if ttl == "Shoreward"
                    else coastal["shoreward_binary"].eq(0)
                )
                line_vals.append(
                    float(pd.to_numeric(coastal.loc[pred_sh, c], errors="coerce").mean(skipna=True)),
                )
            lbl = lab if axi == 0 else None
            ax.plot(xi, line_vals, lw=1.1, marker="o", markersize=3, label=lbl)
        ax.set_title(ttl + " wind")
        ax.set_ylabel("Mean rank participation")
        ax.grid(alpha=0.3)
        ax.set_xticks(xi)
        ax.set_xticklabels(DIST_BANDS, rotation=12)
    axes[-1].set_xlabel("Shipping-distance band")
    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(h0, l0, ncol=4, fontsize=7, bbox_to_anchor=(0.52, -0.02), loc="upper center")
    fig.suptitle(
        "FigD. Participation vs inland distance curve (association / structuring)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIGURES / "FigD_participation_distance_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def correlation_network_plot(df: pd.DataFrame, rho_min: float = 0.12) -> None:
    coastal = df.loc[df["coastal_panel"]]
    cols = [f"rank_participation_{spec[0]}" for spec in INDICATOR_SPECS if f"rank_participation_{spec[0]}" in coastal.columns]
    labs = dict((spec[0], spec[1][:36]) for spec in INDICATOR_SPECS)

    coastal = coastal.copy()
    for c in cols:
        coastal.loc[:, c] = pd.to_numeric(coastal[c], errors="coerce").fillna(0.5)

    sub = coastal[cols].dropna(how="any")
    if len(sub) < 30:
        return
    corr = sub.corr(method="spearman")
    try:
        import networkx as nx
        from matplotlib.patches import Patch
    except ImportError:
        print("[WARN] networkx optional; skipping network graph")
        return
    try:
        G = nx.Graph()
        for ki in corr.columns:
            G.add_node(ki.replace("rank_participation_", ""))
        nodes = corr.columns.to_list()
        for i, ai in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                bi = nodes[j]
                rho = corr.loc[ai, bi]
                if abs(float(rho)) >= rho_min and np.isfinite(rho):
                    G.add_edge(
                        ai.replace("rank_participation_", ""),
                        bi.replace("rank_participation_", ""),
                        weight=float(rho),
                    )
        pos = nx.spring_layout(G, seed=41, weight="weight")

        plt.figure(figsize=(10.5, 8.5))
        node_list = list(G.nodes)
        face_colors = [
            FAMILY_FACE_COLOR.get(INDICATOR_FAMILY.get(n, "composite_stress"), "#e0f2ff")
            for n in node_list
        ]
        edge_colors_nd = [
            FAMILY_EDGE_COLOR.get(INDICATOR_FAMILY.get(n, "composite_stress"), "#003366")
            for n in node_list
        ]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_list,
            node_size=1150,
            node_color=face_colors,
            edgecolors=edge_colors_nd,
            linewidths=1.15,
        )
        edges = G.edges(data=True)
        weights = [abs(d["weight"]) * 3.2 for _, _, d in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.55, edge_color="gray")
        nx.draw_networkx_labels(G, pos, labels={n: labs.get(n, n) for n in G.nodes}, font_size=7)
        leg_handles = [
            Patch(
                facecolor=FAMILY_FACE_COLOR["maritime"],
                edgecolor=FAMILY_EDGE_COLOR["maritime"],
                label="Maritime-linked",
            ),
            Patch(
                facecolor=FAMILY_FACE_COLOR["atmospheric"],
                edgecolor=FAMILY_EDGE_COLOR["atmospheric"],
                label="Atmospheric / directional coupling",
            ),
            Patch(
                facecolor=FAMILY_FACE_COLOR["sentinel_surface"],
                edgecolor=FAMILY_EDGE_COLOR["sentinel_surface"],
                label="Sentinel surface state",
            ),
            Patch(
                facecolor=FAMILY_FACE_COLOR["composite_stress"],
                edgecolor=FAMILY_EDGE_COLOR["composite_stress"],
                label="Composite stress index",
            ),
        ]
        plt.legend(handles=leg_handles, loc="upper left", fontsize=7, framealpha=0.92)
        plt.title(
            f"FigE. Spearman coupling network (|rho|>={rho_min}; participation ranks; not causal)",
            fontsize=10,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIGURES / "FigE_correlation_participation_network.png", dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] network plot failed: {exc}")


def ml_permutation_block(df: pd.DataFrame, min_rows: int = 200) -> tuple[pd.DataFrame, Path | None]:
    coastal = df.loc[df["coastal_panel"]].copy()
    target = "environmental_stress_index"
    if target not in coastal.columns:
        return pd.DataFrame(), None

    feat_rows: list[tuple[str, tuple[str, ...]]] = [
        ("vessel_density_t", ("vessel_density_t",)),
        ("nitrogen_dioxide", ("local_no2_excess", "no2_mean_t", "no2_mean")),
        ("coastal_wind_alignment_score", ("coastal_wind_alignment_score",)),
        ("pollution_transport_wind_alignment_score", ("pollution_transport_wind_alignment_score",)),
        ("oil_slick_probability_t", ("oil_slick_probability_t",)),
        ("ndti", ("sentinel_ndti_mean_t", "ndti_mean_t", "ndti_mean")),
        ("ndwi", ("sentinel_ndwi_mean_t", "ndwi_mean_t", "ndwi_mean")),
        ("ndvi", ("sentinel_ndvi_mean_t", "ndvi_mean_t", "ndvi_mean")),
        ("distance_to_coast_km", ("distance_to_coast_km",)),
        ("distance_to_nearest_high_vessel_density_cell", ("distance_to_nearest_high_vessel_density_cell",)),
    ]
    resolved: list[tuple[str, str]] = []
    seen_sql: set[str] = set()
    for feat_label, cand in feat_rows:
        col = resolve_column(coastal, cand)
        if col is None or col in seen_sql:
            continue
        seen_sql.add(col)
        resolved.append((feat_label, col))

    if len(resolved) < 4:
        print("[WARN] insufficient predictor columns for ML interpretability block")
        return pd.DataFrame(), None

    feat_labels = [r[0] for r in resolved]
    X = coastal[[r[1] for r in resolved]].apply(pd.to_numeric, errors="coerce")
    X.columns = feat_labels
    y = pd.to_numeric(coastal[target], errors="coerce")
    m = y.notna() & X.notna().all(axis=1)
    Xf, yf = X.loc[m], y.loc[m]
    if len(Xf) < min_rows:
        print(f"[WARN] insufficient rows for ML interpretability block (need {min_rows})")
        return pd.DataFrame(), None

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    model = HistGradientBoostingRegressor(max_depth=6, max_iter=180, random_state=42, learning_rate=0.06)
    model.fit(Xf.to_numpy(float), yf.to_numpy(float))
    pi = permutation_importance(
        model,
        Xf.to_numpy(float),
        yf.to_numpy(float),
        n_repeats=20,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    tbl = pd.DataFrame(
        {
            "feature": feat_labels,
            "permutation_mean_delta_neg_mae": pi.importances_mean,
            "permutation_std": pi.importances_std,
        },
    ).sort_values("permutation_mean_delta_neg_mae", ascending=False)
    tbl["note"] = "Higher = higher relative predictive contribution (association only)"
    tbl.to_csv(REPORTS / "indicator_participation_ml_permutation.csv", index=False)

    plt.figure(figsize=(7.2, 5.2))
    order = tbl["feature"][::-1]
    vals = tbl.set_index("feature").loc[order, "permutation_mean_delta_neg_mae"][::-1]
    err = tbl.set_index("feature").loc[order, "permutation_std"][::-1]
    plt.barh(vals.index, vals, xerr=err, color="teal", ecolor="grey", alpha=0.85)
    plt.xlabel("Mean drop in Neg-MAE (permutation)")
    plt.title(f"HistGradientBoosting permutation importance predicting {target}\n(relative predictive participation only)")
    plt.tight_layout()
    outp = FIGURES / "FigF_ml_permutation_importance.png"
    plt.savefig(outp, dpi=200)
    plt.close()
    return tbl, outp


def write_summary_md(_stats: pd.DataFrame) -> None:
    ml_csv = REPORTS / "indicator_participation_ml_permutation.csv"
    fige = FIGURES / "FigE_correlation_participation_network.png"
    figf = FIGURES / "FigF_ml_permutation_importance.png"
    run_notes = [
        "## Run-generated outputs (this build)",
        f"- **FigE (network):** {'present' if fige.is_file() else 'skipped (install `networkx` and re-run)'} — `outputs/figures/indicator_participation/FigE_correlation_participation_network.png`.",
        f"- **FigF / ML permutation:** {'present' if figf.is_file() and ml_csv.is_file() else 'skipped — need enough complete rows (default ≥200, override with `--ml-min-rows`); interpretability is *relative predictive participation*, not causal importance.'}",
        "",
    ]
    lines = [
        "# Indicator participation in coastal exposure",
        "",
        "## Framing (thesis wording)",
        "This layer describes **spatially and regime-dependent environmental participation**: which indicators occupy higher ",
        "relative positions on an outlier-resistant common scale (**coastal percentile rank**) under different ",
        "**shipping-distance** and **directional wind** contexts. Interpret as **association / structuring / behavioural coupling**, ",
        "not causal attribution or source apportionment.",
        "",
        "## Normalisation",
        "- **Participation metric (heatmap / composition / curves):** empirical percentile rank of each indicator **within ",
        "the pooled coastal-analysis panel**. Values near 1 mean that indicator tended to lie high in its empirical distribution ",
        "for that regime relative to coastal grid-week observations.",
        "- **Secondary row in CSV:** robust z-scores `(x-median)/(IQR/1.349)`, clipped ±3.5 for compact regime summaries.",
        "",
        "### Regimes",
        "- Distance bands: 0–3, 3–7, 7–15, 15–30 km from high-density shipping cells.",
        "- Wind: shoreward versus non-shoreward using coastal alignment cosine ≥ cos(45°).",
        "",
        "### Effect sizes vs opposite wind (`cliffs_delta...`) ",
        "- Non-parametric stochastic dominance contrast between regimes **within each band**. Small |δ| ⇒ similar ordinal behaviour.",
        "",
        "### Machine-learning block (optional)",
        "- Histogram gradient boosting with **permutation importance** predicts **environmental stress index** from shipping, ",
        "sentinel, alignment, and distance fields — report as **relative predictive participation** only.",
        "",
        *run_notes,
        "## Thesis-ready figure captions",
        "",
        "**FigA — Indicator participation heatmap (shoreward / non-shoreward).** ",
        "Rows: environmental indicators; columns: shipping-distance bands. ",
        "Colour: mean **coastal-panel percentile rank** of each indicator within the regime subset. ",
        "High values indicate stronger **ordinal participation** in that spatial–wind context (structuring, not source attribution).",
        "",
        "**FigB (thematic) — Stacked exposure composition by distance.** ",
        "Six thesis-facing layers: NO₂, vessel density, combined wind-alignment fields, oil proxy, Sentinel NDTI–NDWI–NDVI, ",
        "and the three exposure indices. Band totals are **sums of mean rank participation** within each theme, then row-normalised ",
        "to show **proportional structuring** (not mass apportionment). Files: `FigB_thematic_stacked_*.png`.",
        "",
        "**FigB (indicator detail) — Full indicator stack.** ",
        "Same construction as the thematic stack but one segment per resolved indicator (11 layers). ",
        "Files: `FigB_stacked_composition_*.png`.",
        "",
        "**FigC — Wind-regime participation comparison.** ",
        "Dual panel: **shoreward** versus **non-shoreward** violins of rank participation by distance band ",
        "for selected indicators (colour / hue). Highlights **regime-dependent structuring** of ordinal participation.",
        "",
        "**FigD — Participation versus inland distance.** ",
        "Lines trace mean rank participation across distance bands under shoreward versus non-shoreward conditions. ",
        "Highlights **persistence or decay** of indicator participation inland — descriptive of coupled environmental behaviour.",
        "",
        "**FigE — Spearman coupling network on rank participation.** ",
        "Edges denote |ρ| above a threshold; **node colour** encodes maritime-linked, atmospheric / directional coupling, ",
        "Sentinel surface state, or composite stress (legend). Integrates **environmental coupling** without implying causality.",
        "",
        "**FigF — Permutation importance (when run).** ",
        "Bars show mean drop in model score when features are permuted — **machine-learned predictive participation**, not causal effect sizes.",
        "",
        "## Figure map",
        "- **FigA** heatmaps: participation matrix by indicator × distance (shore vs non-shoreward). ",
        "- **FigB** thematic stacked composition (NO₂, vessel, wind, oil, Sentinel, indices) plus **indicator-detail** stacks.",
        "- **FigC** violin ridges for curated indicators across bands.",
        "- **FigD** distance curves (line bundles) for inland persistence / decay silhouettes.",
        "- **FigE** coupling network (Spearman on ranks; requires `networkx`). ",
        "- **FigF** permutation-importance bars when ML executes.",
        "",
        "## Interpretive hooks (non-causal)",
        "- **Spatially varying participation:** compare FigA/FigD across bands for maritime-associated vs terrestrial Sentinel signals.",
        "- **Directional environmental coupling:** alignment indices flip rank structure between regimes (CSV: Cliff δ, Mann–Whitney); frame as regime-dependent structuring.",
        "- **Maritime-associated coastal exposure:** vessel density and composite indices typically co-elevate participation near lanes; stress how **composition** (FigB) shifts inland.",
        "",
        "See `indicator_participation_statistics.csv` for means, medians, variance, bootstrap CIs, and regime contrasts.",
        "",
        "## Files",
        "`outputs/reports/indicator_participation_statistics.csv`",
        "`outputs/reports/indicator_participation_ml_permutation.csv` (when ML executes)",
        "",
    ]
    (REPORTS / "indicator_participation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--ne-cache", type=Path, default=_ROOT / "data" / "aux" / "natural_earth_coast_cache")
    ap.add_argument(
        "--wind-features-csv",
        type=Path,
        default=_ROOT / "outputs" / "reports" / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv",
    )
    ap.add_argument(
        "--ml-min-rows",
        type=int,
        default=200,
        help="Minimum complete coastal rows for permutation-importance (lower only for exploratory runs).",
    )
    args = ap.parse_args()

    if not Path(args.input).is_file():
        print(f"[FATAL] missing {args.input}")
        return 1

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    df = merge_wind_vectors(df, args.wind_features_csv)
    df = prepare_panel(df, Path(args.ne_cache))
    df = build_indices(df)
    df, _resolved = build_participation_frame(df)

    stats = regime_statistics(df)
    stats.to_csv(REPORTS / "indicator_participation_statistics.csv", index=False)

    sns.set_theme(style="whitegrid", font_scale=0.95)
    plot_heatmaps(df)
    plot_stacked_composition(df)
    plot_stacked_composition_thematic(df)
    tops = [
        "vessel_density_t",
        "local_no2_excess",
        "coastal_wind_alignment",
        "pollution_transport_alignment",
        "oil_slick_proxy",
        "environmental_stress_index",
    ]
    tops = [k for k in tops if any(spec[0] == k for spec in INDICATOR_SPECS)]
    plot_wind_comparison_violins(df, tops)
    plot_distance_curves(df)
    correlation_network_plot(df)

    outp_ml = ml_permutation_block(df, min_rows=args.ml_min_rows)
    ml_path = outp_ml[1]

    write_summary_md(stats)
    print(f"[OK] {REPORTS / 'indicator_participation_statistics.csv'}")
    print(f"[OK] {REPORTS / 'indicator_participation_summary.md'}")
    print(f"[OK] figures → {FIGURES}")
    if ml_path is not None:
        print(f"[OK] ML permutation figure: {ml_path}")


if __name__ == "__main__":
    raise SystemExit(main())