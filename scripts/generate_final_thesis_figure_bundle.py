#!/usr/bin/env python3
"""Generate thesis figures under outputs/final_thesis_figures/ using only parquet + optional wind CSV."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CAND_PANELS = [
    ROOT / "processed" / "features_ml_ready.parquet",
    ROOT / "outputs" / "processed" / "features_ml_ready_coastal_wind.parquet",
]
WINDCSV = ROOT / "outputs" / "reports" / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv"
OUT = ROOT / "outputs" / "final_thesis_figures"
MID = OUT / "intermediate"

COASTAL_MAP_SCRIPT = ROOT / "scripts" / "generate_geospatial_coastal_exposure_maps.py"
DASHBOARD_SCRIPT = ROOT / "scripts" / "generate_composite_land_exposure_dashboard.py"

PORTS_SHOW = ["Turku", "Mariehamn", "Stockholm"]
DZM_LABELS = ("0–3 km", "3–7 km", "7–15 km", "15–30 km")
DPI = 400
TOP_N = 15

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _resolve_panel_path() -> Path | None:
    for p in CAND_PANELS:
        if p.is_file():
            return p.resolve()
    return None


def setup() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    MID.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 110,
            "savefig.dpi": DPI,
        }
    )
    pq = _resolve_panel_path()
    if pq is None:
        logging.warning("Missing panel parquet; expected one of:\n%s", "\n".join(str(p) for p in CAND_PANELS))
    else:
        logging.info("panel: %s", pq)
    return pq


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", dpi=DPI, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    plt.close(fig)
    logging.info("saved %s", stem)


def load_df(pq: Path) -> pd.DataFrame:
    d = pd.read_parquet(pq)
    d["week_start_utc"] = pd.to_datetime(d["week_start_utc"], utc=True)

    lat_col = "grid_centroid_lat" if "grid_centroid_lat" in d.columns else "latitude"
    lon_col = "grid_centroid_lon" if "grid_centroid_lon" in d.columns else "longitude"
    d["_lat_plot"] = pd.to_numeric(d[lat_col], errors="coerce")
    d["_lon_plot"] = pd.to_numeric(d[lon_col], errors="coerce")

    if WINDCSV.is_file():
        w = pd.read_csv(WINDCSV)
        w["week_start_utc"] = pd.to_datetime(w["week_start_utc"], utc=True)
        keys = ["grid_cell_id", "week_start_utc", "wind_u_mean", "wind_v_mean"]
        if "coastal_wind_alignment_score" in w.columns:
            keys.append("coastal_wind_alignment_score")
        keep = [c for c in keys if c in w.columns]
        w = w[keep]
        d = d.merge(w, on=["grid_cell_id", "week_start_utc"], how="left")
    if {"wind_u_mean", "wind_v_mean"}.issubset(d.columns):
        u = pd.to_numeric(d["wind_u_mean"], errors="coerce")
        v = pd.to_numeric(d["wind_v_mean"], errors="coerce")
        d["wind_speed_m_s"] = np.sqrt(u**2 + v**2)
    elif "wind_speed_m_s" not in d.columns:
        d["wind_speed_m_s"] = np.nan

    d["nearest_port"] = d["nearest_port"].astype(str)
    return d


def esi_series(d: pd.DataFrame) -> pd.Series:
    cols = ["maritime_pressure_index", "NO2_mean", "fai_mean", "ndti_mean", "vessel_density_t", "coastal_exposure_score"]
    if any(c not in d.columns for c in cols):
        return pd.Series(np.nan, index=d.index)
    x = d[cols].apply(pd.to_numeric, errors="coerce")
    zparts = []
    for c in cols:
        sd = float(x[c].std(ddof=0) or 1.0)
        zparts.append((x[c] - x[c].mean()) / sd)
    return pd.concat(zparts, axis=1).mean(axis=1)


def no2_anomaly_series(d: pd.DataFrame, col: str = "NO2_mean") -> pd.Series:
    """Cell-level anomaly: deviation from cell median NO2 (computed from parquet only)."""
    v = pd.to_numeric(d[col], errors="coerce")
    cid = d["grid_cell_id"]
    cell_med = v.groupby(cid).transform("median")
    return v - cell_med


def dist_zone_series(km: pd.Series) -> pd.Series:
    k = pd.to_numeric(km, errors="coerce")
    z = pd.Series(pd.NA, index=k.index, dtype=object)
    z.loc[(k >= 0) & (k < 3)] = DZM_LABELS[0]
    z.loc[(k >= 3) & (k < 7)] = DZM_LABELS[1]
    z.loc[(k >= 7) & (k < 15)] = DZM_LABELS[2]
    z.loc[(k >= 15) & (k <= 30)] = DZM_LABELS[3]
    return z


def pick_target(d: pd.DataFrame) -> str | None:
    for c in (
        "ndi_anomaly_lag01",
        "landi_anomaly_lag01",
        "ndi_anomaly_mean",
        "land_response_index",
        "ndvi_mean",
        "ndti_mean",
    ):
        if c in d.columns and pd.to_numeric(d[c], errors="coerce").notna().sum() > 200:
            return c
    return None


def predictor_columns(d: pd.DataFrame, target: str | None) -> list[str]:
    skip = {"grid_cell_id", "week_start_utc", "_lat_plot", "_lon_plot", "_esi_bundle", "_no2_cell_anomaly", target}
    skip.discard(None)
    prio = (
        "maritime_pressure_index",
        "vessel_density_t",
        "vessel_density",
        "distance_to_port_km",
        "NO2_mean",
        "no2_mean_t",
        "wind_speed_m_s",
        "coastal_wind_alignment_score",
        "pollution_transport_wind_alignment_score",
        "coastal_exposure_score",
        "atmospheric_transfer_index",
        "ndti_mean",
        "fai_mean",
        "ndci_mean",
        "ndwi_mean",
        "ndvi_mean",
        "grid_centroid_lat",
        "grid_centroid_lon",
        "latitude",
        "longitude",
    )
    candid: list[str] = []
    for c in prio:
        if c in d.columns and c not in skip:
            candid.append(c)
    for c in d.columns:
        if c in skip or c in candid:
            continue
        if (
            c.startswith("vessel_x_")
            or c.startswith("no2_x")
            or c.endswith("_seasonality")
            or c.endswith("_spectral_residual")
            or "lag" in c
            or c in {"week_sin", "week_cos", "week_of_year"}
        ):
            candid.append(c)
    out = []
    for c in candid:
        t = pd.to_numeric(d[c], errors="coerce")
        if t.notna().sum() > max(80, len(d) // 400):
            out.append(c)
    return list(dict.fromkeys(out))


def plot_ml(d: pd.DataFrame) -> None:
    target = pick_target(d)
    _empty_imp = pd.DataFrame(columns=["feature", "ridge_coef", "ridge_abs_coef", "hgbr_perm_mean"])
    if target is None:
        logging.warning("ML block skipped: no suitable target column")
        _empty_imp.to_csv(OUT / "feature_importance_table.csv", index=False)
        return

    feats = predictor_columns(d, target)
    if len(feats) < 3:
        logging.warning("ML block skipped: too few predictors")
        _empty_imp.to_csv(OUT / "feature_importance_table.csv", index=False)
        return

    Xdf = d[feats].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(d[target], errors="coerce").to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(Xdf)
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    if len(y) < 400:
        logging.warning("ML block: small n=%s", len(y))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    ridge = Ridge(alpha=100.0).fit(X_trs, y_tr)
    coef_abs = pd.Series(np.abs(ridge.coef_), index=feats).sort_values(ascending=False)
    top_r = coef_abs.head(TOP_N)
    r_norm = top_r / (top_r.max() + 1e-12)

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    names = r_norm.iloc[::-1].index.astype(str).tolist()
    vals = r_norm.iloc[::-1].to_numpy(dtype=float)
    ys = np.arange(len(names))
    pal = sns.color_palette("Blues_r", n_colors=max(len(names), 3))
    cols = [pal[int(np.clip(v * (len(pal) - 1), 0, len(pal) - 1))] for v in vals]
    ax.barh(ys, vals, color=cols)
    ax.set_yticks(ys, labels=names)
    ax.set_xlabel("|β| (standardized scale)")
    ax.set_title(f"Ridge · top {TOP_N} | target={target}")
    ax.axvline(0, color="#666", lw=0.8)
    plt.tight_layout()
    savefig(fig, "ridge_feature_importance")

    hg = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.06, max_iter=200, random_state=0)
    hg.fit(X_tr, y_tr)
    perm = permutation_importance(hg, X_te, y_te, n_repeats=12, random_state=0, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=feats).sort_values(ascending=False).head(TOP_N)
    imp_n = imp / (imp.max() + 1e-12)

    fig2, ax2 = plt.subplots(figsize=(9.0, 6.2))
    names2 = imp_n.iloc[::-1].index.astype(str).tolist()
    vals2 = imp_n.iloc[::-1].to_numpy(dtype=float)
    ys2 = np.arange(len(names2))
    pal2 = sns.color_palette("Greens_r", n_colors=max(len(names2), 3))
    cols2 = [pal2[int(np.clip(v * (len(pal2) - 1), 0, len(pal2) - 1))] for v in vals2]
    ax2.barh(ys2, vals2, color=cols2)
    ax2.set_yticks(ys2, labels=names2)
    ax2.set_xlabel("Permutation Δ loss (normalized)")
    ax2.set_title(f"HistGradientBoosting · top {TOP_N} permutation | target={target}")
    plt.tight_layout()
    savefig(fig2, "hgbr_permutation_importance")

    ix = {f: i for i, f in enumerate(feats)}
    comb = pd.DataFrame({"feature": feats})
    comb["ridge_coef"] = comb["feature"].map(lambda f: float(ridge.coef_[ix[f]]))
    comb["ridge_abs_coef"] = comb["ridge_coef"].abs()
    comb["hgbr_perm_mean"] = comb["feature"].map(lambda f: float(perm.importances_mean[ix[f]]))
    comb = comb.sort_values("ridge_abs_coef", ascending=False)
    comb.to_csv(OUT / "feature_importance_table.csv", index=False)


def plot_correlation_network(d: pd.DataFrame) -> None:
    cols = [
        c
        for c in (
            "maritime_pressure_index",
            "vessel_density_t",
            "NO2_mean",
            "ndti_mean",
            "fai_mean",
            "ndvi_mean",
            "distance_to_port_km",
            "coastal_exposure_score",
        )
        if c in d.columns
    ]
    if len(cols) < 3:
        logging.warning("Correlation network skipped")
        return
    work = d[cols].apply(pd.to_numeric, errors="coerce")
    work["esi_bundle"] = esi_series(d)
    cols2 = cols + ["esi_bundle"]
    rho = work.corr(method="spearman")
    n = len(work.dropna(how="any"))
    thresh = 0.18

    G = nx.Graph()
    labels = []
    for c in cols2:
        G.add_node(c)
        labels.append(c.replace("_", "\n"))

    edges = []
    for i, a in enumerate(cols2):
        for j, b in enumerate(cols2):
            if j <= i:
                continue
            r = rho.loc[a, b]
            if np.isnan(r) or abs(r) < thresh:
                continue
            G.add_edge(a, b, weight=abs(r), sign=np.sign(r))
            edges.append((a, b, r))

    edge_df = pd.DataFrame([(a, b, float(r)) for a, b, r in edges], columns=["a", "b", "rho_spearman"])
    edge_df["n_complete"] = n
    edge_df.to_csv(MID / "environmental_correlation_edges.csv", index=False)

    if G.number_of_edges() == 0:
        logging.warning("Correlation network: no edges above threshold %.2f", thresh)
        fig, ax = plt.subplots(figsize=(8.5, 6.8))
        ax.text(0.5, 0.5, f"No pairwise |Spearman ρ| ≥ {thresh}\n(edge list empty)", ha="center", va="center")
        ax.axis("off")
        savefig(fig, "environmental_correlation_network")
        return

    pos = nx.spring_layout(G, seed=3, weight="weight")
    fig, ax = plt.subplots(figsize=(9.8, 7.8))
    for (u, v, dat) in G.edges(data=True):
        col = "#c0392b" if dat.get("sign", 1) < 0 else "#2980b9"
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=max(1.0, dat["weight"] * 4),
            alpha=0.72,
            edge_color=col,
            ax=ax,
        )
    nx.draw_networkx_nodes(G, pos, node_color="#ecf0f1", edgecolors="#34495e", node_size=1200, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n.replace("_", "\n") for n in G.nodes()}, font_size=8.8, ax=ax)
    pos_h = plt.Line2D([], [], color="#2980b9", lw=3, label="Positive ρ")
    neg_h = plt.Line2D([], [], color="#c0392b", lw=3, label="Negative ρ")
    ax.legend(handles=[pos_h, neg_h], loc="upper left")
    ax.set_title("Environmental correlation network · Spearman | ρ | ≥ {:.2f}".format(thresh))
    ax.axis("off")
    plt.tight_layout()
    savefig(fig, "environmental_correlation_network")


def plot_distance_decay(d: pd.DataFrame) -> None:
    if "distance_to_port_km" not in d.columns:
        logging.warning("Distance decay skipped")
        return

    esi = esi_series(d)
    no2a = (
        pd.to_numeric(d["NO2_anomaly"], errors="coerce")
        if "NO2_anomaly" in d.columns
        else no2_anomaly_series(d, "NO2_mean")
    )

    dd = pd.DataFrame(
        {
            "nearest_port": d["nearest_port"],
            "distance_zone": dist_zone_series(d["distance_to_port_km"]),
            "mei": pd.to_numeric(d["maritime_pressure_index"], errors="coerce"),
            "no2_anomaly_plot": pd.to_numeric(no2a, errors="coerce"),
            "esi": pd.to_numeric(esi, errors="coerce"),
        }
    )
    plc = dd["nearest_port"].astype(str).str.strip().str.casefold().isin({p.casefold() for p in PORTS_SHOW})
    dd = dd.loc[plc].dropna(subset=["distance_zone"])

    agg = (
        dd.groupby(["nearest_port", "distance_zone"], observed=True)[["mei", "no2_anomaly_plot", "esi"]]
        .median()
        .reset_index()
    )
    agg.to_csv(MID / "distance_decay_medians.csv", index=False)

    metrics = ["mei", "no2_anomaly_plot", "esi"]
    titles = dict(mei="MEI (median)", no2_anomaly_plot="NO₂ anomaly (median)", esi="ESI (median, z-score blend)")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.85), sharex=False)
    x_idx = np.arange(len(DZM_LABELS))
    for ax, metric in zip(axes, metrics):
        for port in PORTS_SHOW:
            sub = agg[(agg["nearest_port"] == port) & (agg["distance_zone"].isin(DZM_LABELS))].copy()
            sub["_x"] = sub["distance_zone"].map(dict(zip(DZM_LABELS, x_idx)))
            sub = sub.sort_values("_x")
            ax.plot(sub["_x"], sub[metric], marker="o", label=port, lw=1.85)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(list(DZM_LABELS), rotation=15, ha="right")
        ax.set_title(titles[metric])
        ax.grid(alpha=0.28, linestyle=":")
    axes[0].legend(frameon=False, fontsize=9.8)
    fig.suptitle("Comparative distance decay · 0–30 km bins", fontsize=14, y=1.03)
    plt.tight_layout()
    savefig(fig, "comparative_distance_decay")


def plot_temporal_persistence(d: pd.DataFrame) -> None:
    esi = esi_series(d)
    no2a = (
        pd.to_numeric(d["NO2_anomaly"], errors="coerce")
        if "NO2_anomaly" in d.columns
        else no2_anomaly_series(d, "NO2_mean")
    )

    blk = pd.DataFrame(
        {
            "week_start_utc": d["week_start_utc"],
            "MEI": pd.to_numeric(d["maritime_pressure_index"], errors="coerce"),
            "NO₂ anomaly": pd.to_numeric(no2a, errors="coerce"),
            "NDTI": pd.to_numeric(d["ndti_mean"], errors="coerce"),
            "ESI": pd.to_numeric(esi, errors="coerce"),
        }
    )
    ws = blk.groupby("week_start_utc", sort=True).median(numeric_only=True).sort_index()
    if ws.empty:
        logging.warning("Temporal persistence skipped")
        return
    ws.index = ws.index.strftime("%Y-%m-%d")
    z = ws.copy()
    for c in z.columns:
        arr = z[c].to_numpy(dtype=float)
        z[c] = (arr - np.nanmean(arr)) / (np.nanstd(arr) or 1.0)
    ws.to_csv(MID / "temporal_persistence_medians_weekly.csv")
    z.to_csv(MID / "temporal_persistence_zscore_weekly.csv")

    fig, ax = plt.subplots(figsize=(14.8, 4.95))
    zt = z.T
    sns.heatmap(zt, cmap="RdBu_r", center=0.0, ax=ax, cbar_kws={"label": "z (row metric, across weeks)"})
    ax.set_title("Temporal persistence · weekly Baltic-wide median (metrics z-scored across weeks)")
    ax.set_xlabel("Week starting (UTC)")
    ax.set_ylabel("Indicator")
    plt.tight_layout()
    savefig(fig, "temporal_persistence_heatmap")


def plot_framework_diagram() -> None:
    """Thesis framework: ASCII labels for reliable PNG text; texts above patches (clip off)."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
        }
    )

    xl = np.array([0.58, 4.035, 7.485], dtype=float)
    bw, bh = 3.54, 0.92

    yr1_b, yr2_b, yr3_b = 5.66, 3.74, 1.93
    y_theme_b = 0.345
    cx = xl + bw / 2.0

    fig_w, fig_h = 12.05, 7.12
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0.0, 12.25)
    ax.set_ylim(0.0, 8.15)
    ax.axis("off")
    ax.set_facecolor("#f5f9fc")
    fig.patch.set_facecolor("white")

    TZ = dict(ha="center", va="center", clip_on=False, zorder=20, color="#0f172a")

    def rnd_box(x: float, y: float, w: float, h: float, text: str, fc: str):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.032,rounding_size=0.045",
            facecolor=fc,
            edgecolor="#1e293b",
            linewidth=1.15,
            zorder=3,
            clip_on=False,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2.0, y + h / 2.0, text, fontsize=9.65, linespacing=1.42, wrap=False, weight="normal", **TZ)

    def v_arrow(ax_, x_: float, y_hi: float, y_lo: float):
        gap = 0.08
        ax_.add_patch(
            FancyArrowPatch(
                (x_, y_hi - gap),
                (x_, y_lo + gap),
                arrowstyle="-|>",
                mutation_scale=13,
                lw=1.45,
                color="#334155",
                zorder=2,
                clip_on=False,
            )
        )

    ax.text(
        6.125,
        7.74,
        "Thesis analytic framework - balanced weekly panel to figures",
        fontsize=13.95,
        fontweight="bold",
        ha="center",
        va="bottom",
        color="#0f172a",
        clip_on=False,
        zorder=25,
        wrap=False,
    )
    ax.text(
        6.125,
        7.45,
        "Three parallel thematic columns converge on artefacts and thematic synthesis.",
        fontsize=9.35,
        ha="center",
        va="bottom",
        style="italic",
        color="#475569",
        clip_on=False,
        zorder=25,
        wrap=False,
    )

    rnd_box(
        xl[0],
        yr1_b,
        bw,
        bh,
        "Balanced ML panel\n(feature-ready parquet)",
        "#e8f4fc",
    )
    rnd_box(
        xl[1],
        yr1_b,
        bw,
        bh,
        "Indices and EO drivers\nMEI, NO2, NDTI, ESI (+ wind CSV)",
        "#fff2e8",
    )
    rnd_box(
        xl[2],
        yr1_b,
        bw,
        bh,
        "ML explainability\nRidge regression and HGBT permutation",
        "#eaf7f3",
    )

    rnd_box(
        xl[0],
        yr2_b,
        bw,
        bh,
        "Distance decay - port rings\n0-30 km strata",
        "#fde5ec",
    )
    rnd_box(xl[1], yr2_b, bw, bh, "Correlation network\nenvironmental linkage", "#efe5fb")
    rnd_box(
        xl[2],
        yr2_b,
        bw,
        bh,
        "Temporal persistence\nweekly median heatmaps",
        "#eef6eb",
    )

    rnd_box(
        xl[0],
        yr3_b,
        bw,
        bh,
        "Integrated exposure maps\nper-port, Baltic overview",
        "#fff7ec",
    )

    x_out = xl[1]
    w_out = xl[2] + bw - x_out + 0.06
    rnd_box(x_out, yr3_b, w_out, bh, "Outputs\nPNG / PDF / intermediate tables", "#ffffff")

    margin = 0.48
    us = 12.25 - 2 * margin
    nw = (us - 2 * 0.12) / 3
    gap_t = 0.12
    themes = (
        "Maritime impact and exposure footprints",
        "Coastal monitoring and temporal persistence",
        "ML explainability and proposal alignment",
    )
    for k, lbl in enumerate(themes):
        xk = margin + k * (nw + gap_t)
        rnd_box(xk, y_theme_b, nw, 0.88, f"Themes:\n{lbl}", "#dfe8f5")

    y2_tp = yr2_b + bh
    y3_tp = yr3_b + bh

    for i in range(3):
        v_arrow(ax, cx[i], yr1_b, y2_tp)

    v_arrow(ax, cx[0], yr2_b, y3_tp)

    out_cx = x_out + w_out / 2.0
    g2 = 0.07
    for x_src in (cx[1], cx[2]):
        ax.add_patch(
            FancyArrowPatch(
                (x_src, yr2_b - g2),
                (out_cx, y3_tp + g2),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=1.32,
                color="#334155",
                zorder=2,
                clip_on=False,
            )
        )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.035)
    png_path = OUT / "thesis_framework_diagram.png"
    pdf_path = OUT / "thesis_framework_diagram.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.35, facecolor="white")
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", pad_inches=0.32, facecolor="white")
    plt.close(fig)
    logging.info("saved %s", png_path.name)


def write_figure_summary() -> None:
    path = OUT / "figure_summary.md"
    text = """# Final thesis figures — thematic map

Artifacts live in `outputs/final_thesis_figures/` (PNG + PDF at {:.0f} DPI) and intermediates under `outputs/final_thesis_figures/intermediate/`.

| Figure | Thesis themes |
| --- | --- |
| `ridge_feature_importance.png` | ML explainability — linear baseline |
| `hgbr_permutation_importance.png` | ML explainability — non-linear permutation signal |
| `environmental_correlation_network.png` | Exposure coupling, coastal monitoring hypotheses |
| `comparative_distance_decay.png` | Exposure gradients, maritime proximity footprint |
| `temporal_persistence_heatmap.png` | Persistence of pressure / response signals in time |
| `integrated_exposure_map_*.png` | Geospatial choropleths (EPSG:3857)—composite exposure lattice + corridors |
| `coastal_exposure_hotspots.png` | Top-decile exposure overlay (association / hotspot semantics) |
| `thesis_framework_diagram.png` | Proposal alignment · pipeline narrative |

`feature_importance_table.csv` (model importance merge: Ridge coef + permutation means) sits in this folder alongside the PNG/PDF outputs. **`intermediate/`** holds correlation edges, decay aggregates, temporal medians/z-scores, `composite_exposure_dataset.parquet`, `exposure_grid.geojson`, `hotspot_cells.geojson`, `dashboard_land_exposure_cells.{parquet,geojson}`, plus per-panel `dashboard_panel_*` exports from the composite poster workflow. Coastal thematic maps plus narrative `map_summary.md` are emitted by **`scripts/generate_geospatial_coastal_exposure_maps.py`**, while the thesis-style dark infographic **`composite_land_exposure_dashboard.{png,pdf}`** is produced by **`scripts/generate_composite_land_exposure_dashboard.py`** — both invoked automatically when the ML panel parquet is present.
""".format(
        float(DPI)
    )
    path.write_text(text, encoding="utf-8")
    logging.info("wrote %s", path)


def main() -> None:
    pq = setup()
    if pq is not None:
        df = load_df(pq)

        esi = esi_series(df)
        no2a_col = pd.to_numeric(df["NO2_anomaly"], errors="coerce") if "NO2_anomaly" in df.columns else no2_anomaly_series(df)
        enriched = df.assign(_esi_bundle=esi, _no2_cell_anomaly=no2a_col)

        logging.info("rows=%s cols=%s", len(enriched), enriched.shape[1])
        plot_ml(enriched)
        plot_correlation_network(enriched)
        plot_distance_decay(enriched)
        plot_temporal_persistence(enriched)

        if COASTAL_MAP_SCRIPT.is_file():
            coastal = subprocess.run([sys.executable, str(COASTAL_MAP_SCRIPT)], cwd=str(ROOT))
            if coastal.returncode != 0:
                logging.warning(
                    "coastal choropleth script exited %s (%s)",
                    coastal.returncode,
                    COASTAL_MAP_SCRIPT.name,
                )
        else:
            logging.warning("Coastal choropleth script missing: %s", COASTAL_MAP_SCRIPT)

        if DASHBOARD_SCRIPT.is_file():
            dash = subprocess.run([sys.executable, str(DASHBOARD_SCRIPT)], cwd=str(ROOT))
            if dash.returncode != 0:
                logging.warning(
                    "composite land exposure dashboard exited %s (%s)",
                    dash.returncode,
                    DASHBOARD_SCRIPT.name,
                )
        else:
            logging.warning("Composite dashboard script missing: %s", DASHBOARD_SCRIPT)

    else:
        logging.warning("Skipping data-driven plots (no panel parquet found).")

    plot_framework_diagram()
    write_figure_summary()


if __name__ == "__main__":
    main()
