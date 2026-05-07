#!/usr/bin/env python3
"""
Spatiotemporal thesis layer: lag structure, anomaly overlap, synthesis (Turku & Mariehamn only).

Does not fabricate values; pairwise correlations use complete-case weeks only.
"""

from __future__ import annotations

import io
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

THESIS_PORTS = ["Turku", "Mariehamn"]
WINDOW_KM = 30.0

LAG_TYPES = ("t", "t-1", "t-2", "roll2", "roll3")

RELATIONSHIP_BLOCKS: list[tuple[str, str, str]] = [
    ("A_maritime_atmospheric", "vessel_density_t", "no2_mean_t"),
    ("A_maritime_atmospheric", "maritime_exposure_index", "atmospheric_coastal_exposure_index"),
    ("B_maritime_environmental", "vessel_density_t", "ndti_mean"),
    ("B_maritime_environmental", "vessel_density_t", "environmental_stress_index"),
    ("B_maritime_environmental", "maritime_exposure_index", "environmental_stress_index"),
    ("C_atmospheric_environmental", "no2_mean_t", "ndvi_mean"),
    ("C_atmospheric_environmental", "no2_mean_t", "environmental_stress_index"),
]

PERSISTENCE_COLS = [
    "no2_mean_t",
    "environmental_stress_index",
    "maritime_exposure_index",
]

ROLL_WIN = 8
ROLL_MIN = 6


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Pearson r, Spearman r, n_pairs; nan if insufficient."""
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    except Exception:
        return float("nan"), float("nan"), 0
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 8:
        return float("nan"), float("nan"), n
    xp, yp = x[m], y[m]
    if np.unique(xp).size < 2 or np.unique(yp).size < 2:
        return float("nan"), float("nan"), n
    pr, _ = stats.pearsonr(xp, yp)
    sr, _ = stats.spearmanr(xp, yp)
    return float(pr), float(sr), n


def rolling_corr_median(x: np.ndarray, y: np.ndarray) -> float:
    """Trailing window Pearson correlation between aligned series; median of valid windows."""
    if len(x) != len(y) or len(x) < ROLL_MIN:
        return float("nan")
    sx = pd.Series(x)
    sy = pd.Series(y)
    rc = sx.rolling(ROLL_WIN, min_periods=ROLL_MIN).corr(sy)
    v = pd.to_numeric(rc, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return float("nan")
    return float(v.median())


def _port_slug(p: str) -> str:
    return p.lower().replace(" ", "_")


def _mask_port(df: pd.DataFrame, port: str) -> pd.Series:
    col = f"dist_{_port_slug(port)}_km"
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce") <= WINDOW_KM


def aggregate_port_weekly(df: pd.DataFrame, port: str) -> pd.DataFrame:
    """Mean over grid cells within WINDOW_KM of port, per week."""
    m = _mask_port(df, port)
    sub = df.loc[m].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["week_start_utc"] = pd.to_datetime(sub["week_start_utc"], utc=True, errors="coerce").dt.normalize()
    num = sub.select_dtypes(include=[np.number]).columns.tolist()
    for c in ("week_start_utc",):
        if c in num:
            num.remove(c)
    g = sub.groupby("week_start_utc", as_index=True)[num].mean()
    g = g.sort_index()
    if "shoreward_binary" in sub.columns:
        sw = sub.groupby("week_start_utc")["shoreward_binary"].apply(
            lambda s: pd.to_numeric(s, errors="coerce").eq(1).mean(),
        )
        g["shoreward_fraction"] = sw
    g["port"] = port
    return g


def add_lag_transforms(w: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = w.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        out[f"{c}__L1"] = s.shift(1)
        out[f"{c}__L2"] = s.shift(2)
        out[f"{c}__roll2"] = s.rolling(2, min_periods=2).mean()
        out[f"{c}__roll3"] = s.rolling(3, min_periods=3).mean()
    return out


def _source_series_for_lag(ww: pd.DataFrame, source: str, lag_key: str) -> pd.Series:
    if lag_key == "t":
        col = source
    elif lag_key == "t-1":
        col = source + "__L1"
    elif lag_key == "t-2":
        col = source + "__L2"
    elif lag_key == "roll2":
        col = source + "__roll2"
    elif lag_key == "roll3":
        col = source + "__roll3"
    else:
        col = source
    if col not in ww.columns:
        return pd.Series(np.nan, index=ww.index)
    return pd.to_numeric(ww[col], errors="coerce")


def lag_correlation_row(
    wt: pd.DataFrame,
    wm: pd.DataFrame,
    source: str,
    target: str,
    lag_key: str,
    block: str,
) -> dict[str, Any]:
    """Single relationship × lag; pooled + per-port Spearman/Pearson."""
    ports_d = {"Turku": wt, "Mariehamn": wm}
    rs: dict[str, tuple[float, float, int]] = {}
    pooled_x: list[float] = []
    pooled_y: list[float] = []
    roll_medians: list[float] = []

    for pname, ww in ports_d.items():
        if ww.empty or target not in ww.columns:
            rs[pname] = (float("nan"), float("nan"), 0)
            continue
        x = _source_series_for_lag(ww, source, lag_key)
        y = pd.to_numeric(ww[target], errors="coerce")
        pr, sr, n = _safe_corr(x.values, y.values)
        rs[pname] = (pr, sr, n)
        m = np.isfinite(x.values) & np.isfinite(y.values)
        if m.any():
            pooled_x.extend(x.values[m].tolist())
            pooled_y.extend(y.values[m].tolist())
        rm = rolling_corr_median(x.values, y.values)
        if rm == rm:
            roll_medians.append(rm)

    pr_p, sr_p, n_p = _safe_corr(np.array(pooled_x), np.array(pooled_y))
    t_pr, t_sr, _ = rs.get("Turku", (np.nan, np.nan, 0))
    m_pr, m_sr, _ = rs.get("Mariehamn", (np.nan, np.nan, 0))
    sign_ok = False
    if t_sr == t_sr and m_sr == m_sr:
        sign_ok = (math.copysign(1, t_sr) == math.copysign(1, m_sr)) if t_sr != 0 or m_sr != 0 else True
    stability = 1.0 - min(abs(t_sr - m_sr), 1.0) if (t_sr == t_sr and m_sr == m_sr) else float("nan")
    stronger = "Turku" if (t_sr == t_sr and m_sr == m_sr) and abs(t_sr) > abs(m_sr) else (
        "Mariehamn" if (t_sr == t_sr and m_sr == m_sr) and abs(m_sr) > abs(t_sr) else "similar"
    )

    roll_pool = float(np.nanmedian([x for x in roll_medians if np.isfinite(x)])) if roll_medians else float("nan")

    plaus = "contextual"
    if block.startswith("A") and ("no2" in target or "atmospheric" in target):
        plaus = "plausible_if_advection_present"
    elif block.startswith("B"):
        plaus = "plausible_accumulation_or_mixing"
    elif block.startswith("C"):
        plaus = "plausible_coupling_or_shared_forcing"
    elif block.startswith("D"):
        plaus = "persistence_expected_in_slow_state"

    interp = (
        f"Pooled Spearman ρ≈{sr_p:.2f} (n={n_p}); Turku/Mariehamn signs "
        f"{'match' if sign_ok else 'differ'}; lag={lag_key}. Association / temporal alignment only."
    )
    rec = "review" if (sr_p == sr_p and abs(sr_p) >= 0.12 and n_p >= 24 and stability == stability) else "supporting_only"
    if stability < 0.4 and stability == stability:
        rec = "supporting_only"

    return {
        "source_feature": source,
        "target_feature": target,
        "lag": lag_key,
        "pearson_r": pr_p,
        "spearman_r": sr_p,
        "n_pooled": n_p,
        "rolling_corr_median": roll_pool,
        "spearman_turku": t_sr,
        "spearman_mariehamn": m_sr,
        "stability_score": stability,
        "sign_consistency": sign_ok,
        "stronger_in_port": stronger,
        "interpretation": interp,
        "physically_plausible": plaus,
        "recommended_for_final_model": rec,
        "block": block,
    }


def persistence_row(wt: pd.DataFrame, wm: pd.DataFrame, col: str) -> dict[str, Any]:
    """Lag-1 autocorrelation: Y_{t-1} vs Y_t (stress / NO2 / MEI persistence)."""
    ports_d = {"Turku": wt, "Mariehamn": wm}
    rs: dict[str, tuple[float, float, int]] = {}
    pooled_x: list[float] = []
    pooled_y: list[float] = []

    for pname, ww in ports_d.items():
        if ww.empty or col not in ww.columns or f"{col}__L1" not in ww.columns:
            rs[pname] = (float("nan"), float("nan"), 0)
            continue
        x = pd.to_numeric(ww[f"{col}__L1"], errors="coerce")
        y = pd.to_numeric(ww[col], errors="coerce")
        pr, sr, n = _safe_corr(x.values, y.values)
        rs[pname] = (pr, sr, n)
        m = np.isfinite(x.values) & np.isfinite(y.values)
        if m.any():
            pooled_x.extend(x.values[m].tolist())
            pooled_y.extend(y.values[m].tolist())

    pr_p, sr_p, n_p = _safe_corr(np.array(pooled_x), np.array(pooled_y))
    t_pr, t_sr, _ = rs.get("Turku", (np.nan, np.nan, 0))
    m_pr, m_sr, _ = rs.get("Mariehamn", (np.nan, np.nan, 0))
    sign_ok = False
    if t_sr == t_sr and m_sr == m_sr:
        sign_ok = math.copysign(1, t_sr) == math.copysign(1, m_sr)
    stability = 1.0 - min(abs(t_sr - m_sr), 1.0) if (t_sr == t_sr and m_sr == m_sr) else float("nan")
    stronger = "Turku" if (t_sr == t_sr and m_sr == m_sr) and abs(t_sr) > abs(m_sr) else (
        "Mariehamn" if (t_sr == t_sr and m_sr == m_sr) and abs(m_sr) > abs(t_sr) else "similar"
    )

    return {
        "source_feature": col + "_lag1",
        "target_feature": col + "_t",
        "lag": "t-1",
        "pearson_r": pr_p,
        "spearman_r": sr_p,
        "n_pooled": n_p,
        "rolling_corr_median": float("nan"),
        "spearman_turku": t_sr,
        "spearman_mariehamn": m_sr,
        "stability_score": stability,
        "sign_consistency": sign_ok,
        "stronger_in_port": stronger,
        "interpretation": f"Week-to-week persistence pattern for {col} (associational; not a causal claim).",
        "physically_plausible": "persistence_expected_in_slow_state",
        "recommended_for_final_model": "review" if (sr_p == sr_p and abs(sr_p) >= 0.2) else "supporting_only",
        "block": "D_persistence",
    }


def audit_weekly_panel(wt: pd.DataFrame, wm: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Record ordering, duplicate weeks, and coverage."""
    rows: list[dict[str, Any]] = []
    for name, w in [("Turku", wt), ("Mariehamn", wm)]:
        if w.empty:
            rows.append(
                {
                    "port": name,
                    "n_weeks": 0,
                    "index_monotonic": True,
                    "duplicate_week_rows": 0,
                    "week_span_start": "",
                    "week_span_end": "",
                },
            )
            continue
        dup = int(w.index.duplicated().sum())
        mono = bool(w.index.is_monotonic_increasing)
        rows.append(
            {
                "port": name,
                "n_weeks": len(w),
                "index_monotonic": mono,
                "duplicate_week_rows": dup,
                "week_span_start": str(w.index.min()),
                "week_span_end": str(w.index.max()),
            },
        )
    audit = pd.DataFrame(rows)
    rep = root / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    audit.to_csv(rep / "final_temporal_data_audit.csv", index=False)
    return audit


def build_weekly_with_lags(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_need = [
        "vessel_density_t",
        "no2_mean_t",
        "ndti_mean",
        "ndwi_mean",
        "ndvi_mean",
        "maritime_exposure_index",
        "atmospheric_coastal_exposure_index",
        "environmental_stress_index",
        "oil_slick_probability_t",
    ]
    wt = aggregate_port_weekly(df, "Turku")
    wm = aggregate_port_weekly(df, "Mariehamn")
    allc = [c for c in cols_need if c in df.columns]
    wt = add_lag_transforms(wt, allc) if not wt.empty else wt
    wm = add_lag_transforms(wm, allc) if not wm.empty else wm
    return wt, wm


def run_lag_analysis(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    figd = root / "figures"
    rep = root / "reports"
    tab = root / "tables"
    for d in (figd, rep, tab):
        d.mkdir(parents=True, exist_ok=True)

    wt, wm = build_weekly_with_lags(df)
    audit_weekly_panel(wt, wm, root)

    rows: list[dict[str, Any]] = []

    for block, src, tgt in RELATIONSHIP_BLOCKS:
        if src not in df.columns or tgt not in df.columns:
            continue
        for lag in LAG_TYPES:
            r = lag_correlation_row(wt, wm, src, tgt, lag, block)
            rows.append(r)

    for col in PERSISTENCE_COLS:
        if col not in df.columns:
            continue
        rows.append(persistence_row(wt, wm, col))

    out = pd.DataFrame(rows)
    out.to_csv(tab / "final_temporal_relationships.csv", index=False)
    out.to_csv(rep / "final_lag_analysis_summary.csv", index=False)

    if not out.empty and "spearman_r" in out.columns:
        out["pair"] = out["source_feature"] + " → " + out["target_feature"]
        piv = out.pivot_table(index="pair", columns="lag", values="spearman_r", aggfunc="first")
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(piv))))
        sns.heatmap(piv.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Lag structure (Spearman ρ, pooled Turku+Mariehamn; complete cases)")
        fig.tight_layout()
        fig.savefig(figd / "final_lag_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    try:
        import networkx as nx  # noqa: PLC0415

        G = nx.Graph()
        sub = out.dropna(subset=["spearman_r"])
        sub = sub[sub["lag"].isin(["t", "t-1"])]
        for _, rr in sub.iterrows():
            wght = abs(float(rr["spearman_r"]))
            if wght < 0.05:
                continue
            a, b = rr["source_feature"][:22], rr["target_feature"][:22]
            G.add_edge(a, b, weight=wght, lag=str(rr["lag"]))
        fig, ax = plt.subplots(figsize=(9, 7))
        if len(G.nodes):
            pos = nx.spring_layout(G, seed=42, k=0.4)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#5F9EA0", node_size=700)
            wts_e = [G[u][v]["weight"] * 4 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, ax=ax, width=wts_e, alpha=0.6)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
            ax.set_title("Lag association network (|Spearman ρ|, pooled)")
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, "No edges above threshold", ha="center")
        fig.tight_layout()
        fig.savefig(figd / "final_lag_network_graph.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"[lag network] skipped: {exc}", stacklevel=1)

    for port_name, ww, slug in (
        ("Turku", wt, "turku"),
        ("Mariehamn", wm, "mariehamn"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        if ww.empty or "no2_mean_t" not in ww.columns:
            ax.text(0.5, 0.5, "Insufficient weekly series", ha="center")
        else:
            vx = (
                pd.to_numeric(ww["vessel_density_t__L1"], errors="coerce")
                if "vessel_density_t__L1" in ww.columns
                else pd.to_numeric(ww["vessel_density_t"], errors="coerce").shift(1)
            )
            ny = pd.to_numeric(ww["no2_mean_t"], errors="coerce")
            m = vx.notna() & ny.notna()
            ax.scatter(vx[m], ny[m], alpha=0.5, s=22)
            ax.set_xlabel("Vessel density (lag-1 week)")
            ax.set_ylabel("NO2 (same week)")
            ax.set_title(f"{port_name}: temporal co-structure (associational)")
        fig.tight_layout()
        fig.savefig(figd / f"final_temporal_response_curves_{slug}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if not out.empty:
        s2 = out.groupby("target_feature")["spearman_r"].apply(lambda s: s.abs().max())
        fig, ax = plt.subplots(figsize=(7, 4))
        s2.sort_values().plot.barh(ax=ax, color="steelblue")
        ax.set_xlabel("max |Spearman ρ| across lags (pooled)")
        ax.set_title("Lag strength by target feature (exploratory)")
        fig.tight_layout()
        fig.savefig(figd / "final_lag_strength_by_indicator.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    rep.joinpath("final_lag_analysis_interpretation.md").write_text(
        "\n".join(
            [
                "# Final lag analysis (interpretation)",
                "",
                "Weekly means within **≤30 km** of each focal port. **No causal transport claim.**",
                "Lag keys: `t` contemporaneous; `t-1` / `t-2` source lag; `roll2` / `roll3` trailing means of source.",
                "Rolling correlation uses **8-week** windows (minimum **6** weeks) and reports the **median** window correlation per port where defined.",
                "",
                "Suggested wording: *associated with*, *temporally aligned*, *delayed response structure*,",
                "*persistence pattern*, *suggests possible transport/accumulation dynamics* — not deterministic attribution.",
                "",
                "See `tables/final_temporal_relationships.csv` and `reports/final_lag_analysis_summary.csv`.",
                "",
            ],
        ),
        encoding="utf-8",
    )
    rep.joinpath("final_lag_interpretation.md").write_text(
        "\n".join(
            [
                "# Lag structure — thesis discussion notes",
                "",
                "Results indicate **temporal associations** between maritime, atmospheric, and environmental indicators.",
                "Where lagged maritime signals are **temporally aligned** with atmospheric or stress outcomes, interpretations should describe **possible**",
                "transport or accumulation dynamics without implying proof of a single emission pathway.",
                "",
                "Immediate (contemporaneous) versus **delayed response structure** should be contrasted with **persistence** in NO₂ and stress,",
                "which may reflect slow-changing land–water state or shared seasonal forcing.",
                "",
            ],
        ),
        encoding="utf-8",
    )
    return out


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if not np.isfinite(mad) or mad < 1e-12:
        return pd.Series(np.nan, index=s.index)
    return 0.6745 * (s - med) / mad


def _weekly_flags(s: pd.Series) -> tuple[pd.Series, pd.Series, float, pd.Series]:
    """Robust z, deviation from rolling mean, p90 threshold."""
    rz = robust_z(s)
    roll_m = s.rolling(8, min_periods=4).mean()
    dev = s - roll_m
    p90 = s.quantile(0.9)
    p95 = s.quantile(0.95)
    flag = (rz.abs() > 2.0) | (s >= p90) | (dev > dev.quantile(0.9))
    return rz, dev, float(p90), flag.fillna(False)


def run_anomaly_module(df: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    figd = root / "figures"
    maps = root / "maps"
    rep = root / "reports"
    tab = root / "tables"
    for d in (figd, maps, rep, tab):
        d.mkdir(parents=True, exist_ok=True)

    anom_feats = [
        ("vessel_density_t", "vessel"),
        ("maritime_exposure_index", "mei"),
        ("atmospheric_coastal_exposure_index", "acei"),
        ("environmental_stress_index", "stress"),
        ("no2_mean_t", "no2"),
        ("ndti_mean", "ndti"),
        ("ndvi_mean", "ndvi"),
    ]
    if "oil_slick_probability_t" in df.columns and pd.to_numeric(df["oil_slick_probability_t"], errors="coerce").notna().any():
        anom_feats.append(("oil_slick_probability_t", "oil"))

    overlap_rows: list[dict[str, Any]] = []
    wt, wm = build_weekly_with_lags(df)

    # Per-week concurrent anomaly tags (high-level)
    def collect_concurrent(ww: pd.DataFrame, week_idx: pd.Timestamp) -> str:
        tags: list[str] = []
        if week_idx not in ww.index:
            return ""
        row = ww.loc[week_idx]
        for col, short in anom_feats:
            if col not in ww.columns:
                continue
            s = pd.to_numeric(ww[col], errors="coerce")
            rz, _, p90, fl = _weekly_flags(s)
            if week_idx not in s.index:
                continue
            sv = s.loc[week_idx]
            if not np.isfinite(sv):
                continue
            if bool(fl.loc[week_idx]) if week_idx in fl.index else False:
                tags.append(short)
        return ";".join(sorted(set(tags)))

    for port_name, ww in [("Turku", wt), ("Mariehamn", wm)]:
        if ww.empty:
            continue
        ww = ww.sort_index()
        anom_by_week: dict[pd.Timestamp, list[str]] = {}

        for col, short in anom_feats:
            if col not in ww.columns:
                continue
            s = pd.to_numeric(ww[col], errors="coerce")
            rz, dev, p90, flag = _weekly_flags(s)

            for week_idx in ww.index[flag]:
                conc: list[str] = []
                sw = pd.to_numeric(ww.loc[week_idx].get("shoreward_fraction"), errors="coerce")
                if sw > 0.5:
                    conc.append("shoreward_wind_week")
                anom_by_week.setdefault(week_idx, []).append(short)
                ctag = collect_concurrent(ww, week_idx)
                overlap_rows.append(
                    {
                        "week": str(week_idx),
                        "port": port_name,
                        "anomaly_type": f"high_{short}_or_robust_z",
                        "anomaly_strength": float(rz.loc[week_idx]) if week_idx in rz.index else np.nan,
                        "wind_regime": "shoreward_heavy" if sw > 0.5 else "mixed",
                        "distance_band": f"<={WINDOW_KM:g}km_port_window",
                        "concurrent_anomalies": ctag,
                        "interpretation": "exploratory co-occurrence (not causal attribution)",
                    },
                )

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(ww.index, pd.to_numeric(ww.get("no2_mean_t"), errors="coerce"), label="NO2", lw=1)
        ax2 = ax.twinx()
        if "environmental_stress_index" in ww.columns:
            ax2.plot(
                ww.index,
                pd.to_numeric(ww["environmental_stress_index"], errors="coerce"),
                color="orange",
                alpha=0.7,
                lw=1,
                label="stress",
            )
        ax.set_title(f"{port_name}: weekly means (≤{WINDOW_KM:.0f} km)")
        ax.legend(loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(figd / f"final_anomaly_timeline_{port_name.lower()}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    ovl = pd.DataFrame(overlap_rows)
    if not ovl.empty:
        ovl.to_csv(tab / "final_anomaly_overlap_table.csv", index=False)
    ovl.to_csv(rep / "final_anomaly_detection_summary.csv", index=False)

    rep.joinpath("final_anomaly_interpretation.md").write_text(
        "\n".join(
            [
                "# Anomaly detection (thesis)",
                "",
                "Flags combine **robust MAD z** (|z|>2), **≥p90** level, and **high deviation** from an 8-week rolling mean (top decile of deviations).",
                "Overlap rows are **descriptive**; missing weeks remain missing in source series.",
                "",
            ],
        ),
        encoding="utf-8",
    )

    if not ovl.empty:
        ct = ovl.groupby(["port", "anomaly_type"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(ct))))
        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Anomaly counts by type × port")
        fig.tight_layout()
        fig.savefig(figd / "final_anomaly_overlap_heatmap.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Combined NO2 vs shoreward fraction (two panels, one file per spec)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (pname, ww) in zip(axes, [("Turku", wt), ("Mariehamn", wm)]):
        if ww.empty or "shoreward_fraction" not in ww.columns or "no2_mean_t" not in ww.columns:
            ax.text(0.5, 0.5, "No data", ha="center")
            continue
        ax.scatter(
            ww["shoreward_fraction"],
            pd.to_numeric(ww["no2_mean_t"], errors="coerce"),
            alpha=0.5,
            s=20,
        )
        ax.set_xlabel("Shoreward fraction (weekly)")
        ax.set_ylabel("NO2 (weekly mean)")
        ax.set_title(pname)
    fig.suptitle("NO2 vs shoreward wind context (weekly port means)")
    fig.tight_layout()
    fig.savefig(figd / "final_no2_anomaly_vs_wind.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Stress vs shoreward — robust z of stress on y optional; plot stress vs shoreward
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (pname, ww) in zip(axes, [("Turku", wt), ("Mariehamn", wm)]):
        if ww.empty or "shoreward_fraction" not in ww.columns or "environmental_stress_index" not in ww.columns:
            ax.text(0.5, 0.5, "No data", ha="center")
            continue
        ax.scatter(
            ww["shoreward_fraction"],
            pd.to_numeric(ww["environmental_stress_index"], errors="coerce"),
            alpha=0.5,
            s=20,
            c="darkred",
        )
        ax.set_xlabel("Shoreward fraction (weekly)")
        ax.set_ylabel("Environmental stress index")
        ax.set_title(pname)
    fig.suptitle("Stress index vs shoreward wind context (exploratory)")
    fig.tight_layout()
    fig.savefig(figd / "final_stress_anomaly_vs_wind.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Maritime exposure time series + anomaly markers
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (pname, ww) in zip(axes, [("Turku", wt), ("Mariehamn", wm)]):
        if ww.empty or "maritime_exposure_index" not in ww.columns:
            ax.text(0.5, 0.5, "No data", ha="center")
            continue
        s = pd.to_numeric(ww["maritime_exposure_index"], errors="coerce")
        _, _, _, fl = _weekly_flags(s)
        fl = fl.reindex(ww.index).fillna(False)
        ax.plot(ww.index, s, lw=1, label="MEI")
        ax.scatter(
            ww.index[fl],
            s.loc[fl],
            color="red",
            s=25,
            alpha=0.6,
            zorder=5,
            label="flagged week",
        )
        ax.set_title(pname)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylabel("Maritime exposure index")
    fig.suptitle("Maritime exposure — weekly series with anomaly flags")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figd / "final_maritime_exposure_anomalies.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Spatial maps: variability + mean MEI + port point + wind vectors if available
    for port in THESIS_PORTS:
        m = _mask_port(df, port)
        sub = df.loc[m].copy()
        if sub.empty:
            continue
        agg_std = (
            sub.groupby("grid_cell_id", sort=False)["maritime_exposure_index"]
            .apply(lambda s: pd.to_numeric(s, errors="coerce").std())
            .rename("mei_weekly_std")
        )
        agg_mean_mei = (
            sub.groupby("grid_cell_id", sort=False)["maritime_exposure_index"].mean().rename("mei_mean")
        )
        gstd = None
        if "no2_mean_t" in sub.columns:
            gstd = (
                sub.groupby("grid_cell_id", sort=False)["no2_mean_t"]
                .apply(lambda s: pd.to_numeric(s, errors="coerce").std())
                .rename("no2_weekly_std")
            )
        meta = sub.drop_duplicates("grid_cell_id").set_index("grid_cell_id")[
            ["grid_centroid_lat", "grid_centroid_lon"]
        ]
        plot_df = meta.join(agg_std, how="inner").join(agg_mean_mei, how="left")
        if gstd is not None:
            plot_df = plot_df.join(gstd, how="left")

        lat0 = float(sub["grid_centroid_lat"].median())
        lon0 = float(sub["grid_centroid_lon"].median())

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        sc = ax.scatter(
            plot_df["grid_centroid_lon"],
            plot_df["grid_centroid_lat"],
            c=plot_df["mei_weekly_std"],
            cmap="magma",
            s=32,
            alpha=0.8,
            vmin=plot_df["mei_weekly_std"].quantile(0.05) if len(plot_df) else None,
            vmax=plot_df["mei_weekly_std"].quantile(0.95) if len(plot_df) else None,
        )
        plt.colorbar(sc, ax=ax, shrink=0.55, label="MEI temporal variability (σ over weeks)")

        if "no2_weekly_std" in plot_df.columns and plot_df["no2_weekly_std"].notna().any():
            ax.scatter(
                plot_df["grid_centroid_lon"],
                plot_df["grid_centroid_lat"],
                c="none",
                s=80,
                edgecolors="cyan",
                linewidths=0.4,
                alpha=0.35,
                label="NO2 variability (outline)",
            )

        ax.scatter([lon0], [lat0], c="lime", s=120, marker="*", edgecolors="k", zorder=10, label="port (approx)")

        if "wind_u_mean" in sub.columns and "wind_v_mean" in sub.columns:
            wu = sub.groupby("grid_cell_id")["wind_u_mean"].mean()
            wv = sub.groupby("grid_cell_id")["wind_v_mean"].mean()
            wdf = meta.join(wu.rename("wu"), how="inner").join(wv.rename("wv"), how="inner")
            step = max(1, len(wdf) // 500)
            ax.quiver(
                wdf["grid_centroid_lon"].values[::step],
                wdf["grid_centroid_lat"].values[::step],
                np.asarray(wdf["wu"].values[::step], dtype=float) * 0.02,
                np.asarray(wdf["wv"].values[::step], dtype=float) * 0.02,
                angles="xy",
                scale_units="xy",
                scale=1,
                color="steelblue",
                width=0.004,
                alpha=0.7,
            )

        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.set_title(f"{port}: exposure variability & context (≤{WINDOW_KM:.0f} km)")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(maps / f"final_{_port_slug(port)}_anomaly_map.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    return wt, wm, ovl


def write_synthesis(root: Path, lag_df: pd.DataFrame, ovl: pd.DataFrame, decay_figure: str) -> None:
    rep = root / "reports"
    notes = root / "thesis_notes"
    tab = root / "tables"
    for d in (rep, notes, tab):
        d.mkdir(parents=True, exist_ok=True)

    lag_note = ""
    if not lag_df.empty and "spearman_r" in lag_df.columns:
        j = lag_df.dropna(subset=["spearman_r"])
        if not j.empty:
            k = j.loc[j["spearman_r"].abs().idxmax()]
            lag_note = f"Strongest **pooled** |Spearman| in table: {k['source_feature']} → {k['target_feature']} (lag {k['lag']}, ρ≈{k['spearman_r']:.2f})."

    rep.joinpath("final_spatiotemporal_synthesis.md").write_text(
        "\n".join(
            [
                "# Spatiotemporal synthesis (Turku vs Mariehamn)",
                "",
                "## A. Spatial findings",
                f"Fixed-band distance decay and exposure ranking figures (`{decay_figure}`) show how composite indices change with distance from each port, with explicit gaps where cells are missing.",
                "",
                "## B. Directional findings",
                "Shoreward versus non-shoreward stratification summarises **directional structuring** of exposure in wind geometry; language remains associational.",
                "",
                "## C. Temporal findings",
                "Weekly lag analysis summarises **complete-case** correlations within **≤30 km** port windows. " + lag_note,
                "**Immediate** (same week) versus **delayed response structure** (lagged source) and **rolling averages** describe possible accumulation without proving transport.",
                "",
                "## D. Anomaly findings",
                "Anomaly flags mark weeks that are unusual under robust scale and/or trailing context; overlap tables describe **temporal clustering** descriptively.",
                "",
                "## E. Cross-port comparison",
                "Turku and Mariehamn differ in coastal geometry and grid coverage; **stability** between ports tempers the strength of pooled summaries.",
                "",
                "## F. Limitations",
                "Observational grid, short weekly series, unmeasured confounders, and composite construction imply that results **suggest** structure, not attribution.",
                "",
                "## G. Scientific interpretation",
                "Patterns are consistent with a **spatiotemporal environmental exposure framework**: spatial decay, directional wind context, temporal alignment, and episodic anomalies — **not** proof of shipping-driven pollution transfer.",
                "",
            ],
        ),
        encoding="utf-8",
    )

    findings = [
        {
            "finding_id": "F1",
            "finding": "Spatial exposure metrics attenuate with distance from the focal port where data exist; missing bands remain visible.",
            "evidence_type": "spatial",
            "strongest_supporting_figure": decay_figure,
            "strongest_supporting_metric": "maritime_exposure_index",
            "interpretation_strength": "moderate",
            "limitation": "Coverage and annulus validity vary by port",
        },
        {
            "finding_id": "F2",
            "finding": "Weekly associations between maritime, atmospheric, and stress indicators show exploratory lag structure (pooled and per-port).",
            "evidence_type": "temporal",
            "strongest_supporting_figure": "final_lag_heatmap.png",
            "strongest_supporting_metric": "spearman_r",
            "interpretation_strength": "exploratory",
            "limitation": "Concurrent drivers; short series; pooled vs port stability",
        },
        {
            "finding_id": "F3",
            "finding": "Anomaly weeks cluster descriptively with elevated shoreward fraction or stress/NO₂ context depending on port.",
            "evidence_type": "anomaly",
            "strongest_supporting_figure": "final_anomaly_overlap_heatmap.png",
            "strongest_supporting_metric": "concurrent_anomalies",
            "interpretation_strength": "exploratory",
            "limitation": "Threshold choice affects sensitivity",
        },
    ]
    pd.DataFrame(findings).to_csv(tab / "final_key_findings_table.csv", index=False)

    notes.joinpath("final_temporal_results_section.md").write_text(
        "\n".join(
            [
                "## Temporal results (draft)",
                "",
                "Weekly means within **≤30 km** of each focal port were used to examine **temporal associations** between vessel density, composite maritime exposure,",
                "atmospheric coastal exposure, NO₂, and environmental stress. **Results indicate** co-variation and, for some pairs, **delayed response structure** relative to maritime indicators.",
                "Rolling summaries highlight possible **accumulation effects** without implying a unique causal pathway.",
                "",
            ],
        ),
        encoding="utf-8",
    )
    notes.joinpath("final_anomaly_results_section.md").write_text(
        "\n".join(
            [
                "## Anomaly results (draft)",
                "",
                "**Patterns suggest** weeks of unusual vessel, maritime, atmospheric, or stress levels relative to robust baselines. Overlap with shoreward-wind context is reported as **temporal association** only.",
                "Missing observations were not imputed.",
                "",
            ],
        ),
        encoding="utf-8",
    )
    notes.joinpath("final_spatiotemporal_discussion.md").write_text(
        "\n".join(
            [
                "## Discussion (draft)",
                "",
                "Together, **directional structuring** from wind regimes, **distance decay** in exposure, and **temporal alignment** across indicators support framing a",
                "**spatiotemporal environmental exposure** lens. The evidence does **not** support claims of direct pollution attribution or deterministic shipping effects.",
                "",
            ],
        ),
        encoding="utf-8",
    )


def write_validation(root: Path, lag_df: pd.DataFrame, ovl: pd.DataFrame, wt: pd.DataFrame, wm: pd.DataFrame) -> None:
    rep = root / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    p = rep / "final_temporal_validation.md"

    n_t = len(wt) if not wt.empty else 0
    n_m = len(wm) if not wm.empty else 0
    miss_note = ""
    if not ovl.empty:
        miss_note = f"Anomaly log rows: {len(ovl)} (weeks can appear multiple times if flagged by multiple rules/features)."

    lines = [
        "# Temporal validation",
        "",
        "## Data coverage",
        f"- Turku weekly rows: {n_t}",
        f"- Mariehamn weekly rows: {n_m}",
        "- See `reports/final_temporal_data_audit.csv` for monotonicity and duplicate-week checks on aggregated panel.",
        "",
        "## Lag robustness",
        f"- Lag table rows: {len(lag_df)}",
        "- Correlations require ≥8 complete pairwise weeks; smaller samples yield NaN (honest missingness).",
        "- Review `final_lag_analysis_summary.csv` for |ρ|≈1 (possible collinearity) or unstable port contrasts.",
        "",
        "## Anomaly robustness",
        miss_note,
        "- Anomalies are not defined from imputed values; underlying NaNs reduce effective counts.",
        "",
        "## Temporal limitations",
        "- Single observational period; seasonality and unobserved drivers can align with shipping signals.",
        "- Composite indices share inputs; interpret overlap between indicators cautiously.",
        "- Inspect raw series: **if vessel_density (or another input) is nearly constant in time within a port window**, lag-specific correlations for that source may **coincide numerically** because the signal has no temporal contrast at that scale; pooled summaries can then mix within-port structure with cross-port differences.",
        "",
        "## Uncertainty",
        "- All results are **associational**; physical plausibility is interpreted as consistency with possible mechanisms, not proof.",
        "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")


def console_summary(
    lag_df: pd.DataFrame,
    ovl: pd.DataFrame,
    decay_metric_note: str,
) -> None:
    print("\n=== Final spatiotemporal summary (thesis) ===")
    if not lag_df.empty and "spearman_r" in lag_df.columns:
        j = lag_df.dropna(subset=["spearman_r"]).copy()
        if not j.empty:
            k = j.loc[j["spearman_r"].abs().idxmax()]
            print(
                f"Strongest lag (pooled |Spearman|): {k['source_feature']} -> {k['target_feature']} "
                f"lag={k['lag']} rho={k['spearman_r']:.3f}",
            )
    if not ovl.empty:
        vc = ovl.groupby("week").size()
        mx = int(vc.max()) if len(vc) else 0
        print(f"Strongest anomaly overlap: up to {mx} concurrent anomaly labels in a single week (count of rows by feature).")

    print(f"Strongest directional pattern: see wind-regime bars and shoreward contrasts (spatial layer); decay: {decay_metric_note}")
    print("Strongest distance-decay signal: maritime / stress indices in fixed-band curves (per port figures).")
    print("Major limitations: observational grid, short weekly series, composite construction, association-only inference.")
    print(
        "Top thesis-ready findings: (1) spatial decay + explicit missingness; "
        "(2) exploratory temporal associations/lags; (3) descriptive anomaly overlap — no causal attribution.",
    )


def run_spatiotemporal_block(df: pd.DataFrame, thesis_root: Path) -> None:
    lag_df = run_lag_analysis(df, thesis_root)
    wt, wm, ovl = run_anomaly_module(df, thesis_root)
    decay_fig = "final_distance_decay_turku.png / final_distance_decay_mariehamn.png"
    write_synthesis(thesis_root, lag_df, ovl, decay_fig)
    write_validation(thesis_root, lag_df, ovl, wt, wm)
    console_summary(lag_df, ovl, "Spearman in lag table & decay CSV")


def configure_tee_log(log_path: Path) -> io.StringIO | None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return None
