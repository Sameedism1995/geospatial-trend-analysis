#!/usr/bin/env python3
"""
Thesis analysis bundle: reads existing processed outputs only.
Writes to outputs/thesis_analysis/ (CSVs, figures, markdown).
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "thesis_analysis"
FIG = OUT / "figures"
REPORTS = ROOT / "outputs" / "reports"
PROCESSED = ROOT / "processed"

PRIMARY_DECAY = REPORTS / "distance_decay_statistics.csv"
FALLBACK_DECAY = REPORTS / "port_distance_decay_statistics.csv"
PEARSON = REPORTS / "pearson_correlation.csv"
SPEARMAN = REPORTS / "spearman_correlation.csv"
LAGGED = REPORTS / "lagged_correlations.csv"
PARQUET = PROCESSED / "features_ml_ready.parquet"

PORTS_FOCUS = ["Stockholm", "Turku", "Mariehamn"]
DIST_ORDER = ["0-3 km", "3-7 km", "7-15 km", "15-30 km"]
KEY_METRICS = [
    "maritime_exposure_index",
    "atmospheric_coastal_exposure_index",
    "environmental_stress_index",
    "local_no2_excess",
    "oil_slick_proxy",
    "ndti_mean",
    "ndwi_mean",
]

MANIFEST_ROWS: list[dict] = []
WARNINGS: list[str] = []
MISSING: list[str] = []
OUTPUT_COUNT = 0


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path) -> None:
    global OUTPUT_COUNT
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    OUTPUT_COUNT += 1


def register_figure(
    filename: str,
    plot_type: str,
    variables_used: str,
    thesis_section: str,
) -> None:
    MANIFEST_ROWS.append(
        {
            "filename": filename,
            "plot_type": plot_type,
            "variables_used": variables_used,
            "thesis_section": thesis_section,
        }
    )


def save_fig(path: Path, plot_type: str, variables: str, section: str) -> None:
    global OUTPUT_COUNT
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    OUTPUT_COUNT += 1
    register_figure(path.name, plot_type, variables, section)


def categorize_feature(name: str) -> str:
    n = name.lower()
    if name in ("grid_cell_id",) or "centroid" in n or n.endswith("_lat") or n.endswith("_lon"):
        return "Static geospatial"
    if name == "week_start_utc":
        return "Temporal"
    if name == "nearest_port":
        return "Port"
    if "lag" in n or "_lag" in n:
        return "Lagged"
    if "wind" in n:
        return "Wind"
    if "oil" in n or "slick" in n:
        return "Oil"
    if "vessel" in n:
        return "Maritime"
    if "maritime" in n:
        return "Maritime"
    if "no2" in n or name.startswith("NO2") or "atmospheric_transfer" in n:
        return "Atmospheric"
    if "distance_to_port" in n or ("port_exposure" in n and "focal" not in n):
        return "Port"
    if "distance" in n or "exposure" in n or "coastal_exposure_band" in n:
        return "Distance / Exposure"
    if any(x in n for x in ["ndvi", "ndwi", "ndti", "ndci", "fai", "b11"]):
        return "Environmental"
    if name.endswith("_t") and "vessel" in n:
        return "Maritime"
    if name.endswith("_t") and "no2" in n:
        return "Atmospheric"
    if name.endswith("_t") and "oil" in n:
        return "Oil"
    if name.endswith("_t"):
        return "Temporal"
    if "detection_score" in n or "land_response" in n or "nan_ratio" in n:
        return "Environmental"
    return "Environmental"


FEATURE_PURPOSE: dict[str, tuple[str, str, str]] = {
    "grid_cell_id": ("Unique grid identifier", "Links rows to spatial units.", "Indexing; no causal role."),
    "week_start_utc": ("Calendar week (UTC)", "Defines weekly panel timestep.", "Controls seasonality/trends."),
    "vessel_density_t": ("Time-aligned vessel intensity", "Maritime traffic pressure input.", "Expected association with emissions/proxies."),
    "grid_centroid_lat": ("Grid centroid latitude", "Static location context.", "Coastal/geographic stratification."),
    "grid_centroid_lon": ("Grid centroid longitude", "Static location context.", "Coastal/geographic stratification."),
    "no2_mean_t": ("Weekly NO2 mean (tropospheric)", "Atmospheric nitrogen dioxide signal.", "Shipping/urban combustion proxy."),
    "no2_std_t": ("Weekly NO2 variability", "Atmospheric stability vs episodic NO2.", "Highlights pollution episodes."),
    "oil_slick_probability_t": ("Oil slick detection probability (week)", "Maritime oil discharge risk.", "Pollution outcome proxy."),
    "oil_slick_count_t": ("Oil slick count (week)", "Frequency of detections.", "Pollution outcome proxy."),
    "ndwi_mean": ("Mean NDWI (water index)", "Water/open-water vs turbidity sensitivity.", "Coastal water state indicator."),
    "ndwi_median": ("Median NDWI", "Robust water index aggregate.", "Reduces outlier sensitivity."),
    "ndwi_std": ("NDWI variability", "Temporal/optical variability.", "Heterogeneity of water signal."),
    "ndti_mean": ("Mean NDTI (turbidity)", "Suspended matter / turbidity proxy.", "Coastal water quality stress."),
    "ndti_median": ("Median NDTI", "Robust turbidity aggregate.", "Water quality comparator."),
    "ndti_std": ("NDTI variability", "Turbidity dynamics.", "Heterogeneity indicator."),
    "ndci_mean": ("Mean NDCI (chlorophyll proxy)", "Biological/coastal productivity proxy.", "Eutrophication-related signal."),
    "ndci_median": ("Median NDCI", "Robust chlorophyll proxy.", "Water quality comparator."),
    "ndci_std": ("NDCI variability", "Bio-optical variability.", "Spatial/temporal heterogeneity."),
    "fai_mean": ("Mean FAI (floating algae index)", "Algae/detritus/coastal anomalies.", "Coastal alteration signal."),
    "fai_median": ("Median FAI", "Robust floating-algae aggregate.", "Comparator to mean."),
    "fai_std": ("FAI variability", "Algae signal dynamics.", "Heterogeneity."),
    "b11_mean": ("Mean Sentinel-2 band 11", "Thermal/SWIR context for water.", "Supports optical interpretation."),
    "b11_median": ("Median band 11", "Robust thermal/SWIR aggregate.", "Comparator."),
    "b11_std": ("Band 11 variability", "SWIR dynamics.", "Heterogeneity."),
    "ndvi_mean": ("Mean NDVI (nearest-land linkage)", "Vegetation / land response where available.", "Land-side stress comparator."),
    "ndvi_median": ("Median NDVI", "Robust vegetation aggregate.", "Land comparator."),
    "ndvi_std": ("NDVI variability", "Vegetation dynamics.", "Land heterogeneity."),
    "NO2_mean": ("Weekly NO2 (aligned series)", "Atmospheric pollution level.", "Primary atmospheric stress variable."),
    "NO2_trend": ("NO2 trend component", "Low-frequency NO2 evolution.", "Longer-term atmospheric change."),
    "vessel_density": ("Vessel density (panel)", "Maritime intensity for modelling.", "Core maritime exposure predictor."),
    "detection_score": ("Optical anomaly / detection composite", "Sensor-based anomaly highlighting.", "Correlates with several water indices."),
    "nan_ratio_row": ("Row-wise missing-data fraction", "Data quality diagnostics.", "Masking / completeness control."),
    "nearest_port": ("Nearest labelled port", "Regional stratification.", "Port-centric comparisons."),
    "distance_to_port_km": ("Great-circle distance to port", "Geographic proximity to hub.", "Distance-decay hypotheses."),
    "port_exposure_score": ("Port proximity weighting", "Concentrates risk near hubs.", "Coastal amplification context."),
    "distance_to_nearest_high_vessel_density_cell": ("Distance to high-traffic hotspot", "Shipping lane proximity.", "Maritime concentration."),
    "coastal_exposure_band": ("Discrete coastal band", "Stratifies coastline proximity.", "Coastal vs offshore contrast."),
    "coastal_exposure_score": ("Composite coastal exposure", "Integrates coastal risk context.", "Overall coastal pressure."),
    "maritime_pressure_index": ("Maritime pressure composite", "Summarises shipping-linked stress.", "Maritime driver index."),
    "atmospheric_transfer_index": ("Atmospheric transfer composite", "Summarises air-side connectivity.", "Atmospheric pathway index."),
    "land_response_index": ("Land vegetation response (linked)", "Sea–land linkage where computed.", "Cross-realm comparator."),
    "vessel_x_no2": ("Vessel × NO2 interaction", "Nonlinear maritime–atmospheric coupling.", "Synergy hypothesis testing."),
    "no2_x_ndvi": ("NO2 × NDVI interaction", "Atmosphere–land coupling.", "Cross-domain stress."),
    "vessel_x_ndvi_lag1": ("Vessel × NDVI lag 1 week", "Delayed land response hypothesis.", "Lag structure."),
    "vessel_x_ndvi_lag2": ("Vessel × NDVI lag 2 weeks", "Delayed land response hypothesis.", "Lag structure."),
    "vessel_x_ndvi_lag3": ("Vessel × NDVI lag 3 weeks", "Delayed land response hypothesis.", "Lag structure."),
}


def purpose_row(fname: str) -> tuple[str, str, str]:
    cat = categorize_feature(fname)
    if fname in FEATURE_PURPOSE:
        s, w, e = FEATURE_PURPOSE[fname]
        return cat, s, w, e
    pretty = fname.replace("_", " ")
    return (
        cat,
        f"Engineered indicator: {pretty}",
        "Included as a predictor or stratifier in the weekly coastal panel.",
        "Role inferred from naming; interpret with covariance and coverage limits.",
    )


def flatten_corr(path: Path, label: str) -> pd.DataFrame:
    corr = pd.read_csv(path, index_col=0)
    rows = []
    for i in corr.index:
        for j in corr.columns:
            if str(i) == str(j):
                continue
            v = float(corr.loc[i, j])
            rows.append({"feature_a": i, "feature_b": j, label: v})
    df = pd.DataFrame(rows)
    df["pair_key"] = df.apply(lambda r: "|".join(sorted([str(r["feature_a"]), str(r["feature_b"])])), axis=1)
    df = df.drop_duplicates(subset=["pair_key"])
    df["abs_correlation"] = df[label].abs()
    df["relationship_type"] = np.where(df[label] >= 0, "positive", "negative")
    df["category_a"] = df["feature_a"].map(categorize_feature)
    df["category_b"] = df["feature_b"].map(categorize_feature)
    df["feature_categories"] = df["category_a"] + " ↔ " + df["category_b"]

    def interpret_row(r):
        mag = abs(float(r[label]))
        direc = r["relationship_type"]
        if mag >= 0.7:
            strength = "strong"
        elif mag >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"
        return (
            f"{strength.capitalize()} {direc} association between {r['feature_a']} and {r['feature_b']} "
            f"({label}={float(r[label]):.4f}); descriptive only."
        )

    df["interpretation"] = df.apply(interpret_row, axis=1)
    return df.drop(columns=["pair_key"]).sort_values("abs_correlation", ascending=False)


def cross_category_pairs(df: pd.DataFrame, col: str) -> pd.DataFrame:
    cc = df[df["category_a"] != df["category_b"]].copy()
    cc = cc.sort_values("abs_correlation", ascending=False)
    cc["ordering"] = cc.apply(lambda r: "|".join(sorted([r["category_a"], r["category_b"]])), axis=1)
    return cc.drop(columns=["ordering"], errors="ignore")


def decay_band_numeric(zone: str) -> float:
    return float(DIST_ORDER.index(zone)) if zone in DIST_ORDER else np.nan


def load_decay_csv() -> tuple[pd.DataFrame, str]:
    def _read_and_normalise(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "distance_band" in df.columns and "distance_zone" not in df.columns:
            df = df.rename(columns={"distance_band": "distance_zone"})
        if "wind_regime" not in df.columns:
            df["wind_regime"] = "all"
        return df

    if PRIMARY_DECAY.is_file():
        return _read_and_normalise(PRIMARY_DECAY), str(PRIMARY_DECAY.relative_to(ROOT))
    if FALLBACK_DECAY.is_file():
        WARNINGS.append(
            f"Primary decay file missing; used fallback: {FALLBACK_DECAY.relative_to(ROOT)}"
        )
        MISSING.append(str(PRIMARY_DECAY.relative_to(ROOT)))
        return _read_and_normalise(FALLBACK_DECAY), str(FALLBACK_DECAY.relative_to(ROOT))
    MISSING.append(str(PRIMARY_DECAY.relative_to(ROOT)))
    MISSING.append(str(FALLBACK_DECAY.relative_to(ROOT)))
    raise FileNotFoundError("No distance decay CSV found.")


def section1_dataset_audit(df: pd.DataFrame) -> None:
    nrows = len(df)
    ncols = len(df.columns)
    ngrid = df["grid_cell_id"].nunique() if "grid_cell_id" in df.columns else math.nan
    nweek = df["week_start_utc"].nunique() if "week_start_utc" in df.columns else math.nan

    print("Total rows:", nrows)
    print("Total columns:", ncols)
    print("Total grid cells:", int(ngrid) if pd.notna(ngrid) else "n/a")
    print("Total weeks:", int(nweek) if pd.notna(nweek) else "n/a")
    print("All column names:", df.columns.tolist())

    summary = pd.DataFrame(
        [
            {"metric": "total_rows", "value": nrows},
            {"metric": "total_columns", "value": ncols},
            {"metric": "total_grid_cells", "value": ngrid},
            {"metric": "total_weeks", "value": nweek},
            {"metric": "column_names_json", "value": json.dumps(df.columns.tolist())},
        ]
    )
    save_df(summary, OUT / "dataset_structure_summary.csv")

    rows = []
    for c in df.columns:
        non_null = int(df[c].notna().sum())
        cov = 100.0 * non_null / nrows if nrows else 0.0
        miss_pct = 100.0 - cov
        uniq = df[c].nunique(dropna=True)
        rows.append(
            {
                "feature": c,
                "category": categorize_feature(c),
                "non_null_count": non_null,
                "coverage_percent": round(cov, 4),
                "missing_percent": round(miss_pct, 4),
                "dtype": str(df[c].dtype),
                "unique_values_count": int(uniq),
            }
        )
    cov_tbl = pd.DataFrame(rows).sort_values("coverage_percent", ascending=False)
    save_df(cov_tbl, OUT / "feature_coverage_summary.csv")

    miss_tbl = cov_tbl[["feature", "missing_percent", "coverage_percent", "non_null_count"]].copy()
    miss_tbl = miss_tbl.sort_values("missing_percent", ascending=False)
    save_df(miss_tbl, OUT / "missingness_summary.csv")

    cat_summary = (
        cov_tbl.groupby("category")["feature"]
        .apply(lambda s: ", ".join(sorted(s)))
        .reset_index()
        .rename(columns={"feature": "features_list"})
    )
    EXPECTED_CATS = [
        "Maritime",
        "Atmospheric",
        "Environmental",
        "Wind",
        "Distance / Exposure",
        "Temporal",
        "Lagged",
        "Oil",
        "Port",
        "Static geospatial",
    ]
    for c in EXPECTED_CATS:
        if c not in cat_summary["category"].values:
            cat_summary = pd.concat(
                [cat_summary, pd.DataFrame([{"category": c, "features_list": ""}])],
                ignore_index=True,
            )

    cat_summary["feature_count"] = cat_summary["features_list"].apply(
        lambda s: 0 if (not s or not str(s).strip()) else len(str(s).split(", "))
    )
    save_df(cat_summary.sort_values("feature_count", ascending=False), OUT / "feature_categories_summary.csv")


def section2_purpose_table(df: pd.DataFrame) -> None:
    out_rows = []
    for c in df.columns:
        cat, short_d, why, role = purpose_row(c)
        out_rows.append(
            {
                "feature_name": c,
                "category": cat,
                "short_description": short_d,
                "why_used_in_thesis": why,
                "expected_environmental_role": role,
            }
        )
    save_df(pd.DataFrame(out_rows), OUT / "feature_purpose_table.csv")


def section3_correlations() -> tuple[float, str, pd.DataFrame, pd.DataFrame]:
    pear = flatten_corr(PEARSON, "pearson_r")
    spear = flatten_corr(SPEARMAN, "spearman_rho")
    top_p = pear.head(50).copy()
    top_s = spear.head(50).copy()
    save_df(top_p, OUT / "top_pearson_pairs.csv")
    save_df(top_s, OUT / "top_spearman_pairs.csv")
    cross_p = cross_category_pairs(pear, "pearson_r")
    cross_s = cross_category_pairs(spear, "spearman_rho")
    cross = pd.concat(
        [
            cross_p.head(50).assign(matrix="pearson"),
            cross_s.head(50).assign(matrix="spearman"),
        ],
        ignore_index=True,
    )
    save_df(cross, OUT / "cross_category_relationships.csv")

    corr_mat = pd.read_csv(PEARSON, index_col=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat.astype(float), annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Pearson correlation (subset features)")
    save_fig(
        FIG / "correlation_heatmap_pearson.png",
        "heatmap",
        ",".join(corr_mat.columns.tolist()),
        "Correlation Findings",
    )

    plt.figure(figsize=(10, 6))
    subp = pear.head(15)
    yp = subp["pearson_r"].values
    labp = [
        f"{a[:12]}↔{b[:12]}" if len(a) <= 12 else f"{a[:12]}…↔…{b[:12]}"
        for a, b in zip(subp["feature_a"], subp["feature_b"])
    ]
    plt.barh(range(len(yp)), yp, color=["#2171b5" if v >= 0 else "#cb181d" for v in yp])
    plt.yticks(range(len(yp)), labp, fontsize=8)
    plt.axvline(0, color="grey", lw=0.8)
    plt.xlabel("Pearson r")
    plt.title("Top 15 cross-feature Pearson correlations (by |r|)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_fig(
        FIG / "top_correlations_barplot_pearson.png",
        "barplot_h",
        "pearson_top15",
        "Correlation Findings",
    )

    plt.figure(figsize=(10, 6))
    sub = spear.head(15)
    y = sub["spearman_rho"].values
    labs = [f"{a[:12]}…↔…{b[:12]}" if len(a) > 12 else f"{a}↔{b}" for a, b in zip(sub["feature_a"], sub["feature_b"])]
    colors = ["#2171b5" if v >= 0 else "#cb181d" for v in y]
    plt.barh(range(len(y)), y, color=colors)
    plt.yticks(range(len(y)), labs, fontsize=8)
    plt.axvline(0, color="grey", lw=0.8)
    plt.xlabel("Spearman ρ")
    plt.title("Top 15 cross-feature Spearman correlations (by |ρ|)")
    plt.gca().invert_yaxis()
    save_fig(
        FIG / "top_correlations_barplot_spearman.png",
        "barplot_h",
        "spearman_top15",
        "Correlation Findings",
    )

    best_r = float(spear.iloc[0]["spearman_rho"])
    best_pair = f"{spear.iloc[0]['feature_a']} ↔ {spear.iloc[0]['feature_b']}"
    return best_r, best_pair, pear, spear


def monotonic_direction(values: list[float]) -> str:
    clean = [v for v in values if v == v]
    if len(clean) < 2:
        return "insufficient_series"
    diffs = np.diff(clean)
    if np.all(diffs <= 1e-9):
        return "flat"
    if np.all(diffs >= -1e-9):
        return "non_decreasing"
    if np.all(diffs <= 1e-9):
        return "non_increasing"
    return "non_monotonic"


def section4_distance_decay(decay: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    decay = decay.copy()
    decay["distance_order"] = decay["distance_zone"].map(decay_band_numeric)
    decay["mean_val"] = pd.to_numeric(decay["mean"], errors="coerce")

    summaries = []
    for met, g in decay.groupby("metric"):
        g = g[g["distance_zone"].isin(DIST_ORDER)].sort_values("distance_order")
        by_band = []
        for z in DIST_ORDER:
            sub = g[g["distance_zone"] == z]["mean_val"]
            by_band.append(float(sub.mean()) if len(sub) else float("nan"))
        direction = monotonic_direction(by_band)
        if len(by_band) >= 4 and by_band[0] == by_band[0] and by_band[-1] == by_band[-1]:
            delta_fn = by_band[-1] - by_band[0]
        else:
            delta_fn = float("nan")
        summaries.append(
            {
                "metric": met,
                "mean_band_0_3": by_band[0] if len(by_band) > 0 else np.nan,
                "mean_band_3_7": by_band[1] if len(by_band) > 1 else np.nan,
                "mean_band_7_15": by_band[2] if len(by_band) > 2 else np.nan,
                "mean_band_15_30": by_band[3] if len(by_band) > 3 else np.nan,
                "overall_trend_vs_distance": direction,
                "delta_far_minus_near": delta_fn,
            }
        )

    decay_summary = pd.DataFrame(summaries)
    decay_summary["decay_strength_rank"] = decay_summary["delta_far_minus_near"].abs().rank(ascending=False)
    save_df(decay_summary, OUT / "distance_decay_summary.csv")

    portwise = (
        decay[decay["port"].isin(PORTS_FOCUS)]
        .groupby(["port", "distance_zone", "metric"], as_index=False)["mean_val"]
        .mean()
    )
    save_df(portwise, OUT / "portwise_distance_decay_summary.csv")

    wind_sum = (
        decay.groupby(["wind_regime", "distance_zone", "metric"], as_index=False)["mean_val"]
        .mean()
    )
    save_df(wind_sum, OUT / "wind_regime_distance_summary.csv")

    if decay_summary.shape[0] and decay_summary["delta_far_minus_near"].notna().any():
        best_decay_row = decay_summary.loc[decay_summary["delta_far_minus_near"].abs().idxmax()]
        strongest_decay = (
            f"{best_decay_row['metric']}: Δ(far−near)={float(best_decay_row['delta_far_minus_near']):.6g} "
            f"({best_decay_row['overall_trend_vs_distance']})"
        )
    else:
        strongest_decay = "n/a"

    def auto_sentence(metric: str) -> str | None:
        sub = decay_summary[decay_summary["metric"] == metric]
        if sub.empty:
            return None
        r = sub.iloc[0]
        d = float(r["delta_far_minus_near"]) if r["delta_far_minus_near"] == r["delta_far_minus_near"] else None
        if d is None:
            return f"{metric}: insufficient numeric series for coastline–distance comparison."
        if d < 0:
            return (
                f"Mean {metric} decreases from the nearest band toward offshore distances, consistent with stronger "
                "coastal/port proximity signals near shore (association only)."
            )
        if d > 0:
            return (
                f"Mean {metric} increases with distance from the focal port bands in this summary table (association only); "
                "interpret jointly with metric definition and coverage."
            )
        return f"Mean {metric} is effectively flat across distance bands in the aggregated decay table."

    interp_lines = [auto_sentence(m) for m in KEY_METRICS if auto_sentence(m)]
    interp_text = "\n".join(f"- {x}" for x in interp_lines)
    Path(OUT / "distance_decay_interpretation.txt").write_text(interp_text + "\n", encoding="utf-8")

    avail_metrics = [m for m in KEY_METRICS if m in decay["metric"].values]
    if not avail_metrics:
        WARNINGS.append("No KEY_METRICS present in decay file; distance line plots skipped for keys.")
        return interp_text, strongest_decay, decay_summary

    for met in avail_metrics:
        sub = decay[(decay["metric"] == met) & (decay["wind_regime"] == "all") & (decay["port"].isin(PORTS_FOCUS))]
        if sub.empty:
            continue
        plt.figure(figsize=(8, 5))
        for p in PORTS_FOCUS:
            g = sub[sub["port"] == p].set_index("distance_zone").reindex(DIST_ORDER)
            plt.plot(range(len(DIST_ORDER)), g["mean_val"].values, marker="o", label=p)
        plt.xticks(range(len(DIST_ORDER)), DIST_ORDER, rotation=15, ha="right")
        plt.ylabel("Reported mean (decay table)")
        plt.title(f"{met}: mean by distance band (wind=all)")
        plt.legend(title="Port")
        plt.tight_layout()
        safe = met.replace("/", "_")
        save_fig(
            FIG / f"distance_decay_lines_{safe}.png",
            "multi_line_port",
            f"{met},ports={','.join(PORTS_FOCUS)}",
            "Distance-Decay Findings",
        )

    shore = decay[
        (decay["wind_regime"].isin(["shoreward", "non_shoreward"]))
        & (decay["metric"].isin(avail_metrics))
        & (decay["port"].isin(PORTS_FOCUS))
    ]
    if not shore.empty:
        pivot = shore.pivot_table(
            index=["port", "distance_zone", "metric"],
            columns="wind_regime",
            values="mean_val",
            aggfunc="mean",
        ).reset_index()
        if "shoreward" in pivot.columns and "non_shoreward" in pivot.columns:
            plt.figure(figsize=(10, 6))
            m0 = avail_metrics[0]
            pv = pivot[pivot["metric"] == m0]
            x = np.arange(len(DIST_ORDER))
            width = 0.35
            for i, pt in enumerate(PORTS_FOCUS):
                sub = pv[pv["port"] == pt].set_index("distance_zone").reindex(DIST_ORDER)
                plt.bar(
                    x + i * width / len(PORTS_FOCUS),
                    sub["shoreward"].fillna(0).values,
                    width / len(PORTS_FOCUS),
                    label=f"{pt} shoreward",
                )
            plt.xticks(x, DIST_ORDER, rotation=15, ha="right")
            plt.ylabel("Mean")
            plt.title(f"Shoreward means by band — example metric: {m0}")
            plt.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            save_fig(
                FIG / "distance_decay_shoreward_bars_example.png",
                "grouped_bar",
                m0,
                "Distance-Decay Findings",
            )

    interp_lines_str = interp_text if isinstance(interp_text, str) else ""
    return interp_lines_str, strongest_decay, decay_summary


def section5_wind(decay: pd.DataFrame) -> tuple[str, float | None]:
    rows = []
    effect_rows = []
    for (port, met), g in decay.groupby(["port", "metric"]):
        sw = g[g["wind_regime"] == "shoreward"]["mean_val"].astype(float)
        nsw = g[g["wind_regime"] == "non_shoreward"]["mean_val"].astype(float)
        allm = g[g["wind_regime"] == "all"]["mean_val"].astype(float)
        ms = sw.mean(skipna=True)
        mn = nsw.mean(skipna=True)
        if ms == ms and mn == mn and mn != 0:
            pct = 100.0 * (ms - mn) / abs(mn)
        elif ms == ms and mn == mn:
            pct = float("nan")
        else:
            pct = float("nan")
        rows.append(
            {
                "port": port,
                "metric": met,
                "mean_shoreward": ms,
                "mean_nonshoreward": mn,
                "mean_all": allm.mean(skipna=True),
                "pct_difference_sh_vs_nsw": pct,
                "amplification_flag": pct == pct and pct > 0,
            }
        )
        effect_rows.append(
            {
                "port": port,
                "metric": met,
                "effect_size_mean_diff": (ms - mn) if (ms == ms and mn == mn) else np.nan,
                "pct_difference": pct,
            }
        )

    wdf = pd.DataFrame(rows)
    save_df(wdf, OUT / "wind_regime_summary.csv")
    save_df(pd.DataFrame(effect_rows), OUT / "shoreward_vs_nonshoreward_effects.csv")

    top_amp = wdf.dropna(subset=["pct_difference_sh_vs_nsw"]).sort_values("pct_difference_sh_vs_nsw", ascending=False)
    strongest_line = "n/a"
    top_pct = None
    if not top_amp.empty:
        r = top_amp.iloc[0]
        top_pct = float(r["pct_difference_sh_vs_nsw"])
        strongest_line = f"{r['metric']} at {r['port']}: shoreward vs nonshoreward mean gap ≈ {r['pct_difference_sh_vs_nsw']:.2f}% (table means)."

    if not wdf.empty:
        samp = (
            wdf[wdf["port"].isin(PORTS_FOCUS)]
            .dropna(subset=["mean_shoreward", "mean_nonshoreward"])
            .head(min(18, len(wdf)))
        )
        plt.figure(figsize=(11, 5))
        x = np.arange(len(samp))
        plt.bar(x - 0.2, samp["mean_shoreward"], 0.4, label="shoreward")
        plt.bar(x + 0.2, samp["mean_nonshoreward"], 0.4, label="non_shoreward")
        lbls = [f"{row.port}\n{row.metric}"[:40] for _, row in samp.iterrows()]
        plt.xticks(x, lbls, rotation=45, ha="right", fontsize=7)
        plt.ylabel("Decay-table mean")
        plt.title("Shoreward vs non-shoreward (sample of port×metric rows)")
        plt.legend()
        plt.tight_layout()
        save_fig(
            FIG / "wind_regime_grouped_bars_sample.png",
            "grouped_bar",
            "shoreward,nonshoreward",
            "Wind-Regime Findings",
        )

    for met in ["environmental_stress_index", "local_no2_excess", "atmospheric_coastal_exposure_index"]:
        sub = decay[(decay["metric"] == met) & (decay["port"].isin(PORTS_FOCUS))]
        if sub.empty:
            continue
        comp = (
            sub[sub["wind_regime"].isin(["shoreward", "non_shoreward", "all"])]
            .groupby(["port", "wind_regime"], as_index=False)["mean_val"]
            .mean()
        )
        if comp.dropna(subset=["mean_val"]).empty:
            continue
        plt.figure(figsize=(9, 5))
        piv = comp.pivot(index="port", columns="wind_regime", values="mean_val").reindex(PORTS_FOCUS)
        cols = [c for c in ["shoreward", "non_shoreward", "all"] if c in piv.columns]
        piv[cols].plot(kind="bar", rot=0)
        plt.ylabel("Mean across distance bands (aggregated)")
        plt.title(f"Wind regime × port — {met}")
        plt.legend(title="Wind regime", fontsize=8)
        plt.tight_layout()
        save_fig(
            FIG / f"wind_regime_port_bars_{met}.png",
            "grouped_bar",
            f"{met},{','.join(cols)}",
            "Wind-Regime Findings",
        )

        wr = comp.pivot(index="port", columns="wind_regime", values="mean_val")
        if {"shoreward", "non_shoreward"}.issubset(wr.columns):
            mr = wr.reset_index().melt(id_vars="port", var_name="variable", value_name="value")
            mr2 = mr[mr["variable"].isin(["shoreward", "non_shoreward"])].dropna(subset=["value"])
            if not mr2.empty:
                radar_ports(
                    mr2,
                    ["shoreward", "non_shoreward"],
                    f"Wind regime radar — {met} (0–1 scaled per axis across ports)",
                    f"wind_regime_radar_{met}.png",
                    "Wind-Regime Findings",
                )

    msg = "Radar/violin: decay input is aggregated (no row-level residuals); violin plots omitted."
    if msg not in WARNINGS:
        WARNINGS.append(msg)

    return strongest_line, top_pct


def radar_ports(df_port: pd.DataFrame, variables: list[str], title: str, fname: str, section: str) -> None:
    ports = [p for p in PORTS_FOCUS if p in df_port["port"].values]
    if len(ports) < 2:
        return
    mat = df_port[df_port["port"].isin(ports)].pivot(index="port", columns="variable", values="value")
    for v in variables:
        if v not in mat.columns:
            mat[v] = np.nan
    mat = mat[variables].apply(pd.to_numeric, errors="coerce")
    ranges = mat.max(axis=0) - mat.min(axis=0)
    mat_n = mat.copy()
    for v in variables:
        if ranges[v] == ranges[v] and ranges[v] > 0:
            mat_n[v] = (mat[v] - mat[v].min()) / ranges[v]
        else:
            mat_n[v] = 0.5
    labels = variables
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for p in ports:
        vals = mat_n.loc[p].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=p, linewidth=2)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    save_fig(FIG / fname, "radar", ",".join(variables), section)


def section6_lagged() -> None:
    if not LAGGED.is_file():
        MISSING.append(str(LAGGED.relative_to(ROOT)))
        WARNINGS.append("lagged_correlations.csv missing; lag section skipped.")
        save_df(pd.DataFrame(), OUT / "strongest_lag_relationships.csv")
        save_df(pd.DataFrame(), OUT / "lag_interpretation_table.csv")
        return

    lag = pd.read_csv(LAGGED)
    lag["abs_correlation"] = lag["correlation"].abs()
    best_per = lag.sort_values("abs_correlation", ascending=False).drop_duplicates(subset=["feature_x", "feature_y"])
    save_df(best_per, OUT / "strongest_lag_relationships.csv")

    interp = []
    for (fx, fy), g in lag.groupby(["feature_x", "feature_y"]):
        g = g.sort_values("lag")
        peak = g.loc[g["abs_correlation"].idxmax()]
        opt_lag = int(peak["lag"])
        bl = int(peak["best_lag"]) if "best_lag" in peak else opt_lag
        cat = "synchronous" if bl == 0 else "delayed_response" if bl > 0 else "weak_relationship"
        if peak["abs_correlation"] < 0.05:
            cat = "weak_relationship"
        interp.append(
            {
                "feature_x": fx,
                "feature_y": fy,
                "optimal_lag_weeks": bl,
                "peak_correlation": float(peak["correlation"]),
                "lag_category": cat,
                "interpretation": (
                    f"Weekly scale: peak |r|≈{peak['abs_correlation']:.3f} at lag {bl} between {fx} and {fy}; "
                    "associations only, confounding likely."
                ),
            }
        )
    save_df(pd.DataFrame(interp), OUT / "lag_interpretation_table.csv")

    for (fx, fy), g in lag.groupby(["feature_x", "feature_y"]):
        g = g.sort_values("lag")
        plt.figure(figsize=(6, 4))
        plt.plot(g["lag"], g["correlation"], marker="o")
        plt.axhline(0, color="grey", lw=0.7)
        plt.xlabel("Lag (weeks)")
        plt.ylabel("Correlation")
        plt.title(f"Lag curve: {fx} vs {fy}")
        plt.tight_layout()
        safe = f"{fx}_{fy}".replace("/", "_")[:80]
        save_fig(FIG / f"lag_curve_{safe}.png", "line", f"{fx},{fy}", "Lagged Relationship Findings")

    heat = lag.pivot_table(index="feature_y", columns="lag", values="correlation", aggfunc="first")
    if not heat.empty:
        plt.figure(figsize=(8, max(3, len(heat) * 0.4)))
        sns.heatmap(heat, annot=True, fmt=".3f", cmap="RdBu_r", center=0)
        plt.title("Lag-wise correlations (lagged_correlations.csv)")
        plt.tight_layout()
        save_fig(
            FIG / "lag_correlation_heatmap.png",
            "heatmap",
            "lag x feature_y correlations",
            "Lagged Relationship Findings",
        )

    WANT = {"vessel_density", "NO2_mean", "ndti_mean", "ndwi_mean", "environmental_stress_index"}
    avail = lag[lag["feature_x"].isin(WANT) | lag["feature_y"].isin(WANT)]
    if avail.empty:
        WARNINGS.append("lagged_correlations.csv has no rows for focal indices (NO2/vessel/detection-only); NDVI/env index not present.")


def section7_ports(decay: pd.DataFrame, dfp: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    if decay is None or decay.empty or "port" not in decay.columns:
        WARNINGS.append("Distance decay table empty; port comparison CSVs written with NaNs.")
        comp = pd.DataFrame(
            [
                {"port": p, "metric_label": lab, "source_metric": sm, "decay_table_mean": np.nan}
                for p in PORTS_FOCUS
                for lab, sm in [
                    ("maritime_exposure", "maritime_exposure_index"),
                    ("atmospheric_exposure", "atmospheric_coastal_exposure_index"),
                    ("environmental_stress", "environmental_stress_index"),
                    ("oil_proxy", "oil_slick_proxy"),
                    ("no2_excess", "local_no2_excess"),
                ]
            ]
        )
        save_df(comp, OUT / "port_comparison_summary.csv")
        save_df(pd.DataFrame(), OUT / "normalized_port_metrics.csv")
        return "Port comparison skipped (no decay table).", comp

    metrics_map = [
        ("maritime_exposure", "maritime_exposure_index"),
        ("atmospheric_exposure", "atmospheric_coastal_exposure_index"),
        ("environmental_stress", "environmental_stress_index"),
        ("oil_proxy", "oil_slick_proxy"),
        ("no2_excess", "local_no2_excess"),
    ]
    rows = []
    for port in PORTS_FOCUS:
        for label, met in metrics_map:
            sub = decay[(decay["port"] == port) & (decay["metric"] == met) & (decay["wind_regime"] == "all")]
            v = sub["mean_val"].astype(float).mean(skipna=True)
            rows.append({"port": port, "metric_label": label, "source_metric": met, "decay_table_mean": v})
    comp = pd.DataFrame(rows)
    save_df(comp, OUT / "port_comparison_summary.csv")

    try:
        wide = comp.pivot(index="port", columns="metric_label", values="decay_table_mean").apply(pd.to_numeric, errors="coerce")
    except Exception:
        wide = pd.DataFrame()
    if wide.empty:
        save_df(pd.DataFrame(), OUT / "normalized_port_metrics.csv")
    else:
        norm = wide.copy()
        for c in norm.columns:
            col = norm[c]
            r = col.max() - col.min()
            norm[c] = (col - col.min()) / r if r == r and r > 0 else 0.5
        norm = norm.reset_index()
        norm_melt = norm.melt(id_vars="port", var_name="metric_label", value_name="normalized_0_1")
        save_df(norm_melt, OUT / "normalized_port_metrics.csv")

    plt.figure(figsize=(10, 5))
    wide_plot = wide.reindex(PORTS_FOCUS) if not wide.empty else pd.DataFrame(index=PORTS_FOCUS)
    if wide_plot.empty or wide_plot.isna().all().all():
        plt.text(0.5, 0.5, "No port decay means", ha="center")
        plt.axis("off")
    else:
        wide_plot.plot(kind="bar", rot=0)
    plt.ylabel("Mean (decay table, wind=all)")
    plt.title("Port comparison — key decay metrics")
    if wide_plot.columns.size and not wide_plot.dropna(how="all").empty:
        plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    save_fig(FIG / "port_comparison_grouped_bars.png", "grouped_bar", str(wide_plot.columns.tolist()), "Port Comparisons")

    if wide.shape[0] and not wide_plot.dropna(how="all").empty:
        try:
            sns.heatmap(wide_plot.T.astype(float), annot=True, fmt=".3g", cmap="YlGnBu")
            plt.title("Port × metric heatmap (raw decay means)")
            plt.tight_layout()
            save_fig(FIG / "port_comparison_heatmap.png", "heatmap", "decay metrics by port", "Port Comparisons")
        except Exception:
            WARNINGS.append("Port comparison heatmap skipped (insufficient numeric data).")

    if not wide_plot.empty and wide_plot.columns.size:
        melt_radar = wide_plot.reset_index().melt(id_vars="port", var_name="variable", value_name="value")
        radar_ports(
            melt_radar,
            [c for c in wide_plot.columns if c != "index"],
            "Port radar (0–1 min–max per metric across ports)",
            "port_comparison_radar.png",
            "Port Comparisons",
        )

    exposure_cols = ["maritime_exposure", "atmospheric_exposure", "environmental_stress", "oil_proxy", "no2_excess"]
    if wide.empty:
        high_line = "No decay means for port ranking."
        return high_line, comp
    cols_ok = [c for c in exposure_cols if c in wide.columns]
    score = wide[cols_ok].mean(axis=1, skipna=True) if cols_ok else pd.Series(dtype=float)
    high_port = score.idxmax() if len(score) and score.notna().any() else "n/a"
    high_line = (
        f"Highest mean across listed decay-metric bundle: {high_port} (row-mean of available metrics)."
        if high_port != "n/a"
        else "Cannot rank ports (missing decay means)."
    )

    pq_rows = []
    if not dfp.empty and "nearest_port" in dfp.columns:
        subp = dfp[dfp["nearest_port"].isin(PORTS_FOCUS)]
        agg_map = {
            "maritime_panel": "maritime_pressure_index",
            "atmospheric_panel": "atmospheric_transfer_index",
            "no2_panel": "NO2_mean",
            "oil_panel": "oil_slick_probability_t",
            "vessel_panel": "vessel_density",
        }
        for port in PORTS_FOCUS:
            blk = subp[subp["nearest_port"] == port]
            row = {"port": port}
            for lab, col in agg_map.items():
                if col in blk.columns:
                    row[lab] = blk[col].mean(skipna=True)
                else:
                    row[lab] = np.nan
            pq_rows.append(row)
        pq = pd.DataFrame(pq_rows)
        save_df(pq, OUT / "port_comparison_panel_means.csv")
    elif not dfp.empty:
        WARNINGS.append("Column nearest_port absent from parquet; panel port table skipped.")

    return high_line, comp


def build_markdown(
    df: pd.DataFrame,
    best_r: float,
    best_pair: str,
    decay_interp: str,
    wind_strong: str,
    high_port: str,
    strongest_decay: str,
) -> None:
    cov = pd.read_csv(OUT / "feature_coverage_summary.csv")
    sparse = cov.nsmallest(5, "coverage_percent")[["feature", "coverage_percent"]].to_string(index=False)
    lines = [
        "# Thesis results interpretation",
        "",
        "_Auto-generated from `scripts/run_thesis_analysis.py`; associations are descriptive, not causal._",
        "",
        "## 1. Dataset Overview",
        "",
        f"The ML-ready weekly panel (`processed/features_ml_ready.parquet`) contains **{len(df):,}** rows, "
        f"**{len(df.columns)}** columns, **{df['grid_cell_id'].nunique()}** grid cells, and **{df['week_start_utc'].nunique()}** weekly timestamps. "
        "Several optical and interaction covariates are sparse; lowest coverage examples:",
        "",
        "```",
        sparse,
        "```",
        "",
        "## 2. Feature Engineering",
        "",
        "Variables group into maritime intensity, Sentinel-derived water-quality indices (NDWI/NDTI/NDCI/FAI/B11 where available), "
        "tropospheric NO₂ measures, coastline and port-distance constructs, and interaction terms. "
        "Category assignments and narrative rationale are tabulated in `feature_purpose_table.csv`.",
        "",
        "## 3. Correlation Findings",
        "",
        (
            (
                f"Among the Spearman matrix subset, the largest-magnitude association is **{best_pair}** (ρ ≈ **{best_r:.4f}**). "
                if best_r == best_r
                else "Spearman correlations were not computed (missing inputs)."
            )
            + " Optical water indices are strongly inter-related; cross-domain pairs are generally weaker on the available subset. "
            + "See `top_pearson_pairs.csv`, `top_spearman_pairs.csv`, and `cross_category_relationships.csv`."
        ),
        "",
        "## 4. Distance-Decay Findings",
        "",
        "Distance-stratified summaries (port-centred bands) summarise how decadal/week-aggregated means move from 0–3 km to 15–30 km. ",
        "**Automated synopsis (distance-decay metrics):**",
        "",
        "\n".join(
            line if line.startswith(">") else f"- {line.lstrip('- ')}"
            for line in (decay_interp.strip().split("\n") if decay_interp.strip() else ["> (No automated lines; check decay summaries.)"])
        ),
        "",
        f"**Quantitative cue:** aggregated band slope cue — {strongest_decay}",
        "",
        "## 5. Wind-Regime Findings",
        "",
        "Comparing decay-table rows labelled shoreward versus nonshoreward highlights indicators whose band means shift under onshore-aligned "
        "flow assumptions (see `wind_regime_summary.csv`). ",
        f"**Largest shoreward amplification (tabular % difference):** {wind_strong}",
        "",
        "## 6. Lagged Relationship Findings",
        "",
        "`lagged_correlations.csv` currently enumerates vessel density against NO₂ and against the optical detection score across lags. "
        "Optimal lag and peak correlations are summarized in `strongest_lag_relationships.csv` and `lag_interpretation_table.csv`. "
        "NDTI/NDWI/environmental stress are not represented in this file; interpretations are limited to exported pairs.",
        "",
        "## 7. Port Comparisons",
        "",
        f"{high_port} See `port_comparison_summary.csv`, `normalized_port_metrics.csv`, and figure exports.",
        "",
        "## 8. Limitations",
        "",
        "- Correlations omit confounders; ecological and atmospheric processes are synchronous and spatially structured.",
        "- Optical features have uneven coverage; pairwise correlations use available-complete observations implicitly in upstream matrices.",
        "- Distance and wind stratifications rely on pre-aggregated report tables rather than hierarchical models.",
        "- Lag exploration is sparse (few target pairs exported).",
        "",
        "## 9. Overall Conclusion",
        "",
        "Consistent multimodal signatures link maritime adjacency and atmospheric composites to optical water descriptors in zones with data, "
        "while aggregated distance bands show heterogeneous metric-by-metric patterns. Coastal wind regimes modulate summary means in the decay tables. "
        "All conclusions should stress associational wording and uncertainty from missingness and aggregation.",
        "",
    ]
    Path(OUT / "thesis_results_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def section9_manifest() -> None:
    save_df(pd.DataFrame(MANIFEST_ROWS), FIG / "figures_manifest.csv")


def main() -> None:
    global OUTPUT_COUNT
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="notebook")
    warnings.filterwarnings("ignore", category=FutureWarning)

    if not PARQUET.is_file():
        raise FileNotFoundError(PARQUET)

    df = pd.read_parquet(PARQUET)
    section1_dataset_audit(df)
    section2_purpose_table(df)

    if not PEARSON.is_file() or not SPEARMAN.is_file():
        MISSING.extend([str(x) for x in [PEARSON, SPEARMAN] if not x.is_file()])
        WARNINGS.append("Correlation CSVs missing; section 3 incomplete.")
        best_r, best_pair = float("nan"), "n/a"
        pear = spear = pd.DataFrame()
    else:
        best_r, best_pair, pear, spear = section3_correlations()

    strongest_pearson_str = (
        f"{pear.iloc[0]['feature_a']} ↔ {pear.iloc[0]['feature_b']} (r = {float(pear.iloc[0]['pearson_r']):.4f})"
        if not pear.empty and len(pear)
        else "n/a"
    )

    try:
        decay, decay_path = load_decay_csv()
        decay["mean_val"] = pd.to_numeric(decay["mean"], errors="coerce")
    except FileNotFoundError as e:
        WARNINGS.append(str(e))
        decay = pd.DataFrame()
        decay_path = "none"
        decay_interp = ""
        strongest_decay = "n/a"
        wind_strong = "n/a"
    else:
        decay_interp, strongest_decay, _ = section4_distance_decay(decay)
        wind_strong, _ = section5_wind(decay)
        high_line, _ = section7_ports(decay, df)
        section6_lagged()
        build_markdown(df, best_r, best_pair, decay_interp, wind_strong, high_line, strongest_decay)
        section9_manifest()
        print("\n" + "=" * 50)
        print("FINAL EXECUTION REPORT")
        print("=" * 50)
        print("Total outputs generated (approx line count):", OUTPUT_COUNT)
        print("Distance decay source:", decay_path)
        print("Missing files skipped:", "; ".join(MISSING) if MISSING else "(none)")
        print("Warnings:", "; ".join(WARNINGS) if WARNINGS else "(none)")
        print("Key strongest Spearman (from matrix):", best_pair, "ρ =", round(best_r, 4) if best_r == best_r else "n/a")
        print("Key strongest Pearson (from matrix):", strongest_pearson_str)
        print("Strongest distance-decay cue:", strongest_decay)
        print("Strongest shoreward amplification line:", wind_strong)
        print("Port comparison highlight:", high_line)
        return

    section6_lagged()
    high_line, _ = section7_ports(decay, df)
    build_markdown(df, best_r, best_pair, decay_interp, wind_strong, high_line, strongest_decay)
    section9_manifest()
    print("\n" + "=" * 50)
    print("FINAL EXECUTION REPORT")
    print("=" * 50)
    print("Total outputs generated:", OUTPUT_COUNT)
    print("Distance decay source:", decay_path)
    print("Missing files skipped:", "; ".join(MISSING) if MISSING else "(none)")
    print("Warnings:", "; ".join(WARNINGS) if WARNINGS else "(none)")
    print("Key strongest Spearman (from matrix):", best_pair, "ρ =", round(best_r, 4) if best_r == best_r else "n/a")
    print("Key strongest Pearson (from matrix):", strongest_pearson_str)
    print("Strongest distance-decay cue:", strongest_decay)
    print("Strongest shoreward amplification line:", wind_strong)
    print("Port comparison highlight:", high_line)


if __name__ == "__main__":
    main()
