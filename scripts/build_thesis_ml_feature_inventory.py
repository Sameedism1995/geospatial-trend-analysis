#!/usr/bin/env python3
"""
Build thesis feature inventory from the actual ML parquet + run_delta_ndti_models META_COLS.
Writes outputs/thesis/feature_inventory.csv and feature_inventory.md
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "modeling_dataset.parquet"
MODEL_RESULTS = ROOT / "data" / "model_results.json"
META_COLS = {"grid_cell_id", "week_start_utc", "delta_ndti", "has_valid_delta_ndti", "ndti_next_target"}
OUT_DIR = ROOT / "outputs" / "thesis"

PURPOSE: dict[str, str] = {
    "week_of_year": "Ordinal week-of-year cyclical index for seasonality.",
    "week_sin": "Sinusoidal seasonal encoding derived from UTC week anchor.",
    "week_cos": "Cosine seasonal encoding (pairs with week_sin for smooth periodicity).",
    "grid_res_deg": "Nominal spatial resolution / grid spacing of the modelling cells (degrees).",
    "grid_centroid_lat": "Static centroid latitude linking each grid cell on the Earth's surface.",
    "grid_centroid_lon": "Static centroid longitude for geographical heterogeneity.",
    "vessel_density_t": "EMODnet-derived aggregate vessel-density signal at calendar week *t* (maritime exposure proxy).",
    "vessel_density_t_minus_1": "Prior-week vessel density (1-week maritime lag).",
    "vessel_density_t_minus_2": "Two-week-lagged maritime intensity.",
    "sentinel_ndvi_mean_t": "Sentinel-2 NDVI spatial mean inside the footprint at week *t* (vegetation/coastal fringe context).",
    "sentinel_ndvi_mean_t_minus_1": "NDVI lagged one week.",
    "sentinel_ndvi_mean_t_minus_2": "NDVI lagged two weeks.",
    "sentinel_ndwi_mean_t": "NDWI spatial mean — open water / inundation-sensitive index.",
    "sentinel_ndwi_mean_t_minus_1": "NDWI lag 1.",
    "sentinel_ndwi_mean_t_minus_2": "NDWI lag 2.",
    "sentinel_evi_mean_t": "Enhanced vegetation index summarising canopy structure / productivity.",
    "sentinel_evi_mean_t_minus_1": "EVI lag 1.",
    "sentinel_evi_mean_t_minus_2": "EVI lag 2.",
    "sentinel_ndti_mean_t": "NDTI turbidity surrogate at week *t* — primary optical water-state covariate (also participates in ΔNDTI target logic upstream).",
    "sentinel_ndti_mean_t_minus_1": "NDTI at *t−1;* central for short-term inertia in water optical properties.",
    "sentinel_ndti_mean_t_minus_2": "NDTI at *t−2* for smoother temporal context.",
    "sentinel_observation_count_t": "Number of Sentinel-2 scenes contributing to composites (observational weight / completeness).",
    "has_sentinel": "Indicator that Sentinel-2-based optical stack was populated for this row.",
    "has_emodnet": "Indicator that EMODnet vessel-density lineage contributed to integration.",
    "has_helcom": "Indicator (metadata lineage) referencing HELCOM-related harmonisation pathway.",
}


def categorize(name: str) -> str:
    n = name.lower()
    if n in {"week_of_year", "week_sin", "week_cos"}:
        return "Temporal"
    if n in {"grid_centroid_lat", "grid_centroid_lon", "grid_res_deg"}:
        return "Spatial"
    if "_minus_" in n:
        return "Lagged"
    if n.startswith("vessel_density"):
        return "Maritime"
    if n.startswith("has_"):
        return "Engineered exposure features"
    if n.startswith("sentinel_"):
        return "Environmental"
    return "Environmental"


def ml_feature_columns(columns: pd.Index) -> list[str]:
    return sorted(c for c in columns if c not in META_COLS)


def markdown_escape(cell: str) -> str:
    return cell.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PARQUET)

    feats = ml_feature_columns(df.columns)
    rows = []
    for f in feats:
        rows.append(
            {
                "feature_name": f,
                "feature_category": categorize(f),
                "short_purpose": PURPOSE.get(f, "See thesis pipeline documentation."),
                "used_in_ridge_x": True,
                "used_in_hist_gradient_boosting_x": True,
                "is_lagged_autoregressive": "_minus_" in f,
                "is_static_spatial_grid": f in {"grid_centroid_lat", "grid_centroid_lon", "grid_res_deg"},
                "is_exposure_context": f.startswith("vessel_density") or f.startswith("has_"),
            }
        )

    inv = pd.DataFrame(rows)

    extras = pd.DataFrame(
        [
            {
                "feature_name": "delta_ndti",
                "role": "target_primary",
                "notes": "NDTI(t+1) − NDTI(t); used as *y* in run_delta_ndti_models (ΔNDTI task). Never part of ridge/HGB input matrix.",
            },
            {
                "feature_name": "ndti_next_target",
                "role": "target_secondary_constructed",
                "notes": "Built at runtime as groupby-shift of sentinel_ndti_mean_t; second supervised task compares level vs change with identical *X*.",
            },
            {
                "feature_name": "grid_cell_id",
                "role": "excluded_predictor_panel_key",
                "notes": "Panel unit key; withheld from ridge/HGB.",
            },
            {
                "feature_name": "week_start_utc",
                "role": "excluded_predictor_panel_key",
                "notes": "Temporal index for time-aware split; withheld from ridge/HGB (encoded instead via seasonal features).",
            },
            {
                "feature_name": "has_valid_delta_ndti",
                "role": "row_filter_helper",
                "notes": "Flags rows where ΔNDTI is defined; filtering only — not fed to ML.",
            },
        ]
    )

    counts = Counter(inv["feature_category"])
    ridge_n = len(feats)

    model_meta: dict | None = None
    if MODEL_RESULTS.is_file():
        with MODEL_RESULTS.open(encoding="utf-8") as f:
            model_meta = json.load(f)

    csv_path = OUT_DIR / "feature_inventory.csv"
    md_path = OUT_DIR / "feature_inventory.md"

    inv.to_csv(csv_path, index=False)

    static_spatial = [f for f in feats if f in {"grid_centroid_lat", "grid_centroid_lon", "grid_res_deg"}]
    lagged_pred = [f for f in feats if "_minus_" in f]
    exposure_like = sorted(
        set(f for f in feats if f.startswith("vessel_density") or f.startswith("has_"))
    )

    md_lines = [
        "# ML feature inventory (thesis-ready)",
        "",
        "## Dataset and supervised setup",
        "",
        f"- **Parquet analysed:** `{PARQUET.relative_to(ROOT)}`",
        f"- **Row / column geometry:** `{len(df):,}` rows × `{len(df.columns)}` columns in file.",
        f"- **ML driver script:** `src/run_delta_ndti_models.py` (default `--input {Path('data/modeling_dataset.parquet').as_posix()}`)",
        "",
        "Predictors are every column except the metadata/target set:",
        "",
        f"`{sorted(META_COLS)}`",
        "",
        "- **ΔNDTI target:** `delta_ndti`",
        "- **Secondary target:** `ndti_next_target = shift(sentinel_ndti_mean_t, −1)` by `grid_cell_id` (constructed after filters in `load_modeling_data`).",
        "",
        "**Ridge regression pipeline:** `Pipeline(SimpleImputer(median), StandardScaler(), Ridge(alpha=1.0))` — Ridge receives the **same** numeric feature columns as HistGradientBoosting (median imputation is internal to Ridge only).",
        "",
        "**HistGradientBoostingRegressor:** All listed predictors fed natively with missing-value handling (`nan` retained for tree splits). Early stopping uses a withheld train fraction.",
        "",
        f"- **Count of predictor columns |X|:** {ridge_n} (identical for Ridge & HistGradientBoosting).",
        "",
        "## Feature counts by category (predictors in *X*)",
        "",
        "| Category | Count |",
        "|----------|-------|",
    ]
    canonical_order = [
        "Environmental",
        "Atmospheric",
        "Maritime",
        "Spatial",
        "Temporal",
        "Lagged",
        "Engineered exposure features",
    ]
    for cat in canonical_order:
        md_lines.append(f"| {cat} | {counts.get(cat, 0)} |")

    md_lines.extend(
        [
            "",
            "## Identifier / target rows (never in ridge/HGB)",
            "",
            "| Name | Role | Notes |",
            "|------|------|-------|",
        ]
    )
    for _, r in extras.iterrows():
        md_lines.append(
            f"| `{r['feature_name']}` | {markdown_escape(str(r['role']))} | {markdown_escape(str(r['notes']))} |"
        )

    md_lines.extend(
        [
            "",
            "## Predictor inventory (used by both Ridge & HistGradientBoosting)",
            "",
            "| Feature | Category | Purpose (short) | Ridge X | HistGB X | Lag | Static geo | Exposure ctx |",
            "|---------|----------|-----------------|---------|----------|-----|------------|--------------|",
        ]
    )

    for _, r in inv.iterrows():
        md_lines.append(
            "| `{0}` | {1} | {2} | yes | yes | {3} | {4} | {5} |".format(
                r["feature_name"],
                markdown_escape(str(r["feature_category"])),
                markdown_escape(str(r["short_purpose"])),
                "yes" if r["is_lagged_autoregressive"] else "no",
                "yes" if r["is_static_spatial_grid"] else "no",
                "yes" if r["is_exposure_context"] else "no",
            )
        )

    md_lines.extend(
        [
            "",
            "## Thesis-oriented flags",
            "",
            f"- **Lagged predictor columns ({len(lagged_pred)}):** `{', '.join(lagged_pred)}`",
            f"- **Static spatial predictors:** `{', '.join(static_spatial)}`",
            f"- **Maritime intensity + lineage flags grouped as operational exposure context:** `{', '.join(exposure_like)}`",
            "",
            "### Atmospheric chemistry",
            "",
            "**None** of the current predictor columns encode tropospheric NO₂ / chemistry; this modelling parquet carries optical + maritime + temporal encodings only.",
            "",
            "## Thesis-ready methodological summary",
            "",
            "### Why these features were selected",
            "",
            "**Optical coherence:** NDVI / NDWI / EVI / NDTI and their lags summarise land–water contrasts, turbidity, and inertia in Sentinel-2 composites while respecting the embargo that ΔNDTI is built from contiguous valid weeks.",
            "**Maritime stress:** Weekly vessel-density composites (with lags) capture industrial shipping pressure hypothesised to co-vary with nearshore perturbations.",
            "**Seasonality:** Harmonic (`week_sin` / `week_cos`) and ordinal `week_of_year` approximate phenology-driven illumination and runoff cycles without handing the model absolute timestamps (which remain reserved for the split).",
            "**Static geography:** Latitude, longitude (and nominal resolution) soak up unresolved spatial stratification omitted from causal covariates.",
            "**Instrumentation hygiene:** Observation counts plus source flags tell the estimator how much independent radiometric evidence existed and whether upstream ingestion paths fired — especially relevant when Ridge relies on deterministic imputation before scaling.",
            "",
            "### Exposure analysis alignment",
            "",
            "Environmental exposure proxies here are multispectral composites interpreted as ecological–physical state (turbidity, vegetation/fringe moisture, generalized productivity). Maritime exposure proxies are summed vessel-density artefacts at the modelling grid centroid. Temporal encodings approximate seasonal forcing on water colour and illuminate regime shifts absent explicit synoptic meteorology.",
            "",
            "### Temporal and spatial dependence",
            "",
            "- **Temporal:** Dedicated lag columns (optical indices and vessel intensity at one- and two-week offsets) propagate short-term memory in water radiometry and activity; cyclic week-of-year harmonics summarise annual illumination and seasonal forcing absent raw timestamps.",
            "- **Spatial dependence:** Latitude/longitude summarise location on the Baltic domain; finer spatial interactions are absorbed non-parametrically by HistGradientBoosting and linear-smoothed (after scaling) by Ridge. Correlation structures across neighbouring cells remain unmodelled (i.i.d.-style rows conditioned on engineered inputs). Panel keys (`grid_cell_id`, `week_start_utc`) constrain train/test leakage through the scripted week-split, not via random row shuffles.",
            "",
        ]
    )

    if model_meta:
        md_lines.extend(
            [
                "## Calibration snapshot (`data/model_results.json`)",
                "",
                "*Use as reporting aid; rerun `python3 src/run_delta_ndti_models.py` after data refresh.*",
                "",
                "- **Objective recorded:** {}".format(markdown_escape(str(model_meta.get("objective", "")))[:280]),
                f"- **Usable labelled rows cited:** `{model_meta.get('n_rows_valid_target')}`",
                "",
            ]
        )

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(csv_path.relative_to(ROOT))
    print(md_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
