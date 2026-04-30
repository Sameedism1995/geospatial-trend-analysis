# Run snapshot — `run_20260430_070441`

This folder contains a complete, self-contained snapshot of the entire project
pipeline executed on 2026-04-30. Every script that produces results was run
against `data/modeling_dataset.parquet` (16 065 rows × 29 cols, weekly grid
2023-01-01 → 2023-12-17) and the cached aux parquets in `data/aux/`.

## How it was run

```bash
# 1) Full multi-source pipeline with every analytical stage enabled
python src/pipeline/run_full_pipeline.py \
    --feature-interaction-map \
    --scientific-validation \
    --anomaly-detection \
    --coastal-impact-score \
    --final-visualization \
    --land-impact

# 2) Delta-NDTI time-aware modeling
python src/run_delta_ndti_models.py \
    --input data/modeling_dataset.parquet \
    --results <run>/modeling/delta_ndti/model_results.json \
    --predictions <run>/modeling/delta_ndti/predictions.parquet \
    --plots-dir <run>/modeling/delta_ndti/plots

# 3) Modeling-dataset EDA visualizations
MPLBACKEND=Agg python src/visualize_dataset.py \
    --input data/modeling_dataset.parquet \
    --out-dir <run>/visualizations/dataset --no-show --no-open

# 4) Interactive Plotly maps
python src/visualize_global_maps.py \
    --input data/modeling_dataset.parquet \
    --out-dir <run>/visualizations/global_maps --only all

# 5) Human-impact / distance / regime analysis
python src/human_impact_distance_analysis.py \
    --input data/modeling_dataset.parquet \
    --out-data-dir <run>/human_impact/data \
    --viz-dir <run>/human_impact/visualizations \
    --model-results <run>/human_impact/model_results.json \
    --enriched-parquet <run>/human_impact/data/modeling_dataset_human_impact.parquet

# 6) Master-dataset QA
python src/master_dataset_qa_report.py \
    --input data/master_dataset.parquet \
    --output <run>/reports/master_dataset_qa_report.json

# 7) Legacy EDA pipeline (sample CSVs in data/)
python src/eda_pipeline.py
```

All console output for each command was tee'd into `logs/`.

## Folder layout

```
plots/                   # main pipeline plots (correlations, feature interactions,
                         #   lag response curves, anomaly map, coastal impact map,
                         #   final environmental pressure map)
reports/                 # all CSV / JSON analytical reports + master QA
eda/                     # eda_stats.json + per-feature distribution PNGs
previews/                # per-source histograms, weekly time series, spatial proxies
modeling/delta_ndti/     # delta_ndti & ndti_next regression results, plots, predictions
human_impact/            # distance / decay analysis, NO2 + oil overlays, regime plots
visualizations/dataset/  # NDTI distribution / heatmap / time series, vessel density map
visualizations/global_maps/  # Plotly HTML interactive maps
logs/                    # stdout for every script + pipeline_run.log
merged_dataset.parquet   # merged source-aligned weekly grid dataset
features_ml_ready.parquet  # ML-ready feature table (43 cols, 16065 rows)
RUN_MANIFEST.json        # machine-readable manifest of this run
```

## Headline results

### Strongest cross-domain Spearman correlations

| pair | spearman |
|---|---|
| ndwi_mean vs ndci_mean | -0.876 |
| ndwi_mean vs b11_mean  | -0.842 |
| ndci_mean vs fai_mean  |  0.796 |
| ndwi_mean vs fai_mean  | -0.789 |
| ndci_mean vs b11_mean  |  0.728 |
| ndwi_mean vs ndti_mean | -0.715 |

(Source: `reports/spearman_correlation.csv`, also visualised in
`plots/correlations/correlation_heatmap.png`.)

### Land-sea lag summary

| pair | best_lag_weeks | spearman | label |
|---|---|---|---|
| NO2_mean → ndvi_mean        | 4 | 0.66 | strong |
| vessel_density → ndvi_mean  | 1 | 0.36 | moderate |

(Source: `reports/land_sea_lag_summary.csv`.)

### Land-impact ML (Random Forest, permutation importance)

* primary model: `r²_test = 0.845`, `r²_cv = 0.677 ± 0.203` (n = 96, 12 features)
* leakage-ceiling: `r²_test = 0.821`, `r²_cv = 0.752 ± 0.208` (n = 96, 17 features)

### Top coastal-impact zones

`g0.100_r1503_c2022 (60.35°N, 22.25°E)` — week 2023-04-09 — score = 0.622
(see `reports/coastal_impact_score.csv`).

### Top spatial-outlier grid-week events

`g0.100_r1503_c2022` 2023-03-05, anomaly = 2.13
(see `reports/anomaly_scores.csv` and `reports/anomaly_scores_temporal.csv`).

## Pipeline-stage status

All stages from `run_full_pipeline.py` completed without error:

* `[SOURCE: vessels]`            16065 rows, 95.2 % non-null
* `[SOURCE: no2]`                16065 rows, 94.9 % non-null
* `[SOURCE: sentinel1]`          16065 rows, 84.6 % non-null
* `[SOURCE: sentinel2_water_quality]` 16065 rows, 27.9 % non-null
* `[SOURCE: land_impact_ndvi]`   16065 rows, 41.4 % non-null
* `[MERGE]` 16065 rows × 32 cols
* `[FEATURES]` enriched to 43 cols (LAND IMPACT extension applied)
* `[VALIDATION]` `data/validation/full_pipeline_validation_report.json`
* `[EDA]` `outputs/eda/eda_stats.json` + `outputs/eda_summary.md`
* `[CORRELATION]` Pearson + Spearman + per-window evaluation written
* `[FEATURE_INTERACTION]` ranked CSV + interaction map PNG
* `[SCIENTIFIC VALIDATION]` lag analysis, temporal stability, baseline FI
* `[ANOMALY]` 1286 spatial-outlier rows, 767 temporal-outlier rows
* `[COASTAL IMPACT]` all 4 components active (corr / lag / exposure / anomaly)
* `[FINAL VIS]` `plots/final_environmental_pressure_map.png`

NDVI coverage warning (2.3 %) is expected — the AOI is maritime-dominant; the
pipeline continues by design.
