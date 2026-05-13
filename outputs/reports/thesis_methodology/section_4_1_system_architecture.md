# Section 4.1 — Overall System Architecture (implementation notes)

## Purpose

The codebase implements a **weekly grid panel** (grid cell × ISO week) over the Baltic study area: remote sensing and atmospheric columns are extracted in **observational** form (no gap-filling in extraction), merged on `(grid_cell_id, week_start_utc)`, enriched with geometric and land–sea interaction features, then analysed (correlations, lag studies, anomalies, coastal composite scores, maps).

## Control plane

- **Primary orchestrator:** `src/run_final_pipeline.py` — probes Google Earth Engine (GEE) auth, invokes the full pipeline, mirrors artefacts into a run folder (`final_run/` or `--run-name`), runs **final dataset validation**, **hub distance-decay analysis**, and writes **`FINAL_RUN_SUMMARY.md`**.
- **Core pipeline entrypoint:** `src/pipeline/run_full_pipeline.py` — `main()` sequences extraction, merge, feature build, land-impact extension, validation JSON, EDA, correlation, optional analytical stages.

## Logical layers

1. **Baseline panel** — `data/modeling_dataset.parquet` supplies the vessel-density spine and grid geometry (built upstream, e.g. from EMODnet Human Activities GeoTIFF sampling; see `src/build_modeling_dataset.py` and `docs/modeling_dataset_schema.md`).
2. **Auxiliary extractors (Earth Engine)** — write **`data/aux/*.parquet`** + sidecar validation JSON: NO₂ (`no2_gee_pipeline.py`), Sentinel-1 oil proxy (`sentinel1_oil_pipeline.py`), Sentinel-2 water quality (`sentinel2_water_quality.py`), optional Sentinel-2 land NDVI (`land_impact/sentinel2_land_metrics.py`).
3. **Harmonization** — `merge_sources()`: outer join on **`grid_cell_id`**, **`week_start_utc`**; normalizes aliases (`grid_id`, `week_start`); deduplicates last-wins.
4. **Feature layer** — `feature_engineering()` (derivatives, aliases, row missingness); **`features/port_proximity.py`** and **`features/port_exposure.py`**; optional **`run_land_impact_extension()`** (buffering + interactions + lags, persistence).
5. **Quality and analytics** — `validation/validate_aux_layers.py` from `global_validation`; `analysis/eda_report.py`, `analysis/correlation_analysis.py`, `analysis/scientific_validation.py`, `analysis/anomaly_detection.py`, `analysis/coastal_impact_score.py`, `visualization/final_visualization`-style exports; **land-impact** reports via `analysis/land_sea_correlation.py`, `land_impact_ml.py`, `land_impact_report.py`.
6. **Parallel research ingestion (not merged by default in `run_full_pipeline`)** — `src/run_ingestion.py` + `src/ingestion/` for EMODnet WMS capabilities catalogue, HELCOM GeoNetwork record XML, Sentinel Hub STAC catalogue POST; writes raw/processed under `data/raw/` with manifests.

## Persistent outputs (project root during run)

| Artefact | Role |
|----------|------|
| `data/intermediate/*.parquet` | Per-source snapshots after extraction |
| `data/aux/*.parquet` | Canonical GEE extractions (reused if not `--force-refresh`) |
| `processed/merged_dataset.parquet` | Post-merge multi-source table |
| `processed/features_ml_ready.parquet` | ML-oriented panel (overwritten after land-impact extension when `--land-impact`) |
| `data/validation/full_pipeline_validation_report.json` | Cross-source + aux validation bundle |
| `outputs/preprocessing_diagnostics/` | Per-stage QC (optional exhaustive plots) |
| `outputs/reports/`, `outputs/eda/`, `outputs/plots/`, `outputs/visualizations/` | Analytical exports |

## Optional / downstream analyses (separate CLIs)

Coastal **wind** alignment, exposure indices (MEI / ACEI / ESI), and related figures are driven by **`src/analysis/run_coastal_wind_transport.py`** and **`src/analysis/run_coastal_exposure_analysis.py`**, which consume the panel and/or merged wind CSVs — they are **not** inner steps of `run_full_pipeline.main()` but belong to the same methodological family.

## Dependencies (conceptual)

`run_final_pipeline` → `pipeline.run_full_pipeline` → (`data_sources.*`, `features.*`, `validation.validate_aux_layers`, `analysis.*`, `visualization.*`). The **ingestion** package is invoked only from `run_ingestion.py`, not from `run_full_pipeline`.

## Reproducibility

- Snapshot: orchestrator copies **`config/`**, `requirements.txt`, and key project files into **`config_snapshot/`** inside the run folder.
- **GEE:** `earthengine authenticate`; optional **`GOOGLE_CLOUD_PROJECT`** or **`EE_PROJECT`** for `ee.Initialize(project=...)`.
- **`--force-refresh`** wipes regenerable `data/aux` and `data/intermediate` (keeps `baltic_ports.csv`); if GEE is unavailable, **`run_final_pipeline`** can disable force-refresh and **reuse cached aux** unless **`--no-stale-gee-fallback`**.
