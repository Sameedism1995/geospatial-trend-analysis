# FINAL RUN SUMMARY

- Generated at: `2026-05-09 21:57:19 +0500`
- Final-run root: `full_fresh_20260509`
- Pipeline status: **ok** (elapsed 43060.49s)

## Earth Engine pre-flight
- GEE auth NOT available — `ee.Initialize: no project found. Call with project= or see http://goo.gle/ee-auth.`.
- Consequence: GEE-backed sources (Sentinel-1 oil, Sentinel-2 water/land, NO2) cannot be re-extracted in this environment. The orchestrator therefore reused the most recent cached `data/aux/*.parquet` files (which **are** the latest successful GEE extraction, just committed to the repo). Set `GOOGLE_CLOUD_PROJECT` or `EE_PROJECT` and re-run with `--force-refresh` (the default) to truly re-extract.
- Requested force_refresh: **True** | effective: **True**

## Mirrored artefacts (project root → final_run/)
- `data_aux`: 9 files
- `data_intermediate`: 5 files
- `data_validation`: 10 files
- `modeling_input`: 1 files
- `processed`: 6 files
- `outputs_reports`: 184 files
- `outputs_eda`: 4 files
- `eda_summary_md`: 1 files
- `outputs_plots`: 16 files
- `outputs_preprocessing_diag`: 291 files
- `outputs_previews`: 32 files
- `outputs_visualizations`: 98 files
- `pipeline_log`: 1 files

## Dataset
- ML-ready features parquet: `full_fresh_20260509/processed/features_ml_ready.parquet` (16065 × 46)
- High-missing columns (>30.0%): 24
  - `no2_x_ndvi`: 98.91%
  - `vessel_x_ndvi_lag2`: 98.10%
  - `vessel_x_ndvi_lag3`: 98.10%
  - `vessel_x_ndvi_lag1`: 98.10%
  - `ndvi_mean`: 97.70%
  - `ndvi_median`: 97.70%
  - `ndvi_std`: 97.70%
  - `land_response_index`: 97.70%
  - `ndwi_mean`: 81.77%
  - `ndwi_median`: 81.77%
- Temporal coverage: 2022-12-26/2023-01-01 → 2023-12-11/2023-12-17, 51/51 weeks present, 0 missing
- Spatial: 315 grid cells, 16065 grid-week pairs, 0 duplicates
- Correlation flags (|r|≥0.98): 22
- Validation warnings:
  - 24 columns exceed 30.0% missing.
  - 22 correlation pairs ≥0.98 (possible leakage / redundancy).

## Hub distance-decay analysis
- Hubs analysed: **4** (['Mariehamn', 'Naantali', 'Stockholm', 'Turku']).
- Indicators used: `{'ndti': 'ndti_mean', 'ndwi': 'ndwi_mean', 'ndvi': 'ndvi_mean', 'no2': 'no2_mean_t', 'vessel_density': 'vessel_density_t'}`.
- NO2 200–500 km mean: **1.015e-05** (n=85); <200 km = 1.272e-05; >500 km = 2.173e-05.
- 200–500 km hub composition: `{'Stockholm': 44, 'Mariehamn': 41}`.
- `hub_plots`: `full_fresh_20260509/outputs/visualizations/hub_level_distance_decay`
- `sliding_window_plots`: `full_fresh_20260509/outputs/visualizations/sliding_window_distance_decay`
- `sliding_window_reports`: `full_fresh_20260509/outputs/reports/sliding_window_distance_decay`
- `no2_report`: `full_fresh_20260509/outputs/reports/no2_distance_diagnostic.md`

## Plot quality check
- Plots scanned: 414
- All plots above size threshold.

## Known limitations
- Only 3 distinct nearest ports in the dataset (Mariehamn / Naantali / Turku); the 200–500 km distance band is single-hub (Mariehamn) so cross-hub claims at that range should be made cautiously.
- NDVI is coastal-only; expect low coverage on this column (this is normal, not a defect).
- Sentinel-2 / Sentinel-1 / NO2 sources require valid Earth Engine project credentials for true fresh re-extraction; the run uses cached aux parquets when GEE is unavailable.

## Reproducibility
- Configs snapshot: `config_snapshot/`
- Full log: `logs/full_pipeline.log` (and `logs/pipeline_run.log` if produced).
- Re-run: `python3 src/run_final_pipeline.py`
