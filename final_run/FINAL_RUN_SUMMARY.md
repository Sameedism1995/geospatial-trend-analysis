# FINAL RUN SUMMARY

- Generated at: `2026-05-05 13:32:43 +0500`
- Final-run root: `final_run`
- Pipeline status: **ok** (elapsed 5136.57s)

## Earth Engine pre-flight
- GEE auth OK (project=`clawd-bot-485923`).
- Requested force_refresh: **True** | effective: **True**

## Mirrored artefacts (project root → final_run/)
- `data_aux`: 8 files
- `data_intermediate`: 5 files
- `data_validation`: 10 files
- `modeling_input`: 1 files
- `processed`: 2 files
- `outputs_reports`: 40 files
- `outputs_eda`: 3 files
- `eda_summary_md`: 1 files
- `outputs_plots`: 11 files
- `outputs_previews`: 21 files
- `outputs_visualizations`: 19 files
- `pipeline_log`: 1 files

## Dataset
- ML-ready features parquet: `final_run/processed/features_ml_ready.parquet` (16065 × 46)
- High-missing columns (>30.0%): 26
  - `oil_slick_probability_t`: 100.00%
  - `detection_score`: 100.00%
  - `no2_x_ndvi`: 98.91%
  - `vessel_x_ndvi_lag2`: 98.10%
  - `vessel_x_ndvi_lag3`: 98.10%
  - `vessel_x_ndvi_lag1`: 98.10%
  - `ndvi_mean`: 97.70%
  - `ndvi_median`: 97.70%
  - `ndvi_std`: 97.70%
  - `land_response_index`: 97.70%
- Temporal coverage: 2022-12-26/2023-01-01 → 2023-12-11/2023-12-17, 51/51 weeks present, 0 missing
- Spatial: 315 grid cells, 16065 grid-week pairs, 0 duplicates
- Correlation flags (|r|≥0.98): 19
- Validation warnings:
  - 26 columns exceed 30.0% missing.
  - 19 correlation pairs ≥0.98 (possible leakage / redundancy).

## Hub distance-decay analysis
- Hubs analysed: **3** (['Mariehamn', 'Naantali', 'Turku']).
- Indicators used: `{'ndti': 'ndti_mean', 'ndwi': 'ndwi_mean', 'ndvi': 'ndvi_mean', 'no2': 'no2_mean_t', 'vessel_density': 'vessel_density_t'}`.
- NO2 200–500 km mean: **1.020e-05** (n=133); <200 km = 1.273e-05; >500 km = 2.173e-05.
- 200–500 km hub composition: `{'Mariehamn': 133}`.
- `hub_plots`: `final_run/outputs/visualizations/hub_level_distance_decay`
- `sliding_window_plots`: `final_run/outputs/visualizations/sliding_window_distance_decay`
- `sliding_window_reports`: `final_run/outputs/reports/sliding_window_distance_decay`
- `no2_report`: `final_run/outputs/reports/no2_distance_diagnostic.md`

## Plot quality check
- Plots scanned: 48
- All plots above size threshold.

## Known limitations
- Only 3 distinct nearest ports in the dataset (Mariehamn / Naantali / Turku); the 200–500 km distance band is single-hub (Mariehamn) so cross-hub claims at that range should be made cautiously.
- NDVI is coastal-only; expect low coverage on this column (this is normal, not a defect).
- Sentinel-2 / Sentinel-1 / NO2 sources require valid Earth Engine project credentials for true fresh re-extraction; the run uses cached aux parquets when GEE is unavailable.

## Reproducibility
- Configs snapshot: `config_snapshot/`
- Full log: `logs/full_pipeline.log` (and `logs/pipeline_run.log` if produced).
- Re-run: `python3 src/run_final_pipeline.py`
