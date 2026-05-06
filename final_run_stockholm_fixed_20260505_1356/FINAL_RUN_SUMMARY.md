# FINAL RUN SUMMARY

- Generated at: `2026-05-05 23:04:08 +0500`
- Final-run root: `final_run_stockholm_fixed_20260505_1356`
- Pipeline status: **ok** (elapsed 32840.08s)

## Earth Engine pre-flight
- GEE auth OK (project=`clawd-bot-485923`).
- Requested force_refresh: **True** | effective: **True**

## Mirrored artefacts (project root → final_run/)
- `data_aux`: 9 files
- `data_intermediate`: 5 files
- `data_validation`: 10 files
- `modeling_input`: 1 files
- `processed`: 3 files
- `outputs_reports`: 54 files
- `outputs_eda`: 4 files
- `eda_summary_md`: 1 files
- `outputs_plots`: 16 files
- `outputs_previews`: 23 files
- `outputs_visualizations`: 22 files
- `pipeline_log`: 1 files

## Dataset
- ML-ready features parquet: `final_run_stockholm_fixed_20260505_1356/processed/features_ml_ready.parquet` (42738 × 46)
- High-missing columns (>30.0%): 28
  - `vessel_x_ndvi_lag2`: 99.29%
  - `vessel_x_ndvi_lag3`: 99.29%
  - `vessel_x_ndvi_lag1`: 99.28%
  - `no2_x_ndvi`: 97.67%
  - `ndvi_mean`: 97.21%
  - `ndvi_median`: 97.21%
  - `ndvi_std`: 97.21%
  - `land_response_index`: 97.21%
  - `ndwi_mean`: 79.74%
  - `ndwi_median`: 79.74%
- Temporal coverage: 2022-12-26/2023-01-01 → 2023-12-11/2023-12-17, 51/51 weeks present, 0 missing
- Spatial: 838 grid cells, 42738 grid-week pairs, 0 duplicates
- Correlation flags (|r|≥0.98): 22
- Validation warnings:
  - 28 columns exceed 30.0% missing.
  - 22 correlation pairs ≥0.98 (possible leakage / redundancy).

## Hub distance-decay analysis
- Hubs analysed: **3** (['Mariehamn', 'Naantali', 'Turku']).
- Indicators used: `{'ndti': 'ndti_mean', 'ndwi': 'ndwi_mean', 'ndvi': 'ndvi_mean', 'no2': 'no2_mean_t', 'vessel_density': 'vessel_density_t'}`.
- NO2 200–500 km mean: **1.188e-05** (n=5039); <200 km = 1.278e-05; >500 km = 2.173e-05.
- 200–500 km hub composition: `{'Mariehamn': 5039}`.
- `hub_plots`: `final_run_stockholm_fixed_20260505_1356/outputs/visualizations/hub_level_distance_decay`
- `sliding_window_plots`: `final_run_stockholm_fixed_20260505_1356/outputs/visualizations/sliding_window_distance_decay`
- `sliding_window_reports`: `final_run_stockholm_fixed_20260505_1356/outputs/reports/sliding_window_distance_decay`
- `no2_report`: `final_run_stockholm_fixed_20260505_1356/outputs/reports/no2_distance_diagnostic.md`

## Plot quality check
- Plots scanned: 59
- All plots above size threshold.

## Known limitations
- Only 3 distinct nearest ports in the dataset (Mariehamn / Naantali / Turku); the 200–500 km distance band is single-hub (Mariehamn) so cross-hub claims at that range should be made cautiously.
- NDVI is coastal-only; expect low coverage on this column (this is normal, not a defect).
- Sentinel-2 / Sentinel-1 / NO2 sources require valid Earth Engine project credentials for true fresh re-extraction; the run uses cached aux parquets when GEE is unavailable.

## Reproducibility
- Configs snapshot: `config_snapshot/`
- Full log: `logs/full_pipeline.log` (and `logs/pipeline_run.log` if produced).
- Re-run: `python3 src/run_final_pipeline.py`
