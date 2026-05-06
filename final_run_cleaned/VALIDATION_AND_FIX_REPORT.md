# Validation & fix report — final_run_cleaned/

- Source dataset: `final_run/processed/features_ml_ready.parquet`
- Cleaned root: `final_run_cleaned`
- Anomalies found / addressed: **328**

## Anomalies found and how they were handled

### A. Oil-slick features
- **Status: UNUSABLE** — probability and detection_score are 100% missing; count is all-zero
- Action: excluded from `features_ml_safe.parquet`. Raw columns kept in `features_cleaned_full.parquet` for transparency.
- Thesis rule: **do not claim oil-slick findings from this run.**
  - `oil_slick_probability_t`: {'present': True, 'non_null_count': 0, 'missing_percent': 100.0, 'unique_non_null_values': 0, 'all_zero': False}
  - `detection_score`: {'present': True, 'non_null_count': 0, 'missing_percent': 100.0, 'unique_non_null_values': 0, 'all_zero': False}
  - `oil_slick_count_t`: {'present': True, 'non_null_count': 16065, 'missing_percent': 0.0, 'unique_non_null_values': 1, 'all_zero': True}

### B. Vessel density temporal behaviour
- 100.0% of grid cells have **only 1 unique value** of `vessel_density_t` across all weeks.
- Weekly-mean σ: 1.1102230246251565e-16
- Verdict: **is_static = True** — vessel_density_t behaves as a per-grid SPATIAL pressure proxy, not weekly traffic.
- Action: cleaned dataset exposes `vessel_density_spatial_proxy` (and `*_log1p`) so downstream code can reference it without implying weekly traffic dynamics.
- Thesis rule: describe vessel_density_t as a **spatial pressure proxy**, not weekly vessel traffic.

### C. NO2 weekly outliers (MAD)
- Median weekly mean: 1.628e-05 | MAD: 5.116e-06 | threshold (3·1.4826·MAD): 2.275e-05.
- Outlier weeks (4 of 51):
  - 2023-01-15 00:00:00+00:00
  - 2023-02-19 00:00:00+00:00
  - 2023-02-26 00:00:00+00:00
  - 2023-12-03 00:00:00+00:00
- Action: smoothed columns `no2_mean_t_rolling3`, `no2_mean_t_rolling5` added; raw `no2_mean_t` retained.
- Thesis rule: report rolling-mean curves; flag weeks with extreme negative values rather than deleting them.

### D. Water-quality index validity
- Per-column audit (min/max/p1/p99/out-of-range/extreme):
  - `ndwi_mean`: {'non_null': 2929, 'min': -0.8444969483683549, 'max': 0.9811732478461643, 'p1': -0.6862886192137627, 'p99': 0.8177487958408789, 'n_out_of_range': 0, 'n_extreme_p1_p99': 60}
  - `ndwi_median`: {'non_null': 2929, 'min': -0.9770034836219238, 'max': 1.0, 'p1': -0.7226980641732201, 'p99': 0.8695117475510269, 'n_out_of_range': 0, 'n_extreme_p1_p99': 60}
  - `ndti_mean`: {'non_null': 2929, 'min': -0.9771356153170873, 'max': 0.4277191596659442, 'p1': -0.49880806790745497, 'p99': 0.24443324366997837, 'n_out_of_range': 0, 'n_extreme_p1_p99': 60}
  - `ndti_median`: {'non_null': 2929, 'min': -1.0, 'max': 0.3642679411095152, 'p1': -0.5028024775832202, 'p99': 0.24106653599524874, 'n_out_of_range': 0, 'n_extreme_p1_p99': 60}
  - `ndci_mean`: {'non_null': 2929, 'min': -0.8759683606947217, 'max': 0.9418702059825564, 'p1': -0.3487178745843644, 'p99': 0.405193151697866, 'n_out_of_range': 0, 'n_extreme_p1_p99': 60}
  - `ndci_median`: {'non_null': 2929, 'min': -0.9178098465424313, 'max': 1.0, 'p1': -0.3507009996504196, 'p99': 0.44094367168708803, 'n_out_of_range': 0, 'n_extreme_p1_p99': 60}
  - `ndvi_mean`: {'non_null': 369, 'min': -1.0, 'max': 1.0, 'p1': -0.00011795992597237348, 'p99': 0.6872476579504451, 'n_out_of_range': 0, 'n_extreme_p1_p99': 8}
  - `ndvi_median`: {'non_null': 369, 'min': -1.0, 'max': 1.0, 'p1': 0.0, 'p99': 0.7375360281418016, 'n_out_of_range': 0, 'n_extreme_p1_p99': 11}
  - `fai_mean`: {'non_null': 2929, 'min': -0.042887293709018254, 'max': 0.34076029826478893, 'p1': -0.01031043476611011, 'p99': 0.27910128633995507, 'n_out_of_range': 0, 'n_extreme_p1_p99': 60}
  - `fai_median`: {'non_null': 2929, 'min': -0.042887293709018254, 'max': 0.35747888334073163, 'p1': -0.011117987696637096, 'p99': 0.2860462791382571, 'n_out_of_range': 0, 'n_extreme_p1_p99': 60}
  - `b11_mean`: {'non_null': 2929, 'min': 0.0, 'max': 0.7861866772260647, 'p1': 0.0004884292048958411, 'p99': 0.5876306893975978, 'n_out_of_range': 0, 'n_extreme_p1_p99': 60}
  - `b11_median`: {'non_null': 2929, 'min': 0.0, 'max': 0.8261870358470904, 'p1': 2.5491017039980303e-06, 'p99': 0.6006262439636558, 'n_out_of_range': 0, 'n_extreme_p1_p99': 60}
- Action: winsorized columns at [p1, p99] added (`*_winsor`); flag columns added: `ndwi_extreme_flag`, `ndti_extreme_flag`, `water_quality_extreme_flag`. Raw values preserved.

### E. Skew transforms
- Added 22 columns (log1p / robust_z) for skewed positive features.

### F. High-correlation alias clusters (|r| ≥ 0.98)
- Pairs flagged: 72.
- Alias members dropped from ML-safe set: 37.
- Drop list: `NO2_mean`, `atmospheric_transfer_index`, `b11_mean_winsor`, `b11_median`, `coastal_exposure_score_log1p`, `coastal_exposure_score_robust_z`, `distance_to_nearest_high_vessel_density_cell`, `distance_to_nearest_high_vessel_density_cell_robust_z`, `distance_to_port_km_robust_z`, `fai_median`, `grid_centroid_lat`, `land_response_index`, `maritime_pressure_index`, `maritime_pressure_index_log1p`, `maritime_pressure_index_robust_z`, `ndci_median`, `ndti_median`, `ndti_std_log1p`, `ndti_std_robust_z`, `ndvi_mean_winsor`, `ndvi_median`, `ndvi_std_log1p`, `ndvi_std_robust_z`, `ndwi_median`, `ndwi_std_log1p`, `ndwi_std_robust_z`, `no2_std_t_log1p`, `no2_std_t_robust_z`, `port_exposure_score_log1p`, `port_exposure_score_robust_z`, `vessel_density`, `vessel_density_robust_z`, `vessel_density_t_log1p`, `vessel_density_t_robust_z`, `vessel_x_ndvi_lag2`, `vessel_x_ndvi_lag3`, `vessel_x_no2`

## Cleaned datasets
- Full: `final_run_cleaned/processed/features_cleaned_full.parquet` (raw + flags + transforms)
- ML-safe: `final_run_cleaned/processed/features_ml_safe.parquet` (29 columns; excluded 52)
- Excluded columns: `NO2_mean`, `atmospheric_transfer_index`, `b11_mean`, `b11_mean_winsor`, `b11_median`, `b11_std`, `coastal_exposure_score_log1p`, `coastal_exposure_score_robust_z`, `detection_score`, `distance_to_nearest_high_vessel_density_cell`, `distance_to_nearest_high_vessel_density_cell_robust_z`, `distance_to_port_km_robust_z`, `fai_mean`, `fai_median`, `fai_std`, `grid_centroid_lat`, `land_response_index`, `maritime_pressure_index`, `maritime_pressure_index_log1p`, `maritime_pressure_index_robust_z`, `ndci_mean`, `ndci_median`, `ndci_std`, `ndti_mean`, `ndti_median`, `ndti_std`, `ndti_std_log1p`, `ndti_std_robust_z`, `ndvi_mean_winsor`, `ndvi_median`, `ndvi_std_log1p`, `ndvi_std_robust_z`, `ndwi_mean`, `ndwi_median`, `ndwi_std`, `ndwi_std_log1p`, `ndwi_std_robust_z`, `no2_std_t_log1p`, `no2_std_t_robust_z`, `no2_x_ndvi`, `oil_slick_count_t`, `oil_slick_probability_t`, `port_exposure_score_log1p`, `port_exposure_score_robust_z`, `vessel_density`, `vessel_density_robust_z`, `vessel_density_t_log1p`, `vessel_density_t_robust_z`, `vessel_x_ndvi_lag1`, `vessel_x_ndvi_lag2`, `vessel_x_ndvi_lag3`, `vessel_x_no2`

## Thesis-safe plots
- `final_run_cleaned/outputs/visualizations/temporal_smoothing/smoothing_no2_mean_t.png`
- `final_run_cleaned/outputs/visualizations/temporal_smoothing/smoothing_ndwi_mean.png`
- `final_run_cleaned/outputs/visualizations/temporal_smoothing/smoothing_ndti_mean.png`
- `final_run_cleaned/outputs/visualizations/temporal_smoothing/smoothing_vessel_density_t.png`
- `final_run_cleaned/outputs/visualizations/sliding_window_cleaned/sliding_window_vessel_density_spatial_proxy_log1p.png`
- `final_run_cleaned/outputs/visualizations/sliding_window_cleaned/sliding_window_no2_mean_t_rolling3.png`
- `final_run_cleaned/outputs/visualizations/sliding_window_cleaned/sliding_window_ndwi_mean_winsor.png`
- `final_run_cleaned/outputs/visualizations/sliding_window_cleaned/sliding_window_ndti_mean_winsor.png`
- `final_run_cleaned/outputs/visualizations/sliding_window_cleaned/sliding_window_ndvi_mean_winsor.png`

## Interpretation rules (DO / DON'T)
- **DO** present `vessel_density_t` as a spatial pressure proxy. **DON'T** claim weekly vessel-traffic dynamics from this column.
- **DO NOT** report any oil-slick-based result; the source data is unusable in this run.
- **DO** show NO2 as smoothed rolling means; **DON'T** delete outlier weeks.
- **DO** use winsorized water-quality columns for robust statistics; **DON'T** discard raw values.
- **DO** prefer the recommended feature list (no aliases) for any ML modelling.
