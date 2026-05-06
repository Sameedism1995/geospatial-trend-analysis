# Final dataset validation summary

- Dataset: `/Users/sameedahmed/Documents/geospatial trend analysis/final_run_stockholm_fixed_20260505_1356/processed/features_ml_ready.parquet`
- Rows × cols: **[42738, 46]**

## Missing values (>30.0% threshold)
- `vessel_x_ndvi_lag2`: 99.29% missing
- `vessel_x_ndvi_lag3`: 99.29% missing
- `vessel_x_ndvi_lag1`: 99.28% missing
- `no2_x_ndvi`: 97.67% missing
- `ndvi_mean`: 97.21% missing
- `ndvi_median`: 97.21% missing
- `ndvi_std`: 97.21% missing
- `land_response_index`: 97.21% missing
- `ndwi_mean`: 79.74% missing
- `ndwi_median`: 79.74% missing
- `ndwi_std`: 79.74% missing
- `ndti_mean`: 79.74% missing
- `ndti_median`: 79.74% missing
- `ndti_std`: 79.74% missing
- `ndci_mean`: 79.74% missing
- `ndci_median`: 79.74% missing
- `ndci_std`: 79.74% missing
- `fai_mean`: 79.74% missing
- `fai_median`: 79.74% missing
- `fai_std`: 79.74% missing
- `b11_mean`: 79.74% missing
- `b11_median`: 79.74% missing
- `b11_std`: 79.74% missing
- `vessel_x_no2`: 74.76% missing
- `vessel_density_t`: 71.48% missing
- `vessel_density`: 71.48% missing
- `port_exposure_score`: 71.48% missing
- `maritime_pressure_index`: 71.48% missing

## Infinite values
- Total inf cells: **0**

## Temporal consistency
- Status: **ok**
- Coverage: 2022-12-26/2023-01-01 → 2023-12-11/2023-12-17 (51 of 51 weeks present, 0 missing)

## Spatial consistency
- Status: **ok**
- Duplicate grid-week rows: 0 | unique grid cells: 838 | grid-week pairs: 42738

## Correlation sanity (|corr| ≥ 0.98)
- Status: **high_correlation_flag**
- Pairs flagged: 22
  - `vessel_density_t` ↔ `vessel_density` (r=+1.0000)
  - `vessel_density_t` ↔ `maritime_pressure_index` (r=+1.0000)
  - `no2_mean_t` ↔ `NO2_mean` (r=+1.0000)
  - `no2_mean_t` ↔ `atmospheric_transfer_index` (r=+1.0000)
  - `oil_slick_probability_t` ↔ `detection_score` (r=+1.0000)
  - `ndvi_mean` ↔ `land_response_index` (r=+1.0000)
  - `NO2_mean` ↔ `atmospheric_transfer_index` (r=+1.0000)
  - `vessel_density` ↔ `maritime_pressure_index` (r=+1.0000)
  - `vessel_x_ndvi_lag1` ↔ `vessel_x_ndvi_lag2` (r=+1.0000)
  - `vessel_x_ndvi_lag1` ↔ `vessel_x_ndvi_lag3` (r=+1.0000)
  - `vessel_x_ndvi_lag2` ↔ `vessel_x_ndvi_lag3` (r=+1.0000)
  - `distance_to_port_km` ↔ `distance_to_nearest_high_vessel_density_cell` (r=+0.9998)
  - `oil_slick_probability_t` ↔ `oil_slick_count_t` (r=+0.9995)
  - `oil_slick_count_t` ↔ `detection_score` (r=+0.9995)
  - `vessel_x_no2` ↔ `vessel_x_ndvi_lag1` (r=+0.9955)
  - `vessel_x_no2` ↔ `vessel_x_ndvi_lag2` (r=+0.9955)
  - `vessel_x_no2` ↔ `vessel_x_ndvi_lag3` (r=+0.9955)
  - `ndvi_mean` ↔ `ndvi_median` (r=+0.9951)
  - `ndvi_median` ↔ `land_response_index` (r=+0.9951)
  - `b11_mean` ↔ `b11_median` (r=+0.9845)

## Coverage (per column, % non-null)
| column | coverage % |
|---|---:|
| `vessel_x_ndvi_lag2` | 0.71 |
| `vessel_x_ndvi_lag3` | 0.71 |
| `vessel_x_ndvi_lag1` | 0.72 |
| `no2_x_ndvi` | 2.33 |
| `ndvi_mean` | 2.79 |
| `ndvi_median` | 2.79 |
| `ndvi_std` | 2.79 |
| `land_response_index` | 2.79 |
| `ndwi_mean` | 20.26 |
| `ndwi_median` | 20.26 |
| `ndwi_std` | 20.26 |
| `ndti_mean` | 20.26 |
| `ndti_median` | 20.26 |
| `ndti_std` | 20.26 |
| `ndci_mean` | 20.26 |
| `ndci_median` | 20.26 |
| `ndci_std` | 20.26 |
| `fai_mean` | 20.26 |
| `fai_median` | 20.26 |
| `fai_std` | 20.26 |
| `b11_mean` | 20.26 |
| `b11_median` | 20.26 |
| `b11_std` | 20.26 |
| `vessel_x_no2` | 25.24 |
| `vessel_density_t` | 28.52 |
| `vessel_density` | 28.52 |
| `port_exposure_score` | 28.52 |
| `maritime_pressure_index` | 28.52 |
| `NO2_trend` | 86.45 |
| `no2_mean_t` | 89.62 |
| `no2_std_t` | 89.62 |
| `NO2_mean` | 89.62 |
| `atmospheric_transfer_index` | 89.62 |
| `oil_slick_probability_t` | 97.97 |
| `detection_score` | 97.97 |
| `grid_cell_id` | 100.00 |
| `week_start_utc` | 100.00 |
| `grid_centroid_lat` | 100.00 |
| `grid_centroid_lon` | 100.00 |
| `oil_slick_count_t` | 100.00 |
| `nan_ratio_row` | 100.00 |
| `nearest_port` | 100.00 |
| `distance_to_port_km` | 100.00 |
| `distance_to_nearest_high_vessel_density_cell` | 100.00 |
| `coastal_exposure_band` | 100.00 |
| `coastal_exposure_score` | 100.00 |

NDVI / land_response_index are expected to have low coverage (coastal/inland cells only).

## Warnings
- 28 columns exceed 30.0% missing.
- 22 correlation pairs ≥0.98 (possible leakage / redundancy).
