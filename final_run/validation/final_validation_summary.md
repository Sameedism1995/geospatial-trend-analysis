# Final dataset validation summary

- Dataset: `/Users/sameedahmed/Documents/geospatial trend analysis/final_run/processed/features_ml_ready.parquet`
- Rows × cols: **[16065, 46]**

## Missing values (>30.0% threshold)
- `oil_slick_probability_t`: 100.00% missing
- `detection_score`: 100.00% missing
- `no2_x_ndvi`: 98.91% missing
- `vessel_x_ndvi_lag2`: 98.10% missing
- `vessel_x_ndvi_lag3`: 98.10% missing
- `vessel_x_ndvi_lag1`: 98.10% missing
- `ndvi_mean`: 97.70% missing
- `ndvi_median`: 97.70% missing
- `ndvi_std`: 97.70% missing
- `land_response_index`: 97.70% missing
- `ndwi_mean`: 81.77% missing
- `ndwi_median`: 81.77% missing
- `ndwi_std`: 81.77% missing
- `ndti_mean`: 81.77% missing
- `ndti_median`: 81.77% missing
- `ndti_std`: 81.77% missing
- `ndci_mean`: 81.77% missing
- `ndci_median`: 81.77% missing
- `ndci_std`: 81.77% missing
- `fai_mean`: 81.77% missing
- `fai_median`: 81.77% missing
- `fai_std`: 81.77% missing
- `b11_mean`: 81.77% missing
- `b11_median`: 81.77% missing
- `b11_std`: 81.77% missing
- `vessel_x_no2`: 32.85% missing

## Infinite values
- Total inf cells: **0**

## Temporal consistency
- Status: **ok**
- Coverage: 2022-12-26/2023-01-01 → 2023-12-11/2023-12-17 (51 of 51 weeks present, 0 missing)

## Spatial consistency
- Status: **ok**
- Duplicate grid-week rows: 0 | unique grid cells: 315 | grid-week pairs: 16065

## Correlation sanity (|corr| ≥ 0.98)
- Status: **high_correlation_flag**
- Pairs flagged: 19
  - `vessel_density_t` ↔ `vessel_density` (r=+1.0000)
  - `vessel_density_t` ↔ `maritime_pressure_index` (r=+1.0000)
  - `no2_mean_t` ↔ `NO2_mean` (r=+1.0000)
  - `no2_mean_t` ↔ `atmospheric_transfer_index` (r=+1.0000)
  - `ndvi_mean` ↔ `land_response_index` (r=+1.0000)
  - `NO2_mean` ↔ `atmospheric_transfer_index` (r=+1.0000)
  - `vessel_density` ↔ `maritime_pressure_index` (r=+1.0000)
  - `vessel_x_ndvi_lag1` ↔ `vessel_x_ndvi_lag2` (r=+1.0000)
  - `vessel_x_ndvi_lag1` ↔ `vessel_x_ndvi_lag3` (r=+1.0000)
  - `vessel_x_ndvi_lag2` ↔ `vessel_x_ndvi_lag3` (r=+1.0000)
  - `distance_to_port_km` ↔ `distance_to_nearest_high_vessel_density_cell` (r=+0.9999)
  - `ndvi_mean` ↔ `ndvi_median` (r=+0.9976)
  - `ndvi_median` ↔ `land_response_index` (r=+0.9976)
  - `vessel_x_no2` ↔ `vessel_x_ndvi_lag1` (r=+0.9955)
  - `vessel_x_no2` ↔ `vessel_x_ndvi_lag2` (r=+0.9955)
  - `vessel_x_no2` ↔ `vessel_x_ndvi_lag3` (r=+0.9955)
  - `b11_mean` ↔ `b11_median` (r=+0.9892)
  - `grid_centroid_lat` ↔ `distance_to_nearest_high_vessel_density_cell` (r=-0.9836)
  - `grid_centroid_lat` ↔ `distance_to_port_km` (r=-0.9834)

## Coverage (per column, % non-null)
| column | coverage % |
|---|---:|
| `oil_slick_probability_t` | 0.00 |
| `detection_score` | 0.00 |
| `no2_x_ndvi` | 1.09 |
| `vessel_x_ndvi_lag2` | 1.90 |
| `vessel_x_ndvi_lag3` | 1.90 |
| `vessel_x_ndvi_lag1` | 1.90 |
| `ndvi_mean` | 2.30 |
| `ndvi_median` | 2.30 |
| `ndvi_std` | 2.30 |
| `land_response_index` | 2.30 |
| `ndwi_mean` | 18.23 |
| `ndwi_median` | 18.23 |
| `ndwi_std` | 18.23 |
| `ndti_mean` | 18.23 |
| `ndti_median` | 18.23 |
| `ndti_std` | 18.23 |
| `ndci_mean` | 18.23 |
| `ndci_median` | 18.23 |
| `ndci_std` | 18.23 |
| `fai_mean` | 18.23 |
| `fai_median` | 18.23 |
| `fai_std` | 18.23 |
| `b11_mean` | 18.23 |
| `b11_median` | 18.23 |
| `b11_std` | 18.23 |
| `vessel_x_no2` | 67.15 |
| `vessel_density_t` | 75.87 |
| `vessel_density` | 75.87 |
| `port_exposure_score` | 75.87 |
| `maritime_pressure_index` | 75.87 |
| `NO2_trend` | 86.88 |
| `no2_mean_t` | 89.90 |
| `no2_std_t` | 89.90 |
| `NO2_mean` | 89.90 |
| `atmospheric_transfer_index` | 89.90 |
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
- 26 columns exceed 30.0% missing.
- 19 correlation pairs ≥0.98 (possible leakage / redundancy).
