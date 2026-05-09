# Dataset feature (column) inventory

Automated listing of columns per file. **Parquet** schemas use Arrow; **CSV** headers from the first row.

---

## Coastal wind–merged ML-ready weekly grid (primary thesis panel)

**Path:** `outputs/processed/features_ml_ready_coastal_wind.parquet`  
**Feature count:** 66

**Features:**

- `grid_cell_id`
- `week_start_utc`
- `vessel_density_t`
- `grid_centroid_lat`
- `grid_centroid_lon`
- `no2_mean_t`
- `no2_std_t`
- `oil_slick_probability_t`
- `oil_slick_count_t`
- `ndwi_mean`
- `ndwi_median`
- `ndwi_std`
- `ndti_mean`
- `ndti_median`
- `ndti_std`
- `ndci_mean`
- `ndci_median`
- `ndci_std`
- `fai_mean`
- `fai_median`
- `fai_std`
- `b11_mean`
- `b11_median`
- `b11_std`
- `ndvi_mean`
- `ndvi_median`
- `ndvi_std`
- `NO2_mean`
- `NO2_trend`
- `vessel_density`
- `detection_score`
- `nan_ratio_row`
- `nearest_port`
- `distance_to_port_km`
- `port_exposure_score`
- `distance_to_nearest_high_vessel_density_cell`
- `coastal_exposure_band`
- `coastal_exposure_score`
- `maritime_pressure_index`
- `atmospheric_transfer_index`
- `land_response_index`
- `vessel_x_no2`
- `no2_x_ndvi`
- `vessel_x_ndvi_lag1`
- `vessel_x_ndvi_lag2`
- `vessel_x_ndvi_lag3`
- `coastal_wind_alignment_score`
- `bearing_cell_to_coast_deg`
- `coastal_wind_angle_diff_deg`
- `coastal_wind_shoreward_45deg`
- `pollution_transport_wind_alignment_score`
- `bearing_hotspot_to_coast_deg`
- `pollution_hotspot_lat`
- `pollution_hotspot_lon`
- `pollution_hotspot_type`
- `nearest_coast_ref_distance_km`
- `coastal_wind_alignment_score_cw`
- `bearing_cell_to_coast_deg_cw`
- `coastal_wind_angle_diff_deg_cw`
- `coastal_wind_shoreward_45deg_cw`
- `pollution_transport_wind_alignment_score_cw`
- `bearing_hotspot_to_coast_deg_cw`
- `pollution_hotspot_lat_cw`
- `pollution_hotspot_lon_cw`
- `pollution_hotspot_type_cw`
- `nearest_coast_ref_distance_km_cw`

---

## Root merged grid (`processed/` at repo root)

**Path:** `processed/merged_dataset.parquet`  
**Feature count:** 27

**Features:**

- `grid_cell_id`
- `week_start_utc`
- `vessel_density_t`
- `grid_centroid_lat`
- `grid_centroid_lon`
- `no2_mean_t`
- `no2_std_t`
- `oil_slick_probability_t`
- `oil_slick_count_t`
- `ndwi_mean`
- `ndwi_median`
- `ndwi_std`
- `ndti_mean`
- `ndti_median`
- `ndti_std`
- `ndci_mean`
- `ndci_median`
- `ndci_std`
- `fai_mean`
- `fai_median`
- `fai_std`
- `b11_mean`
- `b11_median`
- `b11_std`
- `ndvi_mean`
- `ndvi_median`
- `ndvi_std`

---

## Root ML-ready features (`processed/`)

**Path:** `processed/features_ml_ready.parquet`  
**Feature count:** 46

**Features:**

- `grid_cell_id`
- `week_start_utc`
- `vessel_density_t`
- `grid_centroid_lat`
- `grid_centroid_lon`
- `no2_mean_t`
- `no2_std_t`
- `oil_slick_probability_t`
- `oil_slick_count_t`
- `ndwi_mean`
- `ndwi_median`
- `ndwi_std`
- `ndti_mean`
- `ndti_median`
- `ndti_std`
- `ndci_mean`
- `ndci_median`
- `ndci_std`
- `fai_mean`
- `fai_median`
- `fai_std`
- `b11_mean`
- `b11_median`
- `b11_std`
- `ndvi_mean`
- `ndvi_median`
- `ndvi_std`
- `NO2_mean`
- `NO2_trend`
- `vessel_density`
- `detection_score`
- `nan_ratio_row`
- `nearest_port`
- `distance_to_port_km`
- `port_exposure_score`
- `distance_to_nearest_high_vessel_density_cell`
- `coastal_exposure_band`
- `coastal_exposure_score`
- `maritime_pressure_index`
- `atmospheric_transfer_index`
- `land_response_index`
- `vessel_x_no2`
- `no2_x_ndvi`
- `vessel_x_ndvi_lag1`
- `vessel_x_ndvi_lag2`
- `vessel_x_ndvi_lag3`

---

## Modeling dataset

**Path:** `data/modeling_dataset.parquet`  
**Feature count:** 29

**Features:**

- `grid_cell_id`
- `week_start_utc`
- `grid_res_deg`
- `grid_centroid_lat`
- `grid_centroid_lon`
- `week_of_year`
- `week_sin`
- `week_cos`
- `vessel_density_t`
- `vessel_density_t_minus_1`
- `vessel_density_t_minus_2`
- `sentinel_ndvi_mean_t`
- `sentinel_ndvi_mean_t_minus_1`
- `sentinel_ndvi_mean_t_minus_2`
- `sentinel_ndwi_mean_t`
- `sentinel_ndwi_mean_t_minus_1`
- `sentinel_ndwi_mean_t_minus_2`
- `sentinel_evi_mean_t`
- `sentinel_evi_mean_t_minus_1`
- `sentinel_evi_mean_t_minus_2`
- `sentinel_ndti_mean_t`
- `sentinel_ndti_mean_t_minus_1`
- `sentinel_ndti_mean_t_minus_2`
- `sentinel_observation_count_t`
- `delta_ndti`
- `has_valid_delta_ndti`
- `has_sentinel`
- `has_emodnet`
- `has_helcom`

---

## Modeling dataset (human impact variant)

**Path:** `data/modeling_dataset_human_impact.parquet`  
**Feature count:** 40

**Features:**

- `grid_cell_id`
- `week_start_utc`
- `grid_res_deg`
- `grid_centroid_lat`
- `grid_centroid_lon`
- `week_of_year`
- `week_sin`
- `week_cos`
- `vessel_density_t`
- `vessel_density_t_minus_1`
- `vessel_density_t_minus_2`
- `sentinel_ndvi_mean_t`
- `sentinel_ndvi_mean_t_minus_1`
- `sentinel_ndvi_mean_t_minus_2`
- `sentinel_ndwi_mean_t`
- `sentinel_ndwi_mean_t_minus_1`
- `sentinel_ndwi_mean_t_minus_2`
- `sentinel_evi_mean_t`
- `sentinel_evi_mean_t_minus_1`
- `sentinel_evi_mean_t_minus_2`
- `sentinel_ndti_mean_t`
- `sentinel_ndti_mean_t_minus_1`
- `sentinel_ndti_mean_t_minus_2`
- `sentinel_observation_count_t`
- `delta_ndti`
- `has_valid_delta_ndti`
- `has_sentinel`
- `has_emodnet`
- `has_helcom`
- `distance_to_coast_km`
- `distance_to_shipping_km`
- `distance_to_urban_proxy_km`
- `no2_mean_t`
- `no2_baseline_t`
- `no2_anomaly_t`
- `oil_slick_probability_t`
- `oil_slick_count_t`
- `activity_regime`
- `coast_distance_bin`
- `shipping_distance_bin`

---

## Aux: Sentinel-2 water quality

**Path:** `data/aux/sentinel2_water_quality.parquet`  
**Feature count:** 17

**Features:**

- `grid_cell_id`
- `week_start_utc`
- `ndwi_mean`
- `ndwi_median`
- `ndwi_std`
- `ndti_mean`
- `ndti_median`
- `ndti_std`
- `ndci_mean`
- `ndci_median`
- `ndci_std`
- `fai_mean`
- `fai_median`
- `fai_std`
- `b11_mean`
- `b11_median`
- `b11_std`

---

## Aux: Sentinel-1 oil slicks

**Path:** `data/aux/sentinel1_oil_slicks.parquet`  
**Feature count:** 4

**Features:**

- `grid_cell_id`
- `week_start_utc`
- `oil_slick_probability_t`
- `oil_slick_count_t`

---

## Master dataset

**Path:** `data/processed/master_dataset.csv`  
**Feature count:** 13

**Features:**

- `date`
- `latitude`
- `longitude`
- `salinity`
- `temperature`
- `turbidity`
- `vessel_density`
- `ndci`
- `ndti`
- `ndwi`
- `port_latitude`
- `port_longitude`
- `b11`

---

## Master dataset enriched

**Path:** `data/processed/master_dataset_enriched.csv`  
**Feature count:** 18

**Features:**

- `date`
- `latitude`
- `longitude`
- `salinity`
- `temperature`
- `turbidity`
- `vessel_density`
- `ndci`
- `ndti`
- `ndwi`
- `port_latitude`
- `port_longitude`
- `b11`
- `density_total_log`
- `sea_lane_flag`
- `traffic_group_25_75`
- `distance_to_lane_km`
- `distance_to_port`

---

## Cleaned merged dataset

**Path:** `data/processed/cleaned_merged_dataset.csv`  
**Feature count:** 9

**Features:**

- `coast_distance_bin`
- `shipping_distance_bin`
- `mean_ndti`
- `mean_ndvi`
- `mean_no2_mean_t`
- `mean_no2_anomaly_t`
- `mean_oil_slick_probability_t`
- `mean_vessel`
- `n`

---

## EMODnet tiff inventory

**Path:** `data/processed/emodnet_tiff_summary.csv`  
**Feature count:** 19

**Features:**

- `file`
- `vessel_code`
- `year`
- `width`
- `height`
- `crs`
- `pixel_count_total`
- `pixel_count_valid`
- `min_density`
- `q25_density`
- `median_density`
- `mean_density`
- `q75_density`
- `max_density`
- `std_density`
- `mean_density_log`
- `sea_lane_flag`
- `traffic_group_25_75`
- `traffic_category_q3`

---

## Nearest-land NDVI linkage

**Path:** `outputs/reports/run_nearest_land_ndvi_linkage/nearest_land_ndvi_linked_dataset.parquet`  
**Feature count:** 53

**Features:**

- `grid_cell_id`
- `week_start_utc`
- `vessel_density_t`
- `grid_centroid_lat`
- `grid_centroid_lon`
- `no2_mean_t`
- `no2_std_t`
- `oil_slick_probability_t`
- `oil_slick_count_t`
- `ndwi_mean`
- `ndwi_median`
- `ndwi_std`
- `ndti_mean`
- `ndti_median`
- `ndti_std`
- `ndci_mean`
- `ndci_median`
- `ndci_std`
- `fai_mean`
- `fai_median`
- `fai_std`
- `b11_mean`
- `b11_median`
- `b11_std`
- `ndvi_mean`
- `ndvi_median`
- `ndvi_std`
- `NO2_mean`
- `NO2_trend`
- `vessel_density`
- `detection_score`
- `nan_ratio_row`
- `nearest_port`
- `distance_to_port_km`
- `port_exposure_score`
- `distance_to_nearest_high_vessel_density_cell`
- `coastal_exposure_band`
- `coastal_exposure_score`
- `maritime_pressure_index`
- `atmospheric_transfer_index`
- `land_response_index`
- `vessel_x_no2`
- `no2_x_ndvi`
- `vessel_x_ndvi_lag1`
- `vessel_x_ndvi_lag2`
- `vessel_x_ndvi_lag3`
- `distance_to_coast_km`
- `nearest_land_grid_cell_id`
- `distance_to_nearest_land_ndvi_km`
- `nearest_land_ndvi_mean`
- `nearest_land_ndvi_median`
- `nearest_land_ndvi_valid_count`
- `shipping_distance_band_refined`

---

## Coastal wind alignment features (weekly grid export)

**Path:** `outputs/reports/run_coastal_wind_transport/coastal_wind_alignment_features.csv`  
**Feature count:** 26

**Features:**

- `grid_cell_id`
- `week_start_utc`
- `nearest_coast_ref_lat`
- `nearest_coast_ref_lon`
- `nearest_coast_ref_distance_km`
- `bearing_cell_to_coast_deg`
- `bearing_hotspot_to_coast_deg`
- `pollution_hotspot_lat`
- `pollution_hotspot_lon`
- `pollution_hotspot_type`
- `wind_u_mean`
- `wind_v_mean`
- `wind_speed_mean`
- `wind_direction_to_degrees`
- `coastal_wind_angle_diff_deg`
- `coastal_wind_alignment_score`
- `coastal_wind_shoreward_45deg`
- `pollution_transport_angle_diff_deg`
- `pollution_transport_wind_alignment_score`
- `distance_to_coast_km`
- `shipping_distance_band_tight`
- `vessel_density_t`
- `local_no2_excess`
- `weekly_no2_anomaly`
- `oil_slick_probability_t`
- `nearest_land_ndvi_mean`

---

## Baltic ports coordinates

**Path:** `data/aux/baltic_ports.csv`  
**Feature count:** 3

**Features:**

- `port_name`
- `latitude`
- `longitude`

---

## Notes

- Duplicate snapshots under **`final_run/`**, **`final_run_cleaned/`**, **`final_run_repaired_sources/`**, and **`final_run_stockholm_fixed_*`** usually share the **same column names** as the canonical **`processed/`** and **`data/aux/`** parquets above; only treat as new features if a run appended columns.
- Hundreds of small **summary CSVs** under **`outputs/reports/`** (decay tables, Mann–Whitney outputs, etc.) were not repeated here; they are **tabular results**, not base feature layers. Those can be listed the same way if you need a full report inventory.

