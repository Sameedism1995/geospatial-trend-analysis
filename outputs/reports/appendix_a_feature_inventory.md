Appendix A. Complete Machine Learning Feature Inventory

This appendix summarises the environmental, atmospheric, maritime, spatial, temporal, and engineered variables used in the machine-learning-ready dataset. The features were derived from Sentinel-2, Sentinel-5P, Sentinel-1 SAR, EMODnet vessel-density layers, Open-Meteo ERA5 wind data, port/coastline GIS layers, and engineered exposure-analysis procedures.

**Primary panel:** `processed/features_ml_ready.parquet` — **16,065** grid-week rows × **46** columns (315 unique `grid_cell_id`, 40 UTC week anchors in this build).

`processed/merged_dataset.parquet` contains **27** columns (observational merge prior to land–sea engineering); all are a subset of or precursor to the **46** columns in `features_ml_ready.parquet`.

### Coverage summary (`features_ml_ready.parquet`)

| Category | Features | Mean_missing_pct |
| --- | --- | --- |
| Atmospheric variable | 4 | 10.9 |
| Engineered exposure index | 5 | 31.2 |
| Environmental indicator | 19 | 80.0 |
| Interaction feature | 5 | 85.2 |
| Maritime variable | 2 | 24.1 |
| Oil slick proxy | 3 | 3.6 |
| Port/geographic attribution feature | 2 | 0.0 |
| Spatial/proximity feature | 5 | 0.0 |
| Temporal feature | 1 | 0.0 |

### A.1 Core machine-learning-ready features

| Feature | Category | Type | Unit / Scale | Source | Purpose |
| --- | --- | --- | --- | --- | --- |
| grid_cell_id | Port/geographic attribution feature | string / object | categorical string (e.g. g0.100_r…_c…) | Pipeline grid lattice (`src/extract_sentinel_weekly_features.py`) | Unique spatial unit identifier linking all weekly observations to a lattice cell. |
| week_start_utc | Temporal feature | datetime64[UTC] | UTC timestamp (ISO week anchor) | Panel index (weekly aggregation) | Defines the weekly timestep for the balanced spatiotemporal panel. |
| vessel_density_t | Maritime variable | float64 | normalised vessel-density units (EMODnet layer; ~0–1 in panel) | EMODnet AIS maritime activity layers | Time-aligned maritime traffic intensity at each grid-week. |
| grid_centroid_lat | Spatial/proximity feature | float64 | degrees north (WGS84) | Grid lattice definition | Cell centroid latitude for mapping, distance metrics, and spatial ML predictors. |
| grid_centroid_lon | Spatial/proximity feature | float64 | degrees east (WGS84) | Grid lattice definition | Cell centroid longitude for mapping and spatial stratification. |
| no2_mean_t | Atmospheric variable | float64 | mol m⁻² (Sentinel-5P tropospheric NO₂ column mean, weekly) | Sentinel-5P NO₂ (GEE pipeline) | Weekly mean tropospheric NO₂ at the grid cell. |
| no2_std_t | Atmospheric variable | float64 | mol m⁻² (within-week std) | Sentinel-5P NO₂ | Sub-weekly NO₂ variability (episodic pollution vs stable background). |
| oil_slick_probability_t | Oil slick proxy | float64 | probability / fraction (0–1 dark-water anomaly proxy) | Sentinel-1 SAR dark-feature detection | Weekly mean probability of oil-slick-like dark features (not confirmed spill attribution). |
| oil_slick_count_t | Oil slick proxy | float64 | count (detections per cell-week) | Sentinel-1 SAR | Count of dark-feature detections supporting the oil proxy. |
| ndwi_mean | Environmental indicator | float64 | dimensionless index (−1 to 1 typical) | Sentinel-2 multispectral (NDWI) | Mean Normalised Difference Water Index — open water vs turbidity sensitivity. |
| ndwi_median | Environmental indicator | float64 | dimensionless | Sentinel-2 NDWI | Robust weekly NDWI aggregate. |
| ndwi_std | Environmental indicator | float64 | dimensionless | Sentinel-2 NDWI | Within-week NDWI dispersion. |
| ndti_mean | Environmental indicator | float64 | dimensionless turbidity index | Sentinel-2 NDTI | Mean Normalised Difference Turbidity Index — suspended matter / turbidity proxy. |
| ndti_median | Environmental indicator | float64 | dimensionless | Sentinel-2 NDTI | Robust NDTI aggregate. |
| ndti_std | Environmental indicator | float64 | dimensionless | Sentinel-2 NDTI | NDTI variability within the week. |
| ndci_mean | Environmental indicator | float64 | dimensionless chlorophyll proxy | Sentinel-2 NDCI | Mean Normalised Difference Chlorophyll Index — productivity / eutrophication-related signal. |
| ndci_median | Environmental indicator | float64 | dimensionless | Sentinel-2 NDCI | Robust NDCI aggregate. |
| ndci_std | Environmental indicator | float64 | dimensionless | Sentinel-2 NDCI | NDCI dispersion. |
| fai_mean | Environmental indicator | float64 | dimensionless floating-algae index | Sentinel-2 FAI | Floating algae / detritus / surface anomaly indicator. |
| fai_median | Environmental indicator | float64 | dimensionless | Sentinel-2 FAI | Robust FAI aggregate. |
| fai_std | Environmental indicator | float64 | dimensionless | Sentinel-2 FAI | FAI dispersion. |
| b11_mean | Environmental indicator | float64 | surface reflectance (Band 11 SWIR) | Sentinel-2 Band 11 | SWIR reflectance context for water / anomaly interpretation. |
| b11_median | Environmental indicator | float64 | reflectance | Sentinel-2 Band 11 | Robust Band-11 aggregate. |
| b11_std | Environmental indicator | float64 | reflectance | Sentinel-2 Band 11 | Band-11 dispersion. |
| ndvi_mean | Environmental indicator | float64 | dimensionless NDVI (−1 to 1) | Sentinel-2 NDVI (nearest-land linkage where computed) | Vegetation / land-response proxy for coastal–terrestrial coupling (sparse coverage). |
| ndvi_median | Environmental indicator | float64 | dimensionless | Sentinel-2 NDVI | Robust NDVI aggregate. |
| ndvi_std | Environmental indicator | float64 | dimensionless | Sentinel-2 NDVI | NDVI dispersion. |
| NO2_mean | Atmospheric variable | float64 | mol m⁻² (alias of weekly NO₂) | Derived from `no2_mean_t` in pipeline harmonisation | Thesis-facing NO₂ column for correlations and land–sea indices. |
| NO2_trend | Atmospheric variable | float64 | mol m⁻² week⁻¹ (first difference within cell) | Engineered from `NO2_mean` | Week-to-week NO₂ change per grid cell. |
| vessel_density | Maritime variable | float64 | normalised density (panel alias) | EMODnet AIS (alias of `vessel_density_t`) | Maritime intensity column used by interaction and min–max indices. |
| detection_score | Oil slick proxy | float64 | probability / score (0–1 typical) | Alias of primary oil-slick probability column | Harmonised detection score for ML and stratified NO₂–oil analysis. |
| nan_ratio_row | Environmental indicator | float64 | fraction in [0, 1] | Engineered QC | [Data-quality diagnostic] Row-wise fraction of missing values across input columns (not an ecological indicator). |
| nearest_port | Port/geographic attribution feature | string / object | categorical (port name) | Port/coastline GIS (`data/aux/baltic_ports.csv`) | Nearest focal port label for Turku / Mariehamn / Naantali stratification. |
| distance_to_port_km | Spatial/proximity feature | float64 | kilometres (geodesic) | Port GIS + grid centroids | Distance from cell centroid to nearest catalogued port — port-distance decay axis. |
| port_exposure_score | Engineered exposure index | float64 | density / km (vessel_density / (1 + distance_km)) | Engineered from EMODnet + port distance | Port-proximity-weighted maritime intensity. |
| distance_to_nearest_high_vessel_density_cell | Spatial/proximity feature | float64 | kilometres (haversine to weekly P90 vessel-density seeds) | EMODnet + lattice geometry | Proximity to high-traffic cells — shipping-lane / corridor structure. |
| coastal_exposure_band | Spatial/proximity feature | string / object | categorical band (0–10 km, 10–50 km, 50+ km) | Engineered from distance to high-density seeds | Discrete coastal exposure stratification. |
| coastal_exposure_score | Engineered exposure index | float64 | unitless score in [0, 1] | Engineered distance-decay exposure function | Piecewise coastal exposure to maritime activity (CES in thesis figures). |
| maritime_pressure_index | Engineered exposure index | float64 | min–max normalised [0, 1] over panel | Engineered from vessel density | Normalised maritime pressure composite (MEI input in Fig. 5.8-style ESI). |
| atmospheric_transfer_index | Engineered exposure index | float64 | min–max normalised [0, 1] | Engineered from NO2_mean | Normalised atmospheric transfer / NO₂ connectivity index. |
| land_response_index | Engineered exposure index | float64 | min–max normalised [0, 1] | Engineered from ndvi_mean | Normalised land-vegetation response where NDVI is available. |
| vessel_x_no2 | Interaction feature | float64 | product of density × mol m⁻² | Engineered interaction | Maritime–atmospheric co-stress interaction term. |
| no2_x_ndvi | Interaction feature | float64 | product of NO₂ × NDVI | Engineered interaction | Atmosphere–land coupling interaction. |
| vessel_x_ndvi_lag1 | Interaction feature | float64 | lagged product (1 week) | Engineered lag interaction | Vessel density at t−1 multiplied by NDVI at t (per grid). |
| vessel_x_ndvi_lag2 | Interaction feature | float64 | lagged product (2 weeks) | Engineered lag interaction | Two-week delayed vessel–vegetation interaction. |
| vessel_x_ndvi_lag3 | Interaction feature | float64 | lagged product (3 weeks) | Engineered lag interaction | Three-week delayed vessel–vegetation interaction. |

#### Example values and missingness (core panel)

| Feature | Missing_pct | Example | Created_in |
| --- | --- | --- | --- |
| grid_cell_id | 0.0 | g0.100_r1038_c1803 | src/extract_sentinel_weekly_features.py |
| week_start_utc | 0.0 | 2023-01-01 00:00:00+00:00 | src/pipeline/run_full_pipeline.py (panel construction) |
| vessel_density_t | 24.13 | 0.0052 | src/data_sources/emodnet_vessel_density.py (ingestion) |
| grid_centroid_lat | 0.0 | 13.8500 | src/extract_sentinel_weekly_features.py |
| grid_centroid_lon | 0.0 | 0.3500 | src/extract_sentinel_weekly_features.py |
| no2_mean_t | 10.1 | 1.0973e-05 | src/data_sources/no2_gee_pipeline.py |
| no2_std_t | 10.1 | 1.2340e-06 | src/data_sources/no2_gee_pipeline.py |
| oil_slick_probability_t | 5.41 | 0.0356 | src/data_sources/sentinel1_oil_pipeline.py |
| oil_slick_count_t | 0.0 | 0.0000e+00 | src/data_sources/sentinel1_oil_pipeline.py |
| ndwi_mean | 81.77 | -0.3475 | src/data_sources/sentinel2_water_quality.py |
| ndwi_median | 81.77 | -0.3476 | src/data_sources/sentinel2_water_quality.py |
| ndwi_std | 81.77 | 0.0546 | src/data_sources/sentinel2_water_quality.py |
| ndti_mean | 81.77 | 0.1979 | src/data_sources/sentinel2_water_quality.py |
| ndti_median | 81.77 | 0.1963 | src/data_sources/sentinel2_water_quality.py |
| ndti_std | 81.77 | 0.0230 | src/data_sources/sentinel2_water_quality.py |
| ndci_mean | 81.77 | 0.0643 | src/data_sources/sentinel2_water_quality.py |
| ndci_median | 81.77 | 0.0615 | src/data_sources/sentinel2_water_quality.py |
| ndci_std | 81.77 | 0.0125 | src/data_sources/sentinel2_water_quality.py |
| fai_mean | 81.77 | 0.0524 | src/data_sources/sentinel2_water_quality.py |
| fai_median | 81.77 | 0.0537 | src/data_sources/sentinel2_water_quality.py |
| fai_std | 81.77 | 0.0203 | src/data_sources/sentinel2_water_quality.py |
| b11_mean | 81.77 | 0.5020 | src/data_sources/sentinel2_water_quality.py |
| b11_median | 81.77 | 0.5176 | src/data_sources/sentinel2_water_quality.py |
| b11_std | 81.77 | 0.0702 | src/data_sources/sentinel2_water_quality.py |
| ndvi_mean | 97.7 | 0.1622 | src/analysis/run_nearest_land_ndvi_linkage.py |
| ndvi_median | 97.7 | 0.1614 | src/analysis/run_nearest_land_ndvi_linkage.py |
| ndvi_std | 97.7 | 0.0096 | src/analysis/run_nearest_land_ndvi_linkage.py |
| NO2_mean | 10.1 | 1.0973e-05 | src/pipeline/run_full_pipeline.py (`_harmonize_feature_aliases`) |
| NO2_trend | 13.12 | 3.1466e-06 | src/pipeline/run_full_pipeline.py (grouped diff on NO2_mean) |
| vessel_density | 24.13 | 0.0052 | src/pipeline/run_full_pipeline.py |
| detection_score | 5.41 | 0.0356 | src/pipeline/run_full_pipeline.py (alias of oil_slick_probability_t) |
| nan_ratio_row | 0.0 | 0.2581 | src/pipeline/run_full_pipeline.py |
| nearest_port | 0.0 | Stockholm | src/features/port_proximity.py |
| distance_to_port_km | 0.0 | 5263.7490 | src/features/port_proximity.py |
| port_exposure_score | 24.13 | 1.8494e-06 | src/features/port_exposure.py |
| distance_to_nearest_high_vessel_density_cell | 0.0 | 5333.8388 | src/features/land_sea_buffering.py |
| coastal_exposure_band | 0.0 | 50+km | src/features/land_sea_buffering.py |
| coastal_exposure_score | 0.0 | 1.1287e-24 | src/features/land_sea_buffering.py |
| maritime_pressure_index | 24.13 | 2.0049e-04 | src/features/land_sea_interactions.py |
| atmospheric_transfer_index | 10.1 | 0.3776 | src/features/land_sea_interactions.py |
| land_response_index | 97.7 | 0.5811 | src/features/land_sea_interactions.py |
| vessel_x_no2 | 32.85 | 6.7591e-08 | src/features/land_sea_interactions.py |
| no2_x_ndvi | 98.91 | 2.8549e-06 | src/features/land_sea_interactions.py |
| vessel_x_ndvi_lag1 | 98.1 | 8.2740e-05 | src/features/land_sea_interactions.py |
| vessel_x_ndvi_lag2 | 98.1 | 2.0719e-04 | src/features/land_sea_interactions.py |
| vessel_x_ndvi_lag3 | 98.1 | 2.0719e-04 | src/features/land_sea_interactions.py |

### A.2 Additional features in `data/modeling_dataset.parquet` (ΔNDTI ML task)

Used by Ridge / HistGradientBoosting models in `src/run_delta_ndti_models.py` (time-aware week split, target `delta_ndti`). Columns below are **not** stored in `features_ml_ready.parquet`.

| Feature | Category | Type | Unit / Scale | Source | Purpose |
| --- | --- | --- | --- | --- | --- |
| grid_res_deg | Spatial/proximity feature | float64 | degrees | Encoded in grid_cell_id prefix | Nominal lattice resolution for ML spatial predictors. |
| week_of_year | Temporal feature | Int64 | integer 1–53 | Derived from week_start_utc | Seasonal cycle encoding for ML. |
| week_sin | Temporal feature | float64 | dimensionless [−1, 1] | Fourier seasonality | Sine component of annual seasonality. |
| week_cos | Temporal feature | float64 | dimensionless [−1, 1] | Fourier seasonality | Cosine component of annual seasonality. |
| vessel_density_t_minus_1 | Temporal feature | float64 | lagged maritime density | EMODnet (1-week lag per grid) | Autoregressive maritime predictor for ΔNDTI models. |
| vessel_density_t_minus_2 | Temporal feature | float64 | lagged maritime density | EMODnet (2-week lag) | Second maritime lag feature. |
| sentinel_ndvi_mean_t | Environmental indicator | float64 | dimensionless NDVI | Sentinel-2 | Current-week NDVI for ML panel. |
| sentinel_ndvi_mean_t_minus_1 | Temporal feature | float64 | NDVI lag 1 | Sentinel-2 | Spectral lag predictor. |
| sentinel_ndvi_mean_t_minus_2 | Temporal feature | float64 | NDVI lag 2 | Sentinel-2 | Spectral lag predictor. |
| sentinel_ndwi_mean_t | Environmental indicator | float64 | NDWI | Sentinel-2 | Current-week NDWI for ML. |
| sentinel_ndwi_mean_t_minus_1 | Temporal feature | float64 | NDWI lag 1 | Sentinel-2 | Spectral lag. |
| sentinel_ndwi_mean_t_minus_2 | Temporal feature | float64 | NDWI lag 2 | Sentinel-2 | Spectral lag. |
| sentinel_evi_mean_t | Environmental indicator | float64 | EVI | Sentinel-2 | Enhanced vegetation index at t. |
| sentinel_evi_mean_t_minus_1 | Temporal feature | float64 | EVI lag 1 | Sentinel-2 | Spectral lag. |
| sentinel_evi_mean_t_minus_2 | Temporal feature | float64 | EVI lag 2 | Sentinel-2 | Spectral lag. |
| sentinel_ndti_mean_t | Environmental indicator | float64 | NDTI | Sentinel-2 | Current-week turbidity for ΔNDTI target construction. |
| sentinel_ndti_mean_t_minus_1 | Temporal feature | float64 | NDTI lag 1 | Sentinel-2 | Spectral lag. |
| sentinel_ndti_mean_t_minus_2 | Temporal feature | float64 | NDTI lag 2 | Sentinel-2 | Spectral lag. |
| sentinel_observation_count_t | Panel metadata / quality | int64 | count | Sentinel-2 | Number of valid S2 observations contributing to the weekly aggregate. |
| delta_ndti | Environmental indicator | float64 | NDTI change (t+1 − t) | Engineered target | Primary ML target: week-ahead turbidity change. |
| has_valid_delta_ndti | Temporal feature | boolean | boolean | Engineered mask | Flags rows usable for ΔNDTI training. |
| has_sentinel | Temporal feature | boolean | boolean | Coverage flag | [ML mask] Sentinel-2 data present at t. |
| has_emodnet | Temporal feature | boolean | boolean | Coverage flag | [ML mask] EMODnet vessel layer present. |
| has_helcom | Temporal feature | boolean | boolean | Coverage flag | [ML mask] HELCOM/auxiliary data flag. |

| Feature | Missing_pct | Example | Created_in |
| --- | --- | --- | --- |
| grid_res_deg | 0.0 | 0.1000 | data/modeling_dataset build (pipeline) |
| week_of_year | 0.0 | 52 | data/modeling_dataset build |
| week_sin | 0.0 | -0.2349 | data/modeling_dataset build |
| week_cos | 0.0 | 0.9720 | data/modeling_dataset build |
| vessel_density_t_minus_1 | 24.13 | 0.0000e+00 | data/modeling_dataset build |
| vessel_density_t_minus_2 | 24.13 | 0.0000e+00 | data/modeling_dataset build |
| sentinel_ndvi_mean_t | 51.22 | -0.2900 | data/modeling_dataset build |
| sentinel_ndvi_mean_t_minus_1 | 52.26 | -0.2900 | data/modeling_dataset build |
| sentinel_ndvi_mean_t_minus_2 | 52.26 | -0.2900 | data/modeling_dataset build |
| sentinel_ndwi_mean_t | 51.22 | 0.4337 | data/modeling_dataset build |
| sentinel_ndwi_mean_t_minus_1 | 52.26 | 0.4337 | data/modeling_dataset build |
| sentinel_ndwi_mean_t_minus_2 | 52.26 | 0.4337 | data/modeling_dataset build |
| sentinel_evi_mean_t | 51.22 | -0.0708 | data/modeling_dataset build |
| sentinel_evi_mean_t_minus_1 | 52.26 | -0.0708 | data/modeling_dataset build |
| sentinel_evi_mean_t_minus_2 | 52.26 | -0.0708 | data/modeling_dataset build |
| sentinel_ndti_mean_t | 51.22 | -0.1651 | data/modeling_dataset build |
| sentinel_ndti_mean_t_minus_1 | 52.26 | -0.1651 | data/modeling_dataset build |
| sentinel_ndti_mean_t_minus_2 | 52.26 | -0.1651 | data/modeling_dataset build |
| sentinel_observation_count_t | 0.0 | 0 | data/modeling_dataset build |
| delta_ndti | 65.66 | 0.1312 | src/run_delta_ndti_models.py / modeling_dataset build |
| has_valid_delta_ndti | 0.0 | False | data/modeling_dataset build |
| has_sentinel | 0.0 | False | data/modeling_dataset build |
| has_emodnet | 0.0 | True | data/modeling_dataset build |
| has_helcom | 0.0 | False | data/modeling_dataset build |

### A.3 Runtime-derived features (coastal exposure & wind merge)

These variables are computed during coastal-exposure and wind-transport analysis and merged at runtime (e.g. `scripts/fix_rq_evidence_pipeline.py`, `src/analysis/run_coastal_exposure_analysis.py`); they are **not** persisted as columns in `features_ml_ready.parquet`.

| Feature | Category | Type | Unit / Scale | Source | Purpose |
| --- | --- | --- | --- | --- | --- |
| maritime_exposure_index (MEI) | Engineered exposure index | float64 | rank percentile [0, 1] | Engineered: vessel × wind-alignment × inverse coast distance | Maritime exposure composite (RQ2, coastal analysis). |
| atmospheric_coastal_exposure_index (ACEI) | Engineered exposure index | float64 | rank percentile [0, 1] | Engineered: local NO₂ excess × coastal wind × transport alignment | Atmospheric coastal exposure composite. |
| environmental_stress_index (ESI) | Engineered exposure index | float64 | rank percentile [0, 1] | Engineered: weekly z-mean of NO₂, vessel, NDTI, oil, wind terms | Experimental multivariate stress composite (Fig. 5.8, sensitivity analysis). |
| coastal_wind_alignment_score | Wind/meteorological feature | float64 | cos(angle) in [−1, 1] | Open-Meteo ERA5 wind + coastline bearing | Alignment of wind toward nearest coast bearing. |
| coastal_wind_shoreward_45deg | Wind/meteorological feature | float64 | binary {0, 1} | ERA5-derived; threshold cos(45°) | Shoreward vs non-shoreward wind regime (§5.4, §5.7). |
| wind_u_mean / wind_v_mean | Wind/meteorological feature | float64 | m s⁻¹ | Open-Meteo ERA5 archive | Mean zonal/meridional wind components for the grid-week. |
| local_no2_excess | Atmospheric variable | float64 | mol m⁻² anomaly | Engineered from no2_mean_t vs 15–30 km band baseline | Weekly NO₂ excess relative to offshore reference band. |

### A.4 Engineering pipeline map

| Stage | Module | Outputs |
| --- | --- | --- |
| Ingestion | `src/data_sources/sentinel2_water_quality.py` | NDWI, NDTI, NDCI, FAI, B11 weekly stats |
| Ingestion | `src/data_sources/no2_gee_pipeline.py` | `no2_mean_t`, `no2_std_t` |
| Ingestion | `src/data_sources/sentinel1_oil_pipeline.py` | `oil_slick_probability_t`, `oil_slick_count_t` |
| Ingestion | EMODnet vessel layers | `vessel_density_t` |
| Merge | `processed/merged_dataset.parquet` | Observational columns only (27 fields) |
| Port GIS | `src/features/port_proximity.py` | `nearest_port`, `distance_to_port_km` |
| Port exposure | `src/features/port_exposure.py` | `port_exposure_score` |
| Land–sea buffer | `src/features/land_sea_buffering.py` | `distance_to_nearest_high_vessel_density_cell`, `coastal_exposure_band`, `coastal_exposure_score` |
| Land–sea interactions | `src/features/land_sea_interactions.py` | MEI/ATI/LRI aliases + interaction terms |
| Harmonisation | `src/pipeline/run_full_pipeline.py` | `NO2_mean`, `NO2_trend`, `vessel_density`, `detection_score`, `nan_ratio_row` |
| Final ML table | `processed/features_ml_ready.parquet` | 46-column thesis panel |
| Wind transport | `src/analysis/run_coastal_wind_transport.py` | Wind alignment CSV → MEI/ACEI/ESI via `build_indices` |
| ML modelling | `data/modeling_dataset.parquet` | Lags, seasonality, `delta_ndti` target |

### A.5 Notes for interpretation

- **Duplicate maritime / NO₂ columns:** `vessel_density_t` and `vessel_density`, `no2_mean_t` and `NO2_mean` are harmonised aliases for the same signals.
- **MEI in figures:** `maritime_pressure_index` in the parquet corresponds to the min–max maritime pressure index; thesis Figure 5.8 ESI uses a related but distinct six-variable z-mean (see `scripts/generate_thesis_sections_5_5_to_5_10.py`).
- **Sparse optical coverage:** `ndvi_*` and land-linked interactions exceed **97%** missing in the coastal panel; interpret as optional land-side comparators.
- **Oil variables:** Treat as **dark-feature proxies**, not confirmed oil-spill detections.

*Generated by `scripts/build_appendix_a_feature_inventory.py`.*
