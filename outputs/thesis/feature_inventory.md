# ML feature inventory (thesis-ready)

## Dataset and supervised setup

- **Parquet analysed:** `data/modeling_dataset.parquet`
- **Row / column geometry:** `16,065` rows × `29` columns in file.
- **ML driver script:** `src/run_delta_ndti_models.py` (default `--input data/modeling_dataset.parquet`)

Predictors are every column except the metadata/target set:

`['delta_ndti', 'grid_cell_id', 'has_valid_delta_ndti', 'ndti_next_target', 'week_start_utc']`

- **ΔNDTI target:** `delta_ndti`
- **Secondary target:** `ndti_next_target = shift(sentinel_ndti_mean_t, −1)` by `grid_cell_id` (constructed after filters in `load_modeling_data`).

**Ridge regression pipeline:** `Pipeline(SimpleImputer(median), StandardScaler(), Ridge(alpha=1.0))` — Ridge receives the **same** numeric feature columns as HistGradientBoosting (median imputation is internal to Ridge only).

**HistGradientBoostingRegressor:** All listed predictors fed natively with missing-value handling (`nan` retained for tree splits). Early stopping uses a withheld train fraction.

- **Count of predictor columns |X|:** 25 (identical for Ridge & HistGradientBoosting).

## Feature counts by category (predictors in *X*)

| Category | Count |
|----------|-------|
| Environmental | 5 |
| Atmospheric | 0 |
| Maritime | 1 |
| Spatial | 3 |
| Temporal | 3 |
| Lagged | 10 |
| Engineered exposure features | 3 |

## Identifier / target rows (never in ridge/HGB)

| Name | Role | Notes |
|------|------|-------|
| `delta_ndti` | target_primary | NDTI(t+1) − NDTI(t); used as *y* in run_delta_ndti_models (ΔNDTI task). Never part of ridge/HGB input matrix. |
| `ndti_next_target` | target_secondary_constructed | Built at runtime as groupby-shift of sentinel_ndti_mean_t; second supervised task compares level vs change with identical *X*. |
| `grid_cell_id` | excluded_predictor_panel_key | Panel unit key; withheld from ridge/HGB. |
| `week_start_utc` | excluded_predictor_panel_key | Temporal index for time-aware split; withheld from ridge/HGB (encoded instead via seasonal features). |
| `has_valid_delta_ndti` | row_filter_helper | Flags rows where ΔNDTI is defined; filtering only — not fed to ML. |

## Predictor inventory (used by both Ridge & HistGradientBoosting)

| Feature | Category | Purpose (short) | Ridge X | HistGB X | Lag | Static geo | Exposure ctx |
|---------|----------|-----------------|---------|----------|-----|------------|--------------|
| `grid_centroid_lat` | Spatial | Static centroid latitude linking each grid cell on the Earth's surface. | yes | yes | no | yes | no |
| `grid_centroid_lon` | Spatial | Static centroid longitude for geographical heterogeneity. | yes | yes | no | yes | no |
| `grid_res_deg` | Spatial | Nominal spatial resolution / grid spacing of the modelling cells (degrees). | yes | yes | no | yes | no |
| `has_emodnet` | Engineered exposure features | Indicator that EMODnet vessel-density lineage contributed to integration. | yes | yes | no | no | yes |
| `has_helcom` | Engineered exposure features | Indicator (metadata lineage) referencing HELCOM-related harmonisation pathway. | yes | yes | no | no | yes |
| `has_sentinel` | Engineered exposure features | Indicator that Sentinel-2-based optical stack was populated for this row. | yes | yes | no | no | yes |
| `sentinel_evi_mean_t` | Environmental | Enhanced vegetation index summarising canopy structure / productivity. | yes | yes | no | no | no |
| `sentinel_evi_mean_t_minus_1` | Lagged | EVI lag 1. | yes | yes | yes | no | no |
| `sentinel_evi_mean_t_minus_2` | Lagged | EVI lag 2. | yes | yes | yes | no | no |
| `sentinel_ndti_mean_t` | Environmental | NDTI turbidity surrogate at week *t* — primary optical water-state covariate (also participates in ΔNDTI target logic upstream). | yes | yes | no | no | no |
| `sentinel_ndti_mean_t_minus_1` | Lagged | NDTI at *t−1;* central for short-term inertia in water optical properties. | yes | yes | yes | no | no |
| `sentinel_ndti_mean_t_minus_2` | Lagged | NDTI at *t−2* for smoother temporal context. | yes | yes | yes | no | no |
| `sentinel_ndvi_mean_t` | Environmental | Sentinel-2 NDVI spatial mean inside the footprint at week *t* (vegetation/coastal fringe context). | yes | yes | no | no | no |
| `sentinel_ndvi_mean_t_minus_1` | Lagged | NDVI lagged one week. | yes | yes | yes | no | no |
| `sentinel_ndvi_mean_t_minus_2` | Lagged | NDVI lagged two weeks. | yes | yes | yes | no | no |
| `sentinel_ndwi_mean_t` | Environmental | NDWI spatial mean — open water / inundation-sensitive index. | yes | yes | no | no | no |
| `sentinel_ndwi_mean_t_minus_1` | Lagged | NDWI lag 1. | yes | yes | yes | no | no |
| `sentinel_ndwi_mean_t_minus_2` | Lagged | NDWI lag 2. | yes | yes | yes | no | no |
| `sentinel_observation_count_t` | Environmental | Number of Sentinel-2 scenes contributing to composites (observational weight / completeness). | yes | yes | no | no | no |
| `vessel_density_t` | Maritime | EMODnet-derived aggregate vessel-density signal at calendar week *t* (maritime exposure proxy). | yes | yes | no | no | yes |
| `vessel_density_t_minus_1` | Lagged | Prior-week vessel density (1-week maritime lag). | yes | yes | yes | no | yes |
| `vessel_density_t_minus_2` | Lagged | Two-week-lagged maritime intensity. | yes | yes | yes | no | yes |
| `week_cos` | Temporal | Cosine seasonal encoding (pairs with week_sin for smooth periodicity). | yes | yes | no | no | no |
| `week_of_year` | Temporal | Ordinal week-of-year cyclical index for seasonality. | yes | yes | no | no | no |
| `week_sin` | Temporal | Sinusoidal seasonal encoding derived from UTC week anchor. | yes | yes | no | no | no |

## Thesis-oriented flags

- **Lagged predictor columns (10):** `sentinel_evi_mean_t_minus_1, sentinel_evi_mean_t_minus_2, sentinel_ndti_mean_t_minus_1, sentinel_ndti_mean_t_minus_2, sentinel_ndvi_mean_t_minus_1, sentinel_ndvi_mean_t_minus_2, sentinel_ndwi_mean_t_minus_1, sentinel_ndwi_mean_t_minus_2, vessel_density_t_minus_1, vessel_density_t_minus_2`
- **Static spatial predictors:** `grid_centroid_lat, grid_centroid_lon, grid_res_deg`
- **Maritime intensity + lineage flags grouped as operational exposure context:** `has_emodnet, has_helcom, has_sentinel, vessel_density_t, vessel_density_t_minus_1, vessel_density_t_minus_2`

### Atmospheric chemistry

**None** of the current predictor columns encode tropospheric NO₂ / chemistry; this modelling parquet carries optical + maritime + temporal encodings only.

## Thesis-ready methodological summary

### Why these features were selected

**Optical coherence:** NDVI / NDWI / EVI / NDTI and their lags summarise land–water contrasts, turbidity, and inertia in Sentinel-2 composites while respecting the embargo that ΔNDTI is built from contiguous valid weeks.
**Maritime stress:** Weekly vessel-density composites (with lags) capture industrial shipping pressure hypothesised to co-vary with nearshore perturbations.
**Seasonality:** Harmonic (`week_sin` / `week_cos`) and ordinal `week_of_year` approximate phenology-driven illumination and runoff cycles without handing the model absolute timestamps (which remain reserved for the split).
**Static geography:** Latitude, longitude (and nominal resolution) soak up unresolved spatial stratification omitted from causal covariates.
**Instrumentation hygiene:** Observation counts plus source flags tell the estimator how much independent radiometric evidence existed and whether upstream ingestion paths fired — especially relevant when Ridge relies on deterministic imputation before scaling.

### Exposure analysis alignment

Environmental exposure proxies here are multispectral composites interpreted as ecological–physical state (turbidity, vegetation/fringe moisture, generalized productivity). Maritime exposure proxies are summed vessel-density artefacts at the modelling grid centroid. Temporal encodings approximate seasonal forcing on water colour and illuminate regime shifts absent explicit synoptic meteorology.

### Temporal and spatial dependence

- **Temporal:** Dedicated lag columns (optical indices and vessel intensity at one- and two-week offsets) propagate short-term memory in water radiometry and activity; cyclic week-of-year harmonics summarise annual illumination and seasonal forcing absent raw timestamps.
- **Spatial dependence:** Latitude/longitude summarise location on the Baltic domain; finer spatial interactions are absorbed non-parametrically by HistGradientBoosting and linear-smoothed (after scaling) by Ridge. Correlation structures across neighbouring cells remain unmodelled (i.i.d.-style rows conditioned on engineered inputs). Panel keys (`grid_cell_id`, `week_start_utc`) constrain train/test leakage through the scripted week-split, not via random row shuffles.

## Calibration snapshot (`data/model_results.json`)

*Use as reporting aid; rerun `python3 src/run_delta_ndti_models.py` after data refresh.*

- **Objective recorded:** Predict delta_ndti and ndti_next (NDTI at t+1); time-aware train/test split; same features
- **Usable labelled rows cited:** `5293`
