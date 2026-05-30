# ML-Based Geospatial Analysis of Coastal Water Quality and Maritime Traffic
## Complete Project Description + Claude Training Prompt

---

# PART 1: FULL PROJECT DESCRIPTION

---

## 1. What This Project Is

This is a research thesis project that uses satellite remote sensing, atmospheric data, and maritime traffic data to study whether shipping activity in the Baltic Sea and North Sea region is spatially associated with changes in coastal water quality and land-environment conditions. It is a full end-to-end geospatial data science pipeline — from raw data ingestion to machine learning models to thesis-quality figures.

The project makes no causal claims. It finds spatial and temporal associations between maritime pressure and environmental indicators, framed explicitly as correlational/associative analysis for a dissertation.

---

## 2. Study Area and Ports

The geographic focus is the Baltic Sea / Northern European coastal zone, centered on five named ports:

| Port | Latitude | Longitude |
|---|---|---|
| Stockholm | 59.3293°N | 18.0686°E |
| Turku | 60.435°N | 22.225°E |
| Naantali | 60.467°N | 22.033°E |
| Mariehamn | 60.097°N | 19.934°E |
| Helsinki | 60.169°N | 24.938°E |

Grid resolution: 0.1° × 0.1° (approximately 11 km × 11 km per cell).
Total study grid: 315 unique grid cells.
Geographic extent: approximately −0.75° to 63.05°N latitude.

---

## 3. Time Period

All data covers 2023 only — specifically 2023-01-01 to 2023-12-17, organized into 51 weekly time steps (ISO calendar weeks). Every data point is indexed by (grid_cell_id, week_start_utc).

---

## 4. Data Sources

### 4.1 Sentinel-2 Water Quality (via Google Earth Engine)
Sentinel-2 multispectral satellite imagery processed to weekly per-grid optical water quality indices:

- NDWI (Normalized Difference Water Index): mean = 0.059, coverage = 18.2%
- NDTI (Normalized Difference Turbidity Index): mean = −0.138, range −1.0 to +0.088
- NDCI (Normalized Difference Chlorophyll Index): mean = 0.038
- FAI (Floating Algae Index): mean = 0.048
- B11 (SWIR Band 11): mean and std available
- Coverage is 18.2% due to cloud masking — optical sensors miss cloudy weeks

### 4.2 Sentinel-1 SAR Oil Slick Proxy (via Google Earth Engine)
Synthetic Aperture Radar (SAR) data processed to generate a dark-water anomaly / oil slick detection proxy:

- oil_slick_probability_t: probability of a dark-water anomaly (proxy for oil slicks / surface films)
- oil_slick_count_t: count of anomalous detections per cell-week
- Mean oil_slick_probability_t = 0.268, coverage = 94.6% (SAR is cloud-independent)
- Explicitly labelled as exploratory — not verified oil spill detection

### 4.3 Sentinel-2 Land NDVI (via Google Earth Engine)
Land-masked NDVI extracted from Sentinel-2 over coastal land cells:

- NDVI mean, median, std per grid-week
- Coverage: 2.3% of grid-weeks (expected — most grid cells are ocean/sea, not land)
- Used in the Land Impact Extension to study how maritime pressure propagates to nearby land vegetation

### 4.4 Tropospheric NO₂ (via Google Earth Engine — Sentinel-5P)
Weekly tropospheric NO₂ column density per grid cell:

- no2_mean_t: mean = 1.53 × 10⁻⁵ mol/m², coverage = 89.9%
- no2_std_t: variability within the grid-week, coverage = 89.9%
- Used as an atmospheric pollution proxy — ships burn fuel and emit NOₓ

### 4.5 Vessel Density / Maritime Traffic (EMODnet / AIS proxy)
Spatial vessel density derived from AIS data via EMODnet:

- vessel_density_t: mean = 0.5255, std = 1.828, coverage = 75.9%
- Also stored as 1-week and 2-week lags: vessel_density_t_minus_1, vessel_density_t_minus_2
- This is a spatial congestion proxy — not weekly AIS-derived traffic counts

### 4.6 HELCOM (Baltic Marine Environment Protection Commission)
Reference environmental monitoring records:

- helcom_record_count: mean = 6.83 records per cell, coverage = 100%
- Up to 10 records per cell
- Used as a validation/reference layer

### 4.7 EMODnet Static Bathymetry / Seabed
EMODnet static records (bathymetry and seabed data):

- emodnet_record_count_static: mean = 0.63, max = 46, coverage = 100%

---

## 5. Dataset Sizes

| Dataset | Rows | Columns |
|---|---|---|
| master_dataset.parquet | 16,065 | 35 |
| modeling_dataset.parquet | 16,065 | 29 |
| features_ml_ready.parquet (final ML features) | 16,065 | 46 |
| NO₂ aux parquet | 16,065 | 4 |
| Sentinel-1 oil slicks parquet | 16,065 | 4 |
| Sentinel-2 water quality parquet | 16,065 | 17 |

The 16,065 rows = 315 grid cells × 51 weeks.

---

## 6. Feature Engineering (54 total features across 6 categories)

| Category | Count | Key Features |
|---|---|---|
| Water Quality | 14 | NDCI, NDTI, NDWI, FAI, B11 SWIR, means/medians/stds |
| Maritime Activity | 8 | vessel_density, total_density, density_total_log, seasonal_metrics variants |
| Atmospheric | 9 | NO₂ mean, NO₂ std, NO₂ trend, detection_score, oil slick count/prob |
| Exposure | 2 | distance_to_port, distance_to_lane |
| Validation | 3 | chlorophyll-a in-situ, EQRS score, water quality class (all missing in current data) |
| Land (extension) | 18 | NDVI mean/std/median, coastal_exposure_score, distance_to_high_vessel_cell, land_response_index, maritime_pressure_index, atmospheric_transfer_index, interaction terms, lags |

Additional engineered features in the 46-column ML-ready dataset:
- NO2_mean, NO2_trend (derived from NO₂ weekly difference)
- vessel_density, detection_score (renamed from source columns)
- nan_ratio_row (fraction of missing values per row)
- nearest_port, distance_to_port_km, port_exposure_score (from 5 Baltic ports)
- coastal_exposure_score, coastal_exposure_band (buffered exposure to vessel density)
- maritime_pressure_index, atmospheric_transfer_index
- land_response_index
- Interaction terms: vessel_x_no2, no2_x_ndvi, vessel_x_ndvi_lag1/2/3

---

## 7. Machine Learning Models

### 7.1 Primary Task: Predicting Delta-NDTI

Target: delta_ndti = NDTI(week t+1) − NDTI(week t) — week-to-week change in turbidity.

Dataset for modeling: 5,293 rows with valid delta-NDTI (34.3% of all grid-weeks).

Time-aware train/test split (no random shuffle — strictly temporal):
- Total distinct weeks: 40
- Train: 30 weeks (2023-01-15 to 2023-08-06) → 4,394 rows
- Test: 10 weeks (2023-08-13 to 2023-11-12) → 899 rows

Features (25 total): grid coordinates (lat, lon, res), temporal encoding (week_of_year, week_sin, week_cos), vessel density (t, t−1, t−2), Sentinel spectral lags (NDVI, NDWI, EVI, NDTI at t, t−1, t−2), sentinel_observation_count_t, boolean flags (has_sentinel, has_emodnet, has_helcom).

**Model 1 — Ridge Regression**
- Preprocessing: SimpleImputer(median) on train only → StandardScaler → Ridge(alpha=1.0)
- Train: RMSE=0.1457, MAE=0.1099, R²=0.446
- Test:  RMSE=0.3042, MAE=0.2628, R²=−0.871

**Model 2 — HistGradientBoostingRegressor (winner)**
- Handles NaN natively; no imputation
- Train: RMSE=0.0366, MAE=0.0251, R²=0.965
- Test:  RMSE=0.2284, MAE=0.1783, R²=−0.055
- Beats Ridge on test RMSE by 0.076

**Top 5 features by permutation importance (HistGBR):**
1. sentinel_ndti_mean_t — 0.188
2. week_of_year — 0.049
3. week_sin — 0.042
4. sentinel_evi_mean_t_minus_1 — 0.030
5. sentinel_ndti_mean_t_minus_1 — 0.026

Vessel density importance: 0.0025 (near zero).

Feature group importances (summed):
- Lagged spectral features: 0.343 (dominant)
- Temporal week encoding: 0.108
- Static geo (lat/lon): 0.046
- Vessel density (all lags): 0.0025
- Boolean flags: 0.0

### 7.2 Secondary Task: Predicting NDTI-Next (absolute level)

Target: NDTI at week t+1 (same features, same split).
- Ridge: test RMSE=0.293, R²=−2.138
- HistGBR: test RMSE=0.296, R²=−2.187
- Both fail to generalize — absolute level is harder to predict than the change.

### 7.3 Rolling Window Cross-Validation (4 folds)

Expanding window: fold 1 trains on 20 weeks, fold 2 on 25, fold 3 on 30, fold 4 on 35. Each tests on the next 5 weeks.

| Model | Mean test RMSE | Std | Mean test R² |
|---|---|---|---|
| Ridge (delta-NDTI) | 0.192 | ±0.054 | −0.128 |
| HistGBR (delta-NDTI) | 0.193 | ±0.055 | −0.085 |
| Ridge (NDTI-next) | 0.178 | ±0.052 | −0.837 |
| HistGBR (NDTI-next) | 0.202 | ±0.103 | −2.033 |

Best single fold: HistGBR fold 1, RMSE=0.132, R²=0.524
Worst single fold: HistGBR fold 3, RMSE=0.242, R²=−1.111
High fold-to-fold dispersion (relative std ~51%) consistent with seasonal/environmental variability.

---

## 8. Correlation Analysis

### 8.1 Pearson Correlations (8 core variables)

| Feature A | Feature B | Pearson r |
|---|---|---|
| ndwi_mean | ndci_mean | −0.844 |
| ndwi_mean | fai_mean | −0.775 |
| ndci_mean | fai_mean | +0.764 |
| ndti_mean | b11_mean | +0.700 |
| ndwi_mean | ndti_mean | −0.680 |
| ndwi_mean | b11_mean | −0.573 |
| ndci_mean | detection_score | −0.454 |
| fai_mean | detection_score | −0.479 |
| NO2_mean | ndci_mean | +0.304 |
| vessel_density | detection_score | −0.117 |
| vessel_density | ndci_mean | +0.144 |
| vessel_density | NO2_mean | +0.039 |

### 8.2 Spearman Correlations (rank-based)

| Feature A | Feature B | Spearman ρ |
|---|---|---|
| ndwi_mean | ndci_mean | −0.876 (strongest) |
| ndwi_mean | b11_mean | −0.842 |
| ndci_mean | fai_mean | +0.796 |
| ndwi_mean | fai_mean | −0.789 |
| ndci_mean | b11_mean | +0.728 |
| ndwi_mean | ndti_mean | −0.715 |
| ndti_mean | b11_mean | +0.663 |
| fai_mean | detection_score | −0.565 |
| ndci_mean | detection_score | −0.550 |
| vessel_density | ndci_mean | +0.101 |
| vessel_density | NO2_mean | +0.096 |
| vessel_density | detection_score | −0.036 |

Key finding: The strongest correlations are between water quality indices themselves (optical physics). Vessel density shows only weak associations with all environmental indicators.

---

## 9. Statistical Tests

### 9.1 High vs Low Traffic t-tests

| Variable | High-traffic mean | Low-traffic mean | t-statistic | p-value |
|---|---|---|---|---|
| NDCI | −1.051 | +1.051 | −3.474 | 0.025 |
| NDTI | +1.241 | −1.038 | +6.008 | 0.0039 |
| NDWI | −1.065 | +1.301 | −5.500 | 0.013 |

All three are statistically significant (p < 0.05). High-traffic cells show higher NDTI (more turbidity) and lower NDWI/NDCI.

### 9.2 Causal Lag Analysis (vessel density → environmental response, lags 0–5 weeks)

- vessel_density → NO2_mean: peak lag at week 4, r=0.041 (very weak, positive) — labelled "no response"
- vessel_density → detection_score: peak at lag 0, r=−0.117 — labelled "immediate coupling"

### 9.3 Land-Sea Lag Analysis

- NO2_mean → ndvi_mean: peak at 4-week lag, Spearman ρ=0.661, Pearson r=0.594, n=360 — ranked strong
- vessel_density → ndvi_mean: peak at 1-week lag, Spearman ρ=0.360, Pearson r=0.156, n=306 — ranked moderate

---

## 10. Feature Interaction Analysis (77 cross-domain pairs, top results)

| Feature 1 | Feature 2 | Correlation | Category pair |
|---|---|---|---|
| b11_mean | distance_to_nearest_high_vessel_density_cell | +0.748 | water quality ↔ land |
| no2_mean_t | ndvi_std | +0.679 | atmospheric ↔ land |
| no2_mean_t | ndvi_median | +0.589 | atmospheric ↔ land |
| oil_slick_probability_t | ndvi_mean | −0.588 | atmospheric ↔ land |
| oil_slick_probability_t | land_response_index | −0.588 | atmospheric ↔ land |
| no2_mean_t | ndvi_mean | +0.583 | atmospheric ↔ land |
| ndti_mean | distance_to_nearest_high_vessel_density_cell | +0.534 | water quality ↔ land |

---

## 11. Anomaly Detection

Isolation Forest applied to the 46-feature space per grid-week (16,065 rows).

**Top spatial-outlier grid-week events:**

| Grid Cell | Week | Lat/Lon | Anomaly Score |
|---|---|---|---|
| g0.100_r1503_c2022 | 2023-03-05 | 60.35°N, 22.25°E | 2.186 |
| g0.100_r1503_c2022 | 2023-01-15 | 60.35°N, 22.25°E | 1.986 |
| g0.100_r1503_c2022 | 2023-02-19 | 60.35°N, 22.25°E | 1.872 |
| g0.100_r1503_c2022 | 2023-04-09 | 60.35°N, 22.25°E | 1.820 |
| g0.100_r1503_c2022 | 2023-01-01 | 60.35°N, 22.25°E | 1.795 |

Grid cell g0.100_r1503_c2022 at 60.35°N, 22.25°E (near Turku/Naantali, Finland) is persistently the most anomalous cell across the year.

**Top temporal anomalies (within-cell z-score):**

| Grid Cell | Week | z-score | Anomaly Score |
|---|---|---|---|
| g0.100_r1496_c2004 | 2023-02-19 | z=5.35 | 0.942 |
| g0.100_r1504_c2020 | 2023-03-05 | z=5.16 | 1.158 |
| g0.100_r1503_c2020 | 2023-03-05 | z=5.09 | 1.158 |

---

## 12. Coastal Impact Score

Composite score (0–1) built from 4 sub-components: correlation, lag response, exposure, and anomaly. Applied to all 16,065 rows.

**Top 10 coastal impact zones (highest scoring distinct cells):**

| Grid Cell | Best Week | Lat/Lon | Score |
|---|---|---|---|
| g0.100_r1503_c2022 | 2023-11-05 | 60.35°N, 22.25°E | 0.682 |
| g0.100_r1366_c1858 | 2023-09-24 | 46.65°N, 5.85°E | 0.634 |
| g0.100_r1498_c1995 | 2023-09-24 | — | 0.615 |
| g0.100_r1354_c1851 | 2023-11-05 | — | 0.608 |
| g0.100_r1496_c2018 | 2023-09-17 | — | 0.601 |
| g0.100_r1506_c2009 | 2023-09-17 | — | 0.600 |
| g0.100_r1503_c2020 | 2023-09-17 | — | 0.598 |
| g0.100_r990_c1803 | 2023-05-14 | — | 0.595 |
| g0.100_r1146_c1763 | 2023-11-05 | — | 0.591 |
| g0.100_r1503_c2021 | 2023-09-17 | — | 0.589 |

---

## 13. Coastal Shipping Impact Analysis

Restricted to grids within ≤30 km of coastline and ≤20 km of high vessel-density cells.

**High vs Low vessel exposure comparison:**

| Indicator | High-exposure mean | Low-exposure mean | p-value (Welch) | Significant? |
|---|---|---|---|---|
| no2_mean_t | 1.499×10⁻⁵ | 1.624×10⁻⁵ | 0.543 | No |
| ndti_mean | −0.140 | −0.165 | 0.259 | No |
| ndwi_mean | +0.070 | −0.030 | 0.105 | No |
| ndvi_mean | 0.260 | 0.441 | 0.108 | No |
| oil_slick_probability_t | 0.194 | 0.064 | 3.5×10⁻¹³ | YES (Cohen's d=0.849) |
| detection_score | 0.194 | 0.064 | 3.5×10⁻¹³ | YES |

The only statistically significant result: oil_slick/SAR detection score is 3× higher in high-vessel-exposure zones.

**Distance bands from coastline:**

| Band | NO₂ mean | NDTI mean | Oil slick prob | NDVI mean |
|---|---|---|---|---|
| 0–5 km | 1.469×10⁻⁵ | −0.151 | 0.167 | 0.276 |
| 5–10 km | 1.361×10⁻⁵ | −0.164 | 0.141 | 0.293 |
| 10–20 km | 1.520×10⁻⁵ | −0.173 | 0.144 | 0.335 |

---

## 14. Land Impact Extension

- Coastal exposure score: mean=0.442, std=0.337, coverage=100%
- Land response index (NDVI-normalized by exposure): mean=0.606, std=0.099, coverage=2.3%
- Distance to nearest high vessel density cell: mean=682 km, std=1,425 km
- Most cells are in the 50+ km coastal exposure band

---

## 15. Output Files

| Type | Location | Count |
|---|---|---|
| Thesis figures (PNG + PDF) | outputs/final_thesis_figures/ | 14 PNG + 14 PDF |
| Report CSVs | outputs/reports/ | 50+ files |
| ML model results | data/model_results.json | 1 |
| Rolling CV results | outputs/ml_cv_results/ | 4 CSV + 1 JSON + PNGs |
| Preprocessing diagnostics | outputs/preprocessing_diagnostics/ | 8 stage folders |
| EDA plots | outputs/plots/ | Multiple |
| Coastal figures | outputs/figures/run_final_coastal_shipping_impact/ | 9 PNGs + 1 HTML map |
| All statistics inventory | outputs/reports/ALL_STATISTICS_INVENTORY.md | 143 source files, 2.5M cells |
| Dashboard | dashboard/app.py (Streamlit, port 8501) | Live |

**14 Thesis figures:**
1. thesis_framework_diagram.png/pdf
2. ridge_feature_importance.png/pdf
3. hgbr_permutation_importance.png/pdf
4. environmental_correlation_network.png/pdf
5. temporal_persistence_heatmap.png/pdf
6. comparative_distance_decay.png/pdf
7. coastal_exposure_hotspots.png/pdf
8. integrated_exposure_map_baltic_overview.png/pdf
9. integrated_exposure_map_mariehamn.png/pdf
10. integrated_exposure_map_stockholm.png/pdf
11. integrated_exposure_map_turku.png/pdf
12. composite_land_exposure_dashboard.png/pdf
13. thesis_dashboard_refined.png/pdf
14. fig_coastal_port_adjacent_lattice_analysis.png/pdf

---

## 16. Project Architecture

```
geospatial-trend-analysis/
├── src/
│   ├── ingestion/          # EMODnet, HELCOM, Sentinel Hub API clients
│   ├── data_sources/       # GEE pipelines (NO2, Sentinel-1, Sentinel-2 water/land)
│   ├── features/           # Port proximity, coastal exposure, land-sea buffering
│   ├── analysis/           # EDA, correlation, anomaly, coastal impact, land impact
│   ├── ml/                 # Rolling window cross-validation
│   ├── pipeline/           # run_full_pipeline.py (master orchestrator)
│   ├── validation/         # Data quality audits
│   └── visualization/      # Impact heatmap, port exposure maps
├── scripts/                # Thesis figure generation scripts (14 scripts)
├── dashboard/              # Streamlit app (app.py)
├── data/
│   ├── aux/                # Pre-extracted parquets (NO2, SAR, water quality, land NDVI)
│   ├── master_dataset.parquet
│   ├── modeling_dataset.parquet
│   └── validation/
├── processed/              # Merged and ML-ready parquets
├── outputs/
│   ├── final_thesis_figures/
│   ├── reports/
│   ├── ml_cv_results/
│   ├── preprocessing_diagnostics/
│   └── plots/
└── logs/
```

---

## 17. Key Findings Summary

1. Water quality indices are strongly intercorrelated with each other (Spearman up to −0.876 between NDWI and NDCI) — optical physics, not a shipping finding.
2. Vessel density has near-zero ML feature importance (0.0025) for predicting NDTI change.
3. The SAR-derived oil slick probability is the only indicator with statistically significant association with vessel exposure zones (p=3.5×10⁻¹³, 3× higher in high-exposure areas).
4. NDTI is higher in high-traffic grid cells (t-test p=0.004) but this does not hold after coastal+lane restriction.
5. NO₂ leads land NDVI by 4 weeks (Spearman ρ=0.661) — strongest temporal lagged relationship found.
6. The Turku/Naantali area (60.35°N, 22.25°E, grid g0.100_r1503_c2022) is the persistently highest anomaly and coastal impact cell across the entire year.
7. ML models do not generalize well — test R² near zero or negative for both models and both tasks.
8. All results are associative, not causal. No wind, current, or confounding terrestrial emission controls have been applied.

---
---

# PART 2: CLAUDE TRAINING PROMPT

Use the text below as a system prompt or knowledge document to train the Claude web version on this project.

---

```
You are an expert assistant on a geospatial data science thesis project. Here is everything about the project:

## Project Name
ML-Based Geospatial Analysis of Coastal Water Quality and Maritime Traffic (Baltic Sea, 2023)

## What It Studies
Spatial and temporal associations between maritime shipping activity, atmospheric NO2, SAR-detected surface anomalies, and coastal water quality (Sentinel-2 optical indices) in the Baltic Sea region. All conclusions are associative — not causal. The project is a master's thesis.

## Study Area and Time Period
- 5 focal ports: Stockholm (59.33N 18.07E), Turku (60.44N 22.23E), Naantali (60.47N 22.03E), Mariehamn (60.10N 19.93E), Helsinki (60.17N 24.94E)
- Grid: 315 cells at 0.1 degree resolution (~11 km per cell)
- Time: 2023-01-01 to 2023-12-17, weekly steps (51 weeks)
- Total dataset: 16,065 rows (315 cells x 51 weeks)

## Data Sources
1. Sentinel-2 (GEE) water quality: NDWI (mean 0.059, 18.2% coverage due to clouds), NDTI (mean -0.138), NDCI (mean 0.038), FAI (mean 0.048), B11 SWIR. Cloud masking causes low coverage.
2. Sentinel-1 SAR (GEE): oil slick probability proxy, mean 0.268, 94.6% coverage (cloud-independent). Exploratory — not verified oil spill detection.
3. Sentinel-5P NO2 (GEE): tropospheric NO2 column, mean 1.53e-5 mol/m2, 89.9% coverage
4. EMODnet/AIS vessel density: spatial congestion proxy (not weekly AIS counts), mean 0.526, 75.9% coverage
5. Sentinel-2 land NDVI (GEE): land-masked NDVI, only 2.3% coverage (most cells are sea)
6. HELCOM reference records: mean 6.83 records/cell, 100% coverage
7. EMODnet static bathymetry: 100% coverage

## Feature Engineering
54 features across 6 categories:
- Water quality: 14 features (NDCI, NDTI, NDWI, FAI, B11 means/medians/stds)
- Maritime activity: 8 features (vessel_density variants)
- Atmospheric: 9 features (NO2 mean/std/trend, detection_score, oil slick count/prob)
- Exposure: 2 features (distance_to_port, distance_to_lane)
- Validation: 3 features (all missing in current data)
- Land extension: 18 features (NDVI mean/std/median, coastal_exposure_score, land_response_index, maritime_pressure_index, atmospheric_transfer_index, interaction terms with lags)
Final ML-ready dataset: 16,065 rows x 46 columns.

## Machine Learning

### Task 1: Predict delta_ndti (NDTI week-to-week change)
- Valid rows: 5,293 (34.3% of all grid-weeks)
- Time-aware split: train on weeks 1-30 (4,394 rows), test on weeks 31-40 (899 rows). No random shuffle.
- Ridge: train R2=0.446, test R2=-0.871, test RMSE=0.304
- HistGradientBoosting: train R2=0.965, test R2=-0.055, test RMSE=0.228 (winner)
- Top feature: sentinel_ndti_mean_t (importance 0.188); vessel_density importance is only 0.0025
- Feature group importances: lagged spectral 0.343, temporal encoding 0.108, geo 0.046, vessel density 0.0025

### Task 2: Predict ndti_next (absolute NDTI at t+1)
- Ridge: test RMSE=0.293, R2=-2.138
- HistGBR: test RMSE=0.296, R2=-2.187
- Both fail to generalize; delta task is more learnable

### Rolling Window CV (4 folds, expanding window)
- Folds: train 20/25/30/35 weeks, test next 5 weeks each
- HistGBR delta-NDTI: mean test RMSE=0.193 +/-0.055, mean R2=-0.085
- Best fold (fold 1): RMSE=0.132, R2=0.524
- Worst fold (fold 3): RMSE=0.242, R2=-1.111
- High fold-to-fold dispersion (~51% relative std)

## Correlation Findings

Pearson (strongest):
- ndwi_mean vs ndci_mean: r=-0.844
- ndti_mean vs b11_mean: r=+0.700
- vessel_density vs NO2_mean: r=+0.039 (near zero)

Spearman (strongest):
- ndwi_mean vs ndci_mean: rho=-0.876 (strongest pair overall)
- ndwi_mean vs b11_mean: rho=-0.842
- ndci_mean vs fai_mean: rho=+0.796
- vessel_density vs detection_score: rho=-0.036
- vessel_density vs NO2_mean: rho=+0.096

## Statistical Tests
High vs low traffic t-tests (all significant):
- NDCI: p=0.025
- NDTI: p=0.004 (higher in high-traffic zones)
- NDWI: p=0.013

Causal lag analysis:
- vessel_density to detection_score peaks at lag 0, r=-0.117 ("immediate coupling")
- vessel_density to NO2_mean peaks at lag 4 weeks, r=0.041 ("no response")

Land-sea lag analysis:
- NO2_mean leads land NDVI by 4 weeks: Spearman rho=0.661, n=360 (ranked "strong")
- vessel_density leads land NDVI by 1 week: rho=0.360, n=306 (ranked "moderate")

## Feature Interactions (top cross-domain pairs)
- b11_mean vs distance_to_nearest_high_vessel_density_cell: r=+0.748
- no2_mean_t vs ndvi_std: r=+0.679
- oil_slick_probability_t vs ndvi_mean: r=-0.588
- ndti_mean vs distance_to_nearest_high_vessel_density_cell: r=+0.534

## Anomaly Detection
Most persistent anomaly cell: g0.100_r1503_c2022 at 60.35N, 22.25E (near Turku/Naantali, Finland)
Highest anomaly score: 2.186 (March 5, 2023)
Top temporal z-score: z=5.35 (February 19, 2023, near Turku)

## Coastal Impact Score (composite 0-1)
Top cell: g0.100_r1503_c2022 at 60.35N, 22.25E — score=0.682 (week 2023-11-05)
4 sub-components: correlation, lag response, exposure, anomaly

## Coastal Shipping Impact Analysis
Restricted to coastal strip (<=30 km from coast, <=20 km from high vessel density cells).

High vs low vessel exposure results:
- oil_slick_probability_t: p=3.5e-13 (YES, significant; high=0.194, low=0.064, Cohen's d=0.849)
- NO2_mean: p=0.543 (not significant)
- NDTI: p=0.259 (not significant)
- NDWI: p=0.105 (not significant)
- NDVI: p=0.108 (not significant)

The ONLY statistically significant difference between high and low vessel exposure is the SAR oil slick proxy (3x higher in high-exposure zones).

## Land Impact Extension
- Coastal exposure score: mean=0.442, std=0.337, coverage=100%
- Land response index: mean=0.606, std=0.099, coverage=2.3% (land cells only)
- Distance to nearest high vessel cell: mean=682 km, std=1425 km

## Outputs Generated
- 14 thesis figures (PNG+PDF): framework diagram, feature importance (Ridge and HistGBR), correlation network, temporal heatmap, distance decay, exposure maps (Baltic overview, Mariehamn, Stockholm, Turku), composite land dashboard, refined dashboard, port lattice analysis
- 50+ report CSVs in outputs/reports/
- Rolling CV results in outputs/ml_cv_results/
- Streamlit dashboard at localhost:8501
- ALL_STATISTICS_INVENTORY.md: 143 source files, 2.5 million cells catalogued

## Key Honest Conclusions
1. Water quality indices are strongly intercorrelated with each other (optical physics), not primarily with shipping
2. Vessel density has near-zero ML feature importance (0.0025) for predicting turbidity change
3. The SAR oil slick detection score is the only indicator with strong, significant association with vessel exposure (p=3.5e-13, 3x higher in high-exposure zones)
4. NO2 precedes land NDVI by 4 weeks (rho=0.661) — strongest temporal lagged finding
5. ML models do not generalize well (test R2 near zero or negative) due to limited 1-year data, cloud gaps, and seasonal distribution shift between train (winter-summer) and test (summer-autumn)
6. The Turku/Naantali coastal zone (60.35N, 22.25E) is consistently the most anomalous and highest-impact area in the study
7. All findings are spatial associations only — no causal attribution is made
8. High-traffic cells show higher NDTI and lower NDWI/NDCI in simple t-tests, but these contrasts do not hold after applying coastal and lane-proximity restrictions

## Technical Stack
Python 3.13, pandas, numpy, scikit-learn (Ridge, HistGradientBoostingRegressor, IsolationForest, permutation_importance), geopandas, shapely, rasterio, earthengine-api, matplotlib, seaborn, plotly, streamlit, pyarrow, scipy, statsmodels, networkx
```
