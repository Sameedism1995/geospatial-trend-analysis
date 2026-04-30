# EDA Summary Report

## Key Statistics
- Total rows: 16,065
- Missing grid coverage (%): 100.0
- VV mean / std: None / None
- VH mean / std: None / None
- VV/VH ratio mean / std: None / None
- Detection positive rate (%): 37.54746342981637
- Weekly VV mean variation (std): nan
- Weekly VH mean variation (std): nan
- Number of grids: 315

## Key Observations
- Dataset contains 16,065 rows.
- Highest missingness features: no2_x_ndvi (98.9%), vessel_x_ndvi_lag2 (98.1%), vessel_x_ndvi_lag3 (98.1%).
- Positive detection rate is 37.55%.
- Temporal variation (std of weekly means): VV=nan, VH=nan.

## Anomalies Detected
- Very high missingness detected in at least one feature (98.9%).
- High grid-level missing coverage (100.0% grids affected).

## Dataset Readiness
- Dataset is not fully ML-ready; review anomalies and missingness before training.