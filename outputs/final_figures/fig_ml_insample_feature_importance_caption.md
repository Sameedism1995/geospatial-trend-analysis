# In-sample ML feature importance (training data only)

**Figure.** Feature weights for predicting **ΔNDTI** under the thesis time-aware split (
first 75% of distinct calendar weeks → train).
**Left:** Ridge absolute standardized coefficients fit on training rows only.
**Right:** HistGradientBoosting permutation importance (ΔRMSE, 10 repeats) evaluated on a
training subsample only. Lower panels isolate **spatial** (`grid_centroid_lat/lon`, `grid_res_deg`)
and **maritime** (`vessel_density_t` lags) predictors. Bar colours encode feature family.

**Interpretation:** Shows which inputs the models emphasised **within the training window**—a
legitimate explainability contribution. Negative test R² under temporally separated holdout
(Chapter 5 ML validation) confirms these weights must not be read as forecast skill.

**Table:** `fig_ml_insample_feature_importance_table.csv`

**Train weeks:** 2023-01-15 00:00:00+00:00 → 2023-08-06 00:00:00+00:00 (30 weeks).
