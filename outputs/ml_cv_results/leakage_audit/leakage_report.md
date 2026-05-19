# `ndti_next` target leakage audit

**Scope:** diagnosis only; pipelines unchanged.

## Executive summary

The thesis modeling code **does not place `ndti_next_target` in X**, and parquet features are constrained to lag *t*, *t−1*, *t−2* per `docs/modeling_dataset_schema.md`.

**Current-week turbidity `sentinel_ndti_mean_t` is an explicit predictor** while **y encodes next-week NDTI** (see target construction). That is **consistent** with the leakage rule (predictors use information up to *t* only; *t+1* appears only via the constructed label). It is **not**, by itself, a bug — but it implies the **level-forecasting task can be easy** if NDTI evolves gradually and joint spectral predictors align. **Always check marginal correlations (Section 5)** and **whether your rolling CV metrics correspond to the same parquet/week span** you audit.

Chronological holdouts in `data/model_results.json` can show **much worse** ndti_next test skill than short expanding windows — that pattern is often **regime / sample-size** shift rather than train/test leakage in code.

### Calendar reconciliation (rolling_window_metrics.csv vs this audit)


{
  "rolling_metrics_csv_example_train_week_from_row1": "2024-01-01 00:00:00+00:00",
  "current_parquet_week_min_after_load_modeling_data": "2023-01-15 00:00:00+00:00",
  "current_parquet_week_max_after_load_modeling_data": "2023-11-12 00:00:00+00:00",
  "overlap_interpretation": "If week ranges disagree, rolling_window_metrics.csv was generated from a different modeling parquet or panel than audited here \u2014 compare artefacts before inferring leakage."
}


## 1. Target alignment & duplicate keys

{
  "statement": "ndti_next_target is defined as groupby(grid_cell_id).sentinel_ndti_mean_t.shift(-1); row filter keeps both sentinel_ndti_mean_t and ndti_next_target non-null.",
  "n_rows_checked": 5069,
  "max_abs_error": 0.0,
  "mean_abs_error": 0.0,
  "fraction_exact_float_match": 1.0
}


Duplicate (`grid_cell_id`, `week_start_utc`) groups:

{
  "duplicate_key_pairs": 0,
  "max_duplicate_count": 1
}


## 2. Forbidden targets in feature list

{
  "META_COLS": [
    "delta_ndti",
    "grid_cell_id",
    "has_valid_delta_ndti",
    "ndti_next_target",
    "week_start_utc"
  ],
  "feature_list_includes_ndti_next_target": false,
  "feature_list_includes_delta_ndti": false,
  "feature_list_includes_has_valid_delta_ndti": false
}

RegExp-based name scan (forward-looking / target-like substrings):

{
  "regex_hits": []
}


## 3. Univariate explanatory power (full modeling panel after `load_modeling_data`)

`sentinel_ndti_mean_t` only (OLS y ~ intercept +):

{
  "n": 5293.0,
  "r2_univariate_ols_intercept": 0.06416703265753687,
  "intercept": -0.15021313947099413,
  "slope": 0.25431615061275187
}

Optional: `delta_ndti` correlation with y (**not used as predictor**):

{
  "n": 5293.0,
  "r2_univariate_ols_intercept": 0.28094819824157125,
  "intercept": -0.19444122402216443,
  "slope": 0.4320183231571749
}


## 4. Rows & feature count post-filter

- n_rows: **5293**
- n_predictors |X|: **25**


## 5. Top predictors correlated with `ndti_next_target`

See `correlations_with_ndti_next.csv` (full table) and 
`correlation_audit_top_predictors_matrix.csv` (pairwise Pearson among top |r| drivers). 
Top 15 by |Pearson| versus y:


```text
                     feature  n_complete  pearson_r     pearson_p  spearman_rho    spearman_p
sentinel_ndvi_mean_t_minus_1        3979   0.377486 5.824329e-135      0.334647 1.015787e-104
sentinel_ndwi_mean_t_minus_1        3979  -0.376389 3.966850e-134     -0.345618 5.063903e-112
sentinel_ndwi_mean_t_minus_2        4149  -0.342894 8.296713e-115     -0.360490 1.503864e-127
sentinel_ndvi_mean_t_minus_2        4149   0.319838  2.562823e-99      0.349000 3.887544e-119
        sentinel_ndwi_mean_t        5293  -0.312713 2.046907e-120     -0.353689 8.861175e-156
sentinel_ndti_mean_t_minus_1        3979   0.270547  1.057929e-67      0.254364  8.511079e-60
        sentinel_ndvi_mean_t        5293   0.256075  5.086804e-80      0.289299 1.432370e-102
        sentinel_ndti_mean_t        5293   0.253312  2.757500e-78      0.318509 4.317281e-125
sentinel_ndti_mean_t_minus_2        4149   0.235327  2.609881e-53      0.295335  2.741378e-84
                    week_cos        5293   0.162215  1.547456e-32      0.220227  3.768412e-59
           grid_centroid_lat        5293   0.131264  8.882430e-22      0.208377  5.208703e-53
                    week_sin        5293  -0.114593  6.137798e-17     -0.164900  1.398798e-33
           grid_centroid_lon        5293   0.111480  4.155824e-16      0.113094  1.551368e-16
    vessel_density_t_minus_2        4907   0.088722  4.795439e-10      0.019618  1.694433e-01
    vessel_density_t_minus_1        4907   0.088722  4.795439e-10      0.019618  1.694433e-01
```



## 6. Near-duplicate / near-identity features vs target

See `equality_audit_vs_target.csv` (exact match ratio between x and y, residual RMSE).


```text
                     feature  n_complete  frac_exact_xy_equal  max_abs_residual  median_abs_residual  rmse_residual_vs_target
                 has_emodnet        5293             0.004534          1.000000             0.175608                 0.259243
            vessel_density_t        4907             0.000204         25.853306             0.400144                 2.446616
    vessel_density_t_minus_1        4907             0.000204         25.853306             0.400144                 2.446616
    vessel_density_t_minus_2        4907             0.000204         25.853306             0.400144                 2.446616
                grid_res_deg        5293             0.000000          1.100000             0.275608                 0.342670
sentinel_ndwi_mean_t_minus_2        4149             0.000000          1.901934             0.426115                 0.568532
                has_sentinel        5293             0.000000          2.000000             1.175608                 1.212174
sentinel_observation_count_t        5293             0.000000      16384.569136          4096.281549              5860.621890
```


## 7. Feature pairs with |Pearson r| ≥ 0.999

```text
               feature_a                feature_b  n_complete  pearson_r
        vessel_density_t vessel_density_t_minus_1        4907        1.0
        vessel_density_t vessel_density_t_minus_2        4907        1.0
vessel_density_t_minus_1 vessel_density_t_minus_2        4907        1.0
```


## 8. Static preprocessing / split order

### Static code audit (no data)

**Target construction** (`run_delta_ndti_models.load_modeling_data`):

1. Filter `has_valid_delta_ndti == True`.
2. Sort by `grid_cell_id`, `week_start_utc`.
3. `ndti_next_target = groupby(grid_cell_id).sentinel_ndti_mean_t.shift(-1)`.
4. Keep rows with non-null `sentinel_ndti_mean_t` and `ndti_next_target`.

**Feature matrix** (`feature_columns`): every column not in `META_COLS` = `{
  "grid_cell_id", "week_start_utc", "delta_ndti", "has_valid_delta_ndti", "ndti_next_target"
}`.

So **`sentinel_ndti_mean_t` (NDTI at week *t*) is an explicit predictor** while **`ndti_next_target` is NDTI at *t+1***.
This is allowed under the leakage rule documented in `docs/modeling_dataset_schema.md` (features ≤ *t*,
target uses *t+1* constructed at load time — not read from parquet as a column).

**Rolling CV** (`src/ml/run_rolling_window_cv.py`):

1. Rows are partitioned by disjoint sets of **calendar weeks** (`week_start_utc`); train weeks precede test weeks on the timeline.
2. Each fold builds `X_train = prepare_X(train_df)`, `X_test = prepare_X(test_df)` — no global fit before split.
3. `fit_ridge` / `fit_hgb` receive **training fold rows only**; `SimpleImputer` + `StandardScaler` are fitted **inside** Ridge’s `pipe.fit(X_train, y_train)` — test fold never participates in scaling/imputation fitting.

Therefore there is **no obvious train/test scaler leakage** introduced by rolling CV.


