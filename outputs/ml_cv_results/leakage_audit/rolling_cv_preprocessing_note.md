**Rolling CV preprocessing order (confirmed in source):**

1. `load_modeling_data` constructs `ndti_next_target`; filter drops tail rows lacking t+1.
2. For each temporal fold: `prepare_X(train_df)`, `prepare_X(test_df)` — no fit.
3. `fit_ridge` fits `Pipeline(imputer→scaler→ridge)` on **X_train, y_train** only.
4. `fit_hgb` fits on **X_train, y_train** only (internally withholds validation fraction **from training rows**, not test fold).
