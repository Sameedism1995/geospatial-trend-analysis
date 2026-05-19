# Chapter 5 — ML figures: rolling-window temporal cross-validation

Renumber the figure labels below to match your thesis sequence (placeholders use **5.a–5.c**).

Artifacts are produced by `src/ml/run_rolling_window_cv.py` (default output: `outputs/ml_cv_results/`). Thesis-ready copies live under `outputs/final_figures/`.

---

## Figure 5.a — Rolling-window RMSE distribution across folds

![Rolling-window test RMSE boxplots by target and model](../final_figures/fig_5_ml_rolling_window_test_rmse_boxplot.png)

**Caption.** Expanding-window cross-validation by calendar week: distribution of **test-set RMSE** over folds for Ridge (median imputation + standardized features + ridge, α = 1) and HistGradientBoosting regressors. Left panel: primary target ΔNDTI (`delta_ndti`); right panel: secondary one-week-ahead level target (`ndti_next`). Boxes summarise fold-wise scores; whiskers extend to quartile-based extremes. Temporal ordering is preserved; folds grow the training history and withhold forward weeks (see methodological script).

**Source files.** `outputs/final_figures/fig_5_ml_rolling_window_test_rmse_boxplot.png` (copy of `outputs/ml_cv_results/rolling_window_boxplot_rmse.png`).

---

## Figure 5.b — Rolling-window R² distribution across folds

![Rolling-window test R² boxplots by target and model](../final_figures/fig_5_ml_rolling_window_test_r2_boxplot.png)

**Caption.** Same fold structure as Figure 5.a: **test R²** across folds for both targets and models. The horizontal reference marks R² = 0 (performance of predicting the fold mean). Negative values indicate poorer skill than that baseline under temporal extrapolation within the withheld window.

**Source files.** `outputs/final_figures/fig_5_ml_rolling_window_test_r2_boxplot.png` (copy of `outputs/ml_cv_results/rolling_window_boxplot_r2.png`).

---

## Figure 5.c — Fold-wise temporal validation performance for ΔNDTI

![Fold-wise train and test RMSE and R² for ΔNDTI](../final_figures/fig_5_ml_rolling_window_delta_ndti_foldwise_lines.png)

**Caption.** Fold-by-fold trajectory for **ΔNDTI** only: training and withheld-window **RMSE** (top row) and **R²** (bottom row), comparing Ridge and HistGradientBoosting. Highlights train–test gaps and instability across withheld periods when the trained history expands.

**Source files.** `outputs/final_figures/fig_5_ml_rolling_window_delta_ndti_foldwise_lines.png` (copy of `outputs/ml_cv_results/rolling_window_lines_foldwise_delta_ndti.png`).

---

## LaTeX drop-in (optional)

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{outputs/final_figures/fig_5_ml_rolling_window_test_rmse_boxplot}
  \caption{Rolling-window test RMSE over expanding calendar-week folds (Ridge vs.\ HistGradientBoosting; $\Delta$NDTI and one-week-ahead level target).}
  \label{fig:rolling_rmse_boxplot}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{outputs/final_figures/fig_5_ml_rolling_window_test_r2_boxplot}
  \caption{Rolling-window test $R^2$ over the same folds (reference line at zero).}
  \label{fig:rolling_r2_boxplot}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{outputs/final_figures/fig_5_ml_rolling_window_delta_ndti_foldwise_lines}
  \caption{Fold-wise training and withheld-window performance for $\Delta$NDTI.}
  \label{fig:rolling_delta_ndti_lines}
\end{figure}
```

*(Adjust `\includegraphics` paths for your thesis project root or use `\graphicspath`.)*
