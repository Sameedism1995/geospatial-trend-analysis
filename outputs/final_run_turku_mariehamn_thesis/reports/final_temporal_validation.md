# Temporal validation

## Data coverage
- Turku weekly rows: 51
- Mariehamn weekly rows: 51
- See `reports/final_temporal_data_audit.csv` for monotonicity and duplicate-week checks on aggregated panel.

## Lag robustness
- Lag table rows: 38
- Correlations require ≥8 complete pairwise weeks; smaller samples yield NaN (honest missingness).
- Review `final_lag_analysis_summary.csv` for |ρ|≈1 (possible collinearity) or unstable port contrasts.

## Anomaly robustness
Anomaly log rows: 212 (weeks can appear multiple times if flagged by multiple rules/features).
- Anomalies are not defined from imputed values; underlying NaNs reduce effective counts.

## Temporal limitations
- Single observational period; seasonality and unobserved drivers can align with shipping signals.
- Composite indices share inputs; interpret overlap between indicators cautiously.
- Inspect raw series: **if vessel_density (or another input) is nearly constant in time within a port window**, lag-specific correlations for that source may **coincide numerically** because the signal has no temporal contrast at that scale; pooled summaries can then mix within-port structure with cross-port differences.

## Uncertainty
- All results are **associational**; physical plausibility is interpreted as consistency with possible mechanisms, not proof.
