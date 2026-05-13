# All pipeline report results (long export)

- Generated UTC: `2026-05-10T05:41:05.972301+00:00`
- Mode: **all columns / all JSON leaves**
- Long-table rows: **2586457** (one row per CSV cell or JSON leaf)
- Source files scanned: **143**

## How to read `ALL_STATISTICS_LONG.csv`

- **source_file**: report path relative to project root.
- **row_index**: row number in the original CSV (0-based); JSON sources use 0.
- **column**: original column name or flattened JSON key path.
- **cell_value**: cell value (stringified).

By default this file includes **every** column from every CSV under `outputs/reports/`, all matching `*.json` there, plus `outputs/eda/eda_stats.json` and `data/validation/full_pipeline_validation_report.json`. Use `--stat-columns-only` for a smaller table that keeps only names matching typical statistics (p-values, means, correlations, etc.).

## Source file inventory

| File | Rows in file | Unique columns | Cells exported | Note |
|------|-------------:|---------------:|---------------:|------|
| `data/validation/full_pipeline_validation_report.json` | 1 | 103 | 103 |  |
| `outputs/eda/eda_stats.json` | 1 | 311 | 311 | eda_stats.json (flattened leaves) |
| `outputs/reports/anomaly_scores.csv` | 16065 | 6 | 96390 |  |
| `outputs/reports/anomaly_scores_temporal.csv` | 16065 | 7 | 112455 |  |
| `outputs/reports/causal_lag_analysis.csv` | 12 | 6 | 72 |  |
| `outputs/reports/causal_lag_summary.csv` | 2 | 9 | 18 |  |
| `outputs/reports/coastal_exposure_statistics.csv` | 64 | 20 | 1280 |  |
| `outputs/reports/coastal_impact_score.csv` | 16065 | 9 | 144585 |  |
| `outputs/reports/correlation_evaluation.csv` | 7 | 9 | 63 |  |
| `outputs/reports/exposure_indices_summary.csv` | 3 | 8 | 24 |  |
| `outputs/reports/feature_importance_baseline.csv` | 9 | 2 | 18 |  |
| `outputs/reports/feature_interactions_ranked.csv` | 77 | 5 | 385 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/anomaly_scores.csv` | 42738 | 6 | 256428 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/anomaly_scores_temporal.csv` | 42738 | 7 | 299166 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/causal_lag_analysis.csv` | 12 | 6 | 72 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/causal_lag_summary.csv` | 2 | 9 | 18 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/coastal_impact_score.csv` | 42738 | 9 | 384642 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/correlation_evaluation.csv` | 7 | 9 | 63 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/feature_importance_baseline.csv` | 9 | 2 | 18 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/feature_interactions_ranked.csv` | 77 | 5 | 385 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/hub_grid_coverage.csv` | 3 | 6 | 18 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/hub_overlap_analysis.csv` | 6 | 6 | 36 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/hub_strategy.csv` | 3 | 5 | 15 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/lagged_correlations.csv` | 12 | 11 | 132 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/land_impact_analysis.csv` | 42738 | 14 | 598332 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/land_impact_ml_feature_importance.csv` | 12 | 4 | 48 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/land_impact_ml_feature_importance_with_s2_bands.csv` | 17 | 4 | 68 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/land_impact_ml_model_summary.json` | 1 | 73 | 73 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/land_impact_ml_model_summary_with_s2_bands.json` | 1 | 98 | 98 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/land_sea_lag_summary.csv` | 2 | 8 | 16 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/land_sea_lagged_correlations.csv` | 16 | 8 | 128 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/local_per_hub_summary.csv` | 45 | 11 | 495 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/local_sliding_window_summary.csv` | 135 | 12 | 1620 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/pearson_correlation.csv` | 8 | 9 | 72 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/port_city_exposure_summary.csv` | 3 | 6 | 18 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/regional_per_hub_summary.csv` | 45 | 12 | 540 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_hub_strategy_turku_naantali_mariehamn_stockholm/hub_grid_coverage.csv` | 3 | 6 | 18 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_hub_strategy_turku_naantali_mariehamn_stockholm/hub_overlap_analysis.csv` | 6 | 6 | 36 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_hub_strategy_turku_naantali_mariehamn_stockholm/hub_strategy.csv` | 3 | 5 | 15 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_hub_strategy_turku_naantali_mariehamn_stockholm/local_per_hub_summary.csv` | 45 | 11 | 495 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_hub_strategy_turku_naantali_mariehamn_stockholm/local_sliding_window_summary.csv` | 135 | 12 | 1620 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_hub_strategy_turku_naantali_mariehamn_stockholm/local_vs_regional_summary.json` | 1 | 1120 | 1120 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_hub_strategy_turku_naantali_mariehamn_stockholm/regional_per_hub_summary.csv` | 45 | 12 | 540 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_stockholm_grid_expanded/current_grid_diagnostics.csv` | 3 | 5 | 15 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_stockholm_grid_expanded/expanded_grid_summary.csv` | 4 | 9 | 36 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_stockholm_grid_expanded/hub_grid_coverage.csv` | 3 | 6 | 18 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_stockholm_grid_expanded/hub_overlap_analysis.csv` | 6 | 6 | 36 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_stockholm_grid_expanded/hub_strategy.csv` | 3 | 5 | 15 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_stockholm_grid_expanded/local_per_hub_summary.csv` | 45 | 11 | 495 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_stockholm_grid_expanded/local_sliding_window_summary.csv` | 135 | 12 | 1620 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_stockholm_grid_expanded/local_vs_regional_summary.json` | 1 | 2734 | 2734 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/run_stockholm_grid_expanded/regional_per_hub_summary.csv` | 45 | 12 | 540 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/sliding_window_distance_decay/no2_sliding_window_200_500km.csv` | 17 | 8 | 136 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/sliding_window_distance_decay/sliding_window_ndti.csv` | 71 | 8 | 568 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/sliding_window_distance_decay/sliding_window_ndvi.csv` | 47 | 8 | 376 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/sliding_window_distance_decay/sliding_window_ndwi.csv` | 71 | 8 | 568 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/sliding_window_distance_decay/sliding_window_no2.csv` | 71 | 8 | 568 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/sliding_window_distance_decay/sliding_window_vessel_density.csv` | 33 | 8 | 264 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/spearman_correlation.csv` | 8 | 9 | 72 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/temporal_stability.csv` | 102 | 5 | 510 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/validation/after_helcom_rerun_comparison.json` | 1 | 60 | 60 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/validation/baseline_before_helcom_rerun.json` | 1 | 30 | 30 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/validation/feature_registry_summary.json` | 1 | 277 | 277 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/validation/final_validation_report.json` | 1 | 509 | 509 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/validation/full_pipeline_run_summary.json` | 1 | 72 | 72 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/validation/full_pipeline_validation_report.json` | 1 | 102 | 102 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/validation/human_impact_integrity_report.json` | 1 | 40 | 40 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/validation/no2_validation_report.json` | 1 | 23 | 23 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/validation/qa_report.json` | 1 | 7205 | 7205 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/validation/sentinel1_oil_debug_log.json` | 1 | 1741 | 1741 |  |
| `outputs/reports/final_run_stockholm_fixed_20260505_1356/validation/sentinel1_oil_validation_report.json` | 1 | 26 | 26 |  |
| `outputs/reports/indicator_participation_statistics.csv` | 176 | 12 | 2112 |  |
| `outputs/reports/lagged_correlations.csv` | 12 | 11 | 132 |  |
| `outputs/reports/land_impact_analysis.csv` | 16065 | 14 | 224910 |  |
| `outputs/reports/land_impact_ml_feature_importance.csv` | 12 | 4 | 48 |  |
| `outputs/reports/land_impact_ml_feature_importance_with_s2_bands.csv` | 17 | 4 | 68 |  |
| `outputs/reports/land_impact_ml_model_summary.json` | 1 | 73 | 73 |  |
| `outputs/reports/land_impact_ml_model_summary_with_s2_bands.json` | 1 | 98 | 98 |  |
| `outputs/reports/land_sea_lag_summary.csv` | 2 | 8 | 16 |  |
| `outputs/reports/land_sea_lagged_correlations.csv` | 16 | 8 | 128 |  |
| `outputs/reports/mariehamn_wind_ingestion_audit.csv` | 23 | 10 | 230 |  |
| `outputs/reports/pearson_correlation.csv` | 8 | 9 | 72 |  |
| `outputs/reports/port_city_exposure_summary.csv` | 4 | 6 | 24 |  |
| `outputs/reports/port_distance_decay_statistics.csv` | 365 | 11 | 4015 |  |
| `outputs/reports/port_exposure_ranking.csv` | 2 | 16 | 32 |  |
| `outputs/reports/port_pairwise_mannwhitney.csv` | 15 | 12 | 180 |  |
| `outputs/reports/run_coastal_wind_transport/coastal_wind_alignment_features.csv` | 13821 | 26 | 359346 |  |
| `outputs/reports/run_coastal_wind_transport/coastal_wind_exposure_summary.csv` | 14 | 6 | 84 |  |
| `outputs/reports/run_final_coastal_shipping_impact/coastal_impact_score_summary.csv` | 4 | 2 | 8 |  |
| `outputs/reports/run_final_coastal_shipping_impact/high_vs_low_vessel_coastal_comparison.csv` | 6 | 13 | 78 |  |
| `outputs/reports/run_final_coastal_shipping_impact/shipping_lane_distance_decay.csv` | 28 | 7 | 196 |  |
| `outputs/reports/run_hub_strategy_turku_naantali_mariehamn_stockholm/hub_grid_coverage.csv` | 3 | 6 | 18 |  |
| `outputs/reports/run_hub_strategy_turku_naantali_mariehamn_stockholm/hub_overlap_analysis.csv` | 6 | 6 | 36 |  |
| `outputs/reports/run_hub_strategy_turku_naantali_mariehamn_stockholm/hub_strategy.csv` | 3 | 5 | 15 |  |
| `outputs/reports/run_hub_strategy_turku_naantali_mariehamn_stockholm/local_per_hub_summary.csv` | 45 | 11 | 495 |  |
| `outputs/reports/run_hub_strategy_turku_naantali_mariehamn_stockholm/local_sliding_window_summary.csv` | 135 | 12 | 1620 |  |
| `outputs/reports/run_hub_strategy_turku_naantali_mariehamn_stockholm/local_vs_regional_summary.json` | 1 | 1120 | 1120 |  |
| `outputs/reports/run_hub_strategy_turku_naantali_mariehamn_stockholm/regional_per_hub_summary.csv` | 45 | 12 | 540 |  |
| `outputs/reports/run_land_pollution_drivers_wind/land_pollution_driver_correlation.csv` | 12 | 5 | 60 |  |
| `outputs/reports/run_land_pollution_drivers_wind/land_pollution_driver_feature_importance.csv` | 27 | 6 | 162 |  |
| `outputs/reports/run_land_pollution_drivers_wind/ndvi_supporting_land_response.csv` | 3 | 10 | 30 |  |
| `outputs/reports/run_land_pollution_drivers_wind/no2_excess_features.csv` | 1530 | 7 | 10710 |  |
| `outputs/reports/run_land_pollution_drivers_wind/no2_wind_aligned_land_impact.csv` | 3 | 10 | 30 |  |
| `outputs/reports/run_land_pollution_drivers_wind/oil_slick_wind_coastal_risk.csv` | 2 | 10 | 20 |  |
| `outputs/reports/run_land_pollution_drivers_wind/wind_alignment_features.csv` | 1530 | 14 | 21420 |  |
| `outputs/reports/run_land_pollution_drivers_wind/wind_features_weekly.csv` | 1530 | 8 | 12240 |  |
| `outputs/reports/run_nearest_land_ndvi_linkage/land_ndvi_candidate_cells.csv` | 127 | 6 | 762 |  |
| `outputs/reports/run_nearest_land_ndvi_linkage/maritime_to_nearest_land_linkage.csv` | 375 | 8 | 3000 |  |
| `outputs/reports/run_nearest_land_ndvi_linkage/nearest_land_distance_decay.csv` | 16 | 7 | 112 |  |
| `outputs/reports/run_nearest_land_ndvi_linkage/nearest_land_high_vs_low_vessel.csv` | 4 | 9 | 36 |  |
| `outputs/reports/run_nearest_land_ndvi_linkage/sea_land_correlation_check.csv` | 5 | 4 | 20 |  |
| `outputs/reports/run_no2_oil_slick_combo/no2_oil_stratified_summary.csv` | 1 | 14 | 14 |  |
| `outputs/reports/run_research_radar_coastal_pipeline/coastal_all_indicators_comparison.csv` | 5 | 10 | 50 |  |
| `outputs/reports/run_research_radar_coastal_pipeline/coastal_distance_decay_all_indicators.csv` | 20 | 7 | 140 |  |
| `outputs/reports/run_research_radar_coastal_pipeline/indicator_group_analysis.csv` | 7 | 7 | 49 |  |
| `outputs/reports/run_research_radar_coastal_pipeline/land_impact_analysis.csv` | 5 | 13 | 65 |  |
| `outputs/reports/run_research_radar_coastal_pipeline/winter_radar_distance_decay.csv` | 8 | 7 | 56 |  |
| `outputs/reports/run_research_radar_coastal_pipeline/winter_radar_high_vs_low.csv` | 2 | 10 | 20 |  |
| `outputs/reports/run_season_aware_coastal_impact/land_impact_by_distance.csv` | 12 | 7 | 84 |  |
| `outputs/reports/run_season_aware_coastal_impact/land_impact_by_vessel.csv` | 3 | 9 | 27 |  |
| `outputs/reports/run_season_aware_coastal_impact/radar_distance_decay_refined.csv` | 4 | 6 | 24 |  |
| `outputs/reports/run_season_aware_coastal_impact/radar_high_vs_low.csv` | 2 | 9 | 18 |  |
| `outputs/reports/run_season_aware_coastal_impact/sea_land_link_analysis.csv` | 6 | 10 | 60 |  |
| `outputs/reports/run_season_aware_coastal_impact/seasonal_indicator_analysis.csv` | 35 | 16 | 560 |  |
| `outputs/reports/run_stockholm_grid_expanded/current_grid_diagnostics.csv` | 3 | 5 | 15 |  |
| `outputs/reports/run_stockholm_grid_expanded/expanded_grid_summary.csv` | 4 | 9 | 36 |  |
| `outputs/reports/run_stockholm_grid_expanded/hub_grid_coverage.csv` | 3 | 6 | 18 |  |
| `outputs/reports/run_stockholm_grid_expanded/hub_overlap_analysis.csv` | 6 | 6 | 36 |  |
| `outputs/reports/run_stockholm_grid_expanded/hub_strategy.csv` | 3 | 5 | 15 |  |
| `outputs/reports/run_stockholm_grid_expanded/local_per_hub_summary.csv` | 45 | 11 | 495 |  |
| `outputs/reports/run_stockholm_grid_expanded/local_sliding_window_summary.csv` | 135 | 12 | 1620 |  |
| `outputs/reports/run_stockholm_grid_expanded/local_vs_regional_summary.json` | 1 | 2734 | 2734 |  |
| `outputs/reports/run_stockholm_grid_expanded/regional_per_hub_summary.csv` | 45 | 12 | 540 |  |
| `outputs/reports/sliding_window_distance_decay/no2_sliding_window_200_500km.csv` | 10 | 8 | 80 |  |
| `outputs/reports/sliding_window_distance_decay/sliding_window_ndti.csv` | 60 | 8 | 480 |  |
| `outputs/reports/sliding_window_distance_decay/sliding_window_ndvi.csv` | 36 | 8 | 288 |  |
| `outputs/reports/sliding_window_distance_decay/sliding_window_ndwi.csv` | 60 | 8 | 480 |  |
| `outputs/reports/sliding_window_distance_decay/sliding_window_no2.csv` | 60 | 8 | 480 |  |
| `outputs/reports/sliding_window_distance_decay/sliding_window_vessel_density.csv` | 33 | 8 | 264 |  |
| `outputs/reports/spearman_correlation.csv` | 8 | 9 | 72 |  |
| `outputs/reports/temporal_stability.csv` | 102 | 5 | 510 |  |
| `outputs/reports/turku_mei_missing_debug.csv` | 1479 | 8 | 11832 |  |
| `outputs/reports/wind_coverage_audit.csv` | 42 | 7 | 294 |  |

## Files skipped or with parse errors

- (none)
