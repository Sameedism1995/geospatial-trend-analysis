# Presentation graphs bundle

Generated under `outputs/presentation_graphs`.

## Figure directories

| Folder | Contents |
|--------|-----------|
| `dataset_overview/` | Baltic extent, ports, temporal coverage, missingness, integration sketch |
| `environmental_indicators/` | Spectral / vessel maps, distributions, correlations |
| `exposure_analysis/` | MPI / coastal scores, decay curves, hotspots |
| `wind_regime/` | Alignment diagnostics, atmospheric indices |
| `machine_learning/` | Split metrics, predictions, residuals, importance |
| `temporal_lag_persistence/` | Lag autocorr, persistence scatter |
| `validation/` | Split diagrams, rolling trends, comparison graphic |
| `anomaly_detection/` | Weekly exceedances, median z-score timeline |
| `comparison_analysis/` | Cross-port standardised contrasts |
| `summary_maps/` | Executive tiles + NDVI footprint summary |

## Source code

- Driver: `scripts/run_presentation_graphs.py`
- Modules: `scripts/presentation_graphs/` (`common.py`, `dataset_environment.py`, `extended.py`, `generate_all.py`)

## Slide placement hints

- **Opening**: dataset_overview Baltic grid + integration diagram.
- **Methods / data**: temporal coverage + missingness.
- **Results environmental**: environmental_indicators maps + distributions.
- **Exposure story**: exposure_analysis decay + hotspots.
- **Wind / atmosphere**: wind_regime alignment plots.
- **Temporal structure**: temporal_lag_persistence heatmaps.
- **ML**: machine_learning predicted-vs-actual + metrics bars.
- **Robustness**: validation rolling vs single-split comparison.
- **Stress events**: anomaly_detection timelines.
- **Ports**: comparison_analysis bars.
- **Closing**: summary_maps infographic.

## Export statistics

- PNG files: **51**
- PDF files: **51**
- Logged failures: **0**


## Generated PNG inventory
### `anomaly_detection/`
- `median_ndti_zscore_timeline.png`
- `weekly_ndti_exceedance_counts.png`
### `comparison_analysis/`
- `cross_port_zscore_medians.png`
### `dataset_overview/`
- `01_baltic_study_area_grid.png`
- `02_port_locations.png`
- `03_integration_architecture_diagram.png`
- `04_temporal_coverage_bars.png`
- `05_weekly_observation_timeline.png`
- `06_missingness_heatmap_weeks.png`
- `07_spatial_coverage_density.png`
### `environmental_indicators/`
- `boxplot_major_spectral.png`
- `correlation_heatmap_spectral.png`
- `env_distributions_grid.png`
- `map_ndci_mean.png`
- `map_ndti_mean.png`
- `map_ndvi_mean.png`
- `map_ndwi_mean.png`
- `map_no2_mean.png`
- `map_vessel_density.png`
- `weekly_median_ndti_ndwi.png`
### `exposure_analysis/`
- `coastal_band_mpi_boxplots.png`
- `composite_coastal_maritime_scatter.png`
- `distance_decay_port_mpi.png`
- `distance_decay_shipping_lane.png`
- `esi_coastal_exposure_score_map.png`
- `exposure_hotspots_top_decile_mpi.png`
- `integrated_exposure_visualization.png`
- `land_response_index_map.png`
- `mei_maritime_pressure_map.png`
### `machine_learning/`
- `baseline_mae_delta_ndti_split.png`
- `baseline_rmse_r2_delta_ndti_split.png`
- `baseline_rmse_r2_ndti_next_split.png`
- `feature_importance_hgb_delta_ndti.png`
- `predicted_vs_actual_four_panel.png`
- `residual_hist_hgb_delta_ndti.png`
### `summary_maps/`
- `infographic_executive_summary_five_tile.png`
- `summary_environmental_exposure_ndti.png`
### `temporal_lag_persistence/`
- `lag_autocorr_heatmap.png`
- `persistence_corr_weekly_medians.png`
- `scatter_ndti_lag1_vs_t.png`
- `weekly_ndti_median_timeline.png`
### `validation/`
- `comparison_standard_vs_rolling_rmse_delta_ndti.png`
- `diagram_rolling_expanding_windows.png`
- `diagram_standard_temporal_split.png`
- `rolling_r2_trend_delta_ndti.png`
- `rolling_rmse_trend_delta_ndti.png`
### `wind_regime/`
- `atmospheric_transfer_map.png`
- `shoreward_vs_other_no2_boxplot.png`
- `wind_alignment_category_counts.png`
- `wind_alignment_score_map.png`
- `wind_exposure_no2_scatter.png`