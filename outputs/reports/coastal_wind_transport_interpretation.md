# Coastal wind transport — interpretation (thesis draft)

## Purpose
Directional exposure and association framing for maritime signals, coastal wind, NO2 and oil proxies. Not proof of deposition, health harm, or emission attribution.

## Methodology (summary)
- **Landward bearing (receptor):** bearing from each grid centroid to the nearest Natural Earth coastline sample (or land-boundary fallback). Compared to wind toward from weekly mean u/v (Open-Meteo ERA5 archive or `--wind-csv`). `coastal_wind_alignment_score = cos(smallest angle difference)`.
- **Hotspot corridor:** each week, top-decile vessel density, NO2, or oil slick proxy define a combined hotspot pool; each row gets the nearest hotspot. Bearing from hotspot to nearest coastline sample yields `pollution_transport_wind_alignment_score` vs wind toward at the receptor.
- **Panel:** same coastal mask as `run_land_pollution_drivers_wind` (distance to coast <=30 km; shipping distance <30 km). 0.1 deg grid via `grid_cell_id`.

## Wind conventions (validation)
- Open-Meteo `wind_direction_10m` is meteorological FROM; u east, v north; wind toward = atan2(u,v) deg from N.
- Initial bearing: forward azimuth point1 to point2, clockwise from north.
- Angles use smallest arc in [0,180] before cosine.

## Assumptions
- Weekly aggregation is informative for NO2 and proxies.
- Cluster-representative wind approximates local flow.
- Nearest coastline sample approximates landward normal for marine-heavy cells; coarse for some land cells.

## Limitations
- Confounding (urban NO2, weather, sensors, SAR speckle).
- 110m shoreline simplification.
- Scalar cosine cannot prove plume paths.
- **Negative correlation is not evidence against transport**; it can reflect mixing, sources, or geometry.

## Quantitative snapshots

- Coastal panel rows: **1,530**, grids **30**, weeks **51**.
- Wind source: **open_meteo_era5_archive**.

## Strongest statistical signals (exploratory)

- **spearman_no2_vs_vessel_times_coastal_alignment**: rho=-0.123, p=8.14e-05, n=1017.
- **spearman_local_no2_excess_vs_pollution_transport_alignment**: rho=-0.101, p=0.000242, n=1314.
- **spearman_weekly_no2_anomaly_vs_coastal_wind_alignment**: rho=-0.086, p=0.00191, n=1314.
- **spearman_local_no2_excess_vs_coastal_wind_alignment**: rho=-0.081, p=0.0035, n=1314.
- **spearman_no2_vs_coastal_alignment_within_high_vessel_p90**: rho=-0.176, p=0.0427, n=133.

### Full test table (`coastal_wind_exposure_summary.csv`)

                                                      test                       statistic         value  p_value    n                                                                   notes
       spearman_local_no2_excess_vs_coastal_wind_alignment                      spearman_r -8.050588e-02 0.003498 1314             Receptor landward bearing vs wind toward (see methodology).
     spearman_weekly_no2_anomaly_vs_coastal_wind_alignment                      spearman_r -8.555144e-02 0.001910 1314             Receptor landward bearing vs wind toward (see methodology).
spearman_local_no2_excess_vs_pollution_transport_alignment                      spearman_r -1.011009e-01 0.000242 1314 Hotspot→coast bearing vs wind toward at receptor (weekly hotspot pool).
            spearman_no2_vs_vessel_times_coastal_alignment                      spearman_r -1.232381e-01 0.000081 1017 Naive maritime×shoreward‑wind coupling (not a formal interaction test).
  spearman_no2_vs_coastal_alignment_within_high_vessel_p90                      spearman_r -1.759980e-01 0.042724  133     Restrict to high vessel-density rows (weekly cross-section pooled).
                  spearman_ndvi_vs_oil_x_coastal_alignment                      spearman_r -2.926954e-03 0.908925 1530          Oil×shorewind vs NDVI (exploratory land response; confounded).
  mannwhitney_no2_excess_shoreward_wind_vs_not_band_0-3_km mean_diff_shoreward_minus_other -3.096611e-06 0.024841  176               Shoreward wind: coastal_wind_alignment_score >= cos(45°).
  mannwhitney_no2_excess_shoreward_wind_vs_not_band_3-7_km mean_diff_shoreward_minus_other -9.544767e-07 0.199942  307               Shoreward wind: coastal_wind_alignment_score >= cos(45°).
 mannwhitney_no2_excess_shoreward_wind_vs_not_band_7-15_km mean_diff_shoreward_minus_other -5.585981e-07 0.240954  355               Shoreward wind: coastal_wind_alignment_score >= cos(45°).
mannwhitney_no2_excess_shoreward_wind_vs_not_band_15-30_km mean_diff_shoreward_minus_other -5.181719e-07 0.319573  476               Shoreward wind: coastal_wind_alignment_score >= cos(45°).
                        decay_profile_mean_no2_band_0-3_km           mean_local_no2_excess  2.384125e-06      NaN  176          Pooled mean NO2 excess by shipping-distance band (not causal).
                        decay_profile_mean_no2_band_3-7_km           mean_local_no2_excess  1.315436e-06      NaN  307          Pooled mean NO2 excess by shipping-distance band (not causal).
                       decay_profile_mean_no2_band_7-15_km           mean_local_no2_excess  2.631127e-06      NaN  355          Pooled mean NO2 excess by shipping-distance band (not causal).
                      decay_profile_mean_no2_band_15-30_km           mean_local_no2_excess  3.016064e-06      NaN  476          Pooled mean NO2 excess by shipping-distance band (not causal).

## Strongest cautions
- Association / exposure language only; avoid causal transport claims.

## Files (repo-relative)
- `outputs/reports/run_coastal_wind_transport/coastal_wind_alignment_features.csv`
- `outputs/reports/run_coastal_wind_transport/coastal_wind_exposure_summary.csv`
- `outputs/visualizations/run_coastal_wind_transport/coastal_pollution_transport_map.html`
- `outputs/figures/run_coastal_wind_transport/` — includes `coastal_pollution_transport_context_map.png` (quiver + shoreline alignment coloring).
- Example augment: `python3 src/analysis/run_coastal_wind_transport.py --augment-parquet outputs/processed/features_ml_ready_coastal_wind.parquet`
