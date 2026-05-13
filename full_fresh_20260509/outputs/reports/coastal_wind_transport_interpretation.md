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

- Coastal panel rows: **13,821**, grids **271**, weeks **51**.
- Wind source: **open_meteo_era5_archive**.

## Strongest statistical signals (exploratory)

- **spearman_ndvi_vs_oil_x_coastal_alignment**: rho=-0.098, p=3.35e-22, n=9741.
- **spearman_local_no2_excess_vs_pollution_transport_alignment**: rho=0.026, p=0.00355, n=12117.
- **spearman_weekly_no2_anomaly_vs_coastal_wind_alignment**: rho=-0.026, p=0.00359, n=12117.
- **spearman_local_no2_excess_vs_coastal_wind_alignment**: rho=-0.022, p=0.0147, n=12117.

### Full test table (`coastal_wind_exposure_summary.csv`)

                                                      test                       statistic         value      p_value     n                                                                   notes
       spearman_local_no2_excess_vs_coastal_wind_alignment                      spearman_r -2.217114e-02 1.466343e-02 12117             Receptor landward bearing vs wind toward (see methodology).
     spearman_weekly_no2_anomaly_vs_coastal_wind_alignment                      spearman_r -2.645337e-02 3.589806e-03 12117             Receptor landward bearing vs wind toward (see methodology).
spearman_local_no2_excess_vs_pollution_transport_alignment                      spearman_r  2.648664e-02 3.547922e-03 12117 Hotspot→coast bearing vs wind toward at receptor (weekly hotspot pool).
            spearman_no2_vs_vessel_times_coastal_alignment                      spearman_r  1.007927e-02 4.650391e-01  5256 Naive maritime×shoreward‑wind coupling (not a formal interaction test).
  spearman_no2_vs_coastal_alignment_within_high_vessel_p90                      spearman_r -4.192105e-02 3.340566e-01   533     Restrict to high vessel-density rows (weekly cross-section pooled).
                  spearman_ndvi_vs_oil_x_coastal_alignment                      spearman_r -9.794784e-02 3.352318e-22  9741          Oil×shorewind vs NDVI (exploratory land response; confounded).
  mannwhitney_no2_excess_shoreward_wind_vs_not_band_0-3_km mean_diff_shoreward_minus_other -6.950547e-07 2.708010e-01   880               Shoreward wind: coastal_wind_alignment_score >= cos(45°).
  mannwhitney_no2_excess_shoreward_wind_vs_not_band_3-7_km mean_diff_shoreward_minus_other -6.534197e-07 3.856163e-01  1015               Shoreward wind: coastal_wind_alignment_score >= cos(45°).
 mannwhitney_no2_excess_shoreward_wind_vs_not_band_7-15_km mean_diff_shoreward_minus_other  4.628461e-09 2.818480e-01  1563               Shoreward wind: coastal_wind_alignment_score >= cos(45°).
mannwhitney_no2_excess_shoreward_wind_vs_not_band_15-30_km mean_diff_shoreward_minus_other -6.796953e-07 8.725274e-02  2074               Shoreward wind: coastal_wind_alignment_score >= cos(45°).
                        decay_profile_mean_no2_band_0-3_km           mean_local_no2_excess  1.976913e-06          NaN   880          Pooled mean NO2 excess by shipping-distance band (not causal).
                        decay_profile_mean_no2_band_3-7_km           mean_local_no2_excess  2.096766e-06          NaN  1015          Pooled mean NO2 excess by shipping-distance band (not causal).
                       decay_profile_mean_no2_band_7-15_km           mean_local_no2_excess  1.496127e-06          NaN  1563          Pooled mean NO2 excess by shipping-distance band (not causal).
                      decay_profile_mean_no2_band_15-30_km           mean_local_no2_excess  1.943931e-07          NaN  2074          Pooled mean NO2 excess by shipping-distance band (not causal).

## Strongest cautions
- Association / exposure language only; avoid causal transport claims.

## Files (repo-relative)
- `outputs/reports/run_coastal_wind_transport/coastal_wind_alignment_features.csv`
- `outputs/reports/run_coastal_wind_transport/coastal_wind_exposure_summary.csv`
- `outputs/visualizations/run_coastal_wind_transport/coastal_pollution_transport_map.html`
- `outputs/figures/run_coastal_wind_transport/` — includes `coastal_pollution_transport_context_map.png` (quiver + shoreline alignment coloring).
- Example augment: `python3 src/analysis/run_coastal_wind_transport.py --augment-parquet outputs/processed/features_ml_ready_coastal_wind.parquet`
