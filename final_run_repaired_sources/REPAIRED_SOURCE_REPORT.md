# Repaired source report — final_run_repaired_sources/

- Source dataset: `final_run/processed/features_ml_ready.parquet`

## 1. Was existing vessel data temporal?
- Status: **spatial_proxy**
- Coverage > 70%: True
- Fraction of grid cells temporally varying: 0.0
- Weekly-mean σ: 1.1212702919885052e-16
- Conclusion: vessel_density_t is **not temporal** (single value per grid; weekly-mean variance ≈ 0). It is a per-cell *spatial* maritime pressure proxy.

## 2. Was external temporal vessel data found?
- EMODnet probe failed: `<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1018)>`. No external vessel layer injected.
- External temporal vessel used: **False**.

## 3. Final vessel feature used
- Final feature: **`vessel_density_spatial_proxy`**.
- Action: spatial_proxy_with_derived_exposure.
- Derived columns added: ['vessel_density_spatial_proxy', 'vessel_density_spatial_proxy_log1p', 'port_pressure', 'shipping_exposure', 'coastal_shipping_pressure'].
- Interpretation: present this as a **spatial maritime pressure proxy**. Do NOT make weekly vessel-traffic claims from this column.

## 4. Was existing oil layer usable?
- Status: **unusable**
- Has non-zero signal: False
- Detection score present: False
- Extra SAR/VV/VH columns: (none)

## 5. Was a Sentinel-1 dark-slick proxy rebuilt?
- GEE not available — `ee.Initialize: no project found. Call with project= or see http://goo.gle/ee-auth.`.
- Action: **marked_unavailable** — Existing oil columns are unusable AND Earth Engine is unauthenticated; no Sentinel-1 dark-slick proxy was computed (we never fabricate oil data).
- We deliberately did NOT fabricate any oil-like values.

## 6. Final oil/slick decision
- Final feature: **`None`**
- Thesis rule: oil/slick exposure is **excluded** from the thesis-safe dataset and from any thesis claims for this run. If a Sentinel-1 dark-slick proxy is later produced, it must be described as *Sentinel-1 dark slick proxy — association, not causality*.

## 7. Thesis-safe feature list
- File: `final_run_repaired_sources/processed/features_thesis_safe.parquet` (18 columns).
  - `grid_cell_id`
  - `week_start_utc`
  - `grid_centroid_lat`
  - `grid_centroid_lon`
  - `nearest_port`
  - `distance_to_port_km`
  - `coastal_exposure_band`
  - `coastal_exposure_score`
  - `no2_mean_t`
  - `no2_std_t`
  - `ndwi_mean`
  - `ndti_mean`
  - `ndvi_mean`
  - `vessel_density_spatial_proxy`
  - `vessel_density_spatial_proxy_log1p`
  - `port_pressure`
  - `shipping_exposure`
  - `coastal_shipping_pressure`

## 8. ML-safe feature list
- The thesis-safe parquet excludes:
  - Oil/Sentinel-1 columns (unusable in this run).
  - Duplicate aliases (NO2_mean ↔ no2_mean_t, ndvi_mean ↔ land_response_index, mean ↔ median, etc. — already pruned in the prior `final_run_cleaned/` deliverable).
  - Columns with >80% missing unless explicitly retained for interpretation (NDVI etc.).

## 9. Plots to use in thesis
- `final_run_repaired_sources/outputs/visualizations/vessel_weekly_mean_raw.png`
- `final_run_repaired_sources/outputs/visualizations/vessel_spatial_proxy_map.png`
- `final_run_repaired_sources/outputs/visualizations/port_pressure_map.png`
- `final_run_repaired_sources/outputs/visualizations/shipping_exposure_map.png`
- `final_run_repaired_sources/outputs/visualizations/distance_decay/sliding_window_vessel_density_spatial_proxy_log1p.png`
- `final_run_repaired_sources/outputs/visualizations/distance_decay/sliding_window_no2_mean_t_rolling3.png`
- `final_run_repaired_sources/outputs/visualizations/distance_decay/sliding_window_ndwi_mean_winsor.png`
- `final_run_repaired_sources/outputs/visualizations/distance_decay/sliding_window_ndti_mean_winsor.png`

## 10. Plots / features to exclude from thesis
- `final_run_repaired_sources/outputs/visualizations/oil_missingness_zero_coverage.png` (use only as a transparency artefact showing the oil layer is unusable; do not draw conclusions).
- All `oil_slick_*` and `detection_score` columns: omit from any thesis claim.
- Weekly-traffic narratives based on `vessel_density_t`: replace with *spatial pressure proxy* language.

## Cautious-language reminders
- *Spatial maritime pressure proxy* (not weekly vessel traffic).
- *Sentinel-1 dark slick proxy* (not confirmed oil spill).
- *Association, not causality.*
