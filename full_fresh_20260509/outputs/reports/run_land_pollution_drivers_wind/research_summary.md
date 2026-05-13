# Land pollution drivers (wind-aware) — summary

Associations only; **not causal proof** of land deposition.

## Framing (careful wording)
- **Drivers associated with land-side pollution** = statistical association of NO₂ anomaly/excess with maritime and wind fields.  
- **Wind-supported transport evidence** = coherence between **wind alignment** (lane→cell bearing vs wind *to* direction) and elevated NO₂ or oil proxy.  
- **Potential coastal exposure pathway** — not direct shoreline proof.

## Analytic takeaways  
- **NO₂ excess vs wind aligned:** see `no2_wind_aligned_land_impact.csv`.  
- **Oil coastal-risk vs wind aligned:** `oil_slick_wind_coastal_risk.csv` — *potential transport / risk*, not proven deposition.  
- **Driver ranking:** `land_pollution_driver_correlation.csv`, `land_pollution_driver_feature_importance.csv`.

## Data notes  
- Wind source: **open_meteo_era5_archive** — default Open‑Meteo **ERA5 archive** at **1** spatial cluster(s) over the coastal panel; weekly means from hourly samples. (**Note:** archived `wind_speed_10m` is typically reported in **km/h**; direction-based alignment uses u/v-derived **bearing**, which is unit-consistent.)
- Alignment uses the **nearest high-vessel-density seed centroid** each week (P90 threshold, same spirit as `land_sea_buffering`).

## NDVI  
See `ndvi_supporting_land_response.csv` — **supporting context only**.

## Verdict

**WEAK — exploratory evidence only on this weekly / cluster-wind aggregation.**

Ridge standardized coefficients on **local_no2_excess**, n=1017. Train R² (ridge): 0.0262; RF/HGB fitted for feature ranking only.

---
Artifacts: `outputs/reports/run_land_pollution_drivers_wind/`, `outputs/figures/run_land_pollution_drivers_wind/`, `outputs/visualizations/run_land_pollution_drivers_wind/`.
