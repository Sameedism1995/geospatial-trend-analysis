# Thesis results interpretation

_Auto-generated from `scripts/run_thesis_analysis.py`; associations are descriptive, not causal._

## 1. Dataset Overview

The ML-ready weekly panel (`processed/features_ml_ready.parquet`) contains **16,065** rows, **46** columns, **315** grid cells, and **51** weekly timestamps. Several optical and interaction covariates are sparse; lowest coverage examples:

```
            feature  coverage_percent
         no2_x_ndvi            1.0893
 vessel_x_ndvi_lag2            1.8985
 vessel_x_ndvi_lag3            1.8985
 vessel_x_ndvi_lag1            1.9048
land_response_index            2.2969
```

## 2. Feature Engineering

Variables group into maritime intensity, Sentinel-derived water-quality indices (NDWI/NDTI/NDCI/FAI/B11 where available), tropospheric NO₂ measures, coastline and port-distance constructs, and interaction terms. Category assignments and narrative rationale are tabulated in `feature_purpose_table.csv`.

## 3. Correlation Findings

Among the Spearman matrix subset, the largest-magnitude association is **ndwi_mean ↔ ndci_mean** (ρ ≈ **-0.8759**).  Optical water indices are strongly inter-related; cross-domain pairs are generally weaker on the available subset. See `top_pearson_pairs.csv`, `top_spearman_pairs.csv`, and `cross_category_relationships.csv`.

## 4. Distance-Decay Findings

Distance-stratified summaries (port-centred bands) summarise how decadal/week-aggregated means move from 0–3 km to 15–30 km. 
**Automated synopsis (distance-decay metrics):**

- maritime_exposure_index: insufficient numeric series for coastline–distance comparison.
- Mean atmospheric_coastal_exposure_index decreases from the nearest band toward offshore distances, consistent with stronger coastal/port proximity signals near shore (association only).
- Mean environmental_stress_index decreases from the nearest band toward offshore distances, consistent with stronger coastal/port proximity signals near shore (association only).
- Mean local_no2_excess decreases from the nearest band toward offshore distances, consistent with stronger coastal/port proximity signals near shore (association only).
- Mean oil_slick_proxy increases with distance from the focal port bands in this summary table (association only); interpret jointly with metric definition and coverage.
- Mean ndti_mean decreases from the nearest band toward offshore distances, consistent with stronger coastal/port proximity signals near shore (association only).
- Mean ndwi_mean increases with distance from the focal port bands in this summary table (association only); interpret jointly with metric definition and coverage.

**Quantitative cue:** aggregated band slope cue — ndwi_mean: Δ(far−near)=0.172176 (non_decreasing)

## 5. Wind-Regime Findings

Comparing decay-table rows labelled shoreward versus nonshoreward highlights indicators whose band means shift under onshore-aligned flow assumptions (see `wind_regime_summary.csv`). 
**Largest shoreward amplification (tabular % difference):** coastal_wind_alignment_score at Stockholm: shoreward vs nonshoreward mean gap ≈ 472.51% (table means).

## 6. Lagged Relationship Findings

`lagged_correlations.csv` currently enumerates vessel density against NO₂ and against the optical detection score across lags. Optimal lag and peak correlations are summarized in `strongest_lag_relationships.csv` and `lag_interpretation_table.csv`. NDTI/NDWI/environmental stress are not represented in this file; interpretations are limited to exported pairs.

## 7. Port Comparisons

Highest mean across listed decay-metric bundle: Turku (row-mean of available metrics). See `port_comparison_summary.csv`, `normalized_port_metrics.csv`, and figure exports.

## 8. Limitations

- Correlations omit confounders; ecological and atmospheric processes are synchronous and spatially structured.
- Optical features have uneven coverage; pairwise correlations use available-complete observations implicitly in upstream matrices.
- Distance and wind stratifications rely on pre-aggregated report tables rather than hierarchical models.
- Lag exploration is sparse (few target pairs exported).

## 9. Overall Conclusion

Consistent multimodal signatures link maritime adjacency and atmospheric composites to optical water descriptors in zones with data, while aggregated distance bands show heterogeneous metric-by-metric patterns. Coastal wind regimes modulate summary means in the decay tables. All conclusions should stress associational wording and uncertainty from missingness and aggregation.
