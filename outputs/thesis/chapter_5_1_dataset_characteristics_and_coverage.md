# Chapter 5.1 Dataset characteristics and coverage

*Machine-learning-ready weekly panel analysed from repository artefacts (generated analysis script: `scripts/build_chapter_5_1_dataset_characteristics.py`).*

## 5.1.1 Final dataset statistics and observation structure

The definitive thesis panel is stored as **`processed/features_ml_ready.parquet`**. It contains **16,065** row observations and **46** tabulated variables. After harmonisation, the panel enumerates **315** distinct `grid_cell_id` polygons and **51** non-overlapping UTC week anchors spanning **2023-01-01** through **2023-12-17**. The product **315 × 51 = 16,065** matches the empirical row cardinality, implying a **complete factorial skeleton** (every cell appears in every week bin). Observation counts therefore constitute neither opportunistic samples nor hierarchical subsampling artefacts; they instantiate a deliberate space–time lattice suitable for longitudinal and spatial–temporal exploratory analyses.

### Table CS-1. Core dimensions (CSV export)

See **`outputs/thesis/tables/chapter_5_1_dataset_core_statistics.csv`**.

### Table CS-2. Thesis-oriented feature-category counts

| Category | Number of columns |
| --- | --- |
| Environmental (optical / water quality) | 19 |
| Atmospheric | 7 |
| Lagged interactions | 3 |
| Maritime traffic | 3 |
| Coastal exposure geometry | 2 |
| Oil / SAR proxy | 2 |
| Port distance / attribution | 2 |
| QC / diagnostics | 2 |
| Spatial coordinates | 2 |
| Maritime composite indices | 1 |
| Regional attribution | 1 |
| Spatial (identifier) | 1 |
| Temporal index | 1 |

_Source file:_ `outputs/thesis/tables/chapter_5_1_feature_category_counts.csv`

**Interpretation.** Categories summarise interpretive taxonomy for narrative exposition; individual columns may logically participate in multiple environmental processes. Maritime and atmospheric composites coexist with Sentinel-derived optical summaries and engineered coastal-exposure constructs, supplying both biophysical and anthropogenic explanatory dimensions.

## 5.1.2 Spatial coverage and structural consistency

Latitude ranges from **-0.7500°** to **63.0500°** and longitude from **-24.3500°** to **22.4500°**, circumscribing the working Baltic littoral lattice encoded in centroid metadata. Consistency is buttressed indirectly by **`nearest_port`** attributions summarised in **`chapter_5_1_spatial_coverage_by_nearest_port.csv`**, which documents row shares concentrated around Mariehamn, Stockholm, Naantali, and Turku—ports that anchor heterogeneous coastal forcing contexts.

Vessel-density spatial structure (Spearman rank correlation of `vessel_density` with latitude and longitude) exhibits **ρ(lat, density) = 0.144** and **ρ(lon, density) = 0.263** over jointly non-missing records. These associations are **descriptive** gradients (not causal effects) but confirm that shipping intensity is heterogeneous across the study grid rather than spatially uniform.

Pipeline validation (`data/validation/full_pipeline_validation_report.json`) reports cross-source **feature completeness ≈ 52.08%** at the integration audit stage; the flag **spatial_flatness = True** cautions that certain weekly aggregates may exhibit limited spatial contrast when evaluated through specific automated checks—an important qualifier when interpreting shipping or anomaly diagnostics.

### Table CS-3. Regional coverage (nearest port × layer availability)

See **`outputs/thesis/tables/chapter_5_1_spatial_coverage_by_nearest_port.csv`** (path: `outputs/thesis/tables/chapter_5_1_spatial_coverage_by_nearest_port.csv`).

## 5.1.3 Temporal coverage, continuity, and seasonal behaviour

Successive calendar weeks are separated by a **median gap of 7.0 days** (maximum 7 days); all inter-week intervals equal seven days, signalling **strict ISO-week cadence** in the stored anchors. Each week contributes **315** rows, mirroring the fixed grid population. **100.0%** of grids register observations in every week slot, so panel attrition is not driven by absent weeks for most locations.

Seasonal optical signals remain **patchy** because Sentinel-2 water-quality stacks populate only a minority of grid-weeks; nevertheless, aggregating `ndti_mean` by ISO week-of-year still reveals slow seasonal modulation (see preprocessing diagnostics and thesis figures under `outputs/eda/` and `outputs/thesis/figures/`). Analysts should therefore separate **panel regularity** (fixed weekly bins) from **sensor observability** (cloud and sea-state driven gaps).

### Table CS-4. Temporal coverage summary metrics

See **`outputs/thesis/tables/chapter_5_1_temporal_coverage_summary.csv`**.

## 5.1.4 Missingness, structural sparsity, and analytical consequences

`nan_ratio_row` provides a row-level digest of input missingness; optical means (`ndwi_*`, `ndti_*`, `ndci_*`, `fai_*`, `b11_*`) exhibit **≈81.8% missingness** at the cell-week level, consistent with strict cloud masking and narrow water footprints. By contrast, **NO₂** and **oil slick proxy** fields retain **≈10–11%** and **≈5.4%** missing fractions respectively, enabling richer multitrophic modelling for those modalities. **`no2_x_ndvi`**, **`vessel_x_ndvi_lag*`** and **`land_response_index`** sit at **>97% missing**, reflecting intermittent land-linkage engineering rather than observational intermittency alone.

Land–sea imbalance is stark: **`ndvi_mean` valid in only ~2.30%** of rows versus **`ndti_mean` ~18.23%**, underscoring that vegetation-side diagnostics target rare coastal linkage cells whereas turbidity composites emphasise aquatic pixels. Optical missingness propagates into correlation and supervised-learning exercises by **shrinking pairwise complete samples**, biasing associative estimates toward cloud-free regimes unless modellers apply explicit censored-data strategies.

The archived Sentinel-1 oil validation record flags **`['insufficient_sparsity', 'no_spatial_signal']`**, signalling cautious interpretation for dark-water surrogates that do not emulate literal slick inventories.

### Table CS-5. Highest missing-rate features (top 20)

| feature | missing_pct | coverage_pct |
| --- | --- | --- |
| no2_x_ndvi | 98.9107 | 1.0893 |
| vessel_x_ndvi_lag3 | 98.1015 | 1.8985 |
| vessel_x_ndvi_lag2 | 98.1015 | 1.8985 |
| vessel_x_ndvi_lag1 | 98.0952 | 1.9048 |
| land_response_index | 97.7031 | 2.2969 |
| ndvi_std | 97.7031 | 2.2969 |
| ndvi_median | 97.7031 | 2.2969 |
| ndvi_mean | 97.7031 | 2.2969 |
| ndci_median | 81.7678 | 18.2322 |
| b11_median | 81.7678 | 18.2322 |
| b11_mean | 81.7678 | 18.2322 |
| fai_std | 81.7678 | 18.2322 |
| fai_median | 81.7678 | 18.2322 |
| fai_mean | 81.7678 | 18.2322 |
| ndci_std | 81.7678 | 18.2322 |
| b11_std | 81.7678 | 18.2322 |
| ndci_mean | 81.7678 | 18.2322 |
| ndti_std | 81.7678 | 18.2322 |
| ndti_median | 81.7678 | 18.2322 |
| ndti_mean | 81.7678 | 18.2322 |

_Full column listing:_ **`outputs/thesis/tables/chapter_5_1_missingness_all_columns.csv`**.

### Table CS-6. NDTI observational persistence (per grid)

Distribution summarised in **`outputs/thesis/tables/chapter_5_1_optical_persistence_summary.csv`**; grid-level CSV: **`chapter_5_1_ndti_persistence_per_grid.csv`**.

## 5.1.5 Descriptive statistics for selected substantive variables

See **`outputs/thesis/tables/chapter_5_1_descriptive_statistics_selected.csv`** for **`vessel_density, vessel_density_t, NO2_mean, ndti_mean, ndwi_mean, coastal_exposure_score, oil_slick_probability_t, nan_ratio_row, detection_score`**.

## 5.1.6 Ancillary evidence in `outputs/reports/` and diagnostic plots

The repository presently enumerates approximately **127** CSV reports and **41** markdown summaries beneath **`outputs/reports/`**, alongside **3** exploratory PNG assets in **`outputs/eda/`** and **283** preprocessing diagnostic figures (recursive PNG search under **`outputs/preprocessing_diagnostics/`**). These artefacts support graphical cross-checking of distributions, detection scores, SAR proxies, and missingness landscapes referenced above.

## 5.1.7 Consolidated strengths and limitations

### Strengths

1. **Orthogonal indexing:** Full grid-week crossing yields transparent sample sizes without hidden replication or unequal temporal weighting.
2. **Multilayer thematic richness:** Atmospheric, SAR, Sentinel-2 water-quality, maritime intensity, coastal geometry, and diagnostics coexist in one harmonised schema, aligning with multimodal coastal-impact hypotheses.
3. **Documented QA lineage:** Serialised validation JSON summaries (`data/validation/`) quantify layer-level coverage and anomaly flags traceable alongside analytical outputs (`outputs/reports/`).
4. **Deterministic reproducibility anchors:** Canonical timestamps and parquet persistence enable exact replay of exploratory analyses.

### Limitations

1. **Optical sparsity & cloud censorship:** Majority-missing Sentinel-based means restrict inference to favourable observing conditions unless models accommodate informative missingness or joint imputation.
2. **Land-side descriptors under-sampled:** NDVI-mediated interactions remain statistically thin.
3. **Validation warnings:** Cross-source completeness <100% and reported `spatial_flatness` imply moderate structural covariance that may limit interpretability of certain cross-layer correlates.
4. **Proxy semantics:** SAR oil probability is a radiometric heuristic, not ground-truthed slick census; associations must be framed as observational co-movements.
5. **Single-scale gridding:** One discrete mesh cannot resolve sub-cell heterogeneity; exposure indices aggregate sub-grid variability.
