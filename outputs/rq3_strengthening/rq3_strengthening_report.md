# RQ3 Strengthening Report — Wind Regimes

**Research question (thesis):** How do coastal wind regimes structure maritime activity and environmental exposure detection in the Turku–Mariehamn corridor?

**Evidence sources (existing outputs only; no ML):**

| File | Role |
|------|------|
| `outputs/reports/coastal_exposure_statistics.csv` | Distance-band bootstrap means; Mann–Whitney + Cliff’s δ; panel Spearman vs wind alignment |
| `outputs/reports/indicator_participation_statistics.csv` | Rank-participation bootstrap CIs; wind-regime contrasts by band |
| `outputs/reports/vessel_density_wind_regime_tests.csv` | Vessel density vs shoreward/non-shoreward (raw, demeaned, port/band slices) |
| `outputs/reports/run_land_pollution_drivers_wind/land_pollution_driver_correlation.csv` | Coastal-panel Spearman: wind speed/alignment vs NO₂ excess and oil proxy |

**Significance threshold:** α = 0.05 (two-sided). **Non-circular** indicators exclude **Maritime Exposure Index (MEI)** and wind-definition fields used only to label regimes.

---

## 1. Statistically significant wind-related findings (non-MEI only)

### 1a. Mann–Whitney: shoreward vs non-shoreward (`coastal_exposure_statistics.csv`, block `mannwhitney_cliffs_delta`)

| Distance band | Indicator | n (shore / other) | Mean shoreward | Mean non-shoreward | Cliff’s δ | Mann–Whitney *p* |
|---------------|-----------|-------------------|----------------|--------------------|-----------|------------------|
| 0–3 km | **local_no2_excess** | 61 / 115 | 4.90×10⁻⁷ | 3.39×10⁻⁶ | −0.18 | **0.049** |
| 0–3 km | **atmospheric_coastal_exposure_index (ACEI)** | 61 / 115 | 0.834 | 0.522 | +0.69 | **6.3×10⁻¹⁴** |
| 0–3 km | **environmental_stress_index (ESI)** | 65 / 139 | 0.724 | 0.666 | +0.21 | **0.016** |
| 3–7 km | **ACEI** | 100 / 207 | 0.828 | 0.540 | +0.64 | **9.0×10⁻²⁰** |
| 3–7 km | **ESI** | 105 / 252 | 0.557 | 0.482 | +0.26 | **1.5×10⁻⁴** |
| 7–15 km | **ACEI** | 144 / 211 | 0.874 | 0.537 | +0.70 | **1.7×10⁻²⁹** |
| 7–15 km | **ESI** | 151 / 257 | 0.628 | 0.532 | +0.31 | **2.0×10⁻⁷** |
| 15–30 km | **ACEI** | 91 / 385 | 0.803 | 0.477 | +0.65 | **1.0×10⁻²²** |
| 15–30 km | **ESI** | 123 / 438 | 0.494 | 0.445 | +0.14 | **0.023** |

**Note:** Inner-band **NO₂ excess** is **lower** under shoreward wind (negative δ), opposite to ACEI/ESI direction—interpret as localized residual structure, not a simple “more pollution when wind blows onshore.”

**Bootstrap band means (non-MW rows):** Shoreward ACEI/ESI bootstrap 95% intervals sit above non-shoreward intervals in every band (e.g. 0–3 km ACEI shore CI [0.791, 0.874] vs non-shore [0.473, 0.572]); full CIs in source file columns `ci95_low`, `ci95_high`.

### 1b. Rank participation by wind regime (`indicator_participation_statistics.csv`, `rank_percentile_coastal_panel`)

Significant contrasts (Mann–Whitney *p* < 0.05; Cliff’s δ vs opposite wind, same band):

| Band | Indicator | Representative shoreward rank CI [low, high] | Cliff’s δ | *p* |
|------|-----------|---------------------------------------------|-----------|-----|
| 0–3 km | **local_no2_excess** | [0.369, 0.502] vs non-shore [0.483, 0.589] | ±0.21 | **0.025** |
| 0–3 km | **sentinel_ndwi** | shore [0.648, 0.875] vs non-shore [0.453, 0.634] | ±0.51 | **0.012** (n = 11 / 35) |
| 0–3 km | **ACEI** | shore [0.652, 0.777] vs non-shore [0.358, 0.442] | ±0.65 | **1.4×10⁻¹²** |
| 0–3 km | **ESI** | shore [0.737, 0.828] vs non-shore [0.663, 0.741] | ±0.20 | **0.021** |
| 3–7 km | **sentinel_ndwi** | *p* = **0.0099** (n = 21 / 56) | ±0.38 | |
| 3–7 km | **ACEI** | *p* = **7.0×10⁻²⁰** | ±0.64 | |
| 3–7 km | **ESI** | *p* = **5.9×10⁻⁵** | ±0.27 | |
| 7–15 km | **ACEI** | *p* = **2.8×10⁻³⁰** | ±0.71 | |
| 7–15 km | **ESI** | *p* = **2.1×10⁻⁸** | ±0.33 | |
| 15–30 km | **ACEI** | *p* = **1.2×10⁻²³** | ±0.67 | |
| 15–30 km | **ESI** | *p* = **0.028** | ±0.14 | |

**Optical caveat:** NDWI contrasts use **small n** (11–56 cells per regime); treat as exploratory.

### 1c. Panel Spearman vs wind alignment (`coastal_exposure_statistics.csv`, block `spearman`)

| Pair | n | ρ | *p* | Non-circular? |
|------|---|-----|-----|----------------|
| local_no2_excess ↔ coastal_wind_alignment_score | 1,314 | −0.085 | **0.0020** | Yes |
| local_no2_excess ↔ **ACEI** | 1,017 | +0.264 | **2.5×10⁻²²** | Yes |
| oil_slick_probability_t ↔ coastal_wind_alignment_score | 1,530 | +0.110 | **1.7×10⁻⁵** | Yes |

Excluded from non-circular list: local_no2_excess ↔ **MEI** (ρ = −0.132, *p* = 2.6×10⁻⁵) — **circular** (see §4).

### 1d. Wind driver correlations (`land_pollution_driver_correlation.csv`)

| Predictor | Outcome | n | ρ | *p* |
|-----------|---------|---|-----|-----|
| **wind_speed_mean** | local_no2_excess | 1,314 | −0.058 | **0.036** |
| **wind_speed_mean** | oil_slick_probability_t | 1,530 | +0.054 | **0.035** |
| wind_alignment_score | local_no2_excess | 1,314 | −0.014 | 0.608 (n.s.) |
| wind_alignment_score | oil_slick_probability_t | 1,530 | −0.047 | 0.065 (n.s.) |

Effect sizes are **weak** (|ρ| < 0.06) but precisely estimated with large *n*.

---

## 2. Wind effects on vessel activity (Section 3a)

**Source:** `vessel_density_wind_regime_tests.csv` (+ vessel rows in `indicator_participation_statistics.csv`).

| Analysis | Port / scope | n (shore / non) | Mean shore | Mean non | Mann–Whitney *p* | Spearman ρ | *p* (ρ) |
|----------|--------------|-----------------|------------|----------|------------------|------------|---------|
| Coastal panel, raw | all_coastal | 384 / 789 | 0.540 | 0.570 | **0.815** | 0.007 | 0.815 |
| Month-median demeaned | all_coastal | 384 / 789 | 0.258 | 0.289 | **0.815** | 0.007 | 0.815 |
| Raw | Turku | 169 / 341 | 0.454 | 0.377 | 0.099 | 0.073 | 0.099 |
| Turku 0–30 km (Fig 5.7 slice) | Turku | 19 / 83 | 0.375 | 0.397 | **0.804** | −0.025 | 0.802 |
| Rank participation **7–15 km** | all_coastal | 151 / 257 | 0.519 | 0.455 | **0.024** | 0.112 | **0.024** |

**Summary (vessel activity):**

- **No significant difference** in **raw or seasonally demeaned** vessel density between shoreward and non-shoreward weeks on the full coastal panel (*p* ≈ 0.815, *n* = 1,173).
- **Turku-only** and **0–30 km** slices remain **non-significant** at α = 0.05 (Turku pooled *p* ≈ 0.099).
- **Only significant vessel finding:** higher **rank participation** for vessel density in the **7–15 km** band under shoreward wind (*p* = 0.024, δ = +0.13)—a **relative** coastal ranking, not proof that wind drives traffic volume.
- **0–3 km** and **3–7 km** rank-participation vessel contrasts: *p* = 0.637 and 0.075 (n.s.).

**Defensible claim:** Wind regime classification **does not** significantly separate vessel density in the coastal panel after pooling; any band-specific rank shift is modest and limited to mid-distance strata.

---

## 3. Wind effects on exposure detection (Section 3b)

**Sources:** Tables in §1; indicators = NO₂ excess, ACEI, ESI, NDWI (sparse), oil proxy.

**Consistent pattern (non-circular):**

1. **ACEI** — Large positive Cliff’s δ (0.64–0.71) and extremely small *p* in **every** distance band under shoreward vs non-shoreward classification.
2. **ESI** — Significant positive δ in all bands (*p* from 1.5×10⁻⁴ to 0.023); composite includes atmospheric and maritime components but is **not** identical to MEI wind alignment.
3. **NO₂ excess** — Significant only at **0–3 km** (MW *p* = 0.049, δ = −0.18); rank-participation contrast at 0–3 km (*p* = 0.025) with **lower** shoreward participation—suggests wind-regime structuring of **localized** NO₂ residuals, not uniform enhancement.
4. **NDWI** — Significant rank contrasts at 0–3 km and 3–7 km with **small samples**.
5. **Oil slick proxy** — Positive Spearman with wind alignment (ρ = 0.11, *p* = 1.7×10⁻⁵); wind speed weakly positive with oil (ρ = 0.054, *p* = 0.035).

**Bootstrap CIs** on band means (coastal_exposure_statistics) confirm higher shoreward **ACEI** and often **ESI**; vessel_density_t band means show overlapping CIs between wind splits (no systematic vessel boost under shoreward wind in raw means).

---

## 4. Circular analyses involving MEI (do not use for causal wind claims)

| Source | Finding | Why circular |
|--------|---------|--------------|
| `coastal_exposure_statistics.csv` | MEI Mann–Whitney: all bands *p* ≈ 10⁻⁶–10⁻⁴², Cliff’s δ ≈ 0.43–1.00 | **MEI** incorporates **coastal wind alignment** (and vessel weighting); shoreward label partly **defines** high MEI |
| `coastal_exposure_statistics.csv` | Spearman: local_no2_excess vs **maritime_exposure_index** ρ = −0.132, *p* = 2.6×10⁻⁵ | MEI wind-linked |
| `indicator_participation_statistics.csv` | MEI rank participation: all bands *p* ≈ 10⁻⁶–10⁻⁵¹, \|δ\| ≈ 0.88–1.00 | Same construction |
| `indicator_participation_statistics.csv` | **coastal_wind_alignment**, **pollution_transport_alignment** *p* ≈ 10⁻³⁰–10⁻⁶⁵, \|δ\| = 1.0 | These are **wind-definition** fields, not independent outcomes |

**Rule for thesis:** Report MEI/wind-alignment contrasts as **internal consistency of regime labeling**, not as independent environmental validation of wind impact.

---

## 5. Final defensible answer to RQ3 (non-circular evidence only)

**Answer status: Partially answered.**

1. **Vessel activity:** On the coastal panel (*n* ≈ 1,173 cell-weeks), **shoreward vs non-shoreward** classification is **not** associated with significantly different **vessel_density_t** (Mann–Whitney *p* ≈ 0.815), including after **month-median demeaning** and in **Turku-focused** subsets (*p* > 0.05). The data **do not support** the claim that wind regimes control shipping intensity in this corridor.

2. **Exposure detection:** Wind regimes **do** significantly structure **non-circular** exposure indicators—especially **ACEI** (large Cliff’s δ, *p* < 10⁻¹² in all distance bands) and **ESI** (significant in all bands)—with **bootstrap-confirmed** higher shoreward rank participation. **Localized NO₂ excess** shows a **significant but small** inner-band contrast (δ ≈ −0.18, *p* ≈ 0.049) and weak panel-level correlation with wind alignment (ρ ≈ −0.09). **Oil proxy** correlates positively with alignment (ρ ≈ 0.11). Effects are **associative** and partly **engineered** into atmospheric composites; they are not proof of causal atmospheric transport.

3. **Synthesis:** RQ3 is best read as: **wind direction structures how exposure is detected and composited along the coast, not how much shipping occurs.** Negative/null vessel findings and positive ACEI/ESI findings can both be true without contradiction.

**Claims to avoid:** Wind drives vessel traffic; MEI proves environmental harm under shoreward wind; causal pollution transport from wind classes alone.

---

## 6. Thesis-ready wording

### Section 5.4 (Results — wind regimes)

Coastal wind regimes were classified as shoreward versus non-shoreward using the 45° alignment rule on ERA5-derived weekly vectors over the coastal panel. Nonparametric comparisons showed **no statistically significant difference** in vessel density between regimes on the pooled coastal sample (Mann–Whitney *p* = 0.815; *n* = 384 shoreward vs 789 non-shoreward cell-weeks; Spearman ρ = 0.007, *p* = 0.815). The same null result held after subtracting monthly medians within the panel (*p* = 0.815). Turku-only subsets and the 0–30 km window used in Figure 5.7 remained non-significant at the 5% level (Turku pooled *p* = 0.099; 0–30 km *p* = 0.804).

In contrast, **wind regime significantly stratified exposure indicators that do not reuse the maritime wind composite.** Atmospheric coastal exposure index (ACEI) was higher under shoreward classification in every distance annulus (Cliff’s δ = 0.64–0.71; Mann–Whitney *p* < 10⁻¹²; shoreward *n* = 61–144 vs non-shoreward *n* = 115–385 depending on band). Environmental stress index (ESI) showed the same directional pattern (δ = 0.14–0.31; *p* ≤ 0.023). Local NO₂ excess differed only in the inner 0–3 km band (δ = −0.18; *p* = 0.049), with **lower** shoreward means—indicating regime-dependent **localized** residuals rather than a uniform onshore NO₂ enhancement. Rank-participation analysis with bootstrap 95% intervals corroborated higher shoreward ACEI and ESI participation (e.g. 0–3 km ACEI *p* = 1.4×10⁻¹²; participation CI shoreward [0.652, 0.777] vs non-shoreward [0.358, 0.442]). Optical NDWI contrasts at 0–3 km and 3–7 km were significant (*p* = 0.012 and 0.010) but based on **small cell counts** (11–56 observations per regime) and are reported as exploratory.

Panel-level Spearman correlations confirmed weak but significant associations between wind alignment and local NO₂ excess (ρ = −0.085, *p* = 0.002; *n* = 1,314) and oil slick probability (ρ = 0.110, *p* = 1.7×10⁻⁵). Wind speed correlated weakly with NO₂ excess (ρ = −0.058, *p* = 0.036) and oil proxy (ρ = +0.054, *p* = 0.035). **Maritime Exposure Index contrasts by wind regime are omitted from substantive interpretation** because MEI embeds wind alignment and therefore duplicates the regime definition (see Section 6.3).

### Section 6.3 (Discussion — wind and limitations)

The wind-regime analysis separates two questions that are often conflated: whether wind **drives shipping volume**, and whether wind **structures observable exposure**. On the first question, the coastal panel provides **no significant evidence** that shoreward weeks concentrate higher vessel density—either in raw units or after month-level demeaning—consistent with AIS seasonality dominating over synoptic wind direction at weekly resolution. The single significant vessel-related contrast (higher vessel **rank participation** at 7–15 km, *p* = 0.024) describes **relative** standing within the coastal distribution, not a large absolute traffic response, and is not reproduced at 0–3 km or 3–7 km.

On the second question, **ACEI and ESI rise systematically under shoreward classification** across distance bands, with large nonparametric effect sizes. That pattern is expected in part because atmospheric composites incorporate alignment-aware weighting; it should be framed as **regime-dependent exposure structuring in the data pipeline**, not as standalone proof of downwind pollution impact. The inner-band NO₂ result (lower shoreward excess, *p* ≈ 0.049) cautions against simplistic “onshore wind equals more pollution” narratives and may reflect mixing, source geography, or residual definition.

**Circular evidence** must be disclosed: Maritime Exposure Index and the coastal wind-alignment fields show near-perfect regime separation (Cliff’s δ → 1, *p* ≪ 10⁻³⁰) because they are **algebraically linked** to the regime label. These statistics validate index construction but **cannot** corroborate independent environmental response. Discussion of RQ3 should rest on ACEI, ESI, localized NO₂, oil proxy, and sparse optical indicators, while MEI-wind figures are relegated to methodological transparency.

Limitations include weekly aggregation, binary shoreward classification, overlapping distance bands, and sparse optical samples. Wind findings remain **associational**; they do not establish causal transport or port-level attribution.

### Section 7 (Conclusions — RQ3)

Regarding **RQ3 (wind regimes)**, the study concludes that **coastal wind direction significantly organizes exposure-detection patterns—particularly atmospheric coastal and environmental stress composites—while it does not significantly separate vessel density** in the Turku–Mariehamn coastal panel. The principal empirical contribution is descriptive: shoreward classification identifies weeks and cells with systematically higher ranked atmospheric and composite stress signals, with weak additional structure in localized NO₂ and oil-proxy fields. The principal negative finding is equally important: **wind regime is not a statistically supported driver of shipping intensity** at the grid-week scale analyzed here. Maritime Exposure Index results tied to wind are **methodologically circular** and are not used as independent confirmation. Overall, RQ3 is **partially answered**: wind matters for **how exposure is observed and composited** along the corridor, not for **how much maritime traffic occurs**.

---

## Appendix: Source file paths

```
outputs/reports/coastal_exposure_statistics.csv
outputs/reports/indicator_participation_statistics.csv
outputs/reports/vessel_density_wind_regime_tests.csv
outputs/reports/run_land_pollution_drivers_wind/land_pollution_driver_correlation.csv
```

*Generated from existing pipeline outputs; no ML; no new data collection.*
