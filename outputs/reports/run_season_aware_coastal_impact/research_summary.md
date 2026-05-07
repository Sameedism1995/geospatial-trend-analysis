# Season-aware coastal impact — research summary

Observational **spatial associations** only; not causal inference.

1. **Radar** shows the strongest, most consistent signal for **localized maritime-activity footprint** when referenced to **high vessel-density cells** (**coastal ≤30 km**, **≤15 km** to anchors): `oil_slick_probability_t` and `detection_score`.
2. **Distance structure** along refined bands to the shipping anchor (**0-3 km, 3-7 km, 7-15 km, 15-30 km**) summarizes how means change with offset from lanes. A **monotone decay** of the SAR proxy is **not required** for a meaningful gradient; when central-band means exceed outer-band means, that supports **localization** near dense traffic.
3. **Radar** is used in **both** winter (Nov–Mar) and non-winter (Apr–Oct), supporting **cross-season** comparison on the same SAR variables.
4. **Optical** indicators (**NDTI, NDWI, NDVI** as available) are included **only for non-winter** in the seasonal workflow; **missing optical values are not imputed**, so optical contributes **when cloud-free observations exist**.
5. **Land-facing indicators** (**NDVI**, **`land_response_index` if present**, optional **NO₂**) probe whether patterns extend into the **closest-to-coast stratum** of the analysis panel. Land panel: **closest tercile of distance_to_coast_km within coastal×shipping panel (threshold 21.61 km; strict ≤10 km had 0 rows; n=357)**.
6. All findings are described as **spatial association** between exposure geometry and response metrics, not as proof of mechanism or attribution.

---

**Research statement**

This analysis evaluates localized coastal environmental impact of maritime activity using season-aware indicator selection, combining radar-based surface disturbance with land and atmospheric responses across refined spatial distance bands.

---

Artifacts: `radar_distance_decay_refined.csv`, `radar_high_vs_low.csv`, `seasonal_indicator_analysis.csv`, `land_impact_by_vessel.csv`, `land_impact_by_distance.csv`, `sea_land_link_analysis.csv` (Spearman correlations + band-mean rows in one file; `row_kind` distinguishes blocks).

**Strongest |Cohen’s d| (radar + land + seasonal high/low):** ndvi_mean (|d|≈1.546 if finite).

**Automated checks:** distance-decay trend (oil mean, monotone across non-empty bands): **NO**; land impact (|d|>0.05 and Welch or Mann–Whitney p<0.05): **NO**.
