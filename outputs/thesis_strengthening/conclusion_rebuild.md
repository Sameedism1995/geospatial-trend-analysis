# Rebuilt Thesis Conclusion (Evidence-Surviving Claims Only)

This document replaces **Chapter 7** substantive claims. It uses only findings that pass scrutiny in `thesis_strengthening_report.md`, `outputs/rq_evidence_fix/rq_evidence_report.md`, and `outputs/rq3_strengthening/rq3_strengthening_report.md`. **No causal language. No predictive ML claims. No NDVI conclusions. No unsupported distance-decay or anomaly claims.**

---

## 7.1 Summary of the study

This thesis developed an **exploratory geospatial exposure framework** for the Turku–Mariehamn–Åland corridor using a balanced weekly panel of approximately 16,000 grid-cell observations (315 cells × 51 weeks) built from Sentinel-2, Sentinel-5P, EMODnet vessel density, wind metadata, and auxiliary geospatial layers. The study area intentionally **excludes Stockholm** because of insufficient spatial coverage in the archived datasets.

The analytical goal was to **describe and compare** spatial and temporal patterns of maritime activity, atmospheric observations, and composite exposure indices—not to establish causal mechanisms or operational forecasting models. Machine learning models (Ridge Regression, Histogram Gradient Boosting) were implemented for methodological completeness but **did not generalize** under temporally separated evaluation (negative test R²); they are **not** used below as evidence for environmental relationships.

**Answer status summary**

| RQ | Topic | Status |
|----|--------|--------|
| RQ1 | Spatial/temporal variation | **Partially answered** |
| RQ2 | Maritime vs exposure associations | **Partially answered** |
| RQ3 | Wind regimes | **Partially answered** |
| RQ4 | Distance from ports/routes | **Partially answered** |
| RQ5 | Temporal lags | **Partially answered** |
| RQ6 | Anomalies / clustering | **Partially answered** (clustering inference **not** established) |
| RQ7 | Turku vs Mariehamn | **Partially answered** (comparative only) |

**None of the seven research questions is fully answered** under strict evidential standards; the thesis contributes an **associational, coverage-aware** characterization of a single-year corridor panel.

---

## 7.2 Summary of key findings (by research question)

### RQ1 — Spatial and temporal variation

Environmental and maritime indicators **do not share equal observability**. NO₂ (~90% coverage) and vessel density (~76%) support robust spatial and weekly variation analysis along Baltic shipping corridors. Optical water indices (NDTI, NDWI, FAI) remain **sparse** (~18% valid cell-weeks after masking). **NDVI (~2.3% valid) is excluded** from substantive conclusions because maritime-dominated grid cells and cloud masking make vegetation inference non-representative.

### RQ2 — Maritime activity and exposure indicators

**Strong associative evidence** links maritime proximity to composite exposure:

- Distance to nearest high vessel-density cell vs coastal exposure score: Spearman ρ ≈ **−0.92** (n = 16,065).
- Distance to port vs coastal exposure score: ρ ≈ **−0.76** (n = 16,065).
- Maritime exposure index vs atmospheric coastal exposure index: ρ ≈ **0.60** (n ≈ 1,017, coastal subset).

These are **cross-sectional and composite-weighted** associations. Direct vessel–optical water-quality correlations are **weak or negligible** at weekly scale. **Machine learning does not validate** these relationships for prediction (negative test R²).

### RQ3 — Wind regimes

**Vessel density does not differ significantly** between shoreward and non-shoreward regimes on the coastal panel (Mann–Whitney **p ≈ 0.815**; n = 1,173 cell-weeks), including after month-level demeaning. Wind regimes **do significantly stratify** non-circular exposure detection—especially **ACEI** and **ESI** (large Cliff’s δ, p ≪ 0.001 across distance bands). Localized NO₂ excess shows a **small, band-limited** contrast (0–3 km only). **Maritime Exposure Index contrasts by wind are circular** (MEI embeds wind alignment) and are not independent confirmation of environmental response.

**Interpretation:** Wind organizes **exposure detection and compositing**, not demonstrated control of shipping volume.

### RQ4 — Distance from ports and shipping routes

**Indicator-specific gradients**—not a single pollution-decay law:

- **Supported:** Vessel density **decreases** with port distance (ρ ≈ **−0.26**, n = 12,189). Coastal exposure score **decreases** strongly (ρ ≈ **−0.76**, n = 16,065).
- **Not supported as uniform decay:** NO₂ mean, MEI, ESI, and NDTI show **positive** Spearman trends with port distance in the archived decay summary—environmental proxies **do not all weaken** nearer ports on that axis.

Portwise annuli (Turku vs Mariehamn) with bootstrap confidence intervals describe **comparative** exposure geography; they do not prove pollutant attenuation causality.

### RQ5 — Temporal lags

**NO₂ persistence** is the clearest temporal signal (lag-1 autocorrelation ρ ≈ **0.48** on the full panel; ρ ≈ 0.41 on weekly corridor medians). Cross-series **maritime → water-quality lags remain weak** (e.g. vessel–NDTI pooled ρ ≈ 0.08–0.13 with small effective n; grid-level vessel–NO₂ peak |r| ≈ 0.04 in lag-robustness outputs). **NDVI lag analysis is omitted** (undefined/sparse). Findings support **short-term atmospheric memory**, not demonstrated delayed maritime forcing of optical water quality.

### RQ6 — Anomalies and clustering

Rolling-baseline rules flag:

- **Episodic** weeks for NO₂ (~9/51), NDTI (~3/51), FAI (~3/51).
- **Persistent** corridor activity for vessel density and MEI (**51/51 weeks**)—this is **baseline shipping presence**, not rare anomalies.

Multi-indicator co-occurrence is **limited** and descriptive. **Formal spatial clustering tests** (e.g. Knox, Moran, Getis) were **not** applied; hotspot maps show **upper-tail concentration** only. The **clustering component of RQ6 is not answered** inferentially.

### RQ7 — Turku vs Mariehamn

**Strongest empirical contrast in the thesis:**

- Mean vessel density: Turku ≈ **1.04** vs Mariehamn ≈ **0.21** (Mann–Whitney **p ≈ 2.5×10⁻⁶⁸**).
- Composite exposure rankings and decay curves show **higher and steeper** gradients near Turku.

**Port-specific pollution attribution is not supported** (~150 km separation, shared corridor plumes, divergent port-level MEI–ESI signs). Results describe **regional comparative structure**, not isolated accountability of either port.

---

## 7.3 Research contributions (defensible)

1. **Integrated weekly geospatial panel** harmonizing satellite, AIS-derived vessel density, NO₂, and exposure-engineered features for a defined Baltic corridor—with explicit missingness reporting.

2. **Associative exposure geography:** quantified port-centred gradients for shipping and composite scores (RQ4, RQ7) and proximity–exposure correlations (RQ2).

3. **Negative methodological result:** temporally honest ML evaluation shows **non-generalizing** models—supporting an exploratory rather than predictive thesis framing.

4. **Wind-regime transparency:** null vessel–wind test plus non-circular ACEI/ESI stratification, with explicit **MEI circularity** disclosure (RQ3).

5. **Uncertainty-aware comparative design:** shared-valid annuli, bootstrap band summaries, and port-pairwise nonparametric tests where archived.

**Practical implication (qualified):** The framework supports **monitoring-oriented exposure mapping and inter-port comparison** in data-sparse coastal zones—not autonomous forecasting or regulatory attribution without additional validation data and source apportionment.

---

## 7.4 Limitations and future work

- Single study year; seasonal confounding (especially wind vs vessel seasonality).
- Structural optical missingness; **NDVI not inferential**.
- Composite indices (MEI, ESI, ACEI, CES) are **experimental summaries**, not validated ecological or human-health metrics.
- Oil slick proxy without ground-truth spill inventory.
- Stockholm excluded by coverage; conclusions **corridor-specific**.
- ML and clustering legs **under-deliver relative to chapter titles**—future work would need longer panels, inferential spatial statistics, and source apportionment—not more complex ML on the same sparse optical record.

---

## 7.5 Closing statement

This thesis demonstrates that **maritime exposure in the Turku–Mariehamn corridor is strongly structured in space**—concentrated on routes and port annuli, contrasting sharply between Turku and Mariehamn, and **weakly or not structured by weekly wind regime in vessel traffic itself**. Atmospheric and composite indicators show **measurable persistence and wind-regime stratification**, but **not** a uniform story that all environmental proxies decay away from ports. Anomaly rules detect **episodic** atmospheric and optical weeks amid **persistent** shipping activity. The work is best classified as an **exploratory geospatial exposure assessment** in which **seven research questions receive partial, uneven answers**—strongest for comparative maritime geography (RQ7, RQ4 vessel/CES), weakest for clustering inference (RQ6), predictive ML (RQ2/RQ5), and wind-driven traffic modulation (RQ3).

---

*Replace corresponding paragraphs in `Thesis_ Final Draft.docx` §7.1–7.4; align Abstract and §6.1–6.3 with this conclusion to remove internal contradictions.*
