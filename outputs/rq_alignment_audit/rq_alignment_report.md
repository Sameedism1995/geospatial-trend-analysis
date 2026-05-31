# Research Question Alignment Audit

**Thesis:** ML-Based Geospatial Analysis of Coastal Water Quality and Urban Exposure  
**Source of research questions:** Chapter 1, Section 1.4 — *Thesis_ Final Draft.docx*  
**Audit date:** 2026-05-30 (revised against Final Draft)  
**Scope:** Existing repository artefacts only; no new data collection suggested.

---

## Executive summary

| Verdict | Count | Research questions |
| --- | ---: | --- |
| **Partially Answered** | 7 | RQ1–RQ7 |
| **Weakly Answered** | 0 | — (RQ3 remains the weakest within “Partially”) |
| **Fully Answered** | 0 | — |
| **Not Answered** | 0 | — |

**Can each research question be defended using the current evidence?**

- **Yes, with explicit caveats** for RQ1, RQ4, RQ6, and **RQ7** — the Final Draft correctly scopes comparison to **Turku vs Mariehamn** and **excludes Stockholm** (§1.5.4, §3.2.1), which aligns with repository coverage.
- **RQ2 and RQ5** are defensible only for **composite/atmospheric** associations; maritime ↔ optical water-quality links remain weak or undefined at port-specific temporal scale.
- **RQ3** is the thinnest: the draft itself reports **Mann–Whitney U = 152,763, *p* = 0.815** for vessel density vs wind regime (§5.4), correctly framing vessel–wind differences as seasonal confound. Directional claims rest mainly on **MEI/CES stratification** (wind is an MEI input), not independent NO₂ transport proof.
- **No RQ is “fully answered”** in a strong inferential sense; all remain associational, single-year, and missingness-constrained.

**Strongest evidence overall:** port-centred **distance-decay** for MEI/vessel density (§5.3), **Turku–Mariehamn ranking** (§5.3, §5.5), **pooled vessel↔ESI ρ = 0.25** (*n* = 102), **NO₂ persistence** (lag-1 ρ ≈ 0.41), and **anomaly co-occurrence** tables (§5.7).

This audit maps repository artefacts to the **Final Draft** chapter structure. Repository figure numbers (e.g. repo Fig. 5.6 decay, 5.7 wind) differ from some Final Draft placements (draft §5.3 cites “Figure 5.7” for shoreline gradients; §5.4 uses 5.7a/5.7b for wind).

---

## 1. Research questions (verbatim from Final Draft §1.4)

> **RQ1.** How do environmental indicators vary spatially and temporally across Baltic coastal port regions?
>
> **RQ2.** What relationships can be identified between maritime activity intensity and environmental exposure indicators?
>
> **RQ3.** How do wind regimes and directional transport dynamics influence coastal environmental exposure patterns?
>
> **RQ4.** How does the intensity of exposure to the environment alter with increasing distance from ports and shipping routes?
>
> **RQ5.** Are there temporal lag correlations between environmental indicators, marine activity and atmospheric exposure that can be measured
>
> **RQ6.** Can spatiotemporal anomaly detection approaches detect anomalous environmental circumstances and clustering behaviour in coastal maritime systems?
>
> **RQ7.** What are the differences in maritime exposure structures and environmental patterns between the port regions of Turku and Mariehamn within the Turku–Mariehamn–Åland corridor?

**Scope note (Final Draft, not RQ text):** Stockholm is **excluded** from final analysis (§1.3.4, §1.5.4, §3.2.1). The corridor is **Turku–Mariehamn–Åland** only.

---

## 2. Evidence matrix

| Research Question | Supporting Analysis | Final Draft § / Figures | Repository Artefacts | Evidence Strength | Main Limitation |
| --- | --- | --- | --- | --- | --- |
| **RQ1** | Panel audit; missingness; distributional & weekly temporal summaries; correlation overview | **§5.1**, **§5.2**; Fig. 5.1–5.3; Table 5.1 | `chapter_5_1_*.md/csv`, `spearman_correlation.csv`, repo Fig. 5.1 map | **Moderate** | Optical ~18–19% valid; **NDVI ~2.3%**; temporal plots reflect observable weeks only |
| **RQ2** | Cross-domain correlations; MEI/ESI/CES; pooled & port-disaggregated Spearman; oil-proxy coastal tests | **§5.2**, **§5.3**, **§5.5**; Fig. 5.1, 5.8 | `final_temporal_relationships.csv`, `port_exposure_ranking.csv`, `cross_category_relationships.csv` | **Moderate** (composites) / **Weak** (optical) | Basin vessel↔NO₂ ρ≈0.096; port temporal vessel↔ESI **NaN**; ML test R² negative (§5.9–5.10) |
| **RQ3** | Shoreward/non-shoreward classification; Mann–Whitney on composites & vessel density; participation by regime | **§5.4**; Fig. 5.7a, 5.7b | `vessel_density_wind_regime_tests.csv`, `coastal_exposure_statistics.csv`, `wind_regime_statistics.csv` | **Weak–Moderate** | Vessel↔wind **p=0.815** (draft acknowledges); MEI wind effect partly ** definitional**; Mariehamn 0 shoreward rows in coastal-panel vessel test |
| **RQ4** | Port annuli; shoreline-distance bands; high-vessel-cell distance; decay curves Turku/Mariehamn | **§5.3**; Fig. 5.7 (shoreline gradients in draft) | `port_distance_decay_statistics_turku_mariehamn.csv`, `final_distance_decay_*.png`, repo Fig. 5.6 | **Moderate** (MEI/NO₂/vessel) / **Weak** (optical) | Inner-band optical *n* tiny; draft claim **30–50 km > 10–30 km** for MEI needs careful defence (see §4) |
| **RQ5** | Lag *t*, *t*−1, *t*−2; roll2/3; persistence; weekly medians; autocorrelation heatmap | **§5.6** | `final_temporal_relationships.csv`, `final_temporal_validation.md`, repo Fig. 5.9a–b | **Moderate** (NO₂, NO₂↔ESI) / **Weak** (maritime↔water quality, NDVI) | NO₂↔NDVI **n=2–4 → NaN**; vessel↔NDTI ρ≈0.08; flat vessel_density within port windows |
| **RQ6** | Rolling-baseline flags; co-occurrence matrix; timeline; hotspot maps | **§5.7** | `final_anomaly_overlap_table.csv`, repo Fig. 5.10a–b | **Moderate** (detection) / **Weak** (clustering inference) | “Clustering” is visual/descriptive; MEI/vessel flagged all 51 weeks (trivial persistence) |
| **RQ7** | Port ranking; cross-port decay; ESI subregion means; shared-valid annulus; synthesis dashboard | **§5.3**, **§5.5**, **§5.6**, **§5.8**; Fig. 5.8b hotspots | `port_exposure_ranking.csv`, `port_pairwise_mannwhitney.csv`, thesis bundle Fig1–Fig7 | **Moderate** | ~150 km port separation confounds attribution (draft §6 acknowledges); MEI–ESI sign reversal Turku (−0.08) vs Mariehamn (+0.26) |

---

## 3. Per-question assessment

### RQ1 — Spatial and temporal variation

**Answer status: Partially Answered**

| Final Draft | Repository |
| --- | --- |
| §5.1: 16,065 obs, 315 cells, 51 weeks | `chapter_5_1_dataset_core_statistics.csv` ✓ |
| Table 5.1 coverage (NO₂ 89.9%, optical 18–19%, NDVI 2.3%) | `chapter_5_1_missingness_all_columns.csv` ✓ |
| Fig. 5.2 distributions, Fig. 5.3 weekly variability | Repo: `figure_5_5_statistical_boxplots.png`, `fig_5_9a_temporal_weekly_medians.png` |
| §5.1: vessel density localized to corridors | `chapter_5_1_baltic_vessel_density_maritime_exposure.png` ✓ |

**Strongest evidence:** Documented panel skeleton and high-coverage maritime/atmospheric spatial heterogeneity.

**Gap:** Full-panel temporal characterisation of NDTI/NDWI/NDVI is not possible; §5.1 correctly states structural missingness but §5.3/6 sometimes discuss optical lags with thin *n*.

**Additional analysis (existing data):** Port-stratified weekly median plots with per-week *n* badges; effective sample size table by indicator×month.

---

### RQ2 — Maritime activity ↔ exposure indicators

**Answer status: Partially Answered**

| Final Draft claim | Repository support |
| --- | --- |
| Pooled vessel↔ESI ρ = 0.25 (*n* = 102) §5.5 | `final_temporal_relationships.csv` row 17: ρ = 0.2507, *n* = 102 ✓ |
| Turku MEI–ESI ρ = −0.08; Mariehamn ρ = 0.26 §5.5 | Spatial/port-disaggregated — consistent with audit; **not** in temporal CSV port columns |
| Vessel-density lag persistence ρ > 0.99 §5.1 | `spearman_correlation.csv` — lag columns if present in panel ✓ |
| ML identifies relationships §1.1.4 | **Contradicted** for predictive use: rolling CV test R² ≈ −0.09 (ΔNDTI), vessel importance ≈ 0.0025 |

**Strongest evidence:** Composite exposure ranking (Turku MEI 0.57 vs Mariehamn 0.32); oil slick proxy in high-exposure zones (*p* ≈ 3.5×10⁻¹³, coastal impact analysis).

**Gap:** Direct maritime ↔ optical water quality near null; draft §5.9–5.10 correctly states negative R² — do not use ML as evidence for RQ2 substantive claims.

**Additional analysis:** Export spatial Spearman with *p* for vessel↔MEI/ESI by port annulus; partial correlation controlling for `week_of_year`.

---

### RQ3 — Wind regimes and directional transport

**Answer status: Partially Answered** *(weakest RQ)*

| Final Draft | Repository |
| --- | --- |
| §5.4: vessel higher in non-shoreward; seasonal confound | Turku coastal raw: non-shoreward mean 0.571 vs shoreward 0.540 — directionally consistent but **not significant** after season adjustment |
| **Mann–Whitney U = 152,763, *p* = 0.815** §5.4 | `vessel_density_wind_regime_tests.csv` row 1 ✓ **exact match** |
| §5.4: shoreward MEI/CES higher inland | `coastal_exposure_statistics.csv`: MEI shoreward > non in bands (*p* tiny) — partly **circular** (wind in MEI) |
| Fig. 5.7a/5.7b directional structuring | Repo `fig_5_7a_*`, `fig_5_7b_*` |

**Strongest defensible claim:** Wind regime **partitions composite exposure ordinally** by band; vessel activity **does not** differ by wind class once season structure is acknowledged.

**Overclaim risk in draft:** “Wind direction contributed to spatial diversity” (§5.4 closing) — soften to **associative stratification of wind-informed indices**, not validated pollutant transport.

**Additional analysis:** ESI/NO₂ (not MEI) shoreward contrasts with month fixed effects; report Mariehamn wind coverage gap explicitly in §5.4.

---

### RQ4 — Distance decay from ports and shipping lanes

**Answer status: Partially Answered**

| Final Draft | Repository |
| --- | --- |
| §5.3: Turku MEI ~0.50 (3–10 km) → ~0.03 (10–50 km) | Directionally supported in decay tables; exact bands differ (repo uses 0–3, 3–7, 7–15, 15–30 km annuli) |
| §5.3: vessel density >13 near-port, <1 distant | Consistent with `port_distance_decay_statistics_turku_mariehamn.csv` order-of-magnitude |
| §5.3 Fig. 5.7: **30–50 km MEI/CES/ESI > 10–30 km** | **Tension:** classic port-centred decay shows **highest MEI near port**, decreasing outward in `coastal_exposure_statistics.csv`. Shoreline-distance lattice (draft Fig. 5.7) uses a **different axis** (shoreline km, not port annulus) — defend only if methodology in §4 clearly separates the two |
| Shipping-lane distance features §1.3.4 | `indicator_participation` distance-to-high-vessel-cell bands |

**Strongest evidence:** Turku port annulus decay for MEI, vessel density, NO₂ (§5.3 narrative matches repo decay figures).

**Gap:** Optical indicators sparse in inner bands; dual distance frameworks (port vs shoreline vs high-vessel cell) easy to conflate in prose.

**Additional analysis:** Single figure/table reconciling port annulus vs shoreline-distance metrics side-by-side with *n* per bin.

---

### RQ5 — Temporal lag correlations

**Answer status: Partially Answered**

| Final Draft | Repository |
| --- | --- |
| §5.6: NO₂ lag-1 autocorrelation ρ = 0.41 | `final_temporal_relationships.csv` persistence block: NO₂ lag1↔NO₂ *t* ρ = 0.4128 ✓ |
| §5.6: NDWI lag ρ = −0.23; NDTI lag-2 ρ ≈ 0.16 | Verify against `lagged_correlations.csv` / thesis lag exports — plausible on optical subset |
| §5.6: NDWI↔NDTI contemporaneous ρ = −0.62 | `spearman_correlation.csv`: ndwi↔ndti ρ = −0.715 (stronger on full pairwise complete sample) |
| Maritime activity lags | `final_temporal_relationships.csv`: vessel↔NDTI ρ ≈ 0.08 (*n* = 31); vessel↔ESI pooled only |

**Strongest evidence:** NO₂ persistence and NO₂↔ESI (ρ ≈ 0.57 contemporaneous, roll2 ρ ≈ 0.43).

**Gap:** NDVI lag leg undefined (NO₂↔NDVI *n* = 2–4); draft §3.x NDVI seasonal confound (N = 369 valid) is appropriately cautious — keep NDVI out of main RQ5 answer.

**Additional analysis:** Export *p*-values from `final_thesis_spatiotemporal.py`; cell-level lag CCF for NO₂ and ESI.

---

### RQ6 — Anomaly detection and clustering

**Answer status: Partially Answered**

| Final Draft | Repository |
| --- | --- |
| §5.7: vessel/MEI anomalies all 51 weeks | Expected if vessel/MEI baselines flag persistent corridor activity — **not** episodic “anomaly” in strict sense |
| §5.7: NO₂ anomalies rare (4 spikes) | Consistent with low co-occurrence in overlap table |
| §5.7: NDTI/FAI episodic | `final_anomaly_overlap_table.csv` |
| “Clustering behaviour” | Co-occurrence heatmap + timeline only — **no** formal cluster test |

**Strongest evidence:** Descriptive co-occurrence structure and episodic multi-indicator weeks (e.g. Turku 2023-02-26: ndti;no2;stress;vessel).

**Gap:** Calling persistent vessel/MEI flags “anomalies” overstates rarity; clustering claim needs qualification.

**Additional analysis:** Permutation test on co-occurrence rates; run-length / recurrence statistics per port.

---

### RQ7 — Turku vs Mariehamn differences (corridor)

**Answer status: Partially Answered** *(upgraded vs prior audit — scope now matches evidence)*

| Final Draft | Repository |
| --- | --- |
| §5.3/5.5: Turku highest MEI, ESI, vessel intensity | `port_exposure_ranking.csv`: Turku rank 1 all composites ✓ |
| §5.5: Turku ESI mean ≈ 0.28 vs Mariehamn ≈ −0.10 | Verify scale — repo ESI is percentile-like ~0.53 mean on coastal subset; **check subregion filtering** if defending exact means |
| §5.5/§6: divergent MEI–ESI signs by port | Qualitatively supported; critical for honest interpretation |
| §6.8: ~150 km separation, no source apportionment | Methodologically appropriate caveat ✓ |
| Stockholm excluded | Matches empty Stockholm rows in repo — **no longer an RQ7 gap** |

**Strongest evidence:** Mann–Whitney port pairs in outer bands (*p* ~ 10⁻²³⁰ for MEI); integrated synthesis dashboard §5.8.

**Gap:** Corridor-level pooling masks port-level sign reversal; spatial proximity prevents port-specific attribution.

**Additional analysis:** Shared-valid-annulus table already in repo — ensure all cross-port figures cite same annulus selection.

---

## 4. Final Draft claims vs repository evidence

| Topic | Final Draft statement | Audit verdict |
| --- | --- | --- |
| **Stockholm** | Excluded §1.5.4, §3.2.1, §6.8 | **Aligned** with repo; prior three-port RQ7 mismatch **resolved** in Final Draft |
| **Vessel vs wind** | *p* = 0.815, seasonal confound §5.4 | **Aligned** — `vessel_density_wind_regime_tests.csv` |
| **Vessel vs ESI** | ρ = 0.25, *n* = 102 §5.5 | **Aligned** — `final_temporal_relationships.csv` |
| **MEI–ESI by port** | Turku −0.08, Mariehamn +0.26 | **Plausible** (spatial disaggregation); not in temporal CSV |
| **ML performance** | Negative R², exploratory only §5.9–5.10 | **Aligned** — `rolling_window_average_metrics.csv` |
| **NDVI** | N = 369, seasonal bias, preliminary §3.x | **Aligned** with ~2.3% panel missingness; do not use for main claims |
| **Wind drives exposure diversity** | §5.4 closing sentence | **Overstated** relative to *p* = 0.815 on vessel and circular MEI input |
| **Shoreline 30–50 km > 10–30 km** | §5.3 Fig. 5.7 narrative | **Needs methodology footnote** — different distance construct than port decay; verify before defence |
| **MEI anomalies all 51 weeks** | §5.7 | **Technically true** but **weak anomaly evidence** — persistent shipping corridor, not rare events |

---

## 5. Chapter 5 crosswalk (Final Draft ↔ repository)

| Final Draft § | Primary RQs | Key repository outputs |
| --- | --- | --- |
| **5.1** Dataset Characteristics | RQ1 | `outputs/thesis/chapter_5_1_*` |
| **5.2** Statistical Characteristics | RQ1, RQ2 | `spearman_correlation.csv`, `cross_category_relationships.csv` |
| **5.3** Maritime Exposure & Distance-Decay | RQ2, RQ4, RQ7 | `port_distance_decay_statistics_turku_mariehamn.csv`, `final_distance_decay_*.png` |
| **5.4** Wind-Regime Results | RQ3 | `vessel_density_wind_regime_tests.csv`, `fig_5_7a/b` |
| **5.5** Environmental Stress & Ecological | RQ2, RQ7 | `fig_5_8a/b`, `final_temporal_relationships.csv` |
| **5.6** Temporal Lag & Persistence | RQ5 | `final_temporal_relationships.csv`, `fig_5_9a/b` |
| **5.7** Anomaly Detection | RQ6 | `final_anomaly_overlap_table.csv`, `fig_5_10a/b` |
| **5.8** Integrated Synthesis | RQ1, RQ7 | `composite_land_exposure_dashboard`, thesis bundle maps |
| **5.9–5.10** ML Performance | Method obj. 1.3.1 (not an RQ) | `outputs/ml_cv_results/` |

---

## 6. Cross-cutting topics

### Distance-to-port / shoreline gradients
- **Port annuli (0–30 km):** Strong composite gradients Turku; supported.
- **Shoreline-distance lattice (draft Fig. 5.7 in §5.3):** Separate construct — report both with distinct labels.
- **NO₂ at large distance:** `no2_distance_diagnostic.md` — regional land-emission confounding.

### MEI / ESI
- MEI: 1,173 coastal rows, mean ≈ 0.73 (`exposure_indices_summary.csv`).
- ESI: experimental composite; NO₂ link strongest temporally (ρ ≈ 0.57).
- Wind alignment embedded in MEI — limits independent wind hypothesis tests.

### Correlation analysis
- Dominated by optical inter-correlations (NDWI↔NDCI ρ ≈ −0.876) — RQ1 structure, not shipping.
- Cross-domain maritime links weak except composites and oil proxy.

### Machine learning
- Rolling CV: ΔNDTI test R² ≈ −0.09 (HistGBR); ndti_next R² ≈ −2.0.
- Draft correctly relegates ML to pattern analysis (§5.9) — **do not cite ML as RQ2/RQ5 evidence**.

### Temporal validation
- `final_temporal_validation.md`: 51 weeks/port; flat-input warning for vessel_density.
- Port stability scores often low for maritime pairs with sign disagreement.

### NDVI missingness
- 97.7% panel missing; draft designates preliminary (§3.x).
- Temporal NO₂↔NDVI: *n* = 2–4 → NaN in `final_temporal_relationships.csv`.
- Nearest-land linkage: 375 maritime→land pairs — exploratory only.

---

## 7. Strongest evidence by research question

| RQ | Strongest defensible claim | Primary artefact |
| --- | --- | --- |
| **RQ1** | Indicators vary spatially (corridor-localized shipping; heterogeneous NO₂/oil) and temporally for high-coverage layers; optical/NDVI observation-limited | §5.1 Table 5.1, Fig. 5.3 |
| **RQ2** | **Associative** links between maritime intensity and **ESI/MEI/oil proxy**; weak links to optical water quality | §5.5 ρ=0.25; `port_exposure_ranking.csv` |
| **RQ3** | Wind regimes **structure composite exposure**; **do not** claim vessel activity differs by wind (*p*=0.815) | §5.4 + `vessel_density_wind_regime_tests.csv` |
| **RQ4** | MEI/vessel/NO₂ **decrease** with port distance (Turku strongest); shipping-lane proximity features show band gradients | §5.3, decay CSV |
| **RQ5** | **Measurable lags** for NO₂ persistence and NO₂–ESI; maritime→NDTI/NDVI lags **not** robust | §5.6, `final_temporal_relationships.csv` |
| **RQ6** | Anomaly **flags and co-occurrence** produced; clustering **descriptive only** | §5.7, overlap table |
| **RQ7** | **Turku > Mariehamn** on exposure composites; divergent local MEI–ESI structure; corridor confounding acknowledged | §5.3, §5.5, §6.8 |

---

## 8. Defence readiness (Final Draft aligned)

| RQ | Defensible? | Recommended posture in viva/examination |
| --- | --- | --- |
| RQ1 | **Yes, with missingness caveats** | Lead with Table 5.1; separate panel regularity from sensor observability |
| RQ2 | **Partially** | Emphasize composites + oil proxy; disclaim optical and ML predictive claims |
| RQ3 | **Partially** | Lead with *p*=0.815 for vessel; frame wind results as index stratification |
| RQ4 | **Partially** | Port annulus decay strong; clarify shoreline vs port distance axes |
| RQ5 | **Partially** | NO₂/ESI lags yes; maritime→water-quality/NDVI no |
| RQ6 | **Partially** | “Detection yes; inferential clustering no”; note MEI/vessel persistent flags |
| RQ7 | **Yes, corridor-scoped** | Turku–Mariehamn contrast robust; cite §6.8 proximity limitation |

---

## 9. Source document note

Research questions extracted from **`/Users/sameedahmed/Downloads/Thesis_ Final Draft.docx`**, Section **1.4** (bullet list under “The study is guided by the following research questions”).

**Change from earlier audit:** The revised PDF (`Thesis_Chapters_1_3_Revised.pdf`) listed a **three-port RQ7 including Stockholm**. The **Final Draft** narrows RQ7 to **Turku vs Mariehamn** and excludes Stockholm throughout — this **improves alignment** with repository evidence.

Evidence paths refer to artefacts under `outputs/` and `processed/` in `/Users/sameedahmed/geospatial-trend-analysis/`.

---

*End of audit.*
