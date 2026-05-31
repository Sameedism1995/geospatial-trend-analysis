# Examiner Attack Points (Florian / Viva Mode)

**Role:** Adversarial examiner stress-test of `Thesis_ Final Draft.docx` against repository evidence.  
**Rule:** Answers may use **only existing outputs**; admit weakness where evidence is thin.

**Legend:** 🟢 Strong answer available | 🟡 Partial answer—needs careful wording | 🔴 Weak answer—acknowledge gap, do not over-defend

---

## Top 20 likely examiner questions

| # | Question | Vulnerable section | Answer (existing evidence only) | Strength |
|---|----------|-------------------|----------------------------------|----------|
| 1 | **You claim ML creates actionable insights—your test R² is negative. Which conclusions survive ML failure?** | Abstract; §7.2; §5.9 | None of the **substantive environmental** conclusions depend on ML. Associations come from Spearman, Mann–Whitney, decay tables. §5.9 already states negligible predictive capacity—move that to Abstract/Conclusion. | 🟢 |
| 2 | **Are your results causal?** | §6.6 (good) vs §6.1/Abstract (bad) | No. Observational panel; §6.6 association-only. All maritime→environment language must be correlational. | 🟢 if unified |
| 3 | **RQ3: Does wind drive shipping? Your title mentions transport dynamics.** | §5.4; §6.2 | **No significant difference** in vessel density by regime (Mann–Whitney p = 0.815, `vessel_density_wind_regime_tests.csv`). Draft §5.4 already notes seasonality confound—examiner wants this as **primary** RQ3 result. | 🟢 |
| 4 | **MEI includes wind alignment. Isn’t wind–MEI significance tautological?** | §5.4; Fig 5.7; §6.3 | Yes for **independent validation**. MEI contrasts prove index construction, not external response. Use ACEI/ESI + disclose circularity (`outputs/rq3_strengthening/`). | 🟢 |
| 5 | **RQ4: You argue distance-decay, but NO₂ increases with port distance (ρ ≈ +0.08). Explain.** | §5.3; §6.1; §7.2 | `rq4_decay_direction_summary.csv`: vessel ρ ≈ −0.26, CES ρ ≈ −0.76, but NO₂/MEI/ESI/NDTI **increase** with port distance—likely regional geography and axis definition, not uniform attenuation. | 🟡 |
| 6 | **What is the difference between port-distance and shoreline-distance decay?** | §5.3; Fig 5.6/5.7 | Two axes in pipeline (`distance_to_port_km` vs shoreline bands). Thesis conflates them in prose. Cite separate outputs: `rq4_decay_direction_summary.csv` vs `port_distance_decay_statistics_turku_mariehamn.csv`. | 🟡 |
| 7 | **NDVI coverage is ~2.3%. Why is NDVI in the ML feature table?** | Table 4.2; §5.1 | Structural sparsity (maritime grid). Keep NDVI in **methods/limitations** only; remove from conclusions and RQ5. | 🟢 |
| 8 | **RQ6 asks about clustering. Where is the clustering test?** | §5.7; §5.5 hotspots; RQ6 Ch.1 | **No inferential clustering test** in repository (`anomaly_detection.py` = rules + z-scores; `hotspot_cells.geojson` = descriptive). Admit RQ6 clustering leg **not answered**; episodic co-occurrence only. | 🔴 → honest partial |
| 9 | **Vessel density is anomalous 51/51 weeks. How is that an anomaly?** | §5.7 | It is **not**—it is persistent corridor activity (`rq6_anomaly_counts.csv`). Relabel in thesis; reserve “anomaly” for NO₂/NDTI/FAI episodic flags. | 🟢 |
| 10 | **Turku vs Mariehamn: can you attribute pollution to Turku port?** | §6.4 (good) vs §7.2 | **No source apportionment.** ~150 km separation, shared plumes. Evidence: comparative means and Mann–Whitney, not attribution. §6.4 text should appear in Conclusion. | 🟢 |
| 11 | **Why exclude Stockholm from RQ7 if it was in earlier pipeline runs?** | §3.2.1; §6.8 | Coverage insufficient in Final Draft scope; empty Stockholm rows in wind stats. Consistent with fixed RQ7 wording (Turku–Mariehamn only). | 🟢 |
| 12 | **Vessel–ESI ρ = 0.25 pooled, but Turku MEI–ESI is negative. Which story is true?** | §5.5; §6.4 | **Both:** pooled corridor association ≠ port-homogeneous mechanism. Heterogeneity is a **finding**, not noise to hide. | 🟡 |
| 13 | **Optical indicators are 81% missing. How can you discuss water quality?** | §5.1–5.2 | Only where valid—~18% cell-weeks. NDTI/NDWI episodic anomalies (3–4 weeks) are **illustrative**, not population inference. | 🟡 |
| 14 | **What is the effective n for lag analysis?** | §5.6 | NO₂ autocorr n ≈ 13,957 (strong). Vessel–NDTI n ≈ 30–31 pooled (`final_temporal_relationships.csv`); port splits often flat/NaN. | 🟢 |
| 15 | **Oil slick proxy: validation flags?** | §6.8; `validate_aux_layers.py` | Proxy only; possible `no_spatial_signal` / sparsity flags in validation JSON. Do not treat as confirmed spills. | 🟢 |
| 16 | **Why should we trust composite indices (MEI, ESI, ACEI)?** | §6.3 | They are **transparent engineered scores** for comparative mapping—not validated stress metrics. Useful descriptively; not ecological ground truth. | 🟡 |
| 17 | **Weekly aggregation—does it destroy the lag signal you claim?** | §5.6; `causal_lag_summary.csv` | Grid-level vessel→NO₂ best |r| ≈ 0.04 suggests weekly aggregation + noise limits cross-lag detection; NO₂ **autocorrelation** still robust. | 🟢 |
| 18 | **Figure 5.7: 30–50 km band higher than 10–30 km—does that prove inland propagation?** | §5.3; §5.4 | **Weak inference.** Band support thresholds and regional geography confound. Present as exploratory spatial pattern, not transport proof. | 🔴 |
| 19 | **What is the single strongest finding in the thesis?** | — | Turku vs Mariehamn vessel contrast (p ≈ 2.5×10⁻⁶⁸) plus CES/port-distance gradients (ρ ≈ −0.76). Lead with these in viva. | 🟢 |
| 20 | **Which RQs would you downgrade if forced to be honest?** | §7 | **Fully answered: none.** Weakest: RQ6 clustering; RQ3 traffic modulation; RQ2 optical/ML legs. Strongest partial: RQ7, RQ4 (vessel/CES), RQ1 (NO₂/vessel). | 🟢 |

---

## Section vulnerability map

| Section | Risk level | Primary issue |
|---------|------------|---------------|
| **Abstract** | 🔴 High | ML + actionable claims; wind→NO₂ oversell |
| **§1.3 Objectives** | 🟡 Medium | Clustering/predictive tone vs results |
| **§5.3 Distance-decay** | 🔴 High | NO₂ “decline” vs positive ρ; inland MEI band |
| **§5.4 Wind** | 🟡 Medium | Good null test buried; MEI circularity under-disclosed |
| **§5.7 Anomalies** | 🔴 High | Persistent mislabeled as anomaly; clustering implied |
| **§5.9–5.10 ML** | 🟢 Low | Honest negative R²—problem is **other chapters ignore it** |
| **§6.1 Discussion decay** | 🔴 High | Overstates uniform decay |
| **§6.2 Wind** | 🟡 Medium | Causal “contributed to” language |
| **§6.3 MEI “successfully identified”** | 🟡 Medium | Sounds validated; is engineered |
| **§6.4 Port comparison** | 🟢 Low | Strong limitations—**underused in Conclusion** |
| **§7.2 Conclusion** | 🔴 High | Repeats unsupported decay, ML, anomaly claims |

---

## Florian-specific concern mapping

| Concern (inferred) | Thesis symptom | Fix (minimum) |
|--------------------|----------------|---------------|
| Conclusions exceed evidence | Abstract vs §5.9 contradiction | Adopt `conclusion_rebuild.md` |
| Wind/transport over-interpreted | MEI + “inland propagation” | RQ3 reframing doc already written |
| Distance-decay oversimplified | One narrative for all indicators | Split indicator table in §5.3/§7.2 |
| Anomaly misuse | 51/51 vessel weeks | Rename + RQ6 partial answer |
| ML oversold | “Actionable” in abstract | Exploratory framework label |

---

## Viva answer template (when caught on weak point)

1. **State the exact statistic** (ρ, p, n, file path).  
2. **State what it does *not* show** (causality, prediction, attribution).  
3. **State RQ status** (“partially answered because…”).  
4. **Point to limitation already in thesis** (or admit if missing and commit to revision).

**Example (RQ6 clustering):**  
“RQ6 is only partially answered. We implemented rule-based exceedance flags (`outputs/rq_evidence_fix/rq6_anomaly_counts.csv`) and descriptive hotspot maps. We did **not** run Knox or spatial scan statistics. I would remove ‘clustering behaviour’ from the conclusion and describe episodic co-occurrence only.”

---

## Questions with currently weak answers (fix before submission)

1. RQ6 clustering validation — **🔴**  
2. Uniform environmental distance-decay — **🔴**  
3. Wind-driven inland MEI propagation (30–50 km band) — **🔴**  
4. ML contribution to scientific conclusions — **🔴** (if Abstract unchanged)  
5. Maritime→water-quality lags — **🟡**  
6. Optical water-quality generalization — **🟡**  
7. MEI as validated stress metric — **🟡**

---

## Recommended viva “lead with” slide (3 bullets)

1. **Turku–Mariehamn contrast** — vessel means 1.04 vs 0.21, p ≈ 2.5×10⁻⁶⁸.  
2. **Spatial exposure geography** — CES vs port distance ρ ≈ −0.76; vs high-traffic cell ρ ≈ −0.92.  
3. **Honest limits** — no predictive ML; wind does not sort vessel density (p = 0.815); anomalies ≠ persistent shipping.

---

*Generated for pre-submission revision. Cross-reference: `thesis_strengthening_report.md`, `rq_claim_audit.csv`, `discussion_focus_ranking.csv`, `conclusion_rebuild.md`.*
