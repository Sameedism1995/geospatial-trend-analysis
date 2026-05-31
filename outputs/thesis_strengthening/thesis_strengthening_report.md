# Thesis Strengthening Report — Scientific Reviewer Mode

**Purpose:** Strengthen defensibility before submission; address examiner concerns (including Florian’s) that some conclusions exceed the evidence.  
**Constraints:** Fixed RQs; no new data; no additional ML; existing `processed/features_ml_ready.parquet`, `outputs/`, `scripts/` only.  
**Thesis texts reviewed:** `Thesis_ Final Draft.docx` (Chapter 1 §1.4 RQs; Chapters 5–7); cross-checked with `outputs/rq_evidence_fix/`, `outputs/rq_alignment_audit/`, `outputs/rq3_strengthening/`.

**Overall verdict:** **No research question is fully answered.** Seven of seven are **partially answered**; **RQ3** and **RQ6** are the most vulnerable to over-interpretation; **RQ7** has the strongest empirical core but must not be pushed into port-level causation.

---

## Minimum set of changes before submission (priority order)

1. **Rewrite Abstract and §7.2** — Remove ML-as-proof framing; state exploratory associational design; negative test R² as methodological finding, not buried caveat.
2. **Fix RQ4 narrative (§5.3, §6.1, §7.2)** — Stop implying uniform “environmental degradation” decay; separate **vessel/CES decay** (supported) from **NO₂/MEI/ESI increase with port distance** (documented, contradicts simple decay story).
3. **Fix RQ6 (§5.7, §7.2)** — Relabel persistent vessel/MEI flags as **baseline corridor activity**; restrict “anomaly” to episodic NO₂/NDTI/FAI; remove “clustering validated” unless downgraded to descriptive hotspots.
4. **Fix RQ3 (§5.4, §6.2, Figure 5.7 captions)** — Lead with **vessel–wind null (p = 0.815)** and **ACEI/ESI stratification**; drop MEI as independent wind validation; soften “wind caused inland propagation.”
5. **Demote NDVI** everywhere except limitations (coverage ~2.3%).
6. **Move ML (§5.9–5.10) to limitations/methodology** — One paragraph in conclusions: models do not generalize; not evidence for RQ2/RQ5 substance.
7. **Add explicit “partially answered” table in §7.1** — Aligns conclusions with evidence matrix (Florian alignment).

Estimated effort: **mostly prose surgery** (2–4 days); **no new analyses required** if existing tables/figures are cited correctly.

---

## RQ1 — Spatial and temporal variation

| Field | Content |
|-------|---------|
| **Status** | **Partially Answered** |
| **Strongest evidence** | 16,065 grid-week panel; NO₂ ~90% and vessel density ~76% coverage; corridor concentration of shipping (`rq1_spatial_summary.csv`, §5.1). |
| **Weakest evidence** | Full-panel optical temporal trends (NDTI/NDWI ~18% valid); **NDVI ~2.3% valid** — cannot support vegetation conclusions. |
| **Unsupported claims in thesis** | Implicit equal weight across indicators in abstract/§1; NDVI as core ML feature in Table 4.2 without coverage warning in conclusions. |
| **Soften** | “Broad environmental variability” → “variation **where observational coverage permits**.” |
| **Remove** | Any conclusion-line use of NDVI; claims that all Sentinel-2 indicators equally inform spatial-temporal RQ1. |

**Results (replacement):** The balanced weekly panel (16,065 cell-weeks) shows strong spatial structure in vessel density and NO₂ where coverage is high. Optical water indices remain sparse after cloud masking (~81% missing at cell-week level; valid optical weeks ~18–19%). Temporal plots reflect observable weeks only and must not be read as complete time series for all indicators.

**Discussion (replacement):** RQ1 is answered for **maritime and atmospheric** indicators; it is only **partially** answered for optical water-quality proxies because missingness is structural (maritime-dominated grid, clouds), not a preprocessing failure.

**Conclusion (replacement):** Documented spatial and temporal heterogeneity in the corridor for NO₂ and vessel density; optical and NDVI legs of RQ1 remain **incomplete** due to coverage.

---

## RQ2 — Maritime activity vs environmental exposure

| Field | Content |
|-------|---------|
| **Status** | **Partially Answered** (strong for **composite/spatial** links; **weak** for optical water quality and ML) |
| **Strongest evidence** | `distance_to_nearest_high_vessel_density_cell` vs CES ρ ≈ −0.923 (n = 16,065); `distance_to_port_km` vs CES ρ ≈ −0.758; MEI vs ACEI ρ ≈ 0.596 (n = 1,017, coastal subset). |
| **Weakest evidence** | Vessel vs NDTI/NDWI (|ρ| negligible); port-week temporal vessel–ESI often undefined/flat; **ML test R² negative** (§5.9). |
| **Unsupported claims** | Abstract: ML demonstrates “strong geographical correlations” as actionable pressure dynamics; §7.2: ML reveals important interrelations for substantive inference; causal maritime→pollution language in §6.1. |
| **Soften** | “Predictors” → “spatial correlates”; “environmental pressure” → “exposure proxy patterns.” |
| **Remove** | ML as validation of environmental relationships; strong vessel–water-quality claims; any NDVI-based association. |

**Results:** Strong **associative** coupling between maritime proximity metrics and composite exposure scores (CES, MEI, ACEI on coastal subset). Direct vessel–optical water-quality correlations are weak at weekly scale. ML models fail temporal generalization (negative test R²) and are reported for transparency only.

**Discussion:** RQ2 supports a **descriptive exposure-mapping** story, not predictive or causal maritime impact. Composites are engineered summaries—useful for comparison, not validated stress metrics.

**Conclusion:** Maritime intensity co-locates with composite exposure indices; optical and predictive ML legs are **not** established.

---

## RQ3 — Wind regimes and directional transport

| Field | Content |
|-------|---------|
| **Status** | **Partially Answered** ( **weak** for traffic modulation; **moderate** for exposure stratification) |
| **Strongest evidence** | Vessel–wind **null**: Mann–Whitney p ≈ 0.815 (384 vs 789); ACEI/ESI shoreward vs non-shoreward: Cliff’s δ ≈ 0.64–0.71, p ≪ 0.001 all bands (`coastal_exposure_statistics.csv`, `outputs/rq3_strengthening/`). |
| **Weakest evidence** | NO₂ regime differences (inner band only, δ = −0.18, lower shoreward); “transport causality”; 30–50 km > 10–30 km MEI inland propagation (§5.3/5.4 — confounded by geography). |
| **Unsupported claims** | §5.4/§6.2: wind “contributed to” exposure without separating circular MEI; §1.3.3 implication that wind **influences** exposure patterns in a transport-causal sense; MEI under shoreward as proof of impact. |
| **Soften** | “Wind-driven structuring” → “wind-regime **stratification of exposure detection**.” |
| **Remove** | Wind controls shipping; MEI validates wind impact independently; strong NO₂ transport conclusions. |

**Results:** Vessel density does not differ significantly by wind regime (p = 0.815). ACEI and ESI differ significantly with large effect sizes; localized NO₂ excess shows a small inner-band contrast only. MEI–wind contrasts are **circular** and reported only as index consistency.

**Discussion:** RQ3 answers **detection and compositing** under wind labels, not **traffic response** to wind. Align with §6.8 seasonality caveat already in draft—make it the headline.

**Conclusion:** Wind regimes organize atmospheric/composite exposure signals; they do **not** significantly organize vessel density in this panel.

---

## RQ4 — Distance from ports and shipping routes

| Field | Content |
|-------|---------|
| **Status** | **Partially Answered** |
| **Strongest evidence** | Vessel vs `distance_to_port_km`: ρ ≈ −0.256 (n = 12,189); CES vs port distance: ρ ≈ −0.758 (n = 16,065); Turku portwise decay curves with bootstrap CIs. |
| **Weakest evidence** | Uniform “degradation/decay” for all indicators; optical inner-band n; NO₂ **increases** with port distance (ρ ≈ +0.084, p ≈ 4.7×10⁻²⁴). |
| **Unsupported claims** | §6.1/§7.2: NO₂ “declined” with distance in same breath as vessel/MEI decay without noting **opposite** NO₂ trend on port-distance axis; “environmental degradation decreases”; §5.3 Mariehamn “deterioration” without metric specification. |
| **Soften** | “Distance-decay of exposure” → “**Port-proximity gradients differ by indicator**.” |
| **Remove** | Single monotonic pollution attenuation story; causal “pressure propagates inland” from 30–50 km MEI band comparison alone. |

**Results:** Maritime proximity metrics (vessel, CES) decrease with distance from port on the port-distance axis. NO₂ mean, MEI, ESI, and NDTI show **positive** Spearman trends with port distance in `rq4_decay_direction_summary.csv`—so **environmental indicators do not share one decay direction**.

**Discussion:** RQ4 is answered for **shipping-intensity and composite exposure geography**; it is **not** answered as a universal law that all pollution proxies weaken away from ports.

**Conclusion:** Strong port-centred gradients for traffic and CES; **mixed/opposite** patterns for NO₂ and several composites on the same axis—requires dual-axis interpretation (port distance vs shoreline distance).

---

## RQ5 — Temporal lag correlations

| Field | Content |
|-------|---------|
| **Status** | **Partially Answered** |
| **Strongest evidence** | NO₂ lag-1 autocorrelation ρ ≈ 0.48 (panel, `rq5_autocorrelation_summary.csv`); weekly median NO₂ lag-1 ρ ≈ 0.41 (§5.6). |
| **Weakest evidence** | Maritime→NDTI/NDVI lags; grid-level vessel→NO₂ peak |r| ≈ 0.04 (`causal_lag_summary.csv`); port-week flat vessel density. |
| **Unsupported claims** | Implied causal delay chains; NDVI/NDVI lag legs; ML lag features as mechanism proof. |
| **Soften** | “Lag correlations measurable” → “**Autocorrelation and weak cross-series alignment** at weekly scale.” |
| **Remove** | NDVI lag conclusions; “maritime drives water-quality delays.” |

**Results:** Clear NO₂ persistence; weak contemporaneous and lagged vessel–NDTI links (ρ ≈ 0.08–0.13, small n); corridor pooled vessel–ESI ρ ≈ 0.25–0.29 with unstable port signs.

**Discussion:** RQ5 is **moderately** answered for atmospheric persistence; **weakly** for maritime-forcing of water-quality lags.

**Conclusion:** Temporal structure is dominated by NO₂ autocorrelation; cross-domain maritime–water lags are not strongly supported.

---

## RQ6 — Anomaly detection and clustering

| Field | Content |
|-------|---------|
| **Status** | **Partially Answered** (detection **moderate**; clustering inference **weak / not answered**) |
| **Strongest evidence** | Episodic flags: NO₂ 9/51 weeks (~18%), NDTI/FAI 3/51 (~6%); pairwise overlap tables (`rq6_anomaly_cooccurrence.csv`). |
| **Weakest evidence** | Formal clustering (no Knox/Moran/Getis in pipeline); vessel 51/51 and MEI persistent (`rq6_anomaly_counts.csv`). |
| **Unsupported claims** | §5.7 “anomalies persisted” for vessel/MEI; §6 “temporal clustering”; §5.5 “clustering behaviour” for ESI hotspots as inferential clustering; RQ6 wording “clustering behaviour” in Ch.1. |
| **Soften** | “Anomaly detection” → “**rule-based exceedance flags**”; “hotspots” → “**upper-tail spatial concentration (descriptive)**.” |
| **Remove** | Rare-event language for vessel/MEI; validated clustering claims. |

**Results:** Rule-based thresholds flag **episodic** atmospheric/optical weeks and **persistent** corridor maritime activity. Co-occurrence is limited (e.g. four weeks with NO₂–MEI overlap in summary tables).

**Discussion:** RQ6 is **partially** answered for **descriptive** co-occurrence; the **clustering** leg is **not answered** with inferential spatial statistics.

**Conclusion:** Use “episodic multi-indicator weeks” not “rare maritime anomalies”; persistent shipping is baseline state, not anomaly.

---

## RQ7 — Turku vs Mariehamn (corridor)

| Field | Content |
|-------|---------|
| **Status** | **Partially Answered** (strongest **comparative** RQ; **not** full attribution) |
| **Strongest evidence** | Turku mean vessel ≈ 1.04 vs Mariehamn ≈ 0.205; Mann–Whitney p ≈ 2.5×10⁻⁶⁸; port decay and pairwise band tests (`port_pairwise_mannwhitney.csv`). |
| **Weakest evidence** | Port-specific pollution causation; single-indicator attribution with shared corridor emissions. |
| **Unsupported claims** | Any Stockholm comparison; implying Turku **causes** higher regional stress without source apportionment (partially acknowledged in §6.4—must reach conclusions). |
| **Soften** | “Turku drives exposure” → “Turku **exhibits higher** exposure metrics in comparative annuli.” |
| **Remove** | Port-level causal attribution; Stockholm in RQ7 scope. |

**Results:** Large, significant differences in vessel density and composite exposure rankings between ports within the corridor; MEI–ESI correlation sign differs by port (Turku negative, Mariehamn positive)—heterogeneity, not uniform mechanism.

**Discussion:** RQ7 is the best-supported comparative question; limitations on spatial separation and shared plumes must appear in **conclusions**, not only §6.4.

**Conclusion:** Clear Turku–Mariehamn **contrasts** in maritime exposure structure; **not** isolated port accountability.

---

## Cross-chapter integrity checks (Florian-aligned)

| Issue | Thesis location | Fix |
|-------|-----------------|-----|
| Abstract promises ML actionable insights | Abstract | Rewrite per `conclusion_rebuild.md` |
| §5.9 says ML negligible then §7.2 cites ML interrelations | §7.2 | Remove substantive ML sentence |
| §5.3 NO₂ “declined” with distance | §5.3, §6.1 | Align with ρ > 0 for NO₂ vs port distance |
| §5.7 persistent = anomaly | §5.7 | Rename persistent vessel/MEI |
| MEI wind circularity | §5.4, Fig 5.7 | Disclose; demote MEI in wind claims |
| Positive abstract on distance-decay | Abstract | Composite + vessel only |

---

## Evidence index (existing outputs)

| RQ | Primary outputs |
|----|-----------------|
| RQ1 | `outputs/rq_evidence_fix/rq1_*`, `outputs/thesis/chapter_5_1_*` |
| RQ2 | `outputs/rq_evidence_fix/rq2_maritime_environment_correlations.csv`, `final_temporal_relationships.csv` |
| RQ3 | `outputs/rq3_strengthening/`, `coastal_exposure_statistics.csv`, `vessel_density_wind_regime_tests.csv` |
| RQ4 | `outputs/rq_evidence_fix/rq4_*`, `port_distance_decay_statistics_turku_mariehamn.csv` |
| RQ5 | `outputs/rq_evidence_fix/rq5_*`, `final_temporal_relationships.csv`, `causal_lag_summary.csv` |
| RQ6 | `outputs/rq_evidence_fix/rq6_*`, `final_anomaly_overlap_table.csv` |
| RQ7 | `port_pairwise_mannwhitney.csv`, `port_exposure_ranking.csv`, `outputs/final_run_turku_mariehamn_thesis/` |

*Reviewer mode: prioritize removing unsupported sentences over adding new analyses.*
