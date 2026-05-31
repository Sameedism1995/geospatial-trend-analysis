# RQ Evidence Report (computed)

Source panel: `/Users/sameedahmed/geospatial-trend-analysis/processed/features_ml_ready.parquet`  
Output directory: `/Users/sameedahmed/geospatial-trend-analysis/outputs/rq_evidence_fix`

## Enrichment
- Attached dist_Turku_km, dist_Mariehamn_km, dist_Stockholm_km via attach_focal_port_distances.
- Merged MEI/ESI/ACEI and shoreward_binary from prepare_panel+build_indices.

## RQ1 — Spatial and temporal variation

**Answer status:** Partially Answered

**Strongest supporting evidence:** Complete 16,065 grid-week panel; high coverage for NO2 (~90%) and vessel density (~76%); spatial concentration along corridors (see rq1_spatial_summary.csv).

**Weakest evidence:** Full-panel temporal trends for NDTI/NDWI/NDVI (optical ~18%, NDVI ~2.3% valid).

**Main limitation:** Sensor missingness, not irregular panel indexing.

**Safe thesis wording:** Environmental indicators show spatial heterogeneity and week-to-week variation where observational coverage permits; optical vegetation indices are sparse.

**Claims to remove or soften:** Any claim that all indicators vary equally across the full panel.

## RQ2 — Maritime activity vs exposure

**Answer status:** Partially Answered

**Strongest supporting evidence:**
- distance_to_nearest_high_vessel_density_cell vs coastal_exposure_score: rho=-0.923, n=16065, strong
- distance_to_port_km vs coastal_exposure_score: rho=-0.758, n=16065, strong
- maritime_exposure_index vs atmospheric_coastal_exposure_index: rho=0.596, n=1017, strong

**Weakest evidence:** Vessel vs optical water quality (typically negligible rho).

**Main limitation:** Cross-sectional dominance; n<30 pairs unreliable.

**Safe thesis wording:** Associative links between maritime intensity and composite exposure/oil proxy; not causal.

**Claims to remove or soften:** ML predictive success; strong vessel–NDTI claims.

## RQ3 — Wind regimes

**Answer status:** Partially Answered

**Strongest supporting evidence:** Stratified composite indices by shoreward regime (MEI partly circular). Vessel vs wind Mann-Whitney p=0.8147.

**Weakest evidence:** Independent proof that wind drives vessel density.

**Main limitation:** MEI includes wind alignment; vessel–wind null after seasonality.

**Safe thesis wording:** Wind regimes structure wind-informed exposure indices; vessel activity does not differ significantly by wind class (p≈0.815).

**Claims to remove or soften:** Wind controls shipping volume; transport causality.

## RQ4 — Distance decay

**Answer status:** Partially Answered

**Strongest supporting evidence:** Monotonic associations with distance_to_port_km for MEI/vessel (Spearman rho≈0.115 if finite). See rq4_distance_band_summary.csv.

**Weakest evidence:** Optical indicators in inner bands (low n).

**Main limitation:** distance_to_port_km mixes regional geography for NO2 at large distances.

**Safe thesis wording:** Exposure composites and vessel density tend to be higher nearer ports on the port-distance axis.

**Claims to remove or soften:** Mixing shoreline-distance and port-distance without labels.

See also: `rq4_shipping_lane_band_summary.csv` (separate axis) and `rq4_decay_direction_summary.csv`.

## RQ5 — Temporal lags

**Answer status:** Partially Answered

**Strongest supporting evidence:** NO2 lag-1 autocorrelation rho≈0.483 (n=13957).

**Weakest evidence:** Maritime→NDTI/NDVI lags; NDVI excluded (coverage).

**Main limitation:** Port-week vessel density often temporally flat.

**Safe thesis wording:** Measurable persistence in NO2 and some composite pairs; weak maritime–water-quality lags.

**Claims to remove or soften:** NDVI lag conclusions; strong predictive lag claims.

## RQ6 — Anomalies

**Answer status:** Partially Answered

**Strongest supporting evidence:** Episodic co-occurrence for NO2/NDTI/FAI; persistent flags for: vessel_density_t.

**Weakest evidence:** Formal spatial clustering tests (no DBSCAN/Knox in this script).

See: `rq6_anomaly_runlength.csv` for temporal run-length only.

**Main limitation:** Persistent vessel/MEI flags are not rare events.

**Safe thesis wording:** Rule-based anomalies detect episodic multi-indicator weeks; not validated clustering.

**Claims to remove or soften:** Rare anomaly language for corridor vessel/MEI; generic clustering claims.

## RQ7 — Turku vs Mariehamn

**Answer status:** Partially Answered

**Strongest supporting evidence:** Turku mean vessel≈1.04 vs Mariehamn≈0.205 (Mann-Whitney p=2.508e-68).

**Weakest evidence:** Port-specific pollution attribution.

**Main limitation:** ~150 km separation; shared corridor emissions.

**Safe thesis wording:** Comparative corridor-level exposure structures differ between Turku and Mariehamn.

**Claims to remove or soften:** Single-port causation; Stockholm comparison.

Port-specific decay: `rq7_port_distance_decay.csv` (dist_turku_km / dist_mariehamn_km axes).

## Summary table

| RQ | Evidence Strength | Defensible Thesis Claim | Claims to Avoid |
| --- | --- | --- | --- |
| RQ1 | Moderate | Spatial/temporal variation documented with coverage caveats | Equal coverage across all indicators |
| RQ2 | Moderate–Weak | Associative maritime–composite links | Strong ML prediction; strong optical links |
| RQ3 | Weak–Moderate | Wind stratifies indices; vessel–wind not significant | Wind drives traffic; independent MEI validation |
| RQ4 | Moderate | Port-distance decay for vessel/MEI/NO2 | Shoreline vs port distance conflation |
| RQ5 | Moderate–Weak | NO2 persistence; weak maritime–water lags | NDVI lags; causal delay claims |
| RQ6 | Moderate–Weak | Descriptive anomaly/co-occurrence | Rare MEI/vessel anomalies; formal clustering |
| RQ7 | Moderate | Turku higher exposure than Mariehamn in corridor | Port-specific attribution; Stockholm |

## Final thesis guidance

**Framing:** Use an **exploratory geospatial exposure framework**, not predictive ML. Section 5.9–5.10 negative R² supports this.

**Central in Discussion:** Distance-decay (RQ4), Turku–Mariehamn contrast (RQ7), pooled vessel–ESI association, NO2 persistence.

**Limitations/appendix:** ML fold metrics; NDVI seasonal bias; shoreline vs port distance methods; Stockholm exclusion.

**NDVI in conclusions:** Exclude from main conclusions (coverage≈2.3%).

**MEI/ESI as main findings:** MEI/ESI suitable as **descriptive composite summaries** if labelled experimental and non-causal; MEI not independent evidence for wind (circular). Do not treat as validated environmental stress metrics.
