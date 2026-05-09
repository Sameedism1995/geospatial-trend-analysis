# Thesis narrative companion: what the code did and what the numbers say

This is a prose summary you can lift from when writing your thesis chapters. Throughout, the wording stays **associational**: patterns, contrasts, and structure—not proof that shipping caused a specific outcome. The figures named here live under **`outputs/`** in the folders that match each analysis bundle; **`outputs/reports/`** holds the spreadsheets if you need the exact bootstrap interval or every stratum-wise *p*-value beyond what is recalled below.

Your implementation builds weekly coastal panels, ranks and rescales variables into composite **exposure indices**, then compares **distance bands**, **wind regimes** (shoreward versus not, in the sense of alignment between wind-toward and landward bearing), and **focal ports**—especially **Turku** and **Mariehamn**. A later layer stacks **time**: lags, short rolling means, persistence from one week to the next, and simple anomaly flags. None of that replaces a controlled experiment; it **organises** what the grid can show under honest missingness.

---

## Building the story: three composites and their overall shape

The heart of the spatial thesis thread is implemented in **`run_coastal_exposure_analysis.py`**. There you define three indices on the same marine-adjacent study frame. The **maritime exposure index** elevates places and weeks where vessel density, coastal wind alignment (after rectification), and proximity to the coast line up; each ingredient is ranked before combination, and the result is turned into a percentile rank between zero and one. The **atmospheric coastal exposure index** does the analogous thing for NO₂ excess in the week, coastal alignment, transport-type alignment, and coast proximity. The **environmental stress index** is explicitly experimental: it averages standardised contributions from NO₂, vessels, an oil-slick proxy, and a water-quality anomaly component, then folds in a wind-rank term, again expressed as a percentile-like score.

Across all grid-week rows that support each index, their marginal distributions reflect that construction. For maritime exposure there are **1 173** non-null values with a mean near **0.50** (standard deviation about **0.29**); the atmospheric index tracks almost the same mean and spread on **1 314** rows. Stress sits slightly higher on average—mean about **0.53** on **1 530** rows—with a narrower spread and a ninety-fifth percentile near **0.82**. Those summary lines come from **`exposure_indices_summary.csv`**. Visually they are unpacked in **`outputs/figures/coastal_exposure/`** (decay curves, behaviour heatmaps, hexbins, dashboards), with interpretive scaffolding in **`coastal_exposure_analysis_summary.md`**.

---

## Port-centred geography: Turku versus Mariehamn

**`run_portwise_coastal_exposure.py`** re-centres the world on canonical port coordinates. It walks outward through rings—**zero to three**, **three to seven**, **seven to fifteen**, and **fifteen to thirty kilometres**—and repeats the stratification where wind is classified as broadly **shore-aligned** versus **not**. For each pocket it reports counts, means, medians where relevant, bootstrap confidence ribbons on pooled zone means where the pipeline supplies them, and nonparametric contrasts. One table asks whether Turku’s and Mariehamn’s distributions **differ** in the **same band and regime** (**`port_pairwise_mannwhitney.csv`**); another stores the full long-form decay (**`port_distance_decay_statistics.csv`**). A narrower “coastal core” band intersects these rings with your coastal-panel mask when you want to stress the harbour–urban interface.

The ranking sheet makes the Baltic contrast blunt. Within thirty kilometres Turku dominates the maritime-story numbers: pooled-band mean maritime index about **0.57** versus Mariehamn’s **0.32**; atmospheric means about **0.54** and **0.28** respectively; stress means about **0.56** and **0.39**. On those heuristic rankings Turku carries rank one for maritime, atmospheric, and stress means; Mariehamn is rank two throughout. Interestingly, Mariehamn shows stronger **separation** of stress index between shoreward and non‑shoreward weeks in this diagnostic (rank **1** versus Turku rank **2**), which becomes something you can narrate as **difference in directional sensitivity** rather than “more pollution.”

Formal two-sample tests amplify the geographic split. Outside the specialised shore-only rows, contrasts in outer bands are uncompromising: for example in the fifteen-to-thirty-kilometre ring the maritime exposure index separates the ports with Mann–Whitney *p*-values on the order of **10⁻²³⁰** and Cliff’s deltas near **0.92**, reflecting almost non-overlapping bulk between Turku (**mean ≈ 0.73**) and Mariehamn (**mean ≈ 0.29**) on paired sample sizes **714** and **1 173**. Even where sample sizes tighten (same band but shoreward strata with **128** versus **235** observations), maritime means stay far apart (**≈ 0.91** vs **0.39**) with astronomically small *p*. Mid-distance rings tell the same story for stress in some strata (**seven to fifteen kilometres**, pooled non‑shoreward: *p* ≈ 4.6 × 10⁻²⁴). Counterexamples matter for honesty: fifteen-to-thirty-kilometre stress in the pooled non‑shoreward comparison differs only modestly (**0.535** versus **0.526**) with *p ≈ 0.18**, so stress is **not** universally “winner takes all” across every band.

Pooling both ports inside the **`coastal_exposure_analysis_summary.md`** excerpt, shoreline wind slicing shreds the maritime composite in the innermost band (**Mann–Whitney *p* roughly 10⁻³⁰**, Cliff’s delta essentially one); stress reacts more gently there (*p ≈ 0.02)*. Your Figures **Fig1–Fig7** under **`outputs/figures/portwise_exposure/`** carry the visual mirror: ranking bars, per-port decay, shared annulus contrasts, focal maps and dashboards—the written captions sit in **`portwise_exposure_analysis_summary.md`**.

---

## Thesis bundle: final decay, methodology text, and time

**`run_final_thesis_turku_mariehamn.py`** stitches the parquet load, merged wind descriptors, shoreline preparation, composite construction, decay tables—and then calls **`final_thesis_spatiotemporal.py`**. That second module restricts attention to weekly means inside **30 km** of each port, aligns calendar weeks without inventing replacements, computes **Pearson** and **Spearman** correlations for contemporaneous (**t**) and shifted (**t − 1**, **t − 2**) exposures and for trailing two- and three-week averages, and records the **median** of short rolling Pearson correlations (eight weeks, needing at least six) when they exist. Persistence is summarized by correlating **last week’s** value with **this week’s** within the port window for NO₂, stress, and maritime exposure.

Associations are exploratory by design yet numerically articulate. Weekly vessel density and weekly NO₂ pooled across both focal ports yields Spearman about **0.28–0.31** depending on lag, typically on **ninety-ish** synchronous weeks—you should caveat that vessel density barely moves from week to week inside some coastal windows so those pooled estimates can mix **spatial** heterogeneity across ports with **temporal** slack. Maritime exposure aligns more strongly with the atmospheric coastal index: contemporaneous pooled Spearman about **0.69** (**n ≈ 91**), still **≈ 0.62** at **t − 1**, but signs **disagree** between Turku and Mariehamn in some instants (**negative ρ** at Turku versus **strong positive** near **0.84** at Mariehamn in the contemporaneous maritime–atmospheric block), stability that matters for wording about **generalisation**.

Maritime-to-stress pooled correlations stay in the twenties to low thirties by Spearman. NO₂ paired with NDVI exposes a **truthful failure**: overlapping weeks collapse to counts of **two to four**, so correlations are deliberately left **missing** rather than interpolated—use that as methodological transparency.

Atmospheric linkage to stress is stronger while still non-causal: NO₂ contemporaneous ρ about **0.57** with stress; rolling NO₂ composites track into the low forties. Week-to-week **persistence**: Spearman of NO₂ on its past self about **0.41** (**89** weeks pooled under the rule), stress about **0.23**, maritime persistence about **0.75** (Pearson climbs near **0.96** because the smoothed maritime rank series is mechanically sticky). Thesis synthesis text and keyed figures—including lag heatmaps, anomaly timelines and maps—are gathered under **`outputs/final_run_turku_mariehamn_thesis/`** beside Markdown interpretation files and **`final_key_findings_table.csv`** that tags **F1–F3** (spatial decay, temporal association layer, anomaly description).

---

## Wind geometry and “transport language” without claiming transport

The coastal wind scripts merge **weekly u/v**, derive landward bearings from shoreline geometry, contrast them against wind toward, and summarise **alignment cosines**. The interpretation note records the panel footprint—**about 271** cells spanning **51** weeks (**13 821** grid-week observations in that configuration), using **ERA5-derived** meteorology—and lists Spearman correlations that hover near **±0.02–0.03** for several NO₂–alignment pairings (**n roughly 12 000**) with classical *p*-values crossing **0.014** down to **0.003**. A comparatively stronger exploratory line ties NDVI negatively to an oil-times-alignment construct (ρ ≈ −0.098, *p* on the order of 10⁻²¹, roughly nine‑thousand paired rows).

Mann–Whitney contrasts for NO₂ comparing shore-aligned weeks to others are **not decisive** everywhere in the archived summary: for instance the pooled **0–3 km** band registers a shoreline–mean difference only on the scale of roughly **−7 × 10⁻⁷** in local NO₂ excess units yet achieves *p* ≈ **0.271**, so directional framing must stay tentative. Artefacts accumulate in **`run_coastal_wind_transport`** under reports and **`outputs/figures`** plus an HTML pollution-transport map you can zoom.

Parallel correlation tests in **`run_land_pollution_drivers_wind`** show modest negative associations between vessels and NO₂ excess (Spearman ρ ≈ −0.077, *p* ≈ 0.014) or between oil probability and NO₂ excess (ρ ≈ −0.110, *p* ≈ 6 × 10⁻⁵); distance to neighbourhoods of historically high vessel density aligns strongly—in the monotone sense tested—with the SAR oil proxy probability (ρ ≈ −0.41, extraordinarily small *p* on roughly one-and-a-half-thousand rows). Flag that the SAR detection score correlates perfectly with oil probability because **one derives from the other** algorithmically—not because the ocean announced a new truth.

---

## Refinements, participations and secondary pipelines

Several scripts **narrow** rather than widen the aperture. **`run_final_coastal_shipping_impact.py`** confines attention to neighbourhoods both coastal and anchored near chronic vessel congestion, then contrasts high and low extremes; the **`professor_summary.md`** therein already speaks in subdued academic tone. Indicator participation summaries stack spectral layers for participation heatmaps (**`indicator_participation_*`** outputs). **`run_nearest_land_ndvi_linkage.py`** attaches nearest land Sentinel greenness summaries for land–sea buffer narrative. Radar-centred notebooks of winter contrast and decadal slicing live under **`run_season_aware_coastal_impact`** and **`run_research_radar_coastal_pipeline`**.

Stratified NO₂-oil interplay tables (**`run_no2_oil_slick_combo`**) and traffic extreme cohort contrasts round out exploratory robustness branches. Scripts whose names begin with **`audit_`** automate transparent QC—wind availability, Mariehamn ingest, Fig3 cross-check traces—backing your thesis appendix with spreadsheets rather than hand-waving.

The **ingestion and Earth Engine** lineage (`process_emodnet_tiffs.py`, Sentinel pipelines, **`no2_gee_pipeline.py`**, validation artefacts such **`no2_gee_validation.json`**) merits a methods subsection on its own enumerating versioning, tiling, masking, atmospheric correction—not retold exhaustively here so the narrative stays analytic.

Lastly the **Streamlit dashboard** (`dashboard/app.py`) is pedagogical scaffolding: sliders and filters regenerate already-saved artefacts in Plotly. It proves nothing scientifically new but communicates your portfolio on a laptop or a classroom screen after **`streamlit run`**.

---

## How to cite your own artefacts without overwriting yourself

Treat **every numeric claim** above as shorthand: before printing in the dissertation, reconcile with the originating CSV cell or markdown table so numbering tracks your final rerun. Maintain the intellectual hygiene already encoded in **`final_temporal_validation.md`** and the portwise README: honour **missing** correlation entries, concede **thin** temporal contrast for NDVI, and distinguish **spatial confounding between ports** from **temporal causal identification**. With that discipline, what you wrote in Python becomes what you argue in English—measurable, restrained, and defensible.

