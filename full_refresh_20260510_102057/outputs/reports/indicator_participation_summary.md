# Indicator participation in coastal exposure

## Framing (thesis wording)
This layer describes **spatially and regime-dependent environmental participation**: which indicators occupy higher 
relative positions on an outlier-resistant common scale (**coastal percentile rank**) under different 
**shipping-distance** and **directional wind** contexts. Interpret as **association / structuring / behavioural coupling**, 
not causal attribution or source apportionment.

## Normalisation
- **Participation metric (heatmap / composition / curves):** empirical percentile rank of each indicator **within 
the pooled coastal-analysis panel**. Values near 1 mean that indicator tended to lie high in its empirical distribution 
for that regime relative to coastal grid-week observations.
- **Secondary row in CSV:** robust z-scores `(x-median)/(IQR/1.349)`, clipped ±3.5 for compact regime summaries.

### Regimes
- Distance bands: 0–3, 3–7, 7–15, 15–30 km from high-density shipping cells.
- Wind: shoreward versus non-shoreward using coastal alignment cosine ≥ cos(45°).

### Effect sizes vs opposite wind (`cliffs_delta...`) 
- Non-parametric stochastic dominance contrast between regimes **within each band**. Small |δ| ⇒ similar ordinal behaviour.

### Machine-learning block (optional)
- Histogram gradient boosting with **permutation importance** predicts **environmental stress index** from shipping, 
sentinel, alignment, and distance fields — report as **relative predictive participation** only.

## Run-generated outputs (this build)
- **FigE (network):** present — `outputs/figures/indicator_participation/FigE_correlation_participation_network.png`.
- **FigF / ML permutation:** skipped — need enough complete rows (default ≥200, override with `--ml-min-rows`); interpretability is *relative predictive participation*, not causal importance.

## Thesis-ready figure captions

**FigA — Indicator participation heatmap (shoreward / non-shoreward).** 
Rows: environmental indicators; columns: shipping-distance bands. 
Colour: mean **coastal-panel percentile rank** of each indicator within the regime subset. 
High values indicate stronger **ordinal participation** in that spatial–wind context (structuring, not source attribution).

**FigB (thematic) — Stacked exposure composition by distance.** 
Six thesis-facing layers: NO₂, vessel density, combined wind-alignment fields, oil proxy, Sentinel NDTI–NDWI–NDVI, 
and the three exposure indices. Band totals are **sums of mean rank participation** within each theme, then row-normalised 
to show **proportional structuring** (not mass apportionment). Files: `FigB_thematic_stacked_*.png`.

**FigB (indicator detail) — Full indicator stack.** 
Same construction as the thematic stack but one segment per resolved indicator (11 layers). 
Files: `FigB_stacked_composition_*.png`.

**FigC — Wind-regime participation comparison.** 
Dual panel: **shoreward** versus **non-shoreward** violins of rank participation by distance band 
for selected indicators (colour / hue). Highlights **regime-dependent structuring** of ordinal participation.

**FigD — Participation versus inland distance.** 
Lines trace mean rank participation across distance bands under shoreward versus non-shoreward conditions. 
Highlights **persistence or decay** of indicator participation inland — descriptive of coupled environmental behaviour.

**FigE — Spearman coupling network on rank participation.** 
Edges denote |ρ| above a threshold; **node colour** encodes maritime-linked, atmospheric / directional coupling, 
Sentinel surface state, or composite stress (legend). Integrates **environmental coupling** without implying causality.

**FigF — Permutation importance (when run).** 
Bars show mean drop in model score when features are permuted — **machine-learned predictive participation**, not causal effect sizes.

## Figure map
- **FigA** heatmaps: participation matrix by indicator × distance (shore vs non-shoreward). 
- **FigB** thematic stacked composition (NO₂, vessel, wind, oil, Sentinel, indices) plus **indicator-detail** stacks.
- **FigC** violin ridges for curated indicators across bands.
- **FigD** distance curves (line bundles) for inland persistence / decay silhouettes.
- **FigE** coupling network (Spearman on ranks; requires `networkx`). 
- **FigF** permutation-importance bars when ML executes.

## Interpretive hooks (non-causal)
- **Spatially varying participation:** compare FigA/FigD across bands for maritime-associated vs terrestrial Sentinel signals.
- **Directional environmental coupling:** alignment indices flip rank structure between regimes (CSV: Cliff δ, Mann–Whitney); frame as regime-dependent structuring.
- **Maritime-associated coastal exposure:** vessel density and composite indices typically co-elevate participation near lanes; stress how **composition** (FigB) shifts inland.

See `indicator_participation_statistics.csv` for means, medians, variance, bootstrap CIs, and regime contrasts.

## Files
`outputs/reports/indicator_participation_statistics.csv`
`outputs/reports/indicator_participation_ml_permutation.csv` (when ML executes)
