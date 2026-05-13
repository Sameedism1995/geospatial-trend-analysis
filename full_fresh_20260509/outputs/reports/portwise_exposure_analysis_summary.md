# Port-wise coastal exposure analysis

**Framing:** association-based coastal and maritime **exposure structuring** around focal Baltic ports. 
Distances are computed to **canonical port coordinates** (WGS84). Comparison **figures** are limited to 
**Turku vs Mariehamn**; **Stockholm** remains in the long CSV tables for diagnostics and in 
`outputs/reports/wind_coverage_audit.csv` (wind merge root cause).

## Zones
- Port-centred annuli: **0–3 km**, **3–7 km**, **7–15 km**, **15–30 km**.
- **0–3 km (coastal core):** intersection of ≤3 km from port and **`coastal_panel`** (marine-adjacent study mask). 
Highlights urban–port/coastal-interface behaviour.

## Outputs
- `port_exposure_ranking.csv` — pooled-band means and heuristic ranks.
- `port_distance_decay_statistics.csv` — long-format means/medians/bootstrap CI per port × zone × wind × metric.
- **`outputs/reports/wind_coverage_audit.csv`** — why wind alignment exists or not by port annulus; 
run `python3 src/analysis/audit_wind_coverage.py` to refresh.

### Statistical notes
- **Mann–Whitney:** non-parametric comparison of pooled grid-week draws between ports in the same zone.
- **Bootstrap CI:** 95% percentile intervals on zone means.
- **Spearman** (in decay table): monotonic association of **focal distance** with each metric within the 
zone/wind filter (descriptive gradient, not transport proof).

## Data coverage caveat
Composite indices (especially **maritime exposure**) require non-missing **vessel density** and **coastal wind 
alignment** fields per grid-week. Cells near **Stockholm** in this build largely lack those inputs, so maritime 
and some atmospheric summaries can be **sparse or empty** while stress and Sentinel means may still compute. 
Use `cells_30km_nonnull_*` columns in `port_exposure_ranking.csv` and per-zone `n` in the decay table—**do not 
interpret missing bars as zero physical exposure**.

## Figure captions
- **Fig1** — Bar comparison (**Turku vs Mariehamn**): mean maritime, atmospheric coastal, and environmental stress indices 
(pooled over standard annuli).
- **Fig2** — **Fixed-band** per-port decay: all standard zones (gaps = missing); indices and NO2 excess; shoreward vs non-shoreward with bootstrap CI.
- **Fig3** — **Shared-valid-annulus** comparison (Turku vs Mariehamn): shoreward vs non-shoreward **per composite**; annulus = first of 0–3 / 3–7 / 7–15 / 15–30 km where **both** ports have n≥1 in **both** wind strata (see `shared_annulus_selection.csv` in thesis bundle). Not the same as forcing a single band for every indicator in Fig2.
- **Fig4** — Focused map around **Turku** or **Mariehamn** (each its own panel).
- **Fig5** — Dashboard for **Turku** or **Mariehamn**.
- **Fig6** — **Indicators only** (no composite indices): Turku vs Mariehamn **pooled** annulus means (0–3 … 15–30 km), one subplot per indicator.
- **Fig7** — Same **raw indicators** vs **distance zone** with overlaid Turku / Mariehamn lines.

## Interpretation hooks (allowed language)
- Compare **spatial decay** of rank-scale indices outward from each hub.
- Discuss **directional environmental association** via shoreward/non-shoreward splits.
- Contrast **maritime-associated structuring** versus Sentinel or NO₂ behaviours where coverage allows.
- Avoid causal transport, deposition, or source apportionment claims.
