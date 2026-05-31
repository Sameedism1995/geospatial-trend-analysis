# Land / coastal exposure figures — thesis integration

This note accompanies artefacts in `outputs/thesis_land_exposure/` generated from the balanced weekly ML panel (`processed/features_ml_ready.parquet`) plus archived wind merges.

## Analytic mask (all quantitative panels)

- Coastal-receiver lattice inside the Baltic study window with **nearest NE110 shoreline distance ≤ 50 km** and a **shipping-proximity screen on `distance_to_nearest_high_vessel_density_cell`**.
- Turku/Mariehamn obey the archival **< 30 km** cutoff to offshore high-density hubs. Stockholm’s terrestrial lattice cells routinely sit **>** 30 km from those Baltic-wide seeds despite active coastal commerce, so the script applies a **bounded Stockholm-only extension (< 7200 km)** on the **same archival column**, documented here so supervisors can adjudicate sensitivity vs excluding the inland capital entirely.
- Littoral Baltic centroids flagged as seawater polygons are retained deliberately (Åland/Helsinki archipelagos rarely register as terrestrial in NE110); CSV exports annotate `strict_land_centroid_ne110` for terrestrial-only overlays.
- Stratification (post-mask): **0–3, 3–10, 10–30, 30–50 km** shoreline-distance bins computed as chord distances to sampled coastlines.

## Figure A — `fig_landward_decay.*` + `landward_decay_summary.csv`

**Suggested placement:** Results chapter — subsection on *landward decay of coastal stressors* after introducing MEI / CES / ESI.

**Caption (draft):** Mean NO₂, MEI, coastal exposure score, and ESI plotted against shoreline-distance bins for archival Turku/Mariehamn/Stockholm slices inside the masked coastal lattice. Connecting lines join port-specific means whose vertical bars encode standard errors of weekly observations pooled within bin × port.

**Key finding:** Summarises shoreline-distance structuring of maritime and atmospheric composites with port-specific footprints; contrasts Turku littoral densities against Mariehamn’s narrow annulus versus Stockholm’s extended shipping-distance caveat (integration note above).

## Figure B — `fig_wind_land_exposure.*`, `fig_wind_land_violin.*`, `wind_land_exposure_summary.csv`

**Placement:** Same chapter subsection on *directional atmospheric coupling* (pairs with existing wind-alignment discussion).

**Caption (bars):** Shoreward (hatched) versus non-shoreward weekly regimes within each shoreline bin; bars are means ± SE for each port assignment in the annulus.

**Caption (violins):** Width-normalised densities of weekly observations with 1–99% trimming to limit extreme tails while preserving structural missingness.

**Key finding:** Contrasts distributional mass under onshore transport windows versus other directions, emphasising land-receiver framing.

## Figure C — `fig_land_hotspots.*` + `land_hotspot_cells.csv`

**Placement:** Spatial results / case-study maps for each focal city.

**Caption:** Discrete lattice choropleths (no continuum smoothing between tiles) depicting cells exceeding **within-port P90 temporal medians** for coastal exposure score (top row) and ESI (bottom row); dashed rims flag the Baltic-wide **P88 vessel-density corridor** on the coincident temporal medians.

**Key finding:** Localises compounded exposure pressure adjoining each coastal metropolis while signalling where shipping-heavy lattice edges overlap elevated CES/ESI medians (`strict_land_centroid_ne110` documents terrestrial vs water centroids in the CSV export).

## Figure D — `fig_wind_transport_landward.*`

**Placement:** Methods–Results bridge for *directional transport interpretation* (Turku vs Åland corridors).

**Caption:** Left: Turku corridor with cell-filling based on median NO₂ weekly anomaly; right: Åland maritime corridor with MEI medians. Arrows show median u/v components (no vector interpolation between unsampled locations).

**Key finding:** Couples spatial gradients of stress with mean flow orientation on the actual ML lattice.

## Figure E — `fig_cross_port_land_exposure.*` + `cross_port_land_exposure_summary.csv`

**Placement:** Comparative synthesis paragraph before discussion.

**Caption:** Port-level means (± SE where applicable) for coastal exposure and ESI, a simple NO₂ “persistence” ratio (30–50 km mean divided by 0–3 km mean), and a weighted linear slope of MEI versus band centre as a compact decay descriptor.

**Key finding:** Summarises which archived port assignment exhibits the strongest land/coastal exposure structure under a shared mask.

---

**Radar / spider chart:** Omitted — mixed units (index vs ratio vs slope) would require arbitrary rescaling; the facetted bar layout keeps comparability transparent.
