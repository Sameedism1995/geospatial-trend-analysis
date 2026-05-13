# Comparison: previous hub approach vs. revised hub strategy

This report contrasts the **previous** distance-decay analysis (file:
`outputs/visualizations/hub_level_distance_decay/` and
`outputs/visualizations/sliding_window_distance_decay/`) with the **revised**
hub strategy in this run folder.

## Previous approach

- **Hubs**: Mariehamn, Turku, Naantali, each treated as an independent port.
- **Distance range**: a single sliding-window curve from 0-1000 km, mixing
  near-port grids and grids hundreds of km away.
- **Implication**: Turku and Naantali (~15 km apart) double-counted the same
  Finnish coastal system.  A single curve from 0-1000 km blurred local
  port-proximity signal with regional background atmospheric/oceanic
  patterns — exactly Florian's concern.

## Revised hub strategy (this run)

- **Turku-Naantali coastal hub** — Turku and Naantali combined; distance is
  the minimum distance to either port.
- **Mariehamn offshore/island hub** — kept independent; serves as a
  low-urban-background contrast.
- **Stockholm urban port hub** — added as an explicit urban-background
  contrast for NO2.
- **Local analysis** restricted to 0-100 km, with a 25 km sliding window
  and 10 km step.
- **Regional analysis** restricted to 100-300, 300-500, 500-1000 km bands,
  always labelled "REGIONAL BACKGROUND PATTERN — NOT PORT IMPACT".

## Which previous conclusions remain valid

- Distance decay is real for **vessel density** in the local 0-100 km
  window: it is highest within ~25 km of a hub and decays toward 100 km.
  This was visible in the previous figures and remains visible here.
- **Water indices (NDTI/NDWI/NDVI)** still vary with hub geography but
  with weak distance-decay; their absolute values are dominated by
  open-water vs. coastal-water contrasts.

## Which previous conclusions need correction

- The previous "NO2 increases between 200-500 km" finding was a binning /
  background-mixing artefact, not a port effect.  In the revised view,
  100-1000 km is **regional background** and is plotted separately.
- The previous "Mariehamn shows the strongest distance-decay" was partly
  a Turku/Naantali double-counting effect — removing the double count
  reduces the relative spread between Mariehamn and the Finnish hub.
- Any single-curve "port effect from 0 to 1000 km" claim should be
  retracted; only the 0-100 km curve is interpretable as a port-proximity
  signal.

## How the new strategy addresses Florian's concern

Florian's concern: combining nearby ports as independent hubs and
extending sliding-window curves to 1000 km confused **local port impact**
with **regional atmospheric / spatial background**.

The new design fixes this by:
1. Treating local-scale (0-100 km) and regional-scale (100-1000 km)
   separately — different plots, different titles, different
   interpretation.
2. Explicitly labelling regional plots as "BACKGROUND PATTERN — NOT PORT
   IMPACT" so they cannot be misread.
3. Combining nearby ports (Turku-Naantali) so a single hub is not
   double-counted and so the local curve represents one coastal system.
4. Adding an urban contrast (Stockholm) so NO2 patterns can be
   attributed to urban/atmospheric rather than ship-source effects.

## Evidence for the urban-NO2 hypothesis (this run)

- Stockholm NO2 decreases with distance (0-50 km): **False**
- Stockholm NO2 higher near port than 50 km out: **False**
- Mariehamn NO2 0-50 km weaker than Stockholm: **False**

Interpretation: **INCONCLUSIVE for 0-50 km: dataset has no grids within 50 km of Stockholm, so the urban-NO2 hypothesis cannot be tested at the intended scale.  In the next available band (50-200 km) NO2 decreases with distance from Stockholm, which is consistent with — but does not prove — an urban-source pattern.**.
