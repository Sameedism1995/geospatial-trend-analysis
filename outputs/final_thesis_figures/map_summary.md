## Coastal composite exposure — interpretation snapshot

Maps are built as **association / exposure composites** combining MEI-related (`maritime_pressure_index`), weekly NO₂, vessel-density intensity, proximity to archived nearest port (1 / [1 + km]), and optionally shoreward-aligned wind proxies when present. **They do not diagnose pollution attribution or deterministic causality**—they summarise co-located gradients useful for Coastal monitoring hypotheses.

### Hotspot anatomy

* **22** Baltic lattice cells occupy the composite **top decile**, emphasised with hatched rims on the thematic panels (plus `coastal_exposure_hotspots.png` for the Baltic extent).
* Optional Getis–Ord GI* overlays are noted when SciPy-compatible spatial stats libs are installed: not computed in this repo build (requires libpysal + esda). Hotspots flagged by global composite ≥ P90 only.

### Port comparisons

**Turku** zooms emphasize the Åland corridor arc and Åbo archipelago littoral stressing; vessels channel westward past Mariehamn, elevating VD-linked exposure bands that align with Baltic Main Lane traffic. **Mariehamn** isolates choke-point behaviour where VD corridors narrow between Sweden and mainland Finland—the composite typically highlights longitudinal streaks tying MEI–NO₂ co-elevation near fairway-aligned cells. **Stockholm** highlights inner-archipelago choke cells where NO₂ residuals and VD interact within short port-range distances—the composite often peaks on cells inside the Sodertalje funnel and eastward littoral wedges.

### Shipping corridor relationship

**Gold linework** approximates dissolve boundaries of lattice cells exceeding the **global 88ᵗʰ %-tile** of aggregated vessel-density; because corridors are drawn as discrete tessellation boundaries, emphasised paths show **spatial adjacency**, not IMO route polylines. They nevertheless align visually with Baltic Main lanes where exposure composites peak.

### Structural limitations

* Week-level medians summarise persistence but smooth synoptic extremes.
* Composite weights are heuristic (0.35 / 0.25 / 0.25 / 0.15 + optional wind).
* Queen-based GI* ignores irregular archipelago fragmentation if libpysal cannot build contiguous weights cleanly.
* No gap-filling in ocean blanks—transparent areas reflect absent composite inputs (`_core_complete` false).
