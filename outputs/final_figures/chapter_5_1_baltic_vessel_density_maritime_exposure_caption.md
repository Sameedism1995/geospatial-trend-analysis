# Figure — Baltic maritime traffic intensity and exposure context (Chapter 5.1)

## Caption

**Figure 5.1.** Baltic Sea study extent showing maritime traffic intensity on the **actual machine-learning lattice** (`processed/features_ml_ready.parquet`). Each tiled polygon is the nominal **square cell** centred on the archived `grid_centroid_lon` / `grid_centroid_lat`, with edge length inferred from **`grid_cell_id`** (prefix `g0.100` ⇒ **≈ 0.100°** per side in this build). Polygon fill reflects the **weekly time-mean `vessel_density_t`**, averaged across **51** UTC week anchors (**211** Baltic-bounded cells retained; malformed off-domain centroids omitted). No continuous ocean interpolation is applied—the figure is strictly a **gridded choropleth**. **Bold navy rims** denote cells at or above the **global 88th percentile** of those means (discrete corridor emphasis across **26** cells). **Dashed rims** tinted by labelled port denote **within-port hotspots**: cells meeting the assigned `nearest_port` **P72** threshold among co-labelled cells **and** `distance_to_port_km` ≤ **42 km** (**39** cells highlighted). Shorelines derive from **Natural Earth 110m** land polygons. **Stockholm**, **Turku**, **Mariehamn**, and **Naantali** use blue, orange, green, and magenta star markers respectively. Horizontal colour bar, **100 km** scale bar, and north arrow follow standard cartographic marginalia.

## Chapter 5.1 — short interpretation (thesis-ready)

Because each observation in the ML panel ties to a **fixed `grid_cell_id`**, the choropleth makes the **empirical support** of vessel-density covariates transparent: intensity is patchy at the resolution of the sampling mesh rather than an artificial smooth sea surface. Spearman correlation of cell means with longitude is about **0.258** and with latitude about **0.121**, underscoring **along-basin heterogeneity** rather than spatial homogeneity. The discrete global-percentile emphasis reveals **elongated high-traffic clusters** consistent with major fairways, while the port-conditioned dashed borders stress **localised adjacency surges** near hub assignments. Together these devices communicate that maritime exposure in the modelling frame is **cell-local and anisotropic**, warranting stratified interpretations relative to Åland-archipelago arcs, Finnish gateway ports, and the Stockholm skerries (**g0.100_r1503_c2022 (Turku, mean=25.71), g0.100_r1501_c2021 (Turku, mean=5.89), g0.100_r1502_c2019 (Naantali, mean=5.36), g0.100_r1503_c2020 (Naantali, mean=4.97), g0.100_r1504_c2020 (Naantali, mean=3.60)** summarise the strongest global cell means).

## Data notes

- Primary attribute: **`mean vessel_density_t` per grid cell**, identical aggregation as used for map coloring.
- No triangulation/KDE smoothing; geometries are rectangles in geographic degrees centred on archival centroids.
- Companion tables: `outputs/final_figures/chapter_5_1_vessel_density_spatial_summary.csv`, `chapter_5_1_highest_vessel_density_cells.csv`, `chapter_5_1_port_adjacent_highest_exposure.csv`, PDF/PNG under the same basename.
- Regenerate: `python3 scripts/generate_chapter_5_1_maritime_map.py`
