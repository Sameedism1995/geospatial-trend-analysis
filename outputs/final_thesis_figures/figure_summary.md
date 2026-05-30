# Final thesis figures — thematic map

Artifacts live in `outputs/final_thesis_figures/` (PNG + PDF at 400 DPI) and intermediates under `outputs/final_thesis_figures/intermediate/`.

| Figure | Thesis themes |
| --- | --- |
| `ridge_feature_importance.png` | ML explainability — linear baseline |
| `hgbr_permutation_importance.png` | ML explainability — non-linear permutation signal |
| `environmental_correlation_network.png` | Exposure coupling, coastal monitoring hypotheses |
| `comparative_distance_decay.png` | Exposure gradients, maritime proximity footprint |
| `temporal_persistence_heatmap.png` | Persistence of pressure / response signals in time |
| `integrated_exposure_map_*.png` | Geospatial choropleths (EPSG:3857)—composite exposure lattice + corridors |
| `coastal_exposure_hotspots.png` | Top-decile exposure overlay (association / hotspot semantics) |
| `thesis_framework_diagram.png` | Proposal alignment · pipeline narrative |

`feature_importance_table.csv` (model importance merge: Ridge coef + permutation means) sits in this folder alongside the PNG/PDF outputs. **`intermediate/`** holds correlation edges, decay aggregates, temporal medians/z-scores, `composite_exposure_dataset.parquet`, `exposure_grid.geojson`, `hotspot_cells.geojson`, `dashboard_land_exposure_cells.{parquet,geojson}`, plus per-panel `dashboard_panel_*` exports from the composite poster workflow. Coastal thematic maps plus narrative `map_summary.md` are emitted by **`scripts/generate_geospatial_coastal_exposure_maps.py`**, while the thesis-style dark infographic **`composite_land_exposure_dashboard.{png,pdf}`** is produced by **`scripts/generate_composite_land_exposure_dashboard.py`** — both invoked automatically when the ML panel parquet is present.
