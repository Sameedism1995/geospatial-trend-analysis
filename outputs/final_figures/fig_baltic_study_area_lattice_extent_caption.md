# Baltic study-area map (neutral lattice footprint)

## Files

- `outputs/final_figures/fig_baltic_study_area_lattice_extent.png`
- `outputs/final_figures/fig_baltic_study_area_lattice_extent.pdf`

## Caption (draft)

**Figure.** Baltic Sea analytic window (see dashed extent) showing **study lattice cell footprints** (n = **237** unique `grid_cell_id` within bounds) reconstructed from centroid coordinates and nominal resolution encoded in cell IDs (`g<deg>_…`). Light grey polygons are coastal land (**Natural Earth 110 m**); star markers denote focal harbours (**Stockholm**, **Turku**, **Mariehamn**, **Naantali**). Derived from **`data/modeling_dataset.parquet`**.

Regenerate: `python3 scripts/generate_baltic_study_area_overview_map.py`
