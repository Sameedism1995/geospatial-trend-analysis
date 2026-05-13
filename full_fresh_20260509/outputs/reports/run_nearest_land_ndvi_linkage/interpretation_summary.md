# Nearest-land NDVI linkage — interpretation

## Why nearest-land linkage was used  
Many maritime/coastal-water grid cells rarely carry valid **pixel NDVI**. To still ask whether **vegetation signals** relate to **maritime exposure**, each eligible maritime centroid is paired with the **closest land-associated grid** that has aggregated NDVI (coast ≤30 km).

## Candidate land cells  
Land NDVI candidates (non-null NDVI & coast ≤30 km): **127** distinct `grid_cell_id` cells (see `land_ndvi_candidate_cells.csv`).

## Maritime linkage  
Maritime/coastal grids (coast ≤50 km and vessel density or oil proxy present): **375** linked to a nearest land cell (`maritime_to_nearest_land_linkage.csv`).  

Median **distance** from maritime centroid to paired land centroid: **11.12 km** (nan if empty).

## Shipping-distance structure  
Bands use `distance_to_nearest_high_vessel_density_cell` (0–3, 3–7, 7–15, 15–30 km). Inspect `nearest_land_distance_decay.csv` for whether linked **nearest_land_ndvi_mean** / **nearest_land_ndvi_median** shift across bands versus **NO₂** and **oil slick** proxy.

## High vs low vessel exposure  
Among linked rows with coast ≤50 km, **top vs bottom decile on log1p(vessel_density_t)** (see `nearest_land_high_vs_low_vessel.csv`).

Strongest explored **Cohen's d** (among those comparisons): **nearest_land_ndvi_median** (|d|≈**0.7475**).

**Automated land-impact flag:** **YES** — YES if any comparison shows |d|>0.05 with Welch or Mann–Whitney p<0.05; INCONCLUSIVE if groups too small / no tests; NO otherwise.

## Sea → land correlations  
Spearman checks on linked weekly rows: `sea_land_correlation_check.csv` (nearest-land NDVI mean vs oil, detection, NO₂, vessel density, lane distance).

## Limitations  
- Nearest land NDVI is a **spatial proxy**, not hydrological or dispersion modeling.  
- NDVI is **seasonal**, **cloud-affected**, and aggregated at coarse grid/week resolution.  
- Results are **associations only**, not causality.  
- A stronger approach is **dedicated Sentinel-2 land-mask extraction along oriented buffers** oriented from shore outward.

Main research question: *Can nearby land vegetation response be linked to maritime exposure by connecting coastal marine cells to the nearest valid land NDVI observations?*  

Answer here is framed by the magnitudes/signs in CSVs/plots—not a causal claim.

Generated artifact: **`nearest_land_ndvi_linked_dataset.parquet`** merges linkage columns onto the full weekly table by `grid_cell_id`.
