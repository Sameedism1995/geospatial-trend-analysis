# Florian wind-data concern — thesis methodology audit

**Document reviewed:** `outputs/thesis_strengthening/Thesis_Final_Draft_revised.docx`  
**Pipeline reference:** `src/analysis/run_coastal_wind_transport.py`, `src/analysis/run_land_pollution_drivers_wind.py`, `src/analysis/run_coastal_exposure_analysis.py`

---

## Executive summary

**Section 3.3.4** already documents most of Florian’s checklist (Open-Meteo ERA5 Archive, hourly native resolution, U/V weekly aggregation, landward bearing, ±45° / cos threshold, shoreward/non-shoreward shares). Gaps: explicit **wind-toward** conversion from meteorological FROM direction; a **corrupted domain sentence** (full-grid latitudes); clarification that **n = 30** refers to **unique coastal-panel cells**, not cell-weeks; and an explicit **reanalysis / non-in-situ limitation** (only partly implied via the SMHI/FMI comparison).

**Section 4.5.3** is **under-specified and partially inconsistent** with the implemented framework: the alignment equation is not printed, symbols refer to “coastline orientation” rather than **landward bearing**, and hourly→weekly / ERA5 / spatial clustering are not restated. This section should be rewritten.

**Discussion / limitations (Chapter 6):** Wind is discussed interpretively (§6.2) but **§6.8 does not state** that ERA5 reanalysis winds are regional-scale transport indicators, not direct coastal in-situ measurements. A short limitation paragraph is recommended.

---

## Checklist against thesis text

| Item | §3.3.4 | §4.5.3 | Ch. 6 limitations | Pipeline |
|------|--------|--------|-------------------|----------|
| Exact wind data source | ✅ Open-Meteo ERA5 Archive API | ❌ Not stated | — | `archive-api.open-meteo.com/v1/archive` |
| Provider / ECMWF ERA5 | ✅ | ❌ | — | Documented in code comments |
| Variables: 10 m speed & direction | ✅ `wind_speed_10m`, `wind_direction_10m` | ❌ | — | Hourly request in `fetch_open_meteo_cluster_wind` |
| Native temporal resolution: hourly | ✅ | ❌ | — | Hourly API → weekly groupby |
| Hourly → weekly aggregation | ✅ U/V mean per ISO week | ❌ | — | `fetch_open_meteo_cluster_wind` |
| Spatial selection (grid/centroid) | ⚠️ Says single cluster; code uses median or k-means (here: **1 cluster** for n≈29 cells) | ❌ | — | `build_wind_clusters` |
| Met FROM → wind-toward conversion | ⚠️ Implied only in classification text | ❌ | — | `wind_uv_from_open_meteo` → `wind_to_direction_deg` |
| Landward / coastline bearing | ✅ NE 1:110m nearest coast | ⚠️ “Coastline orientation” (wording) | — | `initial_bearing_deg` cell → coast ref |
| Shoreward vs non-shoreward rule | ✅ cos(Δθ) ≥ cos(45°) | ⚠️ Qualitative only | — | `coastal_wind_alignment_score` |
| Threshold ±45° | ✅ | ❌ | — | `COS_45 = cos(45°)` |
| % shoreward / non-shoreward | ⚠️ 29.3% / 70.7% (recompute: **29.9% / 70.1%**, n=1,479 cell-weeks, 29 cells) | ❌ | — | Coastal panel `coastal_panel` mask |
| Reanalysis / not in-situ limitation | ⚠️ Implicit (vs SMHI/FMI) | ❌ | ❌ | Should be stated explicitly |

---

## Issues to fix (no change to scientific results)

1. **§3.3.4, paragraph 1 — domain coordinates:** Text reads “latitudes -0.75³-63.05°N” (corrupted). That span is the **full lattice extent**, not the Baltic corridor used for interpretation. Either remove or replace with the verified corridor (§3.2.1) or the coastal-panel footprint.
2. **§3.3.4 — wind-toward:** Add one sentence: meteorological FROM → u,v → toward = atan2(u,v).
3. **§3.3.4 — n = 30:** Clarify **29–30 unique grid cells** × 51 weeks ≈ **1,479 coastal-panel cell-weeks** (not “30 observations”).
4. **§3.3.4 / §6.8 — reanalysis limitation:** State clearly that ERA5 reanalysis is a **gridded regional transport indicator**, not vessel-level or quayside in-situ wind.
5. **§4.5.3 — replace** placeholder “Where: θ_wind … θ_coastline” with the implemented score and classification (below).
6. **Shoreward percentages:** Thesis gives 29.3%; current pipeline on the same panel gives **442 / 1,479 = 29.9%**. Wording “approximately 30%” is defensible; exact share may vary slightly with API gaps.

---

## Proposed replacement — Section 3.3.4 (full subsection)

*Replace existing §3.3.4 body paragraphs (keep heading) with:*

### 3.3.4 Wind and Meteorological Data

Wind fields were obtained from the **Open-Meteo ERA5 Archive** ([https://archive-api.open-meteo.com/v1/archive](https://archive-api.open-meteo.com/v1/archive)), which serves historical **ECMWF ERA5** reanalysis. The variables requested at each query point were **10 m wind speed** (`wind_speed_10m`, m s⁻¹) and **10 m wind direction** (`wind_direction_10m`, degrees), provided at the product’s native **hourly** resolution.

For the **coastal wind-regime panel**, grid cells were retained when distance to the Natural Earth coastline was ≤ 30 km and distance to the nearest weekly high vessel-density seed cell was < 30 km (the same coastal×shipping mask used in `run_coastal_wind_transport.py`). In the 2023 panel this yielded **29 unique grid cells** and **1,479 cell-week records** after temporal alignment. To limit API volume while preserving spatial representativeness, those cells were assigned to a **single wind cluster** (k-means reduces to one centroid when the coastal cell count is small); hourly ERA5 data were retrieved at the cluster representative coordinates and **broadcast to all panel cells** in that cluster for each week.

**Temporal aggregation (hourly → weekly):** For each hour, meteorological direction (wind **from** which direction) and speed were converted to zonal and meridional components with **u positive eastward** and **v positive northward**. Components were averaged over all valid hours in each **ISO calendar week** (Monday 00:00 UTC–Sunday 23:59 UTC). Weekly mean speed and **wind-toward direction** were reconstructed from the mean vector: wind-toward bearing θ_to = atan2(u, v) (degrees clockwise from north, direction toward which the wind blows). Hours with missing speed or direction were excluded list-wise; no separate calm-wind speed filter was applied because shoreward classification uses vector geometry.

**Landward bearing:** For each grid centroid, the **landward bearing** θ_landward was computed as the forward azimuth from the cell to its nearest **Natural Earth 1:110m** coastline sample (great-circle initial bearing). This is a cell-specific onshore-normal direction, not a single regional coastline orientation.

**Shoreward classification:** Let Δθ be the smallest angular difference between θ_to and θ_landward. A cell-week was classified as **shoreward** when

\[
\cos(\Delta\theta) \geq \cos(45^\circ) \approx 0.707,
\]

i.e. when the wind-toward vector lay within a **±45°** sector centred on the landward bearing; otherwise it was **non-shoreward**. In the coastal panel under this rule, **442 of 1,479 cell-weeks (29.9%)** were shoreward and **70.1%** non-shoreward. These labels describe **regional reanalysis transport geometry** relative to the coast; they are not interpreted as causal drivers of vessel traffic.

**Limitation:** ERA5 data are **model reanalysis** on a ~0.25° grid, not direct in-situ observations at ferry berths or coastal stations. They are suited to **directional stratification** of satellite- and AIS-based exposure indicators at the corridor scale, but they cannot resolve micro-scale harbour winds or verify local advection paths. Station archives (e.g. SMHI, FMI) were not used as the primary source because offshore grid cells lack consistent station coverage; Open-Meteo was selected for programmatic, spatially continuous weekly ingestion aligned with the panel workflow.

---

## Proposed replacement — Section 4.5.3 (full subsection)

*Replace existing §4.5.3 (from heading through end of wind-classification paragraph, before §4.5.4 MEI) with:*

### 4.5.3 Wind-Regime Exposure Framework

The wind-regime framework stratifies weekly observations by whether ERA5-derived transport is **oriented toward** or **away from** the nearest coastline, as a cautious indicator of whether maritime emissions are more likely to be advected toward coastal measurement zones. It does **not** assert that wind regimes influence vessel scheduling, traffic volume, or optical water-quality processes.

**Inputs:** Weekly mean zonal and meridional wind components from the Open-Meteo ERA5 Archive (Section 3.3.4), merged to each grid-week in the coastal panel. **Landward geometry** uses the bearing from each cell centroid to the nearest Natural Earth coastline reference point.

**Wind-toward direction:** Meteorological API direction describes the direction **from** which wind blows. Components were formed as u = −speed·sin(from), v = −speed·cos(from), and wind-toward direction θ_to = atan2(u, v).

**Alignment score:** The directional alignment between wind transport and the landward normal was summarised as

\[
\text{coastal\_wind\_alignment\_score} = \cos(\Delta\theta),
\]

where Δθ is the smallest angle between θ_to and θ_landward (both in degrees clockwise from north). Scores near **+1** indicate wind blowing toward the coast along the landward bearing; scores near **0** indicate cross-shore or oblique transport; negative values indicate offshore-directed components relative to that bearing.

**Binary regimes:** Observations with coastal_wind_alignment_score ≥ cos(45°) were labelled **shoreward**; all other valid observations were **non-shoreward**. This threshold matches the ±45° sector used in Section 3.3.4. The binary flag was used for comparative plots (e.g. Section 5.4) and for stratified Mann–Whitney summaries; it was **not** treated as evidence that wind forces changes in vessel density (consistent with the null vessel-density difference by wind class reported in Section 5.4).

**Role in exposure indices:** Shoreward transport enters the **Maritime Exposure Index (MEI)** and **Atmospheric Coastal Exposure Index (ACEI)** as a ranked wind-alignment factor (Section 4.5.4–4.5.5) and as a ranked term in the experimental **Environmental Stress Index (ESI)**. All uses are **associative** and **exploratory**; reanalysis wind cannot validate cell-scale causation or replace in-situ meteorological monitoring.

---

## Proposed addition — Section 6.8 Limitations (new paragraph)

*Insert after the opening limitation paragraphs in §6.8 (e.g. after Stockholm exclusion text):*

**Wind and meteorological data.** Directional wind regimes were derived from **ECMWF ERA5 reanalysis** via the Open-Meteo Archive at ~10 m height and hourly resolution, aggregated to weekly mean vectors and applied to a spatially clustered coastal panel. These fields represent **regional-scale atmospheric transport indicators**, not in-situ wind measurements at ports or monitoring stations. ERA5 cannot resolve local topographic channeling, sea-land breeze cycles at the grid scale, or ship-stack plume dispersion. Shoreward/non-shoreward labels therefore describe **consistent reanalysis geometry relative to the coast** under the ±45° rule, and should not be read as proof of pollutant transport paths or as causal modifiers of maritime activity. Future work could compare ERA5 stratification with SMHI/FMI station records where coastal coverage permits.

---

## Code traceability (for examiner / Florian)

| Step | Implementation |
|------|----------------|
| Hourly ERA5 fetch | `fetch_open_meteo_cluster_wind()` in `run_land_pollution_drivers_wind.py` |
| FROM → u,v | `wind_uv_from_open_meteo()` |
| Weekly mean u,v | `groupby(week_start_utc).mean()` on hourly table |
| Wind-toward | `wind_to_direction_deg(u, v)` |
| Nearest coast point | `grid_nearest_coast_reference_table()` + Natural Earth samples |
| Landward bearing | `initial_bearing_deg(cell_lat, cell_lon, coast_ref_lat, coast_ref_lon)` |
| Δθ and score | `smallest_angle_deg` → `cos(radians(Δθ))` |
| Shoreward flag | `score >= cos(45°)` → `coastal_wind_shoreward_45deg` |
| Coastal panel mask | `distance_to_coast_km ≤ 30` & shipping distance < 30 km |

---

*Audit generated from thesis docx + pipeline recomputation (May 2026). Do not treat as automated docx edit; paste replacements manually after supervisor review.*
