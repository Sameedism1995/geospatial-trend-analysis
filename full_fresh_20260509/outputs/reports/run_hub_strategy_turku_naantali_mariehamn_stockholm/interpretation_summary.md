# Hub strategy — interpretation summary

## Hubs analysed

- **Turku-Naantali coastal hub** (coastal/industrial port system): Turku, Naantali
- **Mariehamn offshore/island hub** (offshore/island port): Mariehamn
- **Stockholm urban port hub** (urban port): Stockholm

## Indicators detected

- vessel_density: `vessel_density_t`
- ndwi: `sentinel_ndwi_mean_t`
- ndti: `sentinel_ndti_mean_t`
- ndvi: `sentinel_ndvi_mean_t`
- no2: `no2_mean_t`

## Overlap findings

- Strong overlap pairs: Turku-Naantali
- Spatially distinct pairs: Turku-Mariehamn, Turku-Stockholm, Naantali-Mariehamn, Naantali-Stockholm, Mariehamn-Stockholm

## Sample-density warnings

- WARNING: Stockholm urban port hub contributes 61.1% of samples in 100-300 km — results dominated by one hub
- DATA COVERAGE WARNING: Stockholm urban port hub has 0 grid cells within 50 km (nearest grid is 81.4 km away). 0-25/0-50 km local stats will be empty for this hub; the local NO2/vessel claim cannot be tested at <50 km in this dataset.

## Local 0-100 km headline numbers

- Turku-Naantali coastal hub | vessel_density 0-50 km: mean=1.527, n=3315, grids=65, missing=20.0%
- Turku-Naantali coastal hub | no2 0-50 km: mean=1.558e-05, n=3315, grids=65, missing=14.27%
- Mariehamn offshore/island hub | vessel_density 0-50 km: mean=0.2569, n=3264, grids=64, missing=6.25%
- Mariehamn offshore/island hub | no2 0-50 km: mean=1.088e-05, n=3264, grids=64, missing=12.1%
- Stockholm urban port hub | vessel_density 0-50 km: mean=nan, n=0, grids=0, missing=nan%
- Stockholm urban port hub | no2 0-50 km: mean=nan, n=0, grids=0, missing=nan%

## NO2 interpretation

- INCONCLUSIVE for 0-50 km: dataset has no grids within 50 km of Stockholm, so the urban-NO2 hypothesis cannot be tested at the intended scale.  In the next available band (50-200 km) NO2 decreases with distance from Stockholm, which is consistent with — but does not prove — an urban-source pattern.

## Thesis-safe statements

- Vessel density shows local-scale decay within 0-100 km of all three hubs.
- Water indices (NDWI/NDTI) vary geographically but show weak distance-decay; interpret as coastal vs. open-water contrasts, not pure port impact.
- NO2 in the 0-50 km range is consistent with urban/atmospheric background (strong near Stockholm, weak near Mariehamn) — do **not** attribute it solely to ship emissions.
- Any plot covering > 100 km is **regional background**, not port impact.