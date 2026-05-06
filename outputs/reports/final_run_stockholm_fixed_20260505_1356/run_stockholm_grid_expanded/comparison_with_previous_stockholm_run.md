# Comparison with previous Stockholm hub run

## What changed
- Previous run (`run_hub_strategy_turku_naantali_mariehamn_stockholm`) used the
  original 315-cell grid built around Mariehamn / Turku / Naantali.
- The previous Stockholm hub had **0** grid cells within 25 km, **0** within
  50 km, only **5** within 100 km, and
  the nearest grid was **81.4 km** away.
- The 0-50 km Stockholm NO2 hypothesis could not be tested.

## What's new in this run
- An expanded 0.1° grid was built around Stockholm (within 150 km of the
  Stockholm port coordinate).  Per-cell NO2, NDWI, NDTI, NDVI features were
  freshly extracted via Google Earth Engine; vessel-density is not available
  for these new cells (no HELCOM/EMODnet coverage in the cached product) and
  is reported transparently as missing.
- Stockholm now has:
    - 31 grids within 25 km
    - 124 grids within 50 km
    - 458 grids within 100 km
    - nearest grid 2.5 km
- The local 0-25 / 0-50 / 0-100 km Stockholm NO2 test is now feasible.

## Key Stockholm NO2 findings (this run)
- Stockholm NO2 decreases with distance (0-100 km):
  **True**
- Stockholm 0-50 km NO2 is higher than Mariehamn 0-50 km:
  **True**
- Urban-atmospheric NO2 interpretation:
  **SUPPORTED**

## Does this improve the thesis?
- Yes: the urban-NO2 contrast (Stockholm vs. Mariehamn) is now testable
  against actual local 0-50 km data instead of a substitute 50-200 km band.
- Vessel-density still cannot be tested at Stockholm because no public
  AIS density product is wired into this run; that limitation should be
  stated explicitly in the thesis.
- Water indices over the Stockholm cells are partially over land
  (Stockholm archipelago + mainland), so NDWI/NDTI/NDVI will look
  qualitatively different — they are reported but should be interpreted
  cautiously.

## Does this change the response sent to Florian?
- Mostly yes for NO2: the 0-50 km Stockholm test is no longer
  inconclusive — the urban-NO2 reading is **SUPPORTED**.
- The sliding-window 25 km / step 10 km smoothing within 0-100 km is now
  consistent across all three hubs.
- The honest caveats remain: tropospheric NO2 column data integrates over
  the column, vessel density at Stockholm is not available for fresh cells,
  and the analysis is association, not causality.
