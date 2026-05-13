# Research summary — radar-first coastal pipeline

Framing: **localized association** only; no causal claims.

## 1. Radar (Sentinel-1–derived proxies)
Winter months (Nov–Mar) under **coastal ≤30 km** and **≤20 km to high vessel-density cells** highlight **surface disturbance** metrics (`oil_slick_probability_t`, `detection_score`). These are **exploratory signals** (look-alikes, wind, speckle) — not confirmed slicks.

## 2. Spatial footprint
Restricting to the coastal strip and shipping-corridor distance emphasizes **short-range co-location** with chronic vessel-density structure rather than port-centric 100–1000 km axes.

## 3. Surface disturbance decay
Inspect `winter_radar_distance_decay.csv`: means often decline with distance band if the **surface disturbance** layer is lane-localized.

## 4. Atmospheric influence
`no2_mean_t` reflects **regional atmospheric** processes; expect **moderate** alignment with vessel extremes relative to SAR oil proxies.

## 5. Optical water column
NDTI / NDWI carry **cloud / season / ill-posedness** noise; treat as **weak / conditional** relative to winter radar.

## 6. Land / coastal vegetation
NDVI and `land_response_index` (if present) test **indirect** land-side response; effects are often **small and confounded** by mixed pixels.

## Artifacts
- `run_research_radar_coastal_pipeline/` — CSV tables and this file
- `run_research_radar_coastal_pipeline/` — figures listed in run log

### Empirical run snapshot (auto)
- Total rows: **42738**
- Winter + coastal + near-shipping: **440**
- Coastal + near-shipping (all year): **1122**
- Strongest |Cohen's d| (step-2 indicators): **oil_slick_probability_t** (d=0.5861)
- Smallest combined p (step-2): **oil_slick_probability_t** (p≈5.98e-07)
- Winter oil proxy mean trend vs distance (0→30 km bands): **downward (mean)**
- NO2 decay vs vessel-high pattern alignment (heuristic) — consistent sign story
