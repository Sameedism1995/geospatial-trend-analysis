# Interpretation — coastal shipping impact

## Scope
We restrict to **within 30 km of the modeled coastline** to represent near-shore grids where terrestrial and maritime signals mix and societal relevance is strongest.

Shipping pressure is summarized by distance to cells flagged as **high vessel density** in the spatial proxy (**≤ 20 km** for the focal cross-section), emphasizing **lanes and shipping corridors** rather than port-centric 100–1000 km axes.

## Indicators used
Supporting water variables (**NDWI, NDVI**) are reported in group comparisons only; **the composite coastal impact score** favors **NO₂, NDTI, oil slick probability**, and **detection score** with stated weights.

Strongest contrasts (Welch / Mann–Whitney among high vs low exposure, coastal + lane-proximity) are listed numerically in `high_vs_low_vessel_coastal_comparison.csv`.

## Distance decay from shipping corridors
Bands use **distance to nearest high vessel-density grid** under the **coastal strip** constraint. Means/medians decline with increasing distance **if atmospheric / slick signals are localized** near dense traffic zones.

## Limitations
1. **Vessel density fields** approximate chronic spatial congestion; they are **not weekly AIS-derived traffic counts**.
2. **Sentinel‑2-derived water indices** exhibit **calendar missingness / cloud avoidance** unrelated to maritime activity.
3. **Oil slick proxy** is exploratory SAR interpretation — **not** verification of spilled oil without field data.
4. **No causal claims** absent wind/current decorrelation and matched temporal AIS / emissions controls.

Artifacts for this interpretation live under `outputs/reports/run_final_coastal_shipping_impact/`.
