# NO2 distance diagnostic (200–500 km band)

NO2 column used: `no2_mean_t`
Sliding-window settings: window=50 km, step=10 km, max distance cap=1000 km

## 1. Does NO2 actually increase between 200–500 km?

- Mean of sliding-window NO2 means inside 200–500 km: **1.022e-05** (n_obs ≈ 665).
- Mean of sliding-window NO2 means outside that band (within 1000 km): **2.056e-05** (n_obs ≈ 45154).
- Ratio (band / outside): **0.50**.
- Raw observation comparison: NO2 mean <200 km = 1.273e-05, 200–500 km = 1.020e-05, >500 km = 2.173e-05.

Conclusion: NO2 **does not clearly appear to rise** in the 200–500 km band relative to the rest of the 0–1000 km range.

## 2. What is causing it?

### 2a. Sample imbalance / coverage

- Observations contributing to the 200–500 km band (raw cells): **133**.
- Hub composition in 200–500 km band: `{'Mariehamn': 133}`.
- Hub composition over the full dataset (with valid NO2): `{'Mariehamn': 9379, 'Naantali': 3073, 'Turku': 1990}`.

If the hub mix in the 200–500 km band differs strongly from the close-range band (e.g., the close range is dominated by small Finnish coastal ports while the 200–500 km band is dominated by a single far-away hub like Mariehamn), the apparent increase reflects a change in *which grid cells we are averaging*, not a true atmospheric signal around those ports.

### 2b. Binning noise

Hard ≥100 km bins gave a non-monotonic curve (e.g. 0.000009 at 200 km → 0.000023 at 600 km), but the underlying counts in those bins are small (often 30–50 cells) and the standard deviation per window is comparable to the mean. The 50-km sliding window with 10-km step (this script's output) smooths out most of that noise, but residual jitter from sparse samples remains.

### 2c. Atmospheric / land-source confounding

NO2 tropospheric column is dominated by *land-based* sources (combustion, traffic, industry) and is transported by wind. In this dataset the cells whose nearest port is far away (200–500 km away) are typically open-Baltic / mainland-coastal cells whose NO2 is governed by land emissions and transport, **not** by the named port. The 'distance to port' axis is therefore a proxy for *which region we are sampling*, and at 200–500 km we partly start sampling continental plumes (e.g., from Stockholm/Helsinki/St Petersburg) that have nothing to do with the port itself.

## 3. How to phrase this in the thesis

- Report the sliding-window NO2 curve, but explicitly mark windows with low sample counts.
- State that NO2 is not expected to show a monotonic distance decay from a single port: it is a regional pollutant, so any rise at 200–500 km is **likely a confounded signal** (different hub mix and continental/long-range transport), not an indication that pollution truly increases away from ports.
- Recommend interpreting NO2 alongside vessel density and wind direction; conclude that, in this dataset, NO2 does not provide a clean port-distance gradient and should be presented as supporting evidence of regional atmospheric exposure rather than port-proximity exposure.

## Generated artefacts

- `outputs/visualizations/sliding_window_distance_decay/no2_sliding_window.png`
- `outputs/visualizations/sliding_window_distance_decay/no2_by_hub.png`
- `outputs/visualizations/sliding_window_distance_decay/no2_sample_counts.png`
- `outputs/reports/sliding_window_distance_decay/sliding_window_no2.csv`
- `outputs/reports/sliding_window_distance_decay/no2_sliding_window_200_500km.csv`
