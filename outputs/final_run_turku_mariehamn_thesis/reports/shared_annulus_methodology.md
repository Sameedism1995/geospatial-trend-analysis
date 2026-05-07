# Shared-valid-annulus methodology (Turku vs Mariehamn)

## Why two distance modes?

### Fixed-band distance decay
All port-centred annuli in the standard list are **always** retained: **0–3 km (coastal core)**, **0–3 km**, **3–7 km**, **7–15 km**, **15–30 km**. Empty or undefined strata appear as **gaps** in the curves or missing bootstrap intervals — **no imputation**.

### Shared-valid-annulus cross-port comparison
For **shoreward vs non-shoreward comparison between Turku and Mariehamn**, forcing one fixed band for every composite can mislead when coverage or valid rank-inputs differ by band (e.g. maritime exposure needs intra-week ranks of vessel density and wind alignment). Comparing bars without aligning supported annuli implies comparability the grid may not support.

So **per composite indicator**, we take the **nearest** annulus in order **0–3 km → 3–7 km → 7–15 km → 15–30 km** such that **both** ports have **shoreward** and **non-shoreward** strata with **n ≥ 1** in the decay table. If none qualify, that panel is **n/a** — **no fabricated values**.

## Data integrity

- Missing bands remain visible in **fixed-band** decay figures.
- Selections are logged in `shared_annulus_selection.csv`.
- **Stockholm** is excluded from this thesis comparative bundle.
