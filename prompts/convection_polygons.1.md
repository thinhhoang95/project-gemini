1) Inputs from ERA5 (hourly)
- Single levels:
  - CAPE: `convective_available_potential_energy` [J kg⁻¹]
  - CIN: `convective_inhibition` [J kg⁻¹] (negative; take magnitude)
  - Convective precipitation: `cp` [m per hour] (accumulated over previous hour)
  - Total column water vapour: `tcwv` [kg m⁻²]
  - 10 m wind gust (optional): `i10fg` [m s⁻¹]
  - Total cloud cover (optional)
- Pressure levels (e.g., 925, 850, 700, 500 hPa):
  - u, v wind components [m s⁻¹]
  - Vertical velocity `w` [Pa s⁻¹] (negative = ascent)

Notes
- Convert convective precipitation to a rate: $P = cp \times 1000$ [mm h⁻¹].
- $1\ \text{kg m}^{-2} = 1\ \text{mm}$ of precipitable water.

2) Derived fields
- Precip rate (convective): $P = cp \times 1000$ [mm h⁻¹].
- CP lightning/CB proxy: $CP = CAPE \times P$ [$\text{J·mm kg}^{-1}\ \text{h}^{-1}$].
- 0–6 km bulk shear (BS06) [m s⁻¹]:
  - Use vector difference between 925 and 500 hPa:
    - $\Delta u = u_{500} - u_{925}$, $\Delta v = v_{500} - v_{925}$,
    - $BS06 = \sqrt{(\Delta u)^2 + (\Delta v)^2}$.
- Vertical motion: $\omega_{500}$, $\omega_{700}$ [Pa s⁻¹] (negative = ascent).
- CIN magnitude: $CIN_{mag} = \max(-CIN, 0)$ [J kg⁻¹].

3) Classify CB-hazard (binary mask + 3-tier severity)
3.1 CB-present mask (binary)
Flag CB present if ALL of the following are true:
- Instability: $CAPE \ge 200\ \text{J kg}^{-1}$
- Inhibition: $CIN_{mag} \le 75\ \text{J kg}^{-1}$
- Moisture + forcing: either
  - Precip signal: $P \ge 0.5\ \text{mm h}^{-1}$, or
  - Lift + moisture: $\omega_{500} \le -0.3\ \text{Pa s}^{-1}$ and $TCWV \ge 20\ \text{kg m}^{-2}$

3.2 Severity score (0–3)
- Start with $S = 0$.
- If CB-present, set $S = 1$ (Low).
- Add $+1$ if $CP \ge CP70\_fallback$, with $CP70\_fallback = 400\ \text{J·mm kg}^{-1}\ \text{h}^{-1}$.
  - Rationale: typical mid-lat convective environment, e.g., $CAPE \approx 800\ \text{J kg}^{-1}$ and $P \approx 0.5\ \text{mm h}^{-1}$ yields $CP \approx 400$.
- Add $+1$ if $BS06 \ge 12.5\ \text{m s}^{-1}$ (organized convection threshold).
- Add $+1$ if $\omega_{500} \le -0.6\ \text{Pa s}^{-1}$ or $TCWV \ge 28\ \text{kg m}^{-2}$ (strong lift or very moist column).
- Clip to $S \in \{0,1,2,3\}$.
- Category mapping:
  - $S=0$: None
  - $S=1$: Low (isolated/weak CB risk)
  - $S=2$: Moderate (scattered/organized CB; likely flow impacts)
  - $S=3$: High (numerous/organized CB; strong planning signal)
- Optional severe-wind bump: add $+1$ (still clip to 3) where gusts exceed a high threshold, e.g., $i10fg \ge 20\ \text{m s}^{-1}$, if you wish to emphasize outflow wind disruption.

4) Post-process to polygons and sector impact
- Morphology:
  - Spatial smooth: e.g., binary 3×3 majority filter on the CB-present mask per hour.
  - Fill single-cell holes; remove speckles below a minimum area (e.g., 2–4 ERA5 grid cells).
  - Dissolve contiguous cells by severity rank into polygons; optionally buffer by one grid spacing to avoid sliver gaps.
- Timing:
  - Generate hourly polygons for D0/D1 horizons.
  - Apply a 3-hour rolling maximum in severity to reduce flicker in planning layers.
- Sector scoring:
  - Intersect polygons with ACC sectors/FABs/TMAs.
  - Compute percent area by severity within each sector per hour.
  - Flag “at risk” if Moderate+High coverage ≥ 25% (tuneable to 20–30% depending on operations).

5) Minimal, corrected code sketch (xarray)
```python
import xarray as xr, numpy as np

# Load ERA5
ds_sfc = xr.open_mfdataset("era5_single_hourly_*.nc")   # CAPE, CIN, cp, tcwv, i10fg
ds_pl  = xr.open_mfdataset("era5_pressure_hourly_*.nc") # u, v, w on 925/850/700/500 hPa

# Fields and units
cape = ds_sfc['cape']                          # J/kg
cin  = ds_sfc['cin']                           # J/kg, negative -> take magnitude
cin_mag = (-cin).clip(min=0)                   # J/kg
P    = (ds_sfc['cp'] * 1000.0).clip(min=0)     # mm/h (cp is m/h accumulated over prev hour)
tcwv = ds_sfc['tcwv']                          # kg/m^2
gust = ds_sfc.get('i10fg')                     # optional, m/s

u500, v500 = ds_pl['u'].sel(level=500), ds_pl['v'].sel(level=500)
u925, v925 = ds_pl['u'].sel(level=925), ds_pl['v'].sel(level=925)
bs06 = np.hypot(u500 - u925, v500 - v925)      # m/s

w500 = ds_pl['w'].sel(level=500)               # Pa/s (neg = ascent)
w700 = ds_pl['w'].sel(level=700)               # Pa/s (optional)

# CP proxy
CP = cape * P                                  # J·mm kg^-1 h^-1

# Tunable constants (fallbacks)
CP70_FALLBACK   = 400.0      # J·mm kg^-1 h^-1
CAPE_MIN        = 200.0      # J/kg
CIN_MAX         = 75.0       # J/kg (magnitude)
P_MIN           = 0.5        # mm/h
OMEGA_PRESENT   = -0.3       # Pa/s
TCWV_PRESENT    = 20.0       # kg/m^2
BS06_ORGANIZED  = 12.5       # m/s
OMEGA_STRONG    = -0.6       # Pa/s
TCWV_VERY_MOIST = 28.0       # kg/m^2
GUST_SEVERE     = 20.0       # m/s (optional)

# CB-present mask
cb_present = (
    (cape >= CAPE_MIN) &
    (cin_mag <= CIN_MAX) &
    (
        (P >= P_MIN) |
        ((w500 <= OMEGA_PRESENT) & (tcwv >= TCWV_PRESENT))
    )
)

# Severity (0–3)
S = xr.zeros_like(CP, dtype=np.int8)
S = xr.where(cb_present, 1, 0)  # 1 = Low when CB-present

S = xr.where(cb_present & (CP >= CP70_FALLBACK), S + 1, S)
S = xr.where(cb_present & (bs06 >= BS06_ORGANIZED), S + 1, S)
S = xr.where(cb_present & ((w500 <= OMEGA_STRONG) | (tcwv >= TCWV_VERY_MOIST)), S + 1, S)

if gust is not None:
    S = xr.where(cb_present & (gust >= GUST_SEVERE), S + 1, S)

severity = S.clip(0, 3)  # 0=None, 1=Low, 2=Moderate, 3=High

# Next steps (not shown): spatial smoothing, polygonization, sector overlap and coverage %, 3-hr rolling max.
```

6) Final threshold summary (consistent units)
- Instability:
  - $CAPE \ge 200\ \text{J kg}^{-1}$ (present)
  - $CIN_{mag} \le 75\ \text{J kg}^{-1}$ (present)
- Moisture/forcing:
  - Present if $P \ge 0.5\ \text{mm h}^{-1}$, or $\omega_{500} \le -0.3\ \text{Pa s}^{-1}$ and $TCWV \ge 20\ \text{kg m}^{-2}$
  - Severity bump if $\omega_{500} \le -0.6\ \text{Pa s}^{-1}$ or $TCWV \ge 28\ \text{kg m}^{-2}$
- Organization:
  - $BS06 \ge 12.5\ \text{m s}^{-1}$ (severity bump)
- Activity proxy:
  - $CP \ge 400\ \text{J·mm kg}^{-1}\ \text{h}^{-1}$ (severity bump; fallback for missing 70th percentile)
- Optional wind-impact emphasis:
  - $i10fg \ge 20\ \text{m s}^{-1}$ (severity bump, still clipped to 3)

Practical tuning notes
- CP70_fallback: If you later obtain a limited archive, recompute a monthly, hour-of-day percentile over a multi-year set and replace the constant. As a quick regional adjustment:
  - Cooler seasons/high latitudes: consider $CP70\_fallback \approx 250$.
  - Warm season/low latitudes: consider $CP70\_fallback \approx 600$–$800$.
- If false alarms occur in dry dynamics (strong ascent but dry), tighten to $TCWV \ge 22\ \text{kg m}^{-2}$ in the CB-present test.
- If missing convective precipitation, you may broaden the lift test to include $\omega_{700}$ with a weaker threshold, e.g., $\omega_{700} \le -0.2\ \text{Pa s}^{-1}$.