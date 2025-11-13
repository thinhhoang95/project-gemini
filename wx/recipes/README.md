The master plan to download weather data can be found in `master_wx_plan.md`. 

# Detailed Step
1. **Download weather data using CDSAPI.** Note that because we do not have license to download all data fields of weather ensembles, we will use the **single re-analysis** as an example, and indirectly model the *uncertainty* through the spatial and temporal capacity drop variations.

    - Code: `download_era5.py`.
    - Inputs: dates to download, API keys.
    - Outputs: `data/wx` directory: single_hourly and pl (pressure level) for each date.

2. **Derive the convection areas.** (This is similar to the weather polygons in CBCFs).
    The goal is to derive the convection risks into three levels: low, medium and high similar to EUMETNET CBCF.

    > The script is `convection_classifier.py`

    Here’s the rule-of-thumb the code uses to turn ERA5 fields into a 0–3 convection severity:

    ### Ingredients it derives (per grid cell & time)

    * **Instability & inhibition:** `CAPE` and `|CIN|` (CIN magnitude).
    * **Moisture & rain:** total column water vapour `TCWV`, convective precip rate `precip_rate` (from `cp` in mm h⁻¹).
    * **Lift:** mid-level vertical velocity `omega500` (Pa s⁻¹; negative = ascent).
    * **Organization:** 0–6 km bulk shear `bs06` = √((u₅₀₀−u₉₂₅)² + (v₅₀₀−v₉₂₅)²).
    * **Lightning/severity proxy:** `cp_proxy = CAPE * precip_rate`.
    * **Optional wind risk:** 10 m instantaneous gust `i10fg`.

    ### Step 1 — Gate: “Is convection present?”

    Flag **cb_present = 1** if **all** of the following hold:

    * CAPE ≥ **200 J kg⁻¹** and |CIN| ≤ **75 J kg⁻¹**, and
    * either precip_rate ≥ **0.5 mm h⁻¹** **or** (omega500 ≤ **−0.3 Pa s⁻¹** and TCWV ≥ **20 kg m⁻²**).

    If this gate fails → **severity = 0 (None)**.

    ### Step 2 — Build severity additively (then clip to 0–3)

    Start at **1 (Low)** when `cb_present = 1`, then add:

    * **+1** if `cp_proxy` ≥ **400 J·mm kg⁻¹ h⁻¹** (instability × rain proxy).
    * **+1** if `bs06` ≥ **12.5 m s⁻¹** (organized storms).
    * **+1** if **either** strong ascent `omega500 ≤ −0.6` **or** very moist `TCWV ≥ 28`.
    * **Optional +1** if gusts ≥ **20 m s⁻¹** (only if gust data present and not disabled).

    Finally, **clip to 0–3**, where:

    * **0** None, **1** Low, **2** Moderate, **3** High.

    ### Outputs

    Alongside `severity`, it returns `cb_present` and the diagnostic fields (`precip_rate`, `cp_proxy`, `bs06`, `omega500`, `tcwv`, `cape`, `cin_mag`) so you can see *why* a cell landed in a given class.

    *(Default thresholds shown above; all are tunable via `Thresholds` / CLI.)*

    - Input: the single-level and pressure-level CDF files.
    - Output: `convection_severity_*.nc`


