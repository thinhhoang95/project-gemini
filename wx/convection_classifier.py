"""
Convection hazard classification from ERA5.

Overview
--------
This module loads ERA5 single-level and pressure-level fields, derives a
set of convective ingredients (e.g., CAPE, CIN, convective precip rate,
0–6 km bulk shear, vertical velocity, moisture), and applies a rule-based
scheme to classify each grid cell into a discrete 0–3 severity scale.

Severity scale
--------------
    0 = None
    1 = Low
    2 = Moderate
    3 = High

Inputs
------
- Single-level ERA5 files with variables such as ``cape``, ``cin``, ``cp``
  (convective precipitation), ``tcwv`` (total column water vapour), and
  optionally ``i10fg`` (instantaneous 10 m gusts).
- Pressure-level ERA5 files with wind components ``u``/``v`` and vertical
  velocity ``w`` at standard pressure levels (uses 925 hPa and 500 hPa).

Key outputs
-----------
- ``severity`` (int8): hazard category 0–3
- ``cb_present`` (int8): binary mask for convective presence
- Derived diagnostic fields (precip rate, CAPE, CIN magnitude, bulk shear,
  omega at 500 hPa, TCWV, and a CAPE*precip proxy)

CLI
---
Run the module as a script to:
1) discover and load input NetCDF files via glob patterns,
2) classify convection severity,
3) write an annotated NetCDF (see ``--output``), and
4) print summary statistics.

Notes
-----
- ERA5 download recipes are outlined in ``wx/recipes/`` and the
  classification rule set is described in ``prompts/convection_polygons.1.md``.
- Single-level inputs from ECMWF may arrive as ``.nc`` zip archives containing
  multiple stepType members; the loader transparently extracts and merges them.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable, Sequence

import json
import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thresholds:
    """Collection of tunable thresholds used in the convection risk logic."""

    cape_min: float = 200.0
    cin_max: float = 75.0
    precip_min: float = 0.5
    omega_present: float = -0.3
    tcwv_present: float = 20.0
    cp70_fallback: float = 400.0
    bs06_organized: float = 12.5
    omega_strong: float = -0.6
    tcwv_very_moist: float = 28.0
    gust_severe: float = 20.0


SEVERITY_LABELS = {
    0: "None",
    1: "Low",
    2: "Moderate",
    3: "High",
}


def _open_single_level_file(path: Path) -> xr.Dataset:
    """
    Open a single-level ERA5 file.

    ECMWF can deliver these files as zip archives (with an .nc extension)
    containing multiple stepType files. We transparently extract and merge
    those members on the fly, returning an in-memory dataset.
    """

    # Basic existence check before attempting to open/inspect the file
    if not path.exists():
        raise FileNotFoundError(path)

    # Handle the ECMWF "NetCDF-in-zip" delivery format by extracting
    # members to a temporary directory and merging them into one Dataset.
    if zipfile.is_zipfile(path):
        LOGGER.debug("Extracting zipped single-level file: %s", path)
        with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as tmpdir:
            datasets: list[xr.Dataset] = []
            for member in zf.namelist():
                if not member.endswith(".nc"):
                    continue
                extracted = Path(tmpdir) / member
                extracted.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(extracted, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                ds = xr.open_dataset(extracted, engine="netcdf4")
                datasets.append(ds.load())
        if not datasets:
            raise ValueError(f"No NetCDF members found inside archive: {path}")
        return xr.merge(datasets, compat="override")

    LOGGER.debug("Opening single-level NetCDF: %s", path)
    return xr.open_dataset(path, engine="netcdf4").load()


def _open_pressure_level_file(path: Path) -> xr.Dataset:
    """Open a pressure-level ERA5 NetCDF file."""

    if not path.exists():
        raise FileNotFoundError(path)

    LOGGER.debug("Opening pressure-level NetCDF: %s", path)
    return xr.open_dataset(path, engine="netcdf4").load()


def _combine_datasets(datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    """
    Concatenate a sequence of datasets along valid_time, dropping duplicates.
    """

    if not datasets:
        raise ValueError("No datasets to combine.")
    if len(datasets) == 1:
        return datasets[0]

    # Concatenate along the time axis, sort, then drop duplicate timestamps
    # using the first occurrence to ensure strictly increasing valid_time.
    combined = xr.concat(datasets, dim="valid_time")
    combined = combined.sortby("valid_time")
    valid_time = combined["valid_time"].values
    _, unique_idx = np.unique(valid_time, return_index=True)
    combined = combined.isel(valid_time=np.sort(unique_idx))
    return combined


def _extract_date_from_path(path: Path) -> str:
    """Extract a YYYYMMDD token from the file stem."""

    match = re.search(r"(\d{8})", path.stem)
    if not match:
        raise ValueError(f"Could not determine date token in filename: {path}")
    return match.group(1)


def _group_paths_by_date(paths: Sequence[Path]) -> dict[str, list[Path]]:
    """Group a sequence of Paths by their YYYYMMDD token."""

    grouped: dict[str, list[Path]] = {}
    for path in sorted(paths):
        date_token = _extract_date_from_path(path)
        grouped.setdefault(date_token, []).append(path)
    return grouped


def _resolve_output_path(base: Path, date_token: str, multiple_dates: bool) -> Path:
    """Derive the destination path for a given date."""

    if not multiple_dates:
        return base

    if base.is_dir():
        return base / f"{date_token}.nc"

    if base.suffix:
        return base.with_name(f"{base.stem}_{date_token}{base.suffix}")

    # Treat suffix-less paths as directories when multiplexing dates.
    return base / f"{date_token}.nc"


def load_single_level_dataset(paths: Iterable[Path]) -> xr.Dataset:
    """Load and concatenate ERA5 single-level datasets."""

    # Map each path through the specialized opener (zip-aware), then combine.
    datasets = [_open_single_level_file(Path(path)) for path in sorted(paths)]
    LOGGER.info("Loaded %d single-level files.", len(datasets))
    return _combine_datasets(datasets)


def load_pressure_level_dataset(paths: Iterable[Path]) -> xr.Dataset:
    """Load and concatenate ERA5 pressure-level datasets."""

    # Use the simpler pressure-level opener and combine chronologically.
    datasets = [_open_pressure_level_file(Path(path)) for path in sorted(paths)]
    LOGGER.info("Loaded %d pressure-level files.", len(datasets))
    return _combine_datasets(datasets)


def classify_convection(
    ds_single: xr.Dataset,
    ds_pressure: xr.Dataset,
    thresholds: Thresholds,
    *,
    apply_gust_bump: bool = True,
) -> xr.Dataset:
    """
    Classify convection severity for each grid cell across time.

    Returns an xarray Dataset containing:
        - severity (int8): 0=None, 1=Low, 2=Moderate, 3=High
        - cb_present (int8): binary mask (0/1)
        - precip_rate (float32): convective precipitation rate (mm h-1)
        - cp_proxy (float32): CAPE * precip rate proxy
        - bs06 (float32): 0–6 km bulk shear magnitude (m s-1)
        - omega500 (float32): vertical velocity at 500 hPa (Pa s-1)
        - tcwv (float32): total column water vapour (kg m-2)
        - cape (float32): convective available potential energy (J kg-1)
        - cin_mag (float32): magnitude of convective inhibition (J kg-1)
    """

    # Align inputs on shared coordinates (time/space), keeping only common points
    ds_single, ds_pressure = xr.align(ds_single, ds_pressure, join="inner")

    # Core single-level ingredients
    cape = ds_single["cape"].fillna(0.0)
    cin = ds_single["cin"]
    cin_mag = xr.where(cin >= 0.0, cin, -cin).fillna(np.inf)

    # Convert convective precipitation accumulation (m per hour) to mm h-1 and enforce non-negative
    precip_rate = ds_single["cp"].fillna(0.0) * 1000.0
    precip_rate = precip_rate.clip(min=0.0)

    tcwv = ds_single["tcwv"].fillna(0.0)
    gust = ds_single.get("i10fg")
    if gust is not None:
        gust = gust.fillna(0.0)

    # Pressure-level wind components for shear and mid-level vertical motion
    u500 = ds_pressure["u"].sel(pressure_level=500.0)
    v500 = ds_pressure["v"].sel(pressure_level=500.0)
    u925 = ds_pressure["u"].sel(pressure_level=925.0)
    v925 = ds_pressure["v"].sel(pressure_level=925.0)
    bs06 = np.hypot(u500 - u925, v500 - v925)  # 0–6 km bulk shear magnitude

    omega500 = ds_pressure["w"].sel(pressure_level=500.0).fillna(0.0)

    # Simple lightning/severity proxy: product of instability and precip rate
    cp_proxy = cape * precip_rate

    # Convective presence gate: sufficient CAPE, manageable CIN, and either
    # active precipitation or supportive ascent/moisture background.
    cb_present = (
        (cape >= thresholds.cape_min)
        & (cin_mag <= thresholds.cin_max)
        & (
            (precip_rate >= thresholds.precip_min)
            | ((omega500 <= thresholds.omega_present) & (tcwv >= thresholds.tcwv_present))
        )
    )

    # Build categorical severity via additive contributions, then clip to [0, 3]
    severity = xr.zeros_like(precip_rate, dtype=np.int8)
    severity = xr.where(cb_present, 1, severity)
    severity = xr.where(cb_present & (cp_proxy >= thresholds.cp70_fallback), severity + 1, severity)
    severity = xr.where(cb_present & (bs06 >= thresholds.bs06_organized), severity + 1, severity)
    severity = xr.where(
        cb_present & ((omega500 <= thresholds.omega_strong) | (tcwv >= thresholds.tcwv_very_moist)),
        severity + 1,
        severity,
    )

    if apply_gust_bump and gust is not None:
        # Optional bump for severe near-surface gust potential
        severity = xr.where(cb_present & (gust >= thresholds.gust_severe), severity + 1, severity)

    severity = severity.clip(min=0, max=3).astype(np.int8)
    cb_mask = cb_present.astype(np.int8)

    # Assemble result dataset with consistently typed variables and metadata
    hazard = xr.Dataset(
        data_vars={
            "severity": severity,
            "cb_present": cb_mask,
            "precip_rate": precip_rate.astype(np.float32),
            "cp_proxy": cp_proxy.astype(np.float32),
            "bs06": bs06.astype(np.float32),
            "omega500": omega500.astype(np.float32),
            "tcwv": tcwv.astype(np.float32),
            "cape": cape.astype(np.float32),
            "cin_mag": cin_mag.astype(np.float32),
        }
    )

    # Human-friendly attributes for downstream tooling / visualization
    hazard["severity"].attrs.update(
        {
            "long_name": "Convection hazard severity",
            "description": "0=None, 1=Low, 2=Moderate, 3=High",
            "categories": json.dumps(SEVERITY_LABELS),
        }
    )
    hazard["cb_present"].attrs.update(
        {
            "long_name": "Convective cloud (CB) presence mask",
            "description": "1 where convection is flagged present, 0 elsewhere",
        }
    )
    hazard["precip_rate"].attrs.update(
        {
            "units": "mm h-1",
            "long_name": "Convective precipitation rate",
        }
    )
    hazard["cp_proxy"].attrs.update(
        {
            "units": "J mm kg-1 h-1",
            "long_name": "CAPE-precipitation lightning proxy",
        }
    )
    hazard["bs06"].attrs.update(
        {
            "units": "m s-1",
            "long_name": "0-6 km bulk shear magnitude",
        }
    )
    hazard["omega500"].attrs.update(
        {
            "units": "Pa s-1",
            "long_name": "Vertical velocity at 500 hPa (negative = ascent)",
        }
    )
    hazard["tcwv"].attrs.update(
        {
            "units": "kg m-2",
            "long_name": "Total column water vapour",
        }
    )
    hazard["cape"].attrs.update({"units": "J kg-1", "long_name": "Convective available potential energy"})
    hazard["cin_mag"].attrs.update({"units": "J kg-1", "long_name": "Convective inhibition magnitude"})

    # Persist thresholds used for reproducibility
    hazard.attrs["thresholds"] = json.dumps(thresholds.__dict__)
    if "pressure_level" in hazard.coords:
        # Drop pressure level coord if carried through from selections
        hazard = hazard.drop_vars("pressure_level")
    return hazard


def run_classification(
    single_glob: str,
    pressure_glob: str,
    output_path: Path | None,
    thresholds: Thresholds,
    *,
    apply_gust_bump: bool = True,
) -> xr.Dataset:
    """High-level helper to load data, classify, and optionally persist output."""

    # Resolve file lists up front; fail fast if patterns match nothing
    single_paths = [Path(p) for p in glob(single_glob)]
    pressure_paths = [Path(p) for p in glob(pressure_glob)]

    if not single_paths:
        raise FileNotFoundError(f"No single-level files matched pattern: {single_glob}")
    if not pressure_paths:
        raise FileNotFoundError(f"No pressure-level files matched pattern: {pressure_glob}")

    single_by_date = _group_paths_by_date(single_paths)
    pressure_by_date = _group_paths_by_date(pressure_paths)

    missing_single = sorted(set(pressure_by_date) - set(single_by_date))
    missing_pressure = sorted(set(single_by_date) - set(pressure_by_date))

    if missing_single:
        raise FileNotFoundError(f"Missing single-level files for dates: {', '.join(missing_single)}")
    if missing_pressure:
        raise FileNotFoundError(f"Missing pressure-level files for dates: {', '.join(missing_pressure)}")

    common_dates = sorted(single_by_date)
    LOGGER.info(
        "Processing convection hazard for %d date(s) (%d single-level files, %d pressure-level files)…",
        len(common_dates),
        len(single_paths),
        len(pressure_paths),
    )

    hazard_slices: list[xr.Dataset] = []
    multiple_dates = len(common_dates) > 1

    for date_token in common_dates:
        single_group = single_by_date[date_token]
        pressure_group = pressure_by_date[date_token]

        LOGGER.info(
            "Loading inputs for %s (%d single-level file(s), %d pressure-level file(s))…",
            date_token,
            len(single_group),
            len(pressure_group),
        )
        ds_single = load_single_level_dataset(single_group)
        ds_pressure = load_pressure_level_dataset(pressure_group)

        LOGGER.info("Classifying convection risk for %s…", date_token)
        hazard = classify_convection(ds_single, ds_pressure, thresholds, apply_gust_bump=apply_gust_bump)

        if output_path is not None:
            destination = _resolve_output_path(output_path, date_token, multiple_dates)
            destination.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Writing hazard dataset for %s to %s", date_token, destination)
            hazard.to_netcdf(destination)

        hazard_slices.append(hazard)

        _summarize(hazard, label=date_token)

    combined = _combine_datasets(hazard_slices)
    return combined


def _summarize(hazard: xr.Dataset, *, label: str | None = None) -> None:
    """Log a concise summary of severity coverage."""

    severity = hazard["severity"]
    total_points = int(np.prod(list(severity.sizes.values())))

    suffix = f" for {label}" if label else ""
    LOGGER.info("Severity distribution%s (all times / grid cells):", suffix)
    for level, severity_label in SEVERITY_LABELS.items():
        count = int((severity == level).sum().item())
        pct = (count / total_points) * 100 if total_points else 0.0
        LOGGER.info("  %s (%d): %d cells (%.2f%%)", severity_label, level, count, pct)

    # Count time slices where at least one grid point reaches Moderate+
    hours_with_moderate = int(((severity >= 2).any(dim=("latitude", "longitude"))).sum().item())
    LOGGER.info("Hours with Moderate+ coverage somewhere%s: %d", suffix, hours_with_moderate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify convection polygons from ERA5 data.")
    parser.add_argument(
        "--single",
        default="data/wx/era5_single_hourly_*.nc",
        help="Glob pattern for ERA5 single-level files (default: %(default)s)",
    )
    parser.add_argument(
        "--pressure",
        default="data/wx/era5_pl_hourly_*.nc",
        help="Glob pattern for ERA5 pressure-level files (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wx/convection_severity.nc"),
        help="Path to write the resulting hazard dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--cp70-fallback",
        type=float,
        default=Thresholds().cp70_fallback,
        help="Fallback CP (CAPE*precip) threshold for the 70th percentile (default: %(default)s)",
    )
    parser.add_argument(
        "--no-gust-bump",
        action="store_true",
        help="Disable the optional severe gust bump in the severity calculation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    thresholds = Thresholds(cp70_fallback=args.cp70_fallback)
    hazard = run_classification(
        single_glob=args.single,
        pressure_glob=args.pressure,
        output_path=args.output,
        thresholds=thresholds,
        apply_gust_bump=not args.no_gust_bump,
    )

    _summarize(hazard)


if __name__ == "__main__":
    main()
