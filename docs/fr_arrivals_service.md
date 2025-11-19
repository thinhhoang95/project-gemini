# FRArrivalMomentsService README

This guide describes `gemini.arr.fr_arrivals_service`, a library-oriented wrapper
around the FR arrivals CLI that lets you compute arrival moments directly from
Python using pre-built configuration objects. Use it when you want to
preconfigure artifact paths once, pass `GroundJitterConfig` / `GroundHoldConfig`
instances in memory, and receive an in-memory `pandas.DataFrame` with arrival
moments instead of working through CSV files.

## Key Components

- `FRArrivalArtifacts`: Lightweight dataclass with the four artifact paths the
  builder needs: FR demand, FR route catalogue, Gemini flight metadata CSV, and
  the TVTW indexer JSON. These paths are normalized into `pathlib.Path`
  instances and cached by the service.
- `FRArrivalMomentsService`: Stateful helper that caches FR segments, the list
  of traffic volumes, and the TVTW indexer. Each call to
  `get_arrival_moments(...)` accepts jitter/ground-hold configuration objects
  and returns a DataFrame containing `volume_id`, `time_bin`, `lambda_mean`,
  `lambda_var`, and `gamma_lag1`. When you only need the deterministic ground
  hold impacts, `get_per_flight_ground_hold_delay(...)` returns every flight ID
  with its assigned delay minutes.
- `arrival_moments_to_dataframe(...)`: Utility that converts a raw
  `ArrivalMoments` instance into the DataFrame shape the CLI traditionally
  writes to disk.

## Example Usage

```python
from gemini.arr.fr_arrivals_service import FRArrivalArtifacts, FRArrivalMomentsService
from gemini.arrivals.ground_jitter_config import GroundJitterConfig
from gemini.arrivals.ground_hold_config import GroundHoldConfig

# 1. Preconfigure artifact locations once for your environment.
artifacts = FRArrivalArtifacts(
    fr_demand_path="data/fr/gem_artifacts_demand_all",
    fr_route_catalogue_path="data/fr/gem_artifacts_route_catalogue_all",
    flights_csv_path="/mnt/d/project-tailwind/output/flights_20230717_0000-2359.csv",
    tvtw_indexer_path="/mnt/d/project-tailwind/output/tvtw_indexer.json",
)

# 2. Instantiate the service (caches segments + TVTW bins lazily).
service = FRArrivalMomentsService(artifacts)

# 3. Load configuration objects however you like (JSON/YAML/string IO/etc.).
jitter_config = GroundJitterConfig.from_json("data/fr/ground_jitter_default.json")
ground_hold_config = GroundHoldConfig.from_yaml("data/fr/ground_hold_example.yaml")

# 4. Build arrival moments as a DataFrame.
arrival_df = service.get_arrival_moments(
    jitter_config,
    ground_hold_config=ground_hold_config,  # optional
    tail_tolerance=1e-6,
    volume_ids=["KPVD_1", "KJFK_3"],        # optional subset; defaults to all volumes
)

print(arrival_df.head())

# 5. Retrieve deterministic per-flight ground hold delays (optional).
per_flight_delays = service.get_per_flight_ground_hold_delay(ground_hold_config)
print(per_flight_delays.head())
```

## Per-Flight Ground Hold Delays

`get_per_flight_ground_hold_delay(...)` surfaces the deterministic ground hold
component without running the full FR arrival pipeline. Pass the same
`GroundHoldConfig` you would supply to `get_arrival_moments(...)` and the method
returns a DataFrame with two columns:

- `flight_id`: All identifiers present in the Gemini flights CSV
- `ground_hold_delay_min`: Minutes of delay imposed by the deterministic hold

The hold operator is re-run each time you call this helper so you always get a
fresh per-flight view whenever the configuration changes.

## Building Configuration Objects

You are not limited to loading JSON/YAML from disk. Both configuration types
offer programmatic constructors so you can build test fixtures or dynamically
generated scenarios entirely in memory.

### GroundJitterConfig

`GroundJitterConfig.from_mapping(...)` expects a nested mapping where each key
represents an airport and the values describe time-of-day windows with
Hurdle/Bulk/Splice parameters. The `default` block is mandatory and applies to
any airport without an override.

```python
from gemini.arrivals.ground_jitter_config import GroundJitterConfig

jitter_mapping = {
    "default": {
        "00:00-06:00": {"p_hurdle": 0.05, "mean": 4.0, "std": 2.0},
        "06:00-22:00": {"p_hurdle": 0.25, "mu": 10.0, "sigma": 6.0, "threshold": 45.0},
        "22:00-24:00": {"p_hurdle": 0.1, "mean": 6.0, "std": 3.0, "tail_scale": 12.0},
    },
    "KJFK": {
        # Airport-specific overrides; times wrap around midnight automatically.
        "05:00-12:00": {"p_hurdle": 0.35, "mean": 12.0, "std": 8.0, "shift": 3.0},
        "12:00-05:00": {"p_hurdle": 0.15, "mean": 6.0, "std": 3.5},
    },
}

jitter_config = GroundJitterConfig.from_mapping(jitter_mapping)
```

- Time windows follow the `"HH:MM-HH:MM"` format; `"24:00"` is allowed.
- Parameter names are the same ones used in the JSON configs (see
  `GroundJitterConfig.HBSParameters` for the full list).
- You can still call `GroundJitterConfig.from_json(...)` when you prefer to read
  from a file; both methods produce identical objects.

### GroundHoldConfig

`GroundHoldConfig` bundles `GroundHoldWindow` objects grouped by airport. Each
window specifies a `[start, end)` ISO timestamp, a rate limit in flights per
hour, and an optional regulation ID.

```python
from datetime import datetime, timezone
from gemini.arrivals.ground_hold_config import GroundHoldConfig, GroundHoldWindow

kjfk_window = GroundHoldWindow(
    start=datetime(2023, 7, 17, 10, tzinfo=timezone.utc),
    end=datetime(2023, 7, 17, 12, tzinfo=timezone.utc),
    rate_fph=30.0,
    airport="KJFK",
    regulation_id="NYC-GH-01",
)

kmia_window = GroundHoldWindow(
    start=datetime(2023, 7, 17, 14, tzinfo=timezone.utc),
    end=datetime(2023, 7, 17, 15, tzinfo=timezone.utc),
    rate_fph=24.0,
    airport="KMIA",
)

ground_hold_config = GroundHoldConfig(
    windows_by_airport={
        "KJFK": [kjfk_window],
        "KMIA": [kmia_window],
    },
    version="demo-2023-07-17",
)
```

- Datetimes can be naive or timezone-aware, but the start/end pair must match.
- Multiple windows per airport are allowed; they will be processed independently
  and their delays will be accumulated if a flight falls into more than one.
- Use `GroundHoldConfig.from_yaml(...)` to load production schedules from disk,
  or `to_yaml(...)` to persist the in-memory configuration defined above.

### Notes

- `service.load_segments()` loads and caches the FR artifacts; the CLI uses this
  result to feed a progress bar but ordinary consumers can rely on the lazy
  loading inside `get_arrival_moments`.
- Pass a custom iterable through the `segments` parameter if you want to stream
  rows with your own progress hook; otherwise the cached segments are reused.
- The returned DataFrame is ready to serialize to CSV using
  `DataFrame.to_csv(...)` or to feed directly into downstream Gemini models.
