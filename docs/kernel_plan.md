Please implement the following Plan: Empirical hourly kernels from 4D segments 

## 1. Understand inputs and mapping to the theoretical objects

- Clarify how each file maps to Section 6 / 6.2.1 notation in `docs/gemini_master_guide.md`

- `flights_20230717_0000-2359.csv`: realized 4D trajectories for all flights on the day (ORIGINAL group are those whose chosen route is labelled `ORIGINAL` in `strictly_better_routes.csv`).
- `strictly_better_routes.csv`: map `flight_identifier` to route labels; partition flights into ORIGINAL vs nonorig; for nonorig, expose all candidate route labels.
- `nonorig_4d_segments_dir` (`all_segs`): 4D segments for nonorig flights, one file per subset; verify that each row has flight id, route id/label, time, and position, in a schema consistent with the master list.
- `wxm_sm_ih_maxpool.geojson`: regulated traffic volumes; extract a stable `volume_id` for each polygon.
- `tvtw_indexer.json`: discover the bin length Δ (expected 15 min), the total number of bins T, and, critically, how per-flight/per-segment entries into volumes are indexed (e.g. per flight: ordered list of `(volume_id, entry_time, exit_time)` or per 4D point: `(flight_id, time_bin, volume_id)`).
- Decide on a common time convention:
- Convert all relevant times to minutes from midnight of the local planning day and then to bin indices `t = floor(time_min / Δ)`.
- Define `H = 60 / Δ` and hour index `h(t) = floor(t / H)` as in §6.1.
- Verify whether `tvtw_indexer` already gives entry times per volume or only per-bin occupancy; choose the simplest mapping that yields, for each traversal of volume `v` by flight `i`, an entry time `τ^{entry}_{v,i,d}` and an ordering of volumes along the route.

## 2. Core data model and classes in `src/gemini/propagation`

- Implement small, focused dataclasses for clarity:
- `Volume`: `id`, optional metadata (name, capacity) as available from the geojson.
- `EdgeId`: `(upstream_volume_id, downstream_volume_id)` with `__hash__` and `__str__` for grouping and output.
- `TraversalRecord`: `edge: EdgeId`, `dep_bin: int`, `arr_bin: int`, `lag_bins: int`, `hour_index: int`, plus optional tags (`flight_id`, `route_label`, `group` = ORIGINAL/nonorig) for diagnostics.
- Implement a `VolumeGraph` helper:
- Maintains known `Volume` objects and can optionally track degree statistics, but its primary role is just to ensure consistent `EdgeId` creation.
- Implement an `HourlyKernelEstimator` class:
- Configuration: `delta_minutes`, `num_bins`, `max_lag_bins` (per-edge cap `L_e` or a global `L_max`), shrinkage parameter `M` from §6.2.1, and optional minimum-count thresholds.
- Internal state: dictionaries keyed by `(edge, hour)` accumulating:
- `N_eh` (traversal count in hour `h` for edge `e`),
- `N_eh_k[k]` (count with lag `k`), and global per-edge totals `N_e` and `N_e_k[k]`.
- Public methods:
- `add_traversal(record: TraversalRecord)`: update counts.
- `finalize_kernels()`: compute per-edge, per-hour kernels and return a table-like structure (e.g. rows for CSV).

## 3. From 4D trajectories to edge traversals (implementing Step K0 conceptually)

- Use an extractor component (e.g. `TraversalExtractor`) that, given flight-level 4D data and the `tvtw_indexer`, yields `TraversalRecord` instances:
- For each flight-day and route (realized ORIGINAL from master CSV; candidate nonorig routes from `all_segs`):
- Obtain the ordered list of volumes intersected, together with entry times. Prefer to:  
- Read from `tvtw_indexer` if it already stores per-flight volume entry indices, else reconstruct an ordered list from per-bin occupancy by grouping contiguous bins of the same volume and taking the first bin as entry.
- For each consecutive pair of volumes `(u, v)` on the route, define an edge `e = (u → v)`.
- Define the edge departure time as the entry time into `u` and the edge arrival time as the entry time into `v` (we care about when flow leaves `u` and reaches `v`).
- Convert times to bins:
- `s = floor(τ_dep / Δ)`; `t = floor(τ_arr / Δ)`; `ℓ = t - s`.
- `H = 60 / Δ`; `h = floor(s / H)`.
- Apply constraints in line with §6.2 assumptions:
- Retain only strictly causal traversals with `ℓ ≥ 1` (if `ℓ ≤ 0` due to rounding or data quirks, discard or log them, since `K_{e,h}(0)=0`).
- Apply edge-specific or global maximum lag: if `ℓ > L_max`, either clip to `L_max` (with caution) or drop the sample; the safer default is to drop and record diagnostics, and compute `L_max` generously from the empirical distribution (e.g. 99.5th percentile of observed lags).
- Exclude traversals whose dep/arr bins fall outside the planning horizon `[0, T-1]` (e.g. flights starting before 00:00 or arriving after 24:00 UTC) or handle them with truncation but mark as partial.
- Emit one `TraversalRecord` per edge traversal and feed these into `HourlyKernelEstimator.add_traversal`.

## 4. Counting and computing hourly kernels (K1–K3 implementation)

- Implement K1 (hourly counts per lag):
- For each traversal record on edge `e` with hour index `h` and lag `ℓ` in `[1, L_e]`:
- Increment `N_{e,h}` and `N_{e,h}(ℓ)`.
- After processing all data, compute per-edge totals:
- `N_e = Σ_h N_{e,h}`, `N_e(k) = Σ_h N_{e,h}(k)`.
- Implement K2 (raw empirical kernel):
- For each `(e,h)` with `N_{e,h} > 0`, define
- `K_raw[e,h,k] = N_{e,h}(k) / N_{e,h}` for `k = 1..L_e`.
- Ensure numerical robustness: if all `N_{e,h}(k)` are zero because all lags are outside `[1, L_e]`, treat this as `N_{e,h}=0` for kernel purposes and fall back to shrinkage.
- Implement K3 (shrinkage and optional smoothing):
- Compute per-edge global kernel (over all hours with any data):
- `K_bar[e,k] = (Σ_h N_{e,h}(k)) / (Σ_h N_{e,h})` for edges with `Σ_h N_{e,h} > 0`.
- For each `(e,h)` pair:
- Let `α_{e,h} = N_{e,h} / (N_{e,h} + M)` for a chosen `M` (CLI parameter, default e.g. `M=50`–`100`).
- If `N_{e,h} = 0`, set `α_{e,h} = 0` (purely global shape).
- Define the final kernel
- `K[e,h,k] = α_{e,h} * K_raw[e,h,k] + (1 - α_{e,h}) * K_bar[e,k]` for `k = 1..L_e`.
- Keep hour-wise smoothing (`\tilde{K}_{e,h}` as in the notes) optional:
- Initially skip inter-hour smoothing for simplicity and transparency.
- If later needed, add a small moving-average in hour space controlled by a CLI flag (neighbourhood `\mathcal{N}(h)` and weights `w_{h,h'}`) while keeping the same output schema.
- Prepare output table for CSV:
- Rows with columns like:  
- `edge_u`, `edge_v`, `hour_index`, `lag_bins`, `kernel_value`,  
- and helpful diagnostics: `delta_minutes`, `lag_minutes = lag_bins * delta_minutes`, `N_eh`, `N_eh_k` (optional), `N_e`.
- An edge is included only if it has at least one traversal across all hours (`Σ_h N_{e,h} > 0`).

## 5. CLI design and wiring

- Create a CLI module (e.g. `src/gemini/propagation/compute_hourly_kernels_cli.py`):
- Use `argparse` with defaults so it can be run directly from the IDE:
- `--master-flights` (default `/mnt/d/project-tailwind/output/flights_20230717_0000-2359.csv`).
- `--strictly-better-routes` (default `/mnt/d/project-gemini/data/strictly_better_routes.csv`).
- `--nonorig-4d-segments-dir` (default `/mnt/d/project-silverdrizzle/tmp/all_segs`).
- `--volumes-geojson` (default `/mnt/d/project-tailwind/output/wxm_sm_ih_maxpool.geojson`).
- `--tvtw-indexer` (default `/mnt/d/project-tailwind/output/tvtw_indexer.json`).
- `--output-kernels` (default `/mnt/d/project-gemini/data/hourly_kernels.csv`).
- Optional tuning flags: `--max-lag-bins`, `--shrinkage-M`, `--min-traversals-per-edge`.
- CLI flow:

1. Parse arguments.
2. Load `tvtw_indexer.json`, extract `delta_minutes`, `num_bins`, and the structure required for volume entry times.
3. Load volumes from the geojson and build `VolumeGraph` (volume ids, no heavy geometry operations).
4. Load `strictly_better_routes.csv` to build maps: ORIGINAL flights vs nonorig, and nonorig route labels.
5. Stream master flights CSV and `all_segs` CSV.GZ files in chunks to build per-flight, per-route ordered volume sequences using `tvtw_indexer`, emitting `TraversalRecord`s into `HourlyKernelEstimator`.
6. Call `finalize_kernels()` to obtain the kernel table.
7. Write the table to CSV in the requested `edge_id,hour,lag,K` style (with explicit columns for `edge_u`, `edge_v`, `hour_index`, `lag_bins`, `kernel_value`, and optional diagnostics).

## 6. Edge cases, assumptions, and delicacies to document and handle

- Time handling and day boundaries:
- Assume all times refer to a single planning day with horizon `[0, 24h)` in a consistent time zone (probably UTC); any segments outside this window are either clipped or discarded and logged.
- If `t_arr` lands in the next day (e.g. after 24h), decide to either discard these traversals or fold into an extended horizon if supported by `tvtw_indexer`; default to discarding with a warning to keep the 24h kernel interpretation clear.
- Multiple crossings and self-loops:
- If a flight crosses the same edge multiple times (due to holding patterns or loops), treat each traversal independently; this naturally increases `N_{e,h}` and broadens `K_{e,h}(k)`.
- If the route data suggests `u == v` (self-loop edge), either exclude these from kernels or treat them separately; default is to exclude them, since their interpretation as inter-volume edges is ambiguous.
- ORIGINAL vs nonorig data usage:
- For ORIGINAL flights, use exactly the realized 4D segments from the master list.
- For nonorig flights, treat each route’s 4D segments present in `all_segs` as distinct empirical traversals; we do not sample from Type-1/Type-2 models or weight by route probabilities in this implementation.
- Document that this makes the kernel estimation conditional on the mix of candidate routes encoded in `all_segs` (rather than the true historical route-choice distribution).
- Sparse data and shrinkage behaviour:
- For `(e,h)` with very few traversals (e.g. `N_{e,h} < 5`), the estimator leans heavily on the global `K_bar[e]` because `α_{e,h}` is small; document the default `M` and allow tuning via CLI.
- If an edge `e` has `Σ_h N_{e,h} = 0` (no traversals at all), exclude it from the output altogether.
- Numerical and performance considerations:
- Use chunked reading for the large CSV / CSV.GZ files and avoid holding all 4D segments in memory at once; `TraversalExtractor` should operate in a streaming fashion.
- Ensure integer arithmetic for bins, hours, and lag indices to avoid floating-point off-by-one errors; explicitly cap indices to `[0, T-1]` and `[0, H_day-1]` where needed.
- Validation hooks:
- Add lightweight sanity checks (optionally triggered by a CLI flag):
- Confirm that for a sample of edges, `Σ_k K[e,h,k] ≤ 1` for each hour, with any missing mass interpreted as flights exiting the regulated network on that edge.
- Plot or dump summaries for a few heavily used edges (mean/median lag per hour, counts) to compare qualitatively with expectations from the ATFM domain knowledge.

This plan yields a modular, class-based implementation that computes purely empirical hourly kernels from the available 4D segments and exposes them via a configurable CLI, while making all key assumptions and edge-case treatments explicit.

# Plan Context
The reference material is Section 6 of docs/gemini_master_guide.md (particularly 6.2.1 - Empirical estimation of hourly kernel).

# Overall Instructions
1. The task should be implemented with classes modularized for better readability and code organization, in `src/gemini/propagation`. 
2. Create a CLI in a separate Python file that will take in necessary files (detailed below) so we can run it (populate the default parameters so we can click run in the IDE).

# Inputs and Data Wiring
The master flight list can be found in `/mnt/d/project-tailwind/output/flights_20230717_0000-2359.csv`. However, all of these flights (identified by the `flight_identifier` column) will be divided into two groups:

- The ORIGINAL group, which includes flights that are filed with certainty. The 4D segments will be retrieved directly from the master flight list CSV above. This group will only observe **Type-2 uncertainty** in the @gemini_master_guide.md's Section 2.

- The non-ORIGINAL group (nonorig for short), includes flights where there are multiple routes that are potentially better given the latest weather condition update. This group will observe **both Type-1 and Type-2 uncertainties**. 

- The CLI will ask for the `strictly_better_routes.csv` (default to `data/strictly_better_routes.csv`), where you will find flight identifiers marked with ORIGINAL in the `route` column for the ORIGINAL group. If it is not `ORIGINAL` then it belongs to the nonorig group.

- The CLI also asks for the nonorig's 4D segments (for all route options). Because of its size, it's splitted into many `csv.gz` files located in nonorig_4d_segments_dir defaults to `/mnt/d/project-silverdrizzle/tmp/all_segs`. In there, you will find (after inspection) the route options and the associated 4d trajectory segment information, in the same format as the master flight list.

- The traffic volume definition defaults to `/mnt/d/project-tailwind/output/wxm_sm_ih_maxpool.geojson`.

- The traffic volume time bin index file is `/mnt/d/project-tailwind/output/tvtw_indexer.json`. You could also get the bin length (in minutes) from here - for our case, for your context it will be 15 minutes.

> The script to compute demand for each traffic volume (ie., entry counting for traffic volume) is located in `scripts/demand_by_entry_counts_from_so6_with_entry_time.py`. You should use this script as reference to implement the necessary interpretation of it to fulfill your task.