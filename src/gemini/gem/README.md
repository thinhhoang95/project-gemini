> The reference for the complete algorithm can be found in the [master guide](docs/gemini_master_guide.md).

## High-level overview

The `atfm_network` module implements a **chronological ATFM (Air Traffic Flow Management) propagation** over a network of traffic volumes.  
Conceptually:

- **Inputs**:
  - Exogenous arrival stats per volume and time-bin (`ArrivalMoments`).
  - Hourly traversal kernels between volumes (`HourlyKernelTable`).
  - An optional capacity regulation plan per volume and time-bin (`RegulationPlan`).
  - Time-binning logic (`TVTWIndexer`) and volume metadata (`VolumeGraph`).
- **Core engine**:
  - `ATFMNetworkModel` walks through the day bin-by-bin and volume-by-volume, updating:
    - Arrival mean/variance into each volume.
    - Queue mean/variance at that volume.
    - Departure mean/variance from that volume.
- **Outputs**:
  - Per-volume time series (`VolumeTimeSeries`) and network-level delay metrics (`ATFMRunResult`).
  - Optional CSV of all time series and a printed summary.

---

## Main components and their roles

- **`TVTWIndexer`** (`tvtw_indexer.py`)
  - Defines the **time discretisation**: `time_bin_minutes`, `num_bins`, `bins_per_hour`.
  - Provides helpers for mapping calendar time to bin indices (e.g. `bin_of_datetime`).
  - In this module it is used solely for:
    - Ensuring kernels and arrivals share the same bin size.
    - Knowing how many bins exist (`num_bins`) and how many bins per hour (`bins_per_hour`).

- **`VolumeGraph` & `VolumeLocator`** (`volume_graph.py`)
  - `VolumeGraph` is a **registry of traffic volumes** (nodes in the network), each a `Volume` from `domain_types`.
  - In `atfm_network.py`, `_build_volume_graph` constructs a simple volume graph from a list of volume IDs (with dummy geometries).
  - `VolumeLocator` is a spatial helper; for this module it is *not used* in the propagation, only for having a valid `VolumeGraph` structure.

- **`HourlyKernelTable` & `EdgeKernel`** (`hourly_kernel_table.py`)
  - Represents **probabilistic traversal kernels** between volumes:
    - Each directed edge `EdgeId(upstream, downstream)` has an `EdgeKernel`.
    - For each hour `h` and lag in bins `lag_bins`, kernel stores a probability `K_edge,h(lag)`.
  - Also provides **adjacency queries**:
    - `get_incoming(volume_id)` → all edges whose downstream is `volume_id`.
    - `get_outgoing(volume_id)` → all edges whose upstream is `volume_id`.
  - In `ATFMNetworkModel`, these kernels are used to **propagate departures from an upstream volume into arrivals at downstream volumes with appropriate delay and hourly variation.**

- **`ArrivalMoments`** (`arrival_moments.py`)
  - A **sparse container** of exogenous arrival statistics:
    - `lambda_ext[(volume_id, bin)]` = mean arrival count in that bin.
    - `nu_ext[(volume_id, bin)]` = variance.
    - `gamma_ext[(volume_id, bin)]` = covariance between bin `t` and `t+1` (lag-1 covariance).
  - `ATFMNetworkModel` queries:
    - `.mean(volume_id, t)`, `.variance(volume_id, t)`, `.covariance_lag1(volume_id, t)`
    - and then adds propagated upstream effects on top.

- **`RegulationPlan` & `VolumeRegulation`** (`regulation_plan.py`)
  - Represents **capacity limits per volume over time**.
  - YAML spec is loaded via `RegulationPlan.load(...)`, which:
    - Aligns to `TVTWIndexer.time_bin_minutes`.
    - Converts human-readable times (`HH:MM`) to `start_bin`/`end_bin`.
    - Converts `capacity_per_hour` to `capacity_per_bin`.
  - `RegulationPlan.build_capacity_matrix(volumes, num_bins)` returns:
    - `Dict[volume_id, List[Optional[float]]]`, per-bin capacity:
      - `None` ⇒ **unregulated** (no binding capacity in that bin).
      - `float` ⇒ **capacity** (flights per bin).

- **`VolumeTimeSeries` and `ATFMRunResult`** (`atfm_network.py`)
  - `VolumeTimeSeries`: for each volume, holds per-bin time series:
    - `lambda_mean`, `lambda_var`: arrival stats.
    - `queue_mean`, `queue_var`: queue length stats (one extra element for state at `T`).
    - `departure_mean`, `departure_var`: departure stats.
  - `ATFMRunResult`:
    - `by_volume`: mapping `volume_id → VolumeTimeSeries`.
    - `total_delay_mean`, `total_delay_var`: integrated network delay over all volumes and time.

- **`ATFMNetworkModel`** (`atfm_network.py`)
  - The **core propagation engine** that:
    - Builds capacities from the regulation plan.
    - Runs a chronological loop over time bins.
    - Uses queueing approximations (Level 0/1) to compute queue and departure statistics.

- **CLI helpers** (`atfm_network.py`)
  - `_parse_cli_args`: parse input file paths and options.
  - `_load_arrival_moments_csv`: read arrivals CSV and build `ArrivalMoments`.
  - `_collect_volume_ids`: compute the set of volume IDs from kernels, arrivals, and plan.
  - `_build_volume_graph`: synthetic `VolumeGraph` for these volume IDs.
  - `_write_volume_series_csv`: dump full time series to CSV.
  - `_print_summary`: print network-wide delay metrics and top delay volumes.
  - `main`: ties everything together for a command-line run.

---

## Step-by-step data flow

### 1. Inputs and indexing

1. **Kernels CSV → `HourlyKernelTable`**
   - Columns (from `compute_hourly_kernels_cli` output) include at least:
     - `edge_u`, `edge_v`, `hour_index`, `lag_bins`, `kernel_value`, `delta_minutes`.
   - `HourlyKernelTable.from_csv(path)`:
     - Verifies a single `delta_minutes` value for all rows.
     - Groups by `(edge_u, edge_v)` and constructs:
       - An `EdgeId(upstream=edge_u, downstream=edge_v)`.
       - An `EdgeKernel` with:
         - `max_lag_bins` = max of `lag_bins` for that edge.
         - `kernels[hour_index][lag_bins] = kernel_value`.
     - Builds:
       - `edges: {EdgeId → EdgeKernel}`
       - `incoming_edges: {volume_id → [EdgeId]}` (downstream adjacency)
       - `outgoing_edges: {volume_id → [EdgeId]}` (upstream adjacency).

2. **Arrivals CSV → `ArrivalMoments`**
   - `_load_arrival_moments_csv(path, num_bins)`:
     - Expects columns: `volume_id`, `time_bin`, `lambda_mean`, `lambda_var`, optionally `gamma_lag1`.
     - For each valid row within `[0, num_bins)`:
       - Fills `lambda_ext[(volume_id, time_bin)]`.
       - Fills `nu_ext[(volume_id, time_bin)]`.
       - Optionally `gamma_ext[(volume_id, time_bin)]`.
   - Result: `ArrivalMoments(lambda_ext, nu_ext, gamma_ext)`.

3. **Regulation plan YAML → `RegulationPlan` (optional)**
   - `RegulationPlan.load(path, tvtw)`:
     - Ensures `time_bin_minutes` matches `tvtw.time_bin_minutes`.
     - Iterates `volumes[volume_id].regulations`.
     - For each reg:
       - Parses `start_time` / `end_time` strings.
       - Converts to `start_bin`, `end_bin`.
       - Converts `capacity_per_hour` to `capacity_per_bin` using `bins_per_hour`.
   - Later, `build_capacity_matrix(volumes, num_bins)` produces:
     - `capacities[volume_id][t]` = `None` or `cap_bin`.

4. **Volume IDs and VolumeGraph**
   - `_collect_volume_ids(kernels, arrivals, plan)`:
     - From kernels: all `edge.upstream` and `edge.downstream`.
     - From arrivals: all volume IDs present in `lambda_ext`.
     - From plan: all `reg.volume_id`.
   - `_build_volume_graph(volume_ids)`:
     - Creates a trivial `Volume` for each ID (no real geometry/capacity).
     - Builds a `VolumeLocator` from a pandas DataFrame with `traffic_volume_id` and `geometry=None`.
   - In this module, `VolumeGraph` is mainly used to:
     - Keep `volume_graph.volumes` (set of volumes).
     - Provide `volume_ids` for iteration.

5. **Time discretisation (`TVTWIndexer`)**
   - If a TVTW JSON is provided: `TVTWIndexer.load(path)`; else: `TVTWIndexer(time_bin_minutes=kernels.delta_minutes)`.
   - Supplies:
     - `num_bins`: number of time bins per day.
     - `bins_per_hour`: how to map a bin index to an hourly index (`t // bins_per_hour`).

---

### 2. Running the ATFM network model

The user-visible entry point is:

- **Programmatic**: `ATFMNetworkModel.run(plan)`
- **CLI**: `python -m gemini.gem.atfm_network ...` (via `main`)

#### 2.1 Capacity matrix

Inside `ATFMNetworkModel.run(plan)`:

- **If `plan` is `None`**:
  - `_build_unregulated_capacities()` returns:
    - `capacities[volume_id][t] = None` for all volumes and bins.
    - Semantically: **no binding capacity — all arrivals can depart immediately**.
- **Else**:
  - `plan.build_capacity_matrix(volume_graph.volumes, num_bins)` returns:
    - `capacities[volume_id][t]` = per-bin capacity or `None` if no regulation.

This matrix is the main *control* input to the queueing model.

#### 2.2 Initialising time series

`_run_with_capacities(capacities)`:

- Builds a `series` dict via `_init_series()`:
  - For each `volume_id`:
    - Pre-allocates lists of zeros for:
      - `lambda_mean[0..T-1]`, `lambda_var[0..T-1]`
      - `queue_mean[0..T]`, `queue_var[0..T]` (queue state at T included)
      - `departure_mean[0..T-1]`, `departure_var[0..T-1]`
- Prepares:
  - `prev_pair_weight[volume_id]` (for smoothing F1 weights).
  - `w_bin[(volume_id, t)]` to store per-bin variance deflation weights.

#### 2.3 Loop over time bins: arrivals and queueing

For each time bin `t` from `0` to `num_bins-1`:

##### Step 1: Compute arrival moments for each volume

For each `volume_id`:

- `_compute_arrival_mean_var(volume_id, t, series)`:

  1. **Start from exogenous arrivals**:
     - `lam = arrivals.mean(volume_id, t)`
     - `nu = arrivals.variance(volume_id, t)`

  2. **Add propagated contributions from upstream volumes**:
     - Get incoming edges: `incoming_edges = kernels.get_incoming(volume_id)`.
     - For each `edge in incoming_edges`:
       - `edge_kernel = kernels.edges[edge]`.
       - `max_lag = min(edge_kernel.max_lag_bins, t)`.
       - For each lag `lag ∈ {1, …, max_lag}`:
         - Let `departure_bin = t - lag`.
         - Compute `hour_index = departure_bin // bins_per_hour`.
         - `kernel_val = edge_kernel.get(hour_index, lag)`:
           - Fraction of flows leaving upstream `edge.upstream` bin `departure_bin` that arrive in this volume at `t`.
         - From upstream series:
           - `dep_mean = upstream_series.departure_mean[departure_bin]`
           - `dep_var = upstream_series.departure_var[departure_bin]`
         - Update mean/variance:
           - Mean:
             - `lam += kernel_val * dep_mean`
           - Variance (Bernoulli thinning + input variance):
             - `nu += kernel_val * (1 - kernel_val) * dep_mean + (kernel_val**2) * dep_var`

  3. **Store**:
     - `series[volume_id].lambda_mean[t] = lam`
     - `series[volume_id].lambda_var[t] = nu`

**Idea**: arrivals at each volume and time-bin are a combination of exogenous inflow and **delayed, probabilistic routing** of departures from upstream volumes.

##### Step 1b: Compute F1 deflation weights (`w_bin`)

For each `volume_id`:

- Goal: build a per-bin weight \( w_{v,t} \in [0.6, 1.0] \) that **deflates the arrival variance** to partially account for temporal correlation when using a single-bin normal approximation.

Key steps:

1. **Base quantities**:
   - `nu_t = series[volume_id].lambda_var[t]` (current variance).
   - If `t < num_bins - 1`, we attempt to estimate correlation with bin `t+1`.

2. **Arrival covariance at lag 1 (`gamma`)**:
   - `_compute_arrival_cov_lag1(volume_id, t, series)`:
     - Start from exogenous `gamma = arrivals.covariance_lag1(volume_id, t)`.
     - For each incoming edge:
       - Use kernel pairs `(k_val, k_next)` for lags `lag` and `lag+1`.
       - Use upstream departure statistics again to adjust `gamma`.
     - Returns an approximation of covariance between arrivals at `t` and `t+1`.

3. **Proxy for next-bin variance `nu_next`**:
   - `_predict_next_variance(...)`:
     - Use `arrivals.variance(volume_id, t+1)` if available.
     - If zero, fallback to current bin variance.

4. **Compute pair weight**:
   - Denominator: `denom = max(nu_t + nu_next, 1e-6)`.
   - Raw pair weight: `pair = 1.0 + 2.0 * gamma / denom`.
   - Clip with `_clip_weight(pair)` to `[0.6, 1.0]`.
   - Smooth with previous pair:
     - If `prev_pair` exists: `w_val = 0.5 * (prev_pair + pair)`.
     - Else: `w_val = pair`.
   - Store: `w_bin[(volume_id, t)] = _clip_weight(w_val)`.
   - For final bin `t = num_bins - 1`, reuse last available `prev_pair` or default `1.0`.

**Idea**: If arrivals in adjacent bins are positively correlated, a naive variance would double count variability.  
The F1 weight partially reduces variance to keep the queue approximation stable and realistic.

##### Step 2: Queue update and departures per volume

For each `volume_id`:

- `_queue_step(volume_id, t, capacities, w_bin, series)`:

  1. **Fetch current state**:
     - From `series[volume_id]`:
       - `lam = lambda_mean[t]`
       - `nu = lambda_var[t]`
       - `q_mean = queue_mean[t]`
       - `q_var = queue_var[t]`
     - `w_t = w_bin.get((volume_id, t), 1.0)`
     - **Deflate variance**:
       - `nu_deflated = w_t * nu`

  2. **Determine capacity for this bin**:
     - `capacity_list = capacities.get(volume_id)`
     - `cap = capacity_list[t]` if defined, else `None`.

  3. **Unregulated case (`cap is None`)**:
     - Intuition: no capacity constraint → no queue can build up.
     - Actions:
       - `queue_mean[t+1] = 0.0`
       - `queue_var[t+1] = 0.0`
       - `departure_mean[t] = lam` (all arrivals depart)
       - `departure_var[t] = max(nu_deflated, 0.0)` (same variance as arrivals)
       - Return.

  4. **Regulated case (`cap` is a float)**:
     - We approximate queue dynamics using a **reflected normal** approximation.

     - Define:
       - `delta = lam - cap` (excess arrival rate)
       - `mu = q_mean + delta` (unconstrained next-period mean queue)
       - `sigma2 = max(q_var + nu_deflated, 0.0)` (variance of unconstrained queue)
       - `sigma = sqrt(max(sigma2, 1e-12))`
       - `a = mu / sigma` (normalised mean)
       - `phi = _std_normal_pdf(a)`
       - `Phi = _std_normal_cdf(a)`

     - **Expected queue next bin**:
       - `E_Q = sigma * phi + mu * Phi`  
         (mean of \( \max(N(\mu, \sigma^2), 0) \))

     - **Second moment and variance**:
       - `EQ2 = (mu^2 + sigma2) * Phi + mu * sigma * phi`
       - `Var_Q = max(EQ2 - E_Q^2, 0.0)`

     - **Update queue state**:
       - `queue_mean[t+1] = E_Q`
       - `queue_var[t+1] = Var_Q`

     - **Departures**:
       - Flow conservation: departures = arrivals + current queue - next queue
       - `D_mean = lam + q_mean - E_Q`
       - `departure_mean[t] = D_mean`

       - Congestion probability:
         - `p_cong = Phi` (approx probability queue is positive).
       - Departure variance:
         - `D_var = max((1.0 - p_cong) * nu_deflated, 0.0)`
       - `departure_var[t] = D_var`

**Idea**: Each bin is treated as a single-step queue update where arrivals plus previous queue either exceed capacity (queue persists) or not (queue drains). The algorithm uses a normal approximation to compute expectations and variances.

#### 2.4 Aggregating network-wide delay

After all time bins are processed:

- `_aggregate_delay(series)`:

  - For each volume’s `volume_series`:
    - `total_mean += delta_minutes * sum(volume_series.queue_mean[:-1])`
    - `total_var += (delta_minutes**2) * sum(volume_series.queue_var[:-1])`

  - `delta_minutes = self.delta_minutes = tvtw.time_bin_minutes`.

- Interpreted as:
  - **Total delay mean** = integral (sum over time bins) of queue length (flights) × bin duration (minutes)  
    ⇒ units: flight-minutes.
  - **Total delay variance** similarly scaled.

Result is packaged into `ATFMRunResult`.

---

## How the CLI ties everything together

The `main` function (`atfm_network.py`) does:

1. **Parse CLI arguments**:
   - `--kernels-csv`: hourly kernels.
   - `--arrivals-csv`: exogenous arrival moments.
   - `--regulation-plan`: optional YAML for capacities.
   - `--tvtw-indexer`: optional JSON; otherwise inferred from kernels.
   - `--output-arrivals-csv`: optional output time series file.
   - `--top-volumes`: how many top-delay volumes to print.

2. **Load models**:
   - `kernels = HourlyKernelTable.from_csv(args.kernels_csv)`
   - `tvtw = TVTWIndexer.load(args.tvtw_indexer)` or `TVTWIndexer(time_bin_minutes=kernels.delta_minutes)`
   - `arrivals = _load_arrival_moments_csv(args.arrivals_csv, tvtw.num_bins)`
   - `plan = RegulationPlan.load(args.regulation_plan, tvtw)` if given.

3. **Build volume graph and model**:
   - `volume_ids = _collect_volume_ids(kernels, arrivals, plan)`
   - `volume_graph = _build_volume_graph(volume_ids)`
   - `model = ATFMNetworkModel(tvtw, volume_graph, kernels, arrivals)`

4. **Run propagation and output**:
   - `result = model.run(plan)`
   - `_print_summary(result, num_bins, delta_minutes, top_k)`
   - Optionally `_write_volume_series_csv(...)`.

---

## Conceptual examples

### Example 1: Simple two-volume network

Imagine:

- Two volumes: `A` and `B`.
- One directed edge: `A → B`.
- 15-minute bins (`time_bin_minutes = 15`): 96 bins per day.

Inputs:

- **Arrivals**:
  - `A` has exogenous arrivals, say `λ_A,t = 10` flights per bin for bins 20–23.
  - `B` has no exogenous arrivals (`λ_B,t = 0`).
- **Kernels**:
  - For edge `A → B`, kernels say:
    - In bin `t`, 80% of departures from `A` arrive in `B` one bin later (lag=1).
    - 20% arrive two bins later (lag=2).
- **Regulation**:
  - `A` has no regulation.
  - `B` has a capacity limit of `5` flights per bin from bins 22–30.

Data flow:

1. At bins 20–23, `A` sees arrivals `≈10`. Without regulation, `A`’s queue clears each bin, so `departure_mean_A,t ≈ 10`.
2. These departures are routed to `B`:
   - At bin 21, `B` receives 80% of `A`’s bin 20 departures: `≈8`.
   - At bin 22, `B` receives `0.2*10` from bin 20 plus `0.8*10` from bin 21, etc.
3. At bin 22 and onwards, `B` has capacity 5:
   - If `arrival_mean_B,22` is >5, a queue starts to form.
   - Each subsequent bin, `B`’s queue evolves according to the queue-step equations.
4. `ATFMNetworkModel` records:
   - `queue_mean_B,t` rising above 0 in regulated bins where `arrival_mean_B,t + previous_queue > cap`.
   - Total delay contribution from `B` as:
     - `sum_t queue_mean_B,t * delta_minutes`.

This illustrates how **propagation of upstream departures** combined with **local capacity limits** generates network-wide delay.

### Example 2: CLI usage sketch

Assume you have the following files:

- `kernels.csv`: output from your hourly kernel computation pipeline.
- `arrivals.csv`: with columns:
  - `volume_id`, `time_bin`, `lambda_mean`, `lambda_var`, optionally `gamma_lag1`.
- `plan.yaml`: regulation plan, e.g.:

```yaml
time_bin_minutes: 15
volumes:
  VOL_A:
    regulations:
      - start_time: "06:00"
        end_time: "08:00"
        capacity_per_hour: 20
  VOL_B:
    regulations:
      - start_time: "07:30"
        end_time: "09:00"
        capacity_per_hour: 10
```

Example command:

```bash
python -m gemini.gem.atfm_network \
  --kernels-csv /path/to/kernels.csv \
  --arrivals-csv /path/to/arrivals.csv \
  --regulation-plan /path/to/plan.yaml \
  --output-arrivals-csv /tmp/atfm_timeseries.csv \
  --top-volumes 10
```

This will:

- Load kernels/arrivals/plan, build a synthetic volume graph.
- Run the ATFM network propagation.
- Print total delay and top delay-contributing volumes.
- Write a CSV with per-volume, per-bin arrivals, queue, and departures.

### Example 3: Programmatic usage

A minimal sketch of using the engine directly from Python:

```python
from gemini.gem.atfm_network import ATFMNetworkModel
from gemini.gem.hourly_kernel_table import HourlyKernelTable
from gemini.gem.arrival_moments import ArrivalMoments
from gemini.gem.regulation_plan import RegulationPlan
from gemini.propagation.tvtw_indexer import TVTWIndexer
from gemini.propagation.volume_graph import VolumeGraph, Volume
import pandas as pd

# 1) Load kernels and create TVTW indexer
kernels = HourlyKernelTable.from_csv("/path/to/kernels.csv")
tvtw = TVTWIndexer(time_bin_minutes=kernels.delta_minutes)

# 2) Load arrivals
num_bins = tvtw.num_bins
arrivals = ...  # or use _load_arrival_moments_csv if you import it

# 3) Build a simple volume graph for relevant IDs
volume_ids = ["VOL_A", "VOL_B"]
volumes = {vid: Volume(id=vid) for vid in volume_ids}
geo_df = pd.DataFrame({"traffic_volume_id": volume_ids, "geometry": [None] * len(volume_ids)})
from gemini.propagation.volume_graph import VolumeLocator
volume_graph = VolumeGraph(volumes=volumes, geo_dataframe=geo_df, locator=VolumeLocator(geo_df))

# 4) Optional regulation plan
plan = RegulationPlan.load("/path/to/plan.yaml", tvtw)

# 5) Run the ATFM network model
model = ATFMNetworkModel(
    tvtw=tvtw,
    volume_graph=volume_graph,
    kernels=kernels,
    arrivals=arrivals,
)
result = model.run(plan)

# 6) Inspect results
print(result.total_delay_mean, result.total_delay_var)
series_A = result.by_volume["VOL_A"]
print(series_A.queue_mean[:10])  # first 10 bins of queue for VOL_A
```

This programmatic view mirrors what the CLI does, but gives you direct access to `VolumeTimeSeries` and lets you integrate ATFM propagation into other pipelines.

