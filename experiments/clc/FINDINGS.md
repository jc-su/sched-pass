# CLC Findings

Local machine: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, `sm_120`.
Driver context: NVIDIA open kernel module `590.48.01`, GPU firmware
`590.48.01`, VBIOS `98.02.8D.00.01`, compute capability `12.0`.
Date: 2026-07-03.

This folder treats Blackwell Cluster Launch Control (CLC) as an observable
hardware primitive. The reverse-engineering scope here is generated PTX/SASS
from our own probes plus black-box hardware behavior, not NVIDIA driver or
firmware blobs. The goal is to learn the contract that matters for sched-pass
policy: which raw CTA ids can be claimed, when claims succeed, how claim order
behaves, and when the extra claim path is worth using.

## Current Artifacts

- Synthetic probe: `experiments/clc/clc_probe.cu`
- Decode-shaped probe: `experiments/clc/clc_decode_probe.cu`
- 2D-grid probe: `experiments/clc/clc_2d_probe.cu`
- Trace probe: `experiments/clc/clc_trace_probe.cu`
- Cluster-size probe: `experiments/clc/clc_cluster_probe.cu`
- 3D tuple probe: `experiments/clc/clc_tuple_probe.cu`
- Partial-participation probe: `experiments/clc/clc_participation_probe.cu`
- Runtime-behavior probe: `experiments/clc/clc_runtime_probe.cu`
- Worker/SM mapping probe: `experiments/clc/clc_mapping_probe.cu`
- Inter-kernel pressure probe: `experiments/clc/clc_pressure_probe.cu`
- CUDA Graph replay probe: `experiments/clc/clc_graph_probe.cu`
- Claim-pipelining probe: `experiments/clc/clc_pipeline_probe.cu`
- Memory-bound decode probe: `experiments/clc/clc_membound_probe.cu`
- Cost-noise probe (arming calibration): `experiments/clc/clc_noise_probe.cu`
- Build helper: `experiments/clc/build_run.sh`
- Synthetic sweeps: `experiments/clc/run_sweep.py`
- Decode sweep: `experiments/clc/run_decode_sweep.py`
- 2D sweep: `experiments/clc/run_2d_sweep.py`
- Cluster sweep: `experiments/clc/run_cluster_sweep.py`
- Adversarial sweep: `experiments/clc/run_adversarial_sweep.py`
- Runtime behavior sweep: `experiments/clc/run_runtime_sweep.py`
- Worker/SM mapping sweep: `experiments/clc/run_mapping_sweep.py`
- Inter-kernel pressure sweep: `experiments/clc/run_pressure_sweep.py`
- Trace analyzer: `experiments/clc/analyze_trace.py`
- Mapping analyzer: `experiments/clc/analyze_mapping.py`
- Profiler permission check: `experiments/clc/check_profiler_access.sh`
- Profiler side-channel runner: `experiments/clc/run_profiler_sidechannels.sh`
- Nsight Compute summarizer: `experiments/clc/summarize_ncu_csv.py`
- Full suite wrapper: `experiments/clc/run_all_sweeps.sh`
- Generated ISA dump: `experiments/clc/dump_generated_isa.sh`

Latest full-run CSVs:

- `experiments/clc/results/clc_threshold_raw_20260703_062250.csv`
- `experiments/clc/results/clc_threshold_summary_20260703_062250.csv`
- `experiments/clc/results/clc_occupancy_raw_20260703_062322.csv`
- `experiments/clc/results/clc_occupancy_summary_20260703_062322.csv`
- `experiments/clc/results/clc_claim-order_raw_20260703_062420.csv`
- `experiments/clc/results/clc_claim-order_summary_20260703_062420.csv`
- `experiments/clc/results/clc_workload_raw_20260703_062434.csv`
- `experiments/clc/results/clc_workload_summary_20260703_062434.csv`
- `experiments/clc/results/clc_decode_all_raw_20260703_062513.csv`
- `experiments/clc/results/clc_decode_all_summary_20260703_062513.csv`
- `experiments/clc/results/clc_2d_all_raw_20260703_063603.csv`
- `experiments/clc/results/clc_2d_all_summary_20260703_063603.csv`
- `experiments/clc/results/clc_trace_events_20260703_064637.csv`
- `experiments/clc/results/clc_trace_summary_20260703_064637.csv`
- `experiments/clc/results/clc_trace_events_full_20260703_080431.csv`
- `experiments/clc/results/clc_trace_summary_full_20260703_080432.csv`
- `experiments/clc/results/clc_trace_analysis_full_20260703_080431.csv`
- `experiments/clc/results/clc_cluster_all_raw_20260703_064751.csv`
- `experiments/clc/results/clc_cluster_all_summary_20260703_064751.csv`
- `experiments/clc/results/clc_tuple_raw_20260703_072243.csv`
- `experiments/clc/results/clc_tuple_summary_20260703_072243.csv`
- `experiments/clc/results/clc_participation_raw_20260703_072247.csv`
- `experiments/clc/results/clc_participation_summary_20260703_072247.csv`
- `experiments/clc/results/clc_runtime_all_raw_20260703_081145.csv`
- `experiments/clc/results/clc_runtime_all_summary_20260703_081145.csv`
- `experiments/clc/results/clc_mapping_all_analysis_20260703_171408.csv`
- `experiments/clc/results/clc_pressure_all_raw_20260703_190731.csv`
- `experiments/clc/results/clc_pressure_all_summary_20260703_190731.csv`
- `experiments/clc/results/clc_graph_stream_summary_20260703_191035.csv`
- `experiments/clc/results/clc_graph_replay_summary_20260703_191035.csv`
- `experiments/clc/results/profiler/ncu_clc_runtime_basic_20260703_183150.csv`
- `experiments/clc/results/profiler/ncu_clc_runtime_basic_sudo_20260703_183844.csv`
- `experiments/clc/results/profiler/ncu_clc_runtime_sched_sudo_20260703_183844.csv`
- `experiments/clc/results/profiler/ncu_clc_runtime_summary_20260703_183844.csv`
- `experiments/clc/results/profiler/ncu_clc_runtime_basic_sudo_20260703_191536.csv`
- `experiments/clc/results/profiler/ncu_clc_runtime_sched_sudo_20260703_191536.csv`
- `experiments/clc/results/profiler/ncu_clc_runtime_summary_20260703_191536.csv`
- `experiments/clc/results/profiler/nsys_clc_runtime_20260703_183220.nsys-rep`
- `experiments/clc/results/profiler/nsys_clc_runtime_20260703_183220_stats_cuda_gpu_kern_sum.csv`
- `experiments/clc/results/profiler/nsys_clc_runtime_20260703_183844.nsys-rep`
- `experiments/clc/results/profiler/nsys_clc_runtime_20260703_183844_stats_cuda_gpu_kern_sum.csv`
- `experiments/clc/results/profiler/nsys_clc_runtime_20260703_191536.nsys-rep`
- `experiments/clc/results/profiler/nsys_clc_runtime_20260703_191536_stats_cuda_gpu_kern_sum.csv`
- `experiments/clc/results/isa_clc_probe_20260703_064841.clc_summary.txt`
- `experiments/clc/results/isa_clc_decode_probe_20260703_064841.clc_summary.txt`
- `experiments/clc/results/isa_clc_2d_probe_20260703_064841.clc_summary.txt`
- `experiments/clc/results/isa_clc_trace_probe_20260703_064841.clc_summary.txt`
- `experiments/clc/results/isa_clc_cluster_probe_20260703_064841.clc_summary.txt`
- `experiments/clc/results/isa_clc_tuple_probe_20260703_071556.clc_summary.txt`
- `experiments/clc/results/isa_clc_participation_probe_20260703_071556.clc_summary.txt`
- `experiments/clc/results/isa_clc_runtime_probe_20260703_081237.clc_summary.txt`
- `experiments/clc/results/isa_clc_mapping_probe_20260703_171422.clc_summary.txt`
- `experiments/clc/results/isa_clc_pressure_probe_20260703_191147.clc_summary.txt`
- `experiments/clc/results/isa_clc_graph_probe_20260703_191147.clc_summary.txt`

Validation summary:

```text
suite          groups  runs  bad groups
threshold      36     108   0
occupancy      63     189   0
claim-order    16      48   0
workload       26     135   0
decode         37     185   0
2D             32      96   0
cluster        20      60   0
trace           1       1   0
3D tuple        3       9   0
participation   5      15   0
runtime        15      45   0
mapping        10      10   0
pressure       13      39   0
graph replay    2      20   0
```

Every core suffix-model row had:

```text
missed == 0
duplicates == 0
duplicate_claims == 0
claim_range_holes == 0
structural_ok == 1
```

The adversarial partial-participation rows had exact-once execution but
intentionally broke clean suffix contiguity when only a subset of CTAs entered
the CLC loop:

```text
missed == 0
duplicates == 0
duplicate_claims == 0
exactly_once == 1
claim_range_holes > 0 when claim_stride > 1
```

## Observable CLC Model

CLC returned a contiguous raw-id suffix, not arbitrary raw CTA ids.

```text
R = cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel, blockDim,
                                                  dynamic_smem)
    * SM_count

raw [0, R) : initial hardware-launched prefix
raw [R, N) : CLC-claimable suffix
```

For `N <= R`:

```text
active_workers = N
successes      = 0
claimed range  = none
```

For `N > R`:

```text
active_workers = R
unique_claimed = N - R
claimed range  = [R, N - 1]
```

This held for synthetic clock work, dynamic shared-memory occupancy changes, and
the decode-shaped global-memory probe. For 2D and 3D grids, the same model held
over CUDA's row-major CTA linearization:

```text
linear_raw = blockIdx.x
           + blockIdx.y * gridDim.x
           + blockIdx.z * gridDim.x * gridDim.y
```

At the policy level, the model is now simple enough to use directly:

```text
if non-clustered:
  R = cudaOccupancyMaxActiveBlocksPerMultiprocessor(...) * SM_count

if clustered:
  R = cudaOccupancyMaxActiveClusters(...) * cluster_size

if N <= R:
  do not expect CLC to help

if N > R:
  prefix = raw [0, R)
  suffix = raw [R, N)
```

## Runtime Attempt Model

The runtime probe measures the behavior visible to CTA code around
`try_cancel`, including success/failure counts, failure timing, SM id, and SM id
stability. Across threshold, thread-count, shared-memory, and work-duration
sweeps, 45/45 runtime rows validated:

```text
attempts  = N
successes = max(0, N - R)
failures  = min(N, R)

if N <= R:
  active_workers = N
  every active worker processes its own CTA and then gets one failed cancel

if N > R:
  active_workers = R
  suffix CTAs are claimed successfully
  after suffix exhaustion, each active worker gets one failed cancel
```

The terminal-failure rule is the key control-flow contract:

```text
terminal failed attempts == active_workers
```

Representative threshold rows, 3 repeats per point, `threads=128`,
`work_cycles=4096`, `R=2256`:

```text
N     active  attempts  successes  failures  claimed range
1024  1024    1024      0          1024      none
2256  2256    2256      0          2256      none
2304  2256    2304      48         2256      2256..2303
8192  2256    8192      5936       2256      2256..8191
```

SM residency behavior:

```text
SM ids observed      = 0..187
active SMs           = 188
SM id changes        = 0 in all runtime rows
workers/SM at R=2256 = 12..12
workers/SM at R=1880 = 10..10
workers/SM at R=940  = 5..5
workers/SM at R=564  = 3..3
```

For `N=1024 < R`, the hardware still spread active CTAs across every SM:

```text
active SMs     = 188
workers/SM     = 5..6
mean workers/SM= 5.447
```

Latency observations from the runtime probe are workload-sensitive because many
CTAs contend for CLC at the same time. They are still useful for policy because
they show the extra path has measurable cost:

```text
N=8192, threads=128, R=2256

work cycles  success mean cycles  failure mean cycles
0            ~1919                ~2333
128          ~1802                ~2200
1024         ~1104                ~1299
4096          ~946                 ~961
```

Occupancy-reducing shared memory reduced the active worker count and changed
claim/failure timing:

```text
smem   R     workers/SM  successes  failures
0      2256  12..12      5936       2256
8192   1880  10..10      6312       1880
16384   940   5..5       7252        940
24576   564   3..3       7628        564
```

Interpretation:

- A participating active CTA attempts cancellation after every processed task.
- A successful cancel means it receives one unlaunched raw CTA id.
- A failed cancel is the loop-exit signal; after the suffix is exhausted, each
  active worker observes exactly one failed cancel.
- In these isolated-kernel runs, every failed cancel was the terminal
  no-more-claimable-work case. CUDA's public documentation also allows failure
  for other scheduling reasons, so sched-pass should treat failure as "stop
  claiming now", not as a universal proof about global scheduler state.
- CTAs did not migrate between SM ids while processing claimed work in these
  runs.
- Initial resident CTAs are distributed evenly across SMs according to the
  occupancy model.

## Worker/SM Mapping

The mapping probe records one row per raw task:

```text
raw task id
original worker CTA id
worker initial SM id
execution SM id
worker-local ordinal
claimed vs initial
```

The policy-relevant finding is that, for `N >= R`, prefix raw ids are striped
across all SMs in a repeating SM order. They are not packed as one contiguous
raw range per SM.

For the local RTX PRO 6000 Blackwell, the first 32 SM ids in the repeated
prefix order were:

```text
0 1 24 25 48 49 72 73 96 97 120 121 144 145 2 3
26 27 50 51 74 75 98 99 122 123 146 147 4 5 28 29
```

For `N=8192`, the same repeated prefix order held across thread counts and
shared-memory occupancy changes:

```text
threads  smem   R     prefix waves  workers/SM  raw stride per SM
64       0      4512  24            24..24      188
128      0      2256  12            12..12      188
256      0      1128   6             6..6       188
128      8192   1880  10            10..10      188
128      16384   940   5             5..5       188
128      24576   564   3             3..3       188
```

For `N=1024 < R`, active CTAs were still balanced across SMs (`5..6` workers
per SM), but the strict one-residue-per-SM pattern did not hold for the partial
tail of the launch. This is not a CLC-helpful regime because `N <= R`.

Suffix mapping:

```text
suffix_worker_is_prefix             = 1
suffix_exec_smid_equals_worker_smid = 1
suffix_active_sms                   = 188 when suffix exists
```

For `N=8192`, `threads=128`, `R=2256`, suffix work per SM was balanced but not
perfectly uniform:

```text
suffix records = 5936
suffix/SM      = 28..34
worker ordinal counts:
  first claimed task  = 2256
  second claimed task = ~2236
  third claimed task  = ~1444
```

Interpretation for `task_order[]`:

- Consecutive raw ids in the resident prefix are already spread across SMs by
  the hardware launch order.
- For `N >= R`, each SM receives one raw id per `SM_count` stride inside the
  prefix; on this machine the stride is `188`.
- Heavy prefix tasks should be distributed across the first `SM_count` raw ids
  and repeated waves, not grouped into one narrow raw-id interval.
- Claimed suffix work executes on the SM of the worker that claimed it. Suffix
  placement should therefore feed all workers with useful tail work, not assume
  a fixed suffix raw-id-to-SM mapping.

## Inter-Kernel Pressure

Pressure artifacts:

- Probe: `experiments/clc/clc_pressure_probe.cu`
- Runner: `experiments/clc/run_pressure_sweep.py`
- Raw CSV: `experiments/clc/results/clc_pressure_all_raw_20260703_190731.csv`
- Summary CSV:
  `experiments/clc/results/clc_pressure_all_summary_20260703_190731.csv`

This probe launches a normal pressure kernel and a CLC kernel in separate
nonblocking streams. CUDA events measure stream queue windows, while device-side
`%globaltimer` min/max stamps measure actual CTA execution windows. The
device-side timing is the important side channel: stream events before a kernel
can complete long before that kernel's CTAs are admitted to SMs.

Observed pressure cases, 3 repeats each:

```text
case                       pressure_sms  active_workers  pressure_us  clc_us   clc_start_delta_us  actual_overlap_us
baseline                            0            2256        0.000   11.435              0.000             0.000
one pressure block, 128t             1            2256    85013.077   11.392          85020.981             0.000
one pressure block, 1024t            1            2256    84966.091   11.467          84973.664             0.000
half device, 1024t                  94            2256    84427.136   11.339          84434.048             0.000
full device, 128t                  188            2256    85372.469   11.392          85379.339             0.000
full device, 1024t                 188            2256    84813.675   11.360          84821.323             0.000
dynamic smem one, 32K                1            2256    85539.424   11.435          85546.677             0.000
dynamic smem full, 32K             188            2256    85546.603   11.403          85554.859             0.000
priority pressure high              1            2256    85489.696   11.509          85496.800             0.000
priority equal                      1            2256    85452.629   11.445          85461.003             0.000
priority CLC high                   1            2256    85476.608   11.531          85484.757             0.000
CLC first, short work                1            2256    85161.877 3453.963          -3461.440             0.000
CLC first, long work                 1            2256    84893.717 33946.752        -33953.920             0.000
```

Interpretation:

- Actual CTA execution overlap was `0.000 us` in every pressure row.
- If pressure was launched first, CLC's actual first CTA started after the
  pressure kernel ended, even with only one pressure block and even when the CLC
  stream had higher priority.
- If CLC was launched first, the pressure kernel started after CLC ended.
- `active_workers` stayed `2256`, equal to standalone
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor(...) * SM_count`.
- The CLC suffix contract stayed intact in every row:

```text
missed == 0
duplicates == 0
duplicate_claims == 0
claim_range_holes == 0
successes == 5936
failures == 2256
claimed_min == 2256
claimed_max == 8191
structural_ok == 1
```

External kernel pressure did not produce a smaller observable CLC resident
prefix in these tests. The runtime serialized the CLC launch with the other
kernel, then admitted CLC with its normal standalone `R`. For sched-pass, do
not design CLC policy around partial inter-kernel co-residency or stream
priority preemption. Treat CLC as intra-grid work stealing after the CLC kernel
is admitted.

## CUDA Graph Replay

Graph artifacts:

- Probe: `experiments/clc/clc_graph_probe.cu`
- Stream replay CSV:
  `experiments/clc/results/clc_graph_stream_summary_20260703_191035.csv`
- CUDA Graph replay CSV:
  `experiments/clc/results/clc_graph_replay_summary_20260703_191035.csv`

Result over 10 repeated launches in each mode:

```text
mode          replays  active_workers  successes  failures  claimed_min  claimed_max  invalid_replays
stream             10            2256       5936      2256         2256         8191                0
cuda graph         10            2256       5936      2256         2256         8191                0
```

CUDA Graph replay does not appear to preserve stale CLC cancel state across
replays. The graph and stream modes both reset to the same observable contract:
same `R`, same suffix, same terminal failed attempts, no missed work, and no
duplicate work.

## Threshold Sweep

Uniform synthetic work, 3 repeats per point:

```text
threads  R     tasks at R  first above R  success at 8192  claim range at 8192
64       4512  no claims   4512..4607    44.922%          4512..8191
128      2256  no claims   2256..2303    72.461%          2256..8191
256      1128  no claims   1128..1535    86.230%          1128..8191
```

The threshold is exact: at `tasks == R`, there were no claims; once
`tasks > R`, the first claimed raw CTA id was `R`.

## Occupancy Sweep

The occupancy sweep varied threads/block and dynamic shared memory, then tested
task counts below, at, and above the predicted `R`.

```text
threads  smem bytes  occ blocks/SM  R
64       0           24             4512
64       4096        19             3572
64       8192        10             1880
64       12288       7              1316
64       16384       5              940
64       24576       3              564
64       32768       3              564

128      0           12             2256
128      4096        12             2256
128      8192        10             1880
128      12288       7              1316
128      16384       5              940
128      24576       3              564
128      32768       3              564

256      0           6              1128
256      4096        6              1128
256      8192        6              1128
256      12288       6              1128
256      16384       5              940
256      24576       3              564
256      32768       3              564
```

`SM_count = 188`, and every observed active-worker count matched
`occ_blocks_per_SM * 188`.

## 2D Grid Behavior

The 2D probe captures the full `get_first_ctaid.v4` result. CLC still claimed a
contiguous suffix, but the suffix is over CUDA's linear CTA order:

```text
linear = x + y * grid_x
```

Shape sweep, 3 repeats per point, `tasks ~= 8192`, `threads=128`,
`R=2256`:

```text
grid       R coord   claimed coord range
8192x1     (2256,0)  (2256,0)..(8191,0)
4096x2     (2256,0)  (2256,0)..(4095,1)
2048x4     (208,1)   (208,1)..(2047,3)
1024x8     (208,2)   (208,2)..(1023,7)
512x16     (208,4)   (208,4)..(511,15)
256x32     (208,8)   (208,8)..(255,31)
128x64     (80,17)   (80,17)..(127,63)
64x128     (16,35)   (16,35)..(63,127)
32x256     (16,70)   (16,70)..(31,255)
17x512     (12,132)  (12,132)..(16,511)
```

Threshold checks:

```text
grid     tasks  result
47x48    2256   no claims
48x48    2304   claimed 2256..2303, coord (0,47)..(47,47)
64x35    2240   no claims
64x36    2304   claimed 2256..2303, coord (16,35)..(63,35)
```

Dynamic shared memory moved the 2D cutoff exactly as the occupancy model
predicts:

```text
grid    smem   occ blocks/SM  R     R coord
64x128  0      12             2256  (16,35)
64x128  8192   10             1880  (24,29)
64x128  16384  5              940   (44,14)
64x128  24576  3              564   (52,8)
```

Implication: sched-pass can treat `task_order[linear_raw]` as the CLC policy
surface even when the real kernel uses 2D launch geometry. The policy still
needs to compute `R` in linear CTA space, then map that boundary back to the
kernel's `(x,y,z)` meaning if task assignment depends on dimensions.

## 3D Tuple Behavior

The tuple probe captures the full `get_first_ctaid.v4` result instead of only
`ctaid.x`. It verifies that the CLC-returned tuple maps to CUDA's normal 3D
linear CTA order:

```text
linear = x + y * grid_x + z * grid_x * grid_y
```

Adversarial shapes, 3 repeats per shape:

```text
grid        tasks  R     claimed range  w range  w nonzero  result
64x16x8     8192   2256  2256..8191    0..0     0          exact suffix
16x16x32    8192   2256  2256..8191    0..0     0          exact suffix
17x19x29    9367   2256  2256..9366    0..0     0          exact suffix
```

Every row had:

```text
processed == tasks
active_workers == R
missed == 0
duplicates == 0
duplicate_claims == 0
claim_range_holes == 0
structural_ok == 1
```

Interpretation:

- For non-clustered CLC, `get_first_ctaid.v4` returns the CTA coordinate tuple
  for the canceled block.
- The fourth lane was always zero in these non-clustered probes; no policy
  signal was observed there.
- The sched-pass policy can be expressed over linear raw CTA id and then mapped
  back to `(x,y,z)` without losing the CLC boundary.

## Trace-Level Claim Order

The trace probe records the first successful claim completions with a global
atomic sequence number. This is not NVIDIA's hidden queue state; it is the order
that ordinary kernel code can observe.

Initial sample run:

```text
grid=8192x1
threads=128
work_cycles=4096
R=2256
trace_recorded=512 of 5936 successful claims
```

Structural behavior still matched the suffix model:

```text
claimed range = 2256..8191
missed        = 0
duplicates    = 0
```

But observed completion order was not monotonic:

```text
first 32 claimed ids:
2256, 2258, 2270, 2268, 2269, 2264, 2267, 2303,
2260, 2263, 2297, 2261, 2329, 2324, 2374, 2373,
2280, 2276, 2279, 2272, 2266, 2315, 2336, 2326,
2349, 2377, 2359, 2281, 2309, 2391, 2310, 2320

events recorded = 512
unique events   = 512
inversions      = 247
```

Full trace run:

```text
grid=8192x1
threads=128
work_cycles=4096
R=2256
trace_recorded=5936 of 5936 successful claims
```

The full trace gives these order and wave metrics:

```text
claimed range           = 2256..8191
unique claims           = 5936
claim range holes       = 0
first observed claim    = 2256
last observed claim     = 8190
adjacent inversions     = 2807
total inversions        = 167982
max backward step       = 286
max forward step        = 336
workers with claims     = 2256
worker claims min/max   = 1..3
worker claims mean      = 2.631
successful claim p50    = 682 cycles
successful claim p90    = 1298 cycles
successful claim p99    = 1628 cycles
```

Wave view by `processed_before`, which is how many tasks that worker had
completed before the successful claim:

```text
wave  count  claimed range  holes inside range
1     2256   2256..5764     1253
2     2237   3930..8018     1852
3     1443   6076..8191      673
```

Failure/exhaustion behavior:

```text
attempts                 = 8192
successes                = 5936
attempts - successes     = 2256
active_workers           = 2256
terminal failed attempts = active_workers
```

Interpretation:

- The claimed set is deterministic enough to model as suffix `[R, N)`.
- The order in which workers observe successful claims is wave-like and
  non-monotonic.
- After the claimable suffix is exhausted, each active worker performs one
  terminal failed cancel attempt and exits its CLC loop.
- Policy should rely on region placement, not exact per-claim FIFO order.
- Put useful work across the early suffix window, not only at a single raw id.

## Partial Participation

The normal probes make every initially launched CTA participate in the CLC loop.
The partial-participation probe asks what happens if only CTAs whose
`raw % claim_stride == 0` use CLC and the rest process their initial CTA then
exit.

This still preserved exact-once execution, but it broke the clean suffix
contract:

```text
tasks=8192, threads=128, R=2256, work_cycles=4096, repeats=3

claim_stride  active workers mean  participants mean  claims mean  claimed min range  claimed max range  holes mean
1             2256.0               2256.0             5936.0       2256..2256         8191..8191         0.0
2             4133.3               2070.7             4058.7       2623..2625         8191..8191         1509.3
4             5816.3               1454.7             2375.7       2837..2852         8185..8191         2969.7
8             6904.0                870.3             1288.0       3122..3136         8173..8184         3761.7
16            7544.7                473.0              647.3       3347..3355         8126..8191         4170.7
```

Every row had:

```text
processed == tasks
missed == 0
duplicates == 0
duplicate_claims == 0
exactly_once == 1
```

Interpretation:

- CLC cancellation remains safe: no task was lost or duplicated.
- If nonparticipating CTAs exit, the normal hardware launcher continues to
  start later raw ids. Those raw ids are no longer available for CLC, so the
  claimed set becomes sparse.
- `active_workers` grows above the predicted resident prefix because additional
  CTAs are launched normally after nonparticipants finish.
- The clean policy surface `suffix = [R, N)` requires all resident workers to
  stay in the CLC claim loop.

Scheduler implication:

```text
For predictable task_order[] control, every initial worker that can finish early
must participate in the CLC loop until cancellation fails. If only a subset
participates, CLC still preserves correctness but the suffix policy becomes
sparse and timing-dependent.
```

## Clustered Launch Behavior

Public CLC documentation describes cancellation at cluster granularity. The
cluster probe uses runtime `cudaLaunchKernelEx` cluster dimensions and exactly
one CLC request per running cluster. Each running cluster then covers all CTAs in
the canceled cluster.

For the 128-thread synthetic probe, clustered launch changed the resident prefix
dramatically. Non-clustered 128-thread CLC had `R=2256`; clustered launch had
about `R_cta=376`.

3 repeats per point:

```text
tasks  cluster_x  active clusters  R_cta  claimed CTA range  claimed base range
8192   1          376              376    376..8191          376..8191
8192   2          188              376    376..8191          376..8190
8192   4          94               376    376..8191          376..8188
8192   8          46               368    368..8191          368..8184

16384  1          376              376    376..16383         376..16383
16384  2          188              376    376..16383         376..16382
16384  4          94               376    376..16383         376..16380
16384  8          46               368    368..16383         368..16376
```

Threshold checks:

```text
cluster_x  R_cta  at R       first above R
1          376    no claims  base 376
2          376    no claims  base 376
4          376    no claims  base 376
8          368    no claims  base 368
```

Cluster conclusions:

- When launch cluster dimensions are set, use
  `cudaOccupancyMaxActiveClusters(...) * cluster_size` to estimate the CTA-space
  prefix, not `cudaOccupancyMaxActiveBlocksPerMultiprocessor(...) * SM_count`.
- CLC returns the first CTA id of a canceled cluster.
- Claimed cluster bases are aligned to `cluster_x`.
- The claimed CTA set still covers a contiguous suffix exactly once, but claims
  happen in cluster-sized chunks.
- Clustered launch itself can reduce residency sharply. For this probe,
  `cluster_x=1` with cluster launch attributes had `R_cta=376`, not the
  non-clustered `R=2256`.

## Claim Order

Claimed raw ids were contiguous `[R, N)` in every claim-order run. The order
looks like a suffix queue consumed in waves by the active worker set. Uniform
synthetic rows show the wave structure most clearly:

```text
tasks  smem   R     first-claim range  last-claim range  max processed
8192   0      2256  2256..5787         5746..8191        4
8192   16384  940   940..1879          7220..8191        9
16384  0      2256  2256..5626         12788..16383      8
16384  16384  940   940..1879          15435..16383      18
```

Interpretation:

- The first successful claim is always in the suffix.
- The first wave covers roughly one active-worker set when work is uniform.
- Workers can claim multiple suffix CTAs; `max_processed` grows as `(N - R)`
  grows or as `R` shrinks.
- Layout and task duration perturb the exact first/last claim ranges, but did
  not break suffix contiguity or exactly-once coverage.

Average claim cost in these runs was roughly:

```text
synthetic: ~575-3200 cycles/attempt, depending on block shape and work regime
decode:    ~560-1350 cycles/attempt
```

Claim cost is not free. It is small compared with long resident CTAs, but it can
dominate tiny kernels or grids where `N <= R`.

The 2D latency sub-sweep also showed the measured claim-attempt cost is higher
when CTAs perform almost no useful work before claiming:

```text
grid     work cycles  claim cycles avg
64x128   0            ~1957
64x128   128          ~1848
64x128   1024         ~1087
64x128   4096         ~895
8192x1   0            ~1957
8192x1   128          ~1838
8192x1   1024         ~1071
8192x1   4096         ~894
```

This is not a pure hardware latency microbenchmark because it includes the
probe's loop, synchronization, and memory-recording overhead. It is still useful
for policy: tiny CTA bodies are a bad CLC regime.

## Generated ISA Inspection

The actionable reverse-engineering path here is inspection of our own generated
PTX/SASS plus black-box behavior. Driver or firmware internals are not needed to
derive the sched-pass policy rule.

PTX emitted by clang contains the expected CLC sequence:

```text
mbarrier.init.shared::cta.b64
clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64
mbarrier.try_wait.parity.shared::cta.b64
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128
clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
```

`cuobjdump --dump-sass` lowers the CLC path to instructions including:

```text
UGETNEXTWORKID.SELFCAST
SYNCS.EXCH.64
SYNCS.ARRIVE.TRANS64
SYNCS.PHASECHK.TRANS64.TRYWAIT
FENCE.VIEW.ASYNC.S
MEMBAR.ALL.CTA
S2R/S2UR SR_CgaCtaId
```

Resource usage from the generated cubins:

```text
kernel                   regs  static shared  note
clc_probe                16    1072           synthetic CLC
static_probe             10    1024           synthetic baseline
clc_decode_probe         40    1064           decode-shaped CLC
static_decode            38    0              decode-shaped baseline
clc_2d_probe             20    1072           2D CLC
static_2d_probe          10    1024           2D baseline
clc_trace_probe          26    1064           trace CLC
clc_cluster_probe        24    1048           clustered-launch CLC
clc_tuple_probe          22    1064           3D tuple CLC
clc_participation_probe  14    1064           partial-participation CLC
clc_runtime_probe        20    1064           runtime behavior CLC
clc_mapping_probe        20    1064           worker/SM mapping CLC
clc_pressure_probe       16    1064           inter-kernel pressure CLC
pressure_kernel          16    0              pressure side workload
clc_graph_probe          14    1064           CUDA Graph replay CLC
```

The key ISA-level observation is that CLC is not just a compiler hint in these
probes. The compiled SASS contains a distinct work-id instruction
(`UGETNEXTWORKID.SELFCAST`) plus synchronization/barrier plumbing around the
asynchronous cancel/query flow. That supports treating CLC as a real hardware
work-stealing primitive with observable cost and resource footprint.

## Driver/Firmware Context

Local NVIDIA context recorded from public system interfaces:

```text
GPU              NVIDIA RTX PRO 6000 Blackwell Server Edition
driver           590.48.01
kernel module    NVIDIA UNIX Open Kernel Module for x86_64 590.48.01
GPU firmware     590.48.01
VBIOS            98.02.8D.00.01
compute cap      12.0
```

NVIDIA driver module files are present under the system module tree, but this
study does not disassemble proprietary driver or firmware blobs. For sched-pass,
the useful contract is the observable GPU behavior:

```text
R calculation
claimable raw-id set
claim-order shape
terminal failed-attempt behavior
cluster alignment
CTA tuple mapping
```

Those are all captured by runnable probes and generated PTX/SASS from our own
code. Driver or firmware internals would not give a supported scheduling
contract for sched-pass, and any discovered detail could change across driver,
firmware, or GPU revisions.

## Profiler Side Channels

Local profiler availability:

```text
Nsight Compute: /usr/local/bin/ncu
Nsight Systems: /usr/local/bin/nsys
CUPTI:          /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcupti.so
```

Nsight Compute as the normal user is blocked because the currently loaded
NVIDIA module has admin-only profiling enabled:

```text
RmProfilingAdminOnly: 1
ERR_NVGPUCTRPERM
```

The persistent config for non-admin profiling has been installed:

```text
/etc/modprobe.d/nvidia-profiler-perf-counters.conf
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

That config takes effect after the NVIDIA module is reloaded or after reboot.
For the current loaded module, `sudo ncu` already unlocks counters.

Nsight Compute side-channel result from `sudo ncu` on `clc_runtime_probe`:

```text
Launch:
  grid size                  8192
  block size                 128
  cluster scheduling policy  PolicySpread
  registers/thread           20
  driver shared/block        1024 bytes
  static shared/block        40 bytes
  dynamic shared/block       0 bytes
  SMs                        188
  TPCs                       94
  waves/SM                   3.63

Occupancy:
  theoretical occupancy      100%
  achieved occupancy         ~91.1%
  theoretical warps/SM       48
  achieved warps/SM          ~43.7

Scheduler:
  no eligible                44.31%
  active warps/scheduler     10.97
  eligible warps/scheduler   2.47
  issued warp/scheduler      0.56

Instruction:
  issued instructions        10,440,660
  executed instructions      10,377,412
  local memory spills        0
  shared memory spills       0
```

Nsight Systems succeeded as a timeline side channel:

```text
kernel:  clc_runtime_probe(...)
time:    11872 ns
count:   1 instance
```

Interpretation:

- `nsys` can confirm launch/timeline behavior and kernel-level duration.
- `sudo ncu` can collect occupancy, scheduler, warp-state, instruction, and
  workload metrics now.
- non-sudo `ncu` should work after module reload/reboot because the persistent
  profiling option is installed.
- The kernel-side probes remain the strongest source of CLC-specific behavior
  because they expose claim/failure counts, SM id, task ownership, and claim
  order directly from device code.

## Workload Value

Timing is secondary. The structural behavior was stable; speedup depends on
task duration, memory pressure, and where heavy tasks live.

Representative synthetic clock-work results:

```text
tasks  threads  long_every  layout        short/long cycles  delta mean
4096   128      4           interleaved   1024/32768         +8.89%
8192   128      4           long-suffix   1024/32768         +26.45%
8192   64       8           interleaved   1024/32768         +8.48%
8192   128      8           interleaved   1024/32768         +7.26% noisy
8192   128      4           long-prefix   1024/32768         -4.48%
8192   128      8           short work    256/8192           -8.62%
8192   256      8           short work    256/8192           -8.70%
16384  128      8           long-suffix   1024/32768         +6.40%
```

Synthetic timing confirms that CLC can win, tie, or lose. It wins when there is
useful stealable tail work and enough CTA work to amortize claims. It loses when
claim overhead is large relative to the task body or when the launch is already
well balanced.

## Decode-Shaped Probe

The decode-shaped probe streams global-memory KV blocks with one CTA per raw
task and `D=128` threads. Its structural result matches the synthetic probe:

```text
threads/block = 128
R             = 2256
claimed range = [2256, N - 1] when N > 2256
```

Uniform short decode work, 5 repeats:

```text
tasks  claimed range  delta mean
1024   none           -5.22%
2048   none           -12.84%
2256   none           +5.22% noisy
2304   2256..2303     +6.63% noisy
3072   2256..3071     -13.05%
4096   2256..4095     +4.31%
8192   2256..8191     +65.47%
```

Heterogeneous decode work, `short_blocks=2`, `long_blocks=16`,
`page_tokens=16`:

```text
tasks  long_every  layout        delta mean
4096   8           interleaved   +39.74%
4096   8           long-prefix   +58.42%
4096   8           long-suffix   +46.20%
4096   4           interleaved   +7.15%
4096   4           long-prefix   +8.60%
4096   4           long-suffix   +8.21%

8192   8           interleaved   -0.18%
8192   8           long-prefix   +0.92%
8192   8           long-suffix   +2.95%
8192   4           interleaved   -1.63%
8192   4           long-prefix   -0.73%
8192   4           long-suffix   +0.41%

16384  8           interleaved   -0.66%
16384  8           long-prefix   -0.68%
16384  8           long-suffix   +0.03%
```

Heavier decode work, `short_blocks=4`, `long_blocks=32`, mostly collapsed to
tie:

```text
tasks  long_every  best layout    best delta mean
4096   8           long-suffix    +2.06%
4096   4           long-prefix    -0.22%
8192   8           long-prefix    +0.60%
8192   4           long-suffix    +0.70%
```

Interpretation:

- CLC helps most in a middle regime: enough suffix to steal, but not so much
  memory work that both static and CLC are bandwidth-saturated.
- Light/moderate decode-shaped work around `4096` tasks can see large wins.
- Larger or heavier decode-shaped work trends toward tie.
- Layout matters, but no single global ordering dominates every regime.

## Scheduler Implications

The sched-pass CLC policy should be region-aware:

```text
R = cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel, blockDim,
                                                  dynamic_smem)
    * SM_count

if N <= R:
  do not expect CLC to help

if N > R:
  prefix = mixed heavy + short/medium workers, size R, spread across SM-count
           stripes
  suffix = remaining heavy/high-variance work first, then lighter work
  task_order = prefix + suffix
```

For clustered launches:

```text
R_cluster = cudaOccupancyMaxActiveClusters(kernel, launch_config)
R_cta     = R_cluster * cluster_size

if N <= R_cta:
  do not expect CLC to help

if N > R_cta:
  prefix = linear CTA ids [0, R_cta)
  suffix = cluster-aligned linear CTA ids [R_cta, N)
```

For multidimensional kernels, define the policy over linear CTA id first:

```text
linear_raw = x + y * grid_x + z * grid_x * grid_y
```

Then map `linear_raw` to whatever logical request/block/tile the original
kernel uses.

Why this is different from flat LPT:

- Hardware starts `raw [0, R)` immediately.
- Hardware exposes `raw [R, N)` to CLC cancellation/claiming.
- sched-pass controls `task_order[raw]`, so policy should place logical tasks
  with awareness of those two hardware-visible regions.
- Under clustered launch, CLC steals cluster-sized chunks and returns the first
  CTA id of the canceled cluster, so suffix placement should respect cluster
  alignment.

Full-participation precondition:

```text
Every launched worker should participate in the CLC loop until cancellation
fails. Otherwise the CUDA launcher can start suffix CTAs normally, and the
claimed set becomes sparse instead of the clean [R, N) suffix.
```

Failure-handling precondition:

```text
After a worker observes a failed cancellation, break its CLC loop. Do not issue
another cancellation request from that worker after observing failure.
```

Practical rule:

- Put enough heavy work in the prefix so long requests start early.
- Keep enough short/medium work in the prefix so some CTAs finish and steal.
- Spread prefix heavy work across `SM_count`-sized stripes, because consecutive
  prefix raw ids are striped over SMs in a repeated hardware SM order.
- Put additional heavy or high-variance work early in the suffix so fast prefix
  CTAs can pull useful tail work forward.
- Disable CLC or expect tie/loss when `N <= R`, when task bodies are tiny, or
  when the kernel is already bandwidth-saturated.
- Do not use CLC as an inter-kernel scheduling or preemption mechanism. In the
  pressure sweep, CLC and normal kernels serialized in launch order; CLC did not
  start on idle-looking partial resources and did not gain priority preemption
  from a higher-priority stream.
- Estimate `R` for the CLC kernel in isolation at admission time. The pressure
  sweep did not show a reduced `R` caused by another running kernel, because CLC
  was admitted only after that kernel ended.
- CUDA Graph replay is compatible with the observed CLC contract in the tested
  shape. Captured graph replay reset claim state across 10 replays and matched
  normal stream launches.

The current evidence is sufficient to design a first sched-pass CLC policy
around `R` and `task_order[]`. Further work should validate the same rule on the
actual FlashInfer/SGLang kernel shapes because register count, dynamic shared
memory, and CTA geometry directly change `R`.

## Claim Pipelining and the task_order Substitution (follow-up)

Every probe above (and the `SchedWorkQueue` driver) issues `try_cancel` AFTER
the task body, so measured "CLC ties/loses" numbers were taken with claim
latency fully EXPOSED. Two follow-up probes ask (a) whether pipelining the claim
(NVIDIA's async-early pattern) recovers the loss, and (b) whether CLC buys
anything the sched-pass `task_order` lever does not already give:

- `experiments/clc/clc_pipeline_probe.cu` -- compute-bound (clock-burn), three
  schedules: `static` (grid=tasks), `exposed` (burn; issue+collect), `pipelined`
  (issue-ahead; body overlaps the in-flight cancel). Split `clc_issue()` /
  `clc_collect()` helpers over CUDA `__shared__` res/bar so the body runs between
  the two halves.
- `experiments/clc/clc_membound_probe.cu` -- memory-bound global KV stream,
  UNIFORM per-task work (no ordering advantage possible), swept across memory
  footprint (L2-resident -> DRAM-bound) and bit-exact-checked vs static.

Build (same as the other probes): `clang++-20 -x cuda --cuda-gpu-arch=sm_120
--cuda-path=/usr/local/cuda-12.8 -O2 -std=c++17 -Wno-unknown-cuda-version <p>.cu
-lcudart -o <p>`.

### Finding 1: try_cancel.async overlaps compute (claim latency is hideable)

Mean collect-wait cycles on tid0, `tasks=8192 threads=128 R=2256`, uniform work:

```text
work_cycles   exposed claim   pipelined claim
1024          1259            562
2048           955            106
4096           893            110
8192           874            121
```

Once the body is >= ~1k cycles the async cancel fully overlaps it: the collect
wait drops from ~900 to ~110 cycles. So the current exposed-claim design wastes
~800 cycles/claim. The overlap is real hardware behavior, not a hint.

### Finding 2: naive claim-ahead pipelining BREAKS load balancing

`try_cancel` removes a block from the pool at ISSUE time. Issuing the next claim
before the current body reserves a task while the worker is still busy, so idle
workers cannot steal it -- head-of-line blocking behind stragglers. In the exact
imbalanced regime CLC exists for, pipelining destroys the win it was meant to
help (`tasks=8192`, 25% long x32, `work=1024`):

```text
layout               static_us  exposed_us  (vs st)   pipelined_us  (vs st)
long-suffix (bad)      30.96      24.95     -19.4%      37.56       +21.3%
long-prefix (LPT)      22.56      24.66      +9.3%      24.64        +9.2%
interleaved            34.75      37.01      +6.5%      49.52       +42.5%
```

The wait IS hidden (pipelined claim ~155 cyc vs exposed ~1000) but the
load-balance loss dwarfs it. Design rule: keep LATE BINDING (claim only when
free); do NOT pipeline the CLC claim. On uniform work pipelining is neutral
(no stragglers to block), so the harm is specific to heterogeneity.

### Finding 3: for the ordering win, task_order dominates CLC

CLC's headline win is entirely "rescue a bad task order". Best makespan per
strategy, same strong-imbalance point (`work=1024`):

```text
strategy                              order         makespan   vs best
static (no CLC) + LPT                 longs first    22.56 us   --  (best)
CLC exposed + LPT                     longs first    24.66 us   +9%
CLC exposed (rescues bad order)       longs last     24.95 us   +11%
static + bad order                    longs last     30.96 us   +37%
static + unsorted                     interleaved    34.75 us   +54%
```

The -19% CLC "win" is 30.96 -> 24.95, i.e. recovering from the longs-last order.
But the control plane sorts longs first (LPT), which puts static at 22.56 us --
better than any CLC configuration. CLC and `task_order` are SUBSTITUTES for the
ordering win, and `task_order` is the strictly better one (same front-loading,
zero per-claim overhead). CLC only helps against a launch order you control and
would never choose.

### Finding 4: no memory-system CLC benefit; the +65% decode row does not reproduce

Uniform memory-bound decode, `tasks=8192`, warmed (40 iters), bit-exact vs
static in every cell, swept across the L2 boundary (L2 = 128 MB):

```text
nb=2   footprint   static_us  exposed(vs st)  pipelined(vs st)  static GB/s
       8 MB          74.05      +2.4%           -0.3%            7250   (L2 hit)
       128 MB        81.56      -2.3%           -2.5%            6583   (~L2 edge)
       512 MB       362.79      +0.3%           +0.2%            1480   (DRAM)

nb=1   8 MB          39.06      +1.8%           -1.0%            6872
       128 MB        46.84      -1.1%           -8.1%            5731
       256 MB       186.39      -0.0%           +0.0%            1440

nb=8   8 MB         284.96      +0.5%           -1.0%            7536
       512 MB      1416.15      -1.4%           -4.5%            1516
       2 GB        1418.78      +0.2%           +0.3%            1514
```

CLC ties static from deep L2 residency to deep DRAM saturation, including
`nb=1` (8192 one-page tasks -- maximum block-launch churn, CLC's best case for
amortizing launch). There is NO systematic footprint>L2 win the residency
hypothesis would predict. The `+65.47%` uniform-decode row in "Decode-Shaped
Probe" (whose own neighbors were +4% and -13%) does not reproduce under a
warmed, correctness-checked loop; treat it as a transient outlier.

### Finding 5: cost-noise sweep -- when does late binding beat open-loop pi?

`experiments/clc/clc_noise_probe.cu` degrades the LPT estimate with
multiplicative noise (c_hat = c * max(.05, 1 + eps*u), u ~ U(-1,1), fixed per
task) and runs static+pi(eps) vs CLC+pi(eps), same order table for both.
`tasks=8192 threads=128 work=1024 x32 every 4, R=2256, 40 iters`, seeds {1,7}:

```text
pi          recall%   static_us   clc_us   clc_vs_st     st_vs_oracle
eps=0.00      100.0     20.5        20.5      +-0.2%        --
eps=0.25      100.0     20.5        20.6      +-0.2%        +0%
eps=0.50      100.0     20.5        20.6      +-0.2%        +0%
eps=1.00     ~97        30.8        30.8      +-0.2%        +50%
eps=2.00     ~74        25.5-26.3   23.6-25.7 -2.4..-7.5%   +24-28%
eps=4.00     ~62        22.6-23.3   21.8-21.9 -3.7..-6.0%   +10-14%
identity      25.0      29.9-30.1   27.6-28.0 -7.2..-7.5%   +45-47%
```

(recall% = true longs ranked in the top-n_long slots.)

Four lessons, two of them counter to the naive "CLC = insurance" story:

- LPT ordering is remarkably NOISE-TOLERANT for a bimodal (32x) decode mix:
  +-50% multiplicative error leaves the ranking effectively intact (recall
  100%, +0% penalty). Mild estimator error is NOT a reason to arm CLC.
- A single late-issued straggler is unfixable by CLC: at eps=1.0 recall is
  97% (a few longs mispositioned late) and BOTH schedules pay the same +50%.
  Late binding balances WHO runs a task; it cannot parallelize one long task
  that pi issued last. The only fix is issuing it early -- i.e. per-request
  residual tracking so an observed long is never mispredicted again. The
  exposure is COLD requests, not noisy ones.
- CLC pays only under SEVERE order breakdown (recall <~75%, e.g. cold start /
  no estimate): -2..-7.5% vs static with the same broken pi. Modest, real.
- The arming threshold follows: uncertainty ~= eps/2 for this noise model, so
  arm at uncertainty >~ 0.75 (eps >= ~1.5-2), i.e. only when the estimate is
  close to uninformative. SCHED_CLC_RESID defaults to 0.75 accordingly.

### Revised scheduler implication

On the 188-SM RTX PRO 6000 (sm_120), once `task_order` is in use CLC does not
add a makespan benefit in either regime:

```text
compute-bound: CLC's only lever is ordering, and task_order captures it.
memory-bound:  CLC ties static across L2 -> DRAM (no residency effect).
pipelining:    hides claim latency but breaks balancing; do not use it.
```

Recommendation unchanged from PROJECT_REMAINING but now with a mechanism: lead
with `task_order` + a sparsely-sampled timer. Treat CLC as a correctness-
preserving substrate for full-grid dynamic claim, not a speedup. Its makespan
win lives in the workers<<tasks TICKET regime (A6000 -32%), not full-grid CLC on
a big GPU whose scheduler already balances. Wire baked-ABI CLC for correctness/
portability, but do not expect it to beat `task_order` here.

## Public References

These references were used only to choose safe, reproducible experiments. The
behavioral conclusions above come from the local probes and CSVs.

- NVIDIA CUTLASS Blackwell CLC documentation:
  `https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html`
- NVIDIA CUDA Programming Guide CLC documentation:
  `https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html`
- NVIDIA CUTLASS/CuTe CLC API documentation:
  `https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html`
- LLVM NVPTX backend CLC intrinsic documentation:
  `https://llvm.org/docs/NVPTXUsage.html`
