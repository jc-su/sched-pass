# Blackwell CLC Empirical Study

This folder is intentionally separate from the sched-pass implementation. Its
job is to characterize Cluster Launch Control (CLC) as a hardware work-stealing
primitive before feeding assumptions back into the scheduler design.

The key distinction:

- We do not need the sched-pass LLVM pass here.
- We do need measurement code. "No instrumentation pass" does not mean no
  instrumentation; the standalone kernel records CLC attempts, successes,
  claimed CTA ids, per-task visits, and claim latency.

## Questions

Use this probe to answer observable questions about CLC:

- When does `clusterlaunchcontrol.try_cancel` succeed?
- Which raw `ctaid.x` values does hardware tend to return?
- How much latency does one claim add?
- How does success rate change with grid size, task duration, and imbalance?
- Does CLC improve or hurt makespan compared with static scheduling?
- Which raw id ranges are likely to be stolen, so `task_order[raw]` can be
  arranged intelligently in sched-pass?

CLC does not expose full scheduling control. It can only try to cancel an
unlaunched block from the same grid and return that block's CTA id. The
scheduler controls the launch shape, task granularity, and the mapping from raw
CTA id to logical task.

## Build

On the local Blackwell setup:

```bash
./experiments/clc/build_run.sh
```

Equivalent manual command:

```bash
/usr/bin/clang++-20 -x cuda --cuda-gpu-arch=sm_120 \
  --cuda-path=/usr/local/cuda-12.8 -O2 -std=c++17 \
  -Wno-unknown-cuda-version -L/usr/local/cuda-12.8/lib64 -lcudart \
  experiments/clc/clc_probe.cu -o build/clc_probe
```

To build the decode-shaped probe instead:

```bash
TARGET=clc_decode_probe ./experiments/clc/build_run.sh 4096 8 2 16 16 0
```

To build the 2D-grid probe:

```bash
TARGET=clc_2d_probe ./experiments/clc/build_run.sh 64 128 128 4096 0
```

To build trace, clustered-launch, 3D tuple, partial-participation, runtime
behavior, worker/SM mapping, inter-kernel pressure, and CUDA Graph replay
probes:

```bash
TARGET=clc_trace_probe ./experiments/clc/build_run.sh 8192 1 128 4096 128
TARGET=clc_cluster_probe ./experiments/clc/build_run.sh 8192 128 2 4096
TARGET=clc_tuple_probe ./experiments/clc/build_run.sh 64 16 8 128 4096
TARGET=clc_participation_probe ./experiments/clc/build_run.sh 8192 128 1 4096
TARGET=clc_runtime_probe ./experiments/clc/build_run.sh 8192 128 4096 0
TARGET=clc_mapping_probe ./experiments/clc/build_run.sh 8192 128 4096 0
TARGET=clc_pressure_probe ./experiments/clc/build_run.sh 8192 128 4096 0 128 0 0
TARGET=clc_graph_probe ./experiments/clc/build_run.sh 8192 128 4096 0 5 1
```

## Run

```bash
./build/clc_probe [tasks] [threads] [long_every] [short_iters] [long_iters] [layout] [smem_bytes]
```

Defaults:

```text
tasks=4096 threads=128 long_every=8 short_iters=256 long_iters=8192
layout=0
smem_bytes=0
```

Layouts:

```text
0 = interleaved long tasks: raw % long_every == 0
1 = long-prefix: the first tasks/long_every raw ids are long
2 = long-suffix: the last tasks/long_every raw ids are long
```

Examples:

```bash
./build/clc_probe 64 128 8 256 8192
./build/clc_probe 1024 128 8 256 8192
./build/clc_probe 8192 128 8 256 8192
./build/clc_probe 8192 128 4 256 32768
./build/clc_probe 8192 128 8 1024 32768 1
./build/clc_probe 8192 128 8 1024 32768 2
./build/clc_probe 4096 128 0 8192 8192 0 16384
```

## Sweeps

Run the full standalone characterization suite:

```bash
./experiments/clc/run_all_sweeps.sh
```

The wrapper builds the standalone probes and runs:

```text
threshold sweep       3 repeats
occupancy sweep       3 repeats
claim-order sweep     3 repeats
synthetic workload    5 repeats
decode sweep          5 repeats
2D sweep              3 repeats
cluster sweep         3 repeats
trace capture
full trace analysis
3D tuple capture
partial participation capture
runtime behavior sweep
worker/SM mapping sweep
inter-kernel pressure sweep
CUDA Graph replay checks
generated ISA dump
```

Override repeat counts if needed:

```bash
SYN_REPEATS=5 WORKLOAD_REPEATS=10 DECODE_REPEATS=10 \
ADVERSARIAL_REPEATS=5 RUNTIME_REPEATS=5 PRESSURE_REPEATS=5 \
GRAPH_REPLAYS=20 TRACE_CAP=8192 \
  ./experiments/clc/run_all_sweeps.sh
```

Run individual suites:

```bash
python3 experiments/clc/run_sweep.py --suite threshold --repeats 3
python3 experiments/clc/run_sweep.py --suite occupancy --repeats 3
python3 experiments/clc/run_sweep.py --suite claim-order --repeats 3
python3 experiments/clc/run_sweep.py --suite workload --repeats 5
python3 experiments/clc/run_decode_sweep.py --suite all --repeats 5
python3 experiments/clc/run_2d_sweep.py --suite all --repeats 3
python3 experiments/clc/run_cluster_sweep.py --suite all --repeats 3
python3 experiments/clc/run_adversarial_sweep.py --suite all --repeats 3
python3 experiments/clc/run_runtime_sweep.py --suite all --repeats 3
python3 experiments/clc/run_mapping_sweep.py --suite all
python3 experiments/clc/run_pressure_sweep.py --suite all --repeats 3
CLC_PROBE_CSV=1 ./build/clc_graph_probe 8192 128 4096 0 10 0 \
  > experiments/clc/results/clc_graph_stream_summary.csv
CLC_PROBE_CSV=1 ./build/clc_graph_probe 8192 128 4096 0 10 1 \
  > experiments/clc/results/clc_graph_replay_summary.csv
CLC_TRACE_EVENTS_CSV=1 ./build/clc_trace_probe 8192 1 128 4096 512 \
  > experiments/clc/results/clc_trace_events.csv
CLC_PROBE_CSV=1 ./build/clc_trace_probe 8192 1 128 4096 512 \
  > experiments/clc/results/clc_trace_summary.csv
python3 experiments/clc/analyze_trace.py \
  --events experiments/clc/results/clc_trace_events.csv \
  --summary experiments/clc/results/clc_trace_summary.csv \
  --out experiments/clc/results/clc_trace_analysis.csv
./experiments/clc/dump_generated_isa.sh
```

Optional profiler side channels:

```bash
./experiments/clc/check_profiler_access.sh
./experiments/clc/run_profiler_sidechannels.sh

nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  --output experiments/clc/results/profiler/nsys_clc_runtime \
  ./build/clc_runtime_probe 8192 128 4096 0
```

On the current machine, the loaded NVIDIA module has
`RmProfilingAdminOnly: 1`, so non-sudo `ncu` is blocked. The persistent config
`/etc/modprobe.d/nvidia-profiler-perf-counters.conf` has been installed with
`NVreg_RestrictProfilingToAdminUsers=0`, which takes effect after module reload
or reboot. Until then, `run_profiler_sidechannels.sh` uses `sudo ncu`.
`nsys` timeline capture works without that counter permission.

Core sweep CSV rows include the launch shape, work shape, layout, dynamic
shared memory, occupancy prediction, observed active workers, claim range,
success rate, claim cycles, missed/duplicate counts, and timing delta. The
tuple and participation probes add focused fields for CTA tuple lanes and
participating-worker counts. The runtime probe adds success/failure attempt
counts, SM-id residency, and success/failure claim timing. The mapping probe
adds task-to-worker and worker-to-SM placement. The pressure probe adds
cross-stream and device-global timing to detect actual inter-kernel overlap.
The graph probe checks whether stream launches and CUDA Graph replay reset CLC
state identically. The sweep scripts validate structural acceptance rules and
exit nonzero on any failure.

Interpretation guide:

- If `successes == 0`, CLC had no unlaunched blocks to steal or claims happened
  too late.
- In the runtime probe, `failures == active_workers` means every active worker
  observed one terminal failed cancel and exited the CLC loop.
- If `missed == 0` and `duplicates == 0`, every logical raw CTA id was processed
  exactly once.
- If `structural_ok == 1`, the row matched the current observed contract:
  active workers matched predicted `R`, the claimed range was contiguous, and
  no task was missed or duplicated.
- If CLC is slower than static despite many successes, claim overhead or loss of
  normal scheduler locality may dominate.
- If claimed raw ids cluster in a range, sched-pass can place useful logical
  tasks in that likely-stolen range with `task_order[]`.

## Current Outcome

Current detailed notes are in `experiments/clc/FINDINGS.md`. The short version
from the latest run is:

```text
R = cudaOccupancyMaxActiveBlocksPerMultiprocessor(...) * SM_count

raw [0, R) : initially launched prefix
raw [R, N) : CLC-claimable suffix

if N <= R:
  no useful CLC stealing

if N > R:
  claimed raw ids are [R, N - 1]

runtime behavior:
  attempts  = N
  successes = max(0, N - R)
  failures  = min(N, R)
  terminal failed attempts == active workers
  active workers are balanced across SM ids
  no CTA SM-id migration observed

prefix mapping when N >= R:
  raw ids are striped across all SMs in a repeating SM order
  each SM gets raw ids separated by SM_count, not one contiguous raw range
  claimed suffix work runs on the claiming worker's SM

for 2D grids:
  linear_raw = blockIdx.x + blockIdx.y * gridDim.x

for 3D grids:
  linear_raw = blockIdx.x
             + blockIdx.y * gridDim.x
             + blockIdx.z * gridDim.x * gridDim.y

for clustered launches:
  R_cta = cudaOccupancyMaxActiveClusters(...) * cluster_size
  claimed bases are cluster-aligned

for predictable task_order[] control:
  all resident workers should participate in the CLC loop
```

Do not assume CLC is universally better. The useful result is the boundary:
where CLC is a win, a tie, or a loss.
