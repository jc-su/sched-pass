#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-python3}
SYN_REPEATS=${SYN_REPEATS:-3}
WORKLOAD_REPEATS=${WORKLOAD_REPEATS:-5}
DECODE_REPEATS=${DECODE_REPEATS:-5}
TWO_D_REPEATS=${TWO_D_REPEATS:-3}
CLUSTER_REPEATS=${CLUSTER_REPEATS:-3}
ADVERSARIAL_REPEATS=${ADVERSARIAL_REPEATS:-3}
RUNTIME_REPEATS=${RUNTIME_REPEATS:-3}
PRESSURE_REPEATS=${PRESSURE_REPEATS:-3}
GRAPH_REPLAYS=${GRAPH_REPLAYS:-10}
TRACE_CAP=${TRACE_CAP:-8192}

echo "== build clc_probe =="
./experiments/clc/build_run.sh 4096 128 0 1024 1024 0 0

echo "== build clc_decode_probe =="
TARGET=clc_decode_probe ./experiments/clc/build_run.sh 4096 0 2 16 16 0

echo "== build clc_2d_probe =="
TARGET=clc_2d_probe ./experiments/clc/build_run.sh 64 128 128 4096 0

echo "== build clc_trace_probe =="
TARGET=clc_trace_probe ./experiments/clc/build_run.sh 8192 1 128 4096 128

echo "== build clc_cluster_probe =="
TARGET=clc_cluster_probe ./experiments/clc/build_run.sh 8192 128 2 4096

echo "== build clc_tuple_probe =="
TARGET=clc_tuple_probe ./experiments/clc/build_run.sh 64 16 8 128 4096

echo "== build clc_participation_probe =="
TARGET=clc_participation_probe ./experiments/clc/build_run.sh 8192 128 1 4096

echo "== build clc_runtime_probe =="
TARGET=clc_runtime_probe ./experiments/clc/build_run.sh 8192 128 4096 0

echo "== build clc_mapping_probe =="
TARGET=clc_mapping_probe ./experiments/clc/build_run.sh 8192 128 4096 0

echo "== build clc_pressure_probe =="
TARGET=clc_pressure_probe ./experiments/clc/build_run.sh 8192 128 4096 0 128 0 0

echo "== build clc_graph_probe =="
TARGET=clc_graph_probe ./experiments/clc/build_run.sh 8192 128 4096 0 5 1

echo "== threshold sweep =="
"$PYTHON" experiments/clc/run_sweep.py --suite threshold --repeats "$SYN_REPEATS"

echo "== occupancy sweep =="
"$PYTHON" experiments/clc/run_sweep.py --suite occupancy --repeats "$SYN_REPEATS"

echo "== claim-order sweep =="
"$PYTHON" experiments/clc/run_sweep.py --suite claim-order --repeats "$SYN_REPEATS"

echo "== synthetic workload sweep =="
"$PYTHON" experiments/clc/run_sweep.py --suite workload --repeats "$WORKLOAD_REPEATS"

echo "== decode sweep =="
"$PYTHON" experiments/clc/run_decode_sweep.py --suite all --repeats "$DECODE_REPEATS"

echo "== 2D sweep =="
"$PYTHON" experiments/clc/run_2d_sweep.py --suite all --repeats "$TWO_D_REPEATS"

echo "== cluster sweep =="
"$PYTHON" experiments/clc/run_cluster_sweep.py --suite all --repeats "$CLUSTER_REPEATS"

echo "== trace capture =="
stamp=$(date +%Y%m%d_%H%M%S)
trace_events="experiments/clc/results/clc_trace_events_${stamp}.csv"
trace_summary="experiments/clc/results/clc_trace_summary_${stamp}.csv"
trace_analysis="experiments/clc/results/clc_trace_analysis_${stamp}.csv"
CLC_TRACE_EVENTS_CSV=1 ./build/clc_trace_probe 8192 1 128 4096 "$TRACE_CAP" \
  > "$trace_events"
CLC_PROBE_CSV=1 ./build/clc_trace_probe 8192 1 128 4096 "$TRACE_CAP" \
  > "$trace_summary"
"$PYTHON" experiments/clc/analyze_trace.py --events "$trace_events" \
  --summary "$trace_summary" --out "$trace_analysis"

echo "== adversarial tuple/participation sweeps =="
"$PYTHON" experiments/clc/run_adversarial_sweep.py --suite all \
  --repeats "$ADVERSARIAL_REPEATS"

echo "== runtime behavior sweep =="
"$PYTHON" experiments/clc/run_runtime_sweep.py --suite all \
  --repeats "$RUNTIME_REPEATS"

echo "== worker/SM mapping sweep =="
"$PYTHON" experiments/clc/run_mapping_sweep.py --suite all

echo "== inter-kernel pressure sweep =="
"$PYTHON" experiments/clc/run_pressure_sweep.py --suite all \
  --repeats "$PRESSURE_REPEATS"

echo "== CUDA Graph replay checks =="
stamp=$(date +%Y%m%d_%H%M%S)
graph_stream="experiments/clc/results/clc_graph_stream_summary_${stamp}.csv"
graph_replay="experiments/clc/results/clc_graph_replay_summary_${stamp}.csv"
CLC_PROBE_CSV=1 ./build/clc_graph_probe 8192 128 4096 0 "$GRAPH_REPLAYS" 0 \
  > "$graph_stream"
CLC_PROBE_CSV=1 ./build/clc_graph_probe 8192 128 4096 0 "$GRAPH_REPLAYS" 1 \
  > "$graph_replay"
echo "graph_stream=$graph_stream"
echo "graph_replay=$graph_replay"

echo "== generated ISA dump =="
./experiments/clc/dump_generated_isa.sh
