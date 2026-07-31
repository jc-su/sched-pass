#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

OUTDIR=${OUTDIR:-experiments/clc/results/profiler}
TARGET=${TARGET:-./build/clc_runtime_probe}
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTDIR"

if [ ! -x "$TARGET" ]; then
  echo "missing $TARGET; build clc_runtime_probe first" >&2
  exit 2
fi

basic="$OUTDIR/ncu_clc_runtime_basic_sudo_${STAMP}.csv"
sched="$OUTDIR/ncu_clc_runtime_sched_sudo_${STAMP}.csv"
nsys_base="$OUTDIR/nsys_clc_runtime_${STAMP}"
ncu_summary="$OUTDIR/ncu_clc_runtime_summary_${STAMP}.csv"

echo "== ncu basic =="
sudo -n ncu --set basic --target-processes all --csv --log-file "$basic" \
  "$TARGET" 8192 128 4096 0

echo "== ncu scheduler/warp/instruction =="
sudo -n ncu --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  --section InstructionStats --target-processes all --csv --log-file "$sched" \
  "$TARGET" 8192 128 4096 0

sudo chown "$(id -u):$(id -g)" "$basic" "$sched"

echo "== summarize ncu =="
python3 experiments/clc/summarize_ncu_csv.py --out "$ncu_summary" \
  "$basic" "$sched"

echo "== nsys timeline =="
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true --output "$nsys_base" \
  "$TARGET" 8192 128 4096 0
nsys stats --report cuda_gpu_kern_sum --format csv \
  --output "${nsys_base}_stats" "${nsys_base}.nsys-rep"

echo "ncu_basic=$basic"
echo "ncu_sched=$sched"
echo "ncu_summary=$ncu_summary"
echo "nsys_report=${nsys_base}.nsys-rep"
echo "nsys_summary=${nsys_base}_stats_cuda_gpu_kern_sum.csv"
