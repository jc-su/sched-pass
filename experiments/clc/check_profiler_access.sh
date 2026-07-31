#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "== NVIDIA profiling permission =="
if [ -r /proc/driver/nvidia/params ]; then
  grep -E "RmProfilingAdminOnly" /proc/driver/nvidia/params || true
fi

echo
echo "== modprobe profiler config =="
for f in /etc/modprobe.d/*nvidia* /usr/lib/modprobe.d/*nvidia* /lib/modprobe.d/*nvidia*; do
  [ -e "$f" ] || continue
  if grep -q "RestrictProfilingToAdminUsers" "$f"; then
    echo "-- $f"
    grep "RestrictProfilingToAdminUsers" "$f"
  fi
done

echo
echo "== ncu permission smoke =="
ncu --query-metrics 2>&1 | sed -n '1,40p'
