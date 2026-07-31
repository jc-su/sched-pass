#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build=${NTA_BUILD_DIR:-"${root}/build"}
results=${NTA_RESULTS_DIR:-"${root}/results"}
trials=${NTA_TRIALS:-10}
iterations=${NTA_OVERHEAD_ITERATIONS:-200}
requests=${NTA_OVERHEAD_REQUESTS:-96}
dependencies=${NTA_OVERHEAD_DEPENDENCIES:-4}
tile_bytes=${NTA_OVERHEAD_TILE_BYTES:-65536}
csv=${results}/direct-overhead.csv

if (( trials < 2 )); then
  echo "NTA_TRIALS must be at least 2" >&2
  exit 2
fi
mkdir -p "${results}"
printf 'trial,baseline_gib_s,mechanism_gib_s\n' >"${csv}"

measure() {
  local baseline=$1
  local output
  output=$("${build}/nta-kv-bench" \
    --mode=resident --requests="${requests}" --coalesce=1 \
    --dependencies="${dependencies}" --tile-bytes="${tile_bytes}" \
    --iterations="${iterations}" --baseline="${baseline}")
  printf '%s\n' "${output}" >&2
  sed -n 's/.*logical_GiB\/s=\([0-9.]*\).*/\1/p' <<<"${output}"
}

for ((trial = 1; trial <= trials; ++trial)); do
  if (( trial % 2 == 1 )); then
    baseline=$(measure 1)
    mechanism=$(measure 0)
  else
    mechanism=$(measure 0)
    baseline=$(measure 1)
  fi
  if [[ -z ${baseline} || -z ${mechanism} ]]; then
    echo "failed to parse benchmark throughput" >&2
    exit 1
  fi
  printf '%d,%s,%s\n' "${trial}" "${baseline}" "${mechanism}" | tee -a "${csv}"
done

python3 - "${csv}" <<'PY'
import csv
import math
import statistics
import sys

path = sys.argv[1]
with open(path, newline="", encoding="ascii") as source:
    rows = list(csv.DictReader(source))
baseline = [float(row["baseline_gib_s"]) for row in rows]
mechanism = [float(row["mechanism_gib_s"]) for row in rows]
reduction = [100.0 * (base - mech) / base
             for base, mech in zip(baseline, mechanism)]
t95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

def interval(values):
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values))
    critical = t95.get(len(values) - 1, 1.96)
    return mean, critical * sem

base_mean, base_ci = interval(baseline)
mech_mean, mech_ci = interval(mechanism)
red_mean, red_ci = interval(reduction)
print(f"trials={len(rows)}")
print(f"baseline_GiB/s={base_mean:.2f} +/- {base_ci:.2f} (95% t interval)")
print(f"mechanism_GiB/s={mech_mean:.2f} +/- {mech_ci:.2f} (95% t interval)")
print(f"paired_reduction_pct={red_mean:.2f} +/- {red_ci:.2f} pp (95% t interval)")
print("note=GPU clocks and host interference are not controlled by this script")
PY
