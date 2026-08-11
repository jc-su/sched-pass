#!/bin/bash
# RQ2 mechanism trials: n seeds x budgets at the 16K single-claim shape.
# Durable runner — safe to re-invoke; completed trials are skipped by
# their existing artifact. Artifacts: results/serving/rq2-ladder/.
set -u
cd "$(dirname "$0")/.."
MODEL="${NTA_RQ2_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
OUT=results/serving/rq2-ladder
mkdir -p "$OUT"
for SEED in 20260901 20260902 20260903 20260904 20260905 20260906 20260907 20260908 20260909 20260910; do
  for BUDGET in 64 128; do
    ARTIFACT="$OUT/ladder-b$BUDGET-s$SEED.json"
    if [ -s "$ARTIFACT" ]; then
      echo "skip existing $ARTIFACT"
      continue
    fi
    python3 benchmarks/serving/SglangSelectedLoad.py \
      --model "$MODEL" --selected-budget "$BUDGET" \
      --selection-refresh-interval 1024 --seed "$SEED" \
      --output "$ARTIFACT" > "$OUT/ladder-b$BUDGET-s$SEED.log" 2>&1
    STATUS=$?
    echo "budget=$BUDGET seed=$SEED exit=$STATUS"
    if [ $STATUS -ne 0 ]; then
      echo "trial failed; log tail:"; tail -3 "$OUT/ladder-b$BUDGET-s$SEED.log"
    fi
  done
done
python3 - <<'EOF'
import json, glob, math, statistics as st
rows = {}
for path in sorted(glob.glob("results/serving/rq2-ladder/ladder-*.json")):
    d = json.load(open(path))
    key = d["selected_budget_pages"]
    rows.setdefault(key, []).append(d["tiered_vs_stock"])
for budget, trials in sorted(rows.items()):
    print(f"budget {budget} (n={len(trials)}):")
    for metric in ("external_p95_ttft_ratio", "resident_p99_itl_ratio", "resident_p95_tpot_ratio"):
        values = [t[metric] for t in trials]
        geo = math.exp(st.mean(math.log(v) for v in values))
        print(f"  {metric}: geomean={geo:.3f} min={min(values):.3f} max={max(values):.3f}")
EOF
