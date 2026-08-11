#!/bin/bash
# Quality matrix: task kinds x budgets x refresh intervals.
# Resumable by artifact; artifacts under results/serving/quality-matrix/.
set -u
cd "$(dirname "$0")/.."
MODEL="${NTA_QM_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
OUT=results/serving/quality-matrix
mkdir -p "$OUT"
for REFRESH in 1024 1; do
  ARTIFACT="$OUT/matrix-refresh$REFRESH.json"
  if [ -s "$ARTIFACT" ]; then echo "skip existing $ARTIFACT"; continue; fi
  python3 benchmarks/serving/SglangSelectedQuality.py \
    --model "$MODEL" --task-kinds needle,multikey --task-count 3 \
    --selected-budgets 32,64,128 --selection-refresh-interval "$REFRESH" \
    --max-new-tokens 96 \
    --output "$ARTIFACT" > "$OUT/matrix-refresh$REFRESH.log" 2>&1
  echo "refresh=$REFRESH exit=$?"
done
python3 - <<'PYEOF'
import json, glob
for path in sorted(glob.glob("results/serving/quality-matrix/matrix-*.json")):
    d = json.load(open(path))
    refresh = d.get("selection_refresh_interval", "?")
    print(f"refresh {refresh}: stock by-kind {d['stock']['pass_rate_by_kind']}")
    for budget, arm in sorted(d.get("budgets", d.get("budget_reports", {})).items(), key=lambda kv: int(kv[0])):
        print(f"  budget {budget}: {arm.get('pass_rate_by_kind', arm.get('pass_rate'))}")
PYEOF
