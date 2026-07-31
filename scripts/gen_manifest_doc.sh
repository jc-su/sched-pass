#!/usr/bin/env bash
# gen_manifest_doc.sh -- regenerate MANIFEST.md from the authoritative C++
# capability manifest (SchedManifest.h), so the doc can never drift from code.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PL=${PLUGIN:-$ROOT/build/libSchedPass.so}
OPT=${OPT:-$(command -v opt 2>/dev/null || echo /usr/lib/llvm-22/bin/opt)}
OUT=$ROOT/MANIFEST.md

human=$(SCHED_MANIFEST_DUMP=1 "$OPT" -load-pass-plugin="$PL" \
  -passes=verify -disable-output /dev/null 2>&1 | grep -E '^(==|  )')

{
  echo "# Capability manifest"
  echo
  echo "Generated from \`include/sched/SchedManifest.h\` by"
  echo "\`scripts/gen_manifest_doc.sh\` -- do not edit by hand. One declarative"
  echo "row per woven instrument; the pass emit order FOLLOWS this table"
  echo "(asserted by \`test/py/test_manifest.py\`: order field == row index),"
  echo "the Python control plane mirrors it and derives its cache-key mask from"
  echo "it (same test), and operators read it at server start."
  echo
  echo '```text'
  echo "$human"
  echo '```'
  echo
  echo "Effect types (the safety discipline, THEORY.md 4/10): **E0** hint"
  echo "(bit-identical) · **E1** permute (reorders WHEN not WHAT) · **E2**"
  echo "budget (epsilon, gated on tau>0) · **O** observe (commutative monoid)"
  echo "· **acquire** (launch-model transform). Composition is closed under"
  echo "these types, so adding a row needs a type assignment, not a new proof."
} > "$OUT"
echo "wrote $OUT"
