#!/usr/bin/env bash
# run_ir_tests.sh -- CPU-only golden-IR gates (FileCheck) for the weave
# shapes. No GPU, no CUDA: catches pass regressions at the IR level and
# localizes them (the e2e suite proves behavior; this proves structure).
#
#   ./test/ir/run_ir_tests.sh            # uses build/libSchedPass.so
# Env: LLVM_BIN (/usr/lib/llvm-22/bin), PLUGIN
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LLVM_BIN=${LLVM_BIN:-/usr/lib/llvm-22/bin}
PLUGIN=${PLUGIN:-$ROOT/build/libSchedPass.so}

pass=0; fail=0
for t in "$ROOT"/test/ir/*.ll; do
  name=$(basename "$t")
  # expand the RUN line's %PLUGIN / %s convention
  if "$LLVM_BIN/opt" -load-pass-plugin="$PLUGIN" -passes=sched-weave "$t" -S \
      2>/tmp/ir_gate.err | "$LLVM_BIN/FileCheck" "$t"; then
    echo "  [PASS] $name"; pass=$((pass+1))
  else
    echo "  [FAIL] $name"; fail=$((fail+1))
    head -5 /tmp/ir_gate.err | sed 's/^/     | /'
  fi
done
echo "== $pass passed, $fail failed =="
[ $fail -eq 0 ] && echo "== ALL PASS =="
exit $fail
