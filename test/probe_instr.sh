#!/bin/bash
# Probe which instrumentation PTX each arch's ptxas accepts.
PTXAS=${PTXAS:-/usr/local/cuda-12.9/bin/ptxas}
WD=$(mktemp -d)
declare -a TESTS=(
"ld.ca(L1+L2)|ld.global.ca.f32 %f2, [%rd1];"
"ld.cg(bypassL1)|ld.global.cg.f32 %f2, [%rd1];"
"ld.cs(stream)|ld.global.cs.f32 %f2, [%rd1];"
"ld.lu(lastuse)|ld.global.lu.f32 %f2, [%rd1];"
"ld.cv(volatile)|ld.global.cv.f32 %f2, [%rd1];"
"ld.L2evict_last|ld.global.L2::evict_last.f32 %f2, [%rd1];"
"ld.L2evict_first|ld.global.L2::evict_first.f32 %f2, [%rd1];"
"ld.L1evict_first|ld.global.L1::evict_first.f32 %f2, [%rd1];"
"createpolicy.frac|createpolicy.fractional.L2::evict_last.b64 %rd3, 0.5;"
"ld.L2cache_hint|createpolicy.fractional.L2::evict_last.b64 %rd3, 0.5;\nld.global.L2::cache_hint.f32 %f2, [%rd1], %rd3;"
"discard.L2|discard.global.L2 [%rd1], 128;"
"applypriority.L2|applypriority.global.L2::evict_normal [%rd1], 128;"
"prefetch.L1|prefetch.global.L1 [%rd1];"
"prefetch.L2|prefetch.global.L2 [%rd1];"
"prefetch.L2evict_last|prefetch.global.L2::evict_last [%rd1];"
"prefetchu.L1|prefetchu.L1 [%rd1];"
"st.cs(stream)|st.global.cs.f32 [%rd1], %f2;"
"st.wt(writethru)|st.global.wt.f32 [%rd1], %f2;"
"nanosleep|nanosleep.u32 100;"
"elect.sync|elect.sync _|%p1, 0xffffffff;"
"griddepcontrol.wait|griddepcontrol.wait;"
"griddepcontrol.launch|griddepcontrol.launch_dependents;"
"barrier.cluster.arv|barrier.cluster.arrive;"
"barrier.cluster.wait|barrier.cluster.wait;"
"getctarank|getctarank.u32 %r2, %rd1;"
"mapa.cluster|mapa.shared::cluster.u32 %r2, %r3, %r4;"
"setmaxnreg.inc|setmaxnreg.inc.sync.aligned.u32 24;"
"redux.sync.add|redux.sync.add.u32 %r2, %r3, 0xffffffff;"
"smid|mov.u32 %r2, %smid;"
"clc.try_cancel|.shared .align 16 .b8 cr[16];\n.shared .align 8 .b64 cb;\nmov.u32 %r2,cb;\nmbarrier.init.shared::cta.b64 [%r2],1;\nclusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [cr],[%r2];"
)
for ARCH in sm_86 sm_90 sm_100 sm_120; do
  printf "%-24s" "=== $ARCH"
done; echo
for T in "${TESTS[@]}"; do
  NAME="${T%%|*}"; BODY="${T#*|}"
  printf "%-22s" "$NAME"
  for ARCH in sm_86 sm_90 sm_100 sm_120; do
    F=$WD/t.ptx
    { echo ".version 8.7"; echo ".target $ARCH"; echo ".address_size 64";
      echo ".visible .entry k(.param .u64 p){";
      echo ".reg .b64 %rd<8>; .reg .f32 %f<8>; .reg .b32 %r<8>; .reg .pred %p<4>;";
      echo "ld.param.u64 %rd1,[p];";
      echo -e "$BODY";
      echo "ret; }"; } > $F
    if $PTXAS -arch=$ARCH $F -o $WD/t.o 2>$WD/err; then R="OK"; else R="--"; fi
    printf "%-8s" "$R"
  done
  echo
done
rm -rf $WD
