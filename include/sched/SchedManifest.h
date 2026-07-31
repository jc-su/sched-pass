//===- sched/SchedManifest.h - the capability manifest -----------*- C++ -*-===//
//
// ONE declarative row per woven instrument: the single source of truth for
// its effect type, arch gate, the runtime slots it reads, its compile-time
// knob, its pass/emit ORDER, its JIT cache-key fragment, and the invariants
// that were learned the hard way -- now DATA, asserted, not comments scattered
// across the pass.
//
// This is the D1 refactor's spine. Adding an instrument is adding a row here
// (+ its emit code): the effect-type composition laws (THEORY.md #9-10) then
// guarantee bit-exact-under-composition without a new global proof, and the
// order-rank field encodes the ordering constraints this project paid for:
//   * work-queue BEFORE weave (it clones the body the weave then instruments)
//   * shed BEFORE prefetch     (prefetch splits blocks, invalidating LoopInfo
//                               that shed's canonical-IV counter needs)
//   * claim AFTER body         (late binding; claim-ahead breaks balancing)
//   * pi honored only when order_size == nctaid (bijective under overlap)
//   * acquisition needs a 1D grid (tickets/CLC enumerate one axis)
//
// What is DERIVED from this table vs. ASSERTED against it (be precise, so the
// comments don't overclaim):
//   * DERIVED: dumpManifest() (operator/census/docs, MANIFEST.md); the Python
//     control plane's cache-key disable-mask (sched_rt.py builds it by
//     iterating the mirror -- add a lever here and it keys the cache with no
//     second list).
//   * ASSERTED: the pass emit order (SchedWeave applies pi/shed/policy/timer
//     in this row order; SchedPlugin's addPass sequence puts workqueue before
//     weave) is CHECKED against the `order` field by test/py/test_manifest.py
//     (order == row index), not literally generated from the array -- the emit
//     sites stay explicit for readability, the test is the guard against drift.
//   * MIRRORED + consistency-tested: python/sched_rt.py::MANIFEST matches this
//     row-for-row (name/effect/sm/knob/disable/tag/order) via test_manifest.py.
//
//===----------------------------------------------------------------------===//
#ifndef SCHED_MANIFEST_H
#define SCHED_MANIFEST_H

namespace sched {

// The effect-type discipline (THEORY.md #4, composition laws #10). Every woven
// capability is exactly one of these, and that assignment -- not a per-weave
// argument -- is what makes safety closed under composition.
enum class Effect {
  E0_Hint,     // semantics-invariant hint (prefetch/discard/PDL): bit-identical
  E1_Permute,  // permutation of disjoint sub-DAGs (pi): reorders WHEN not WHAT
  E2_Budget,   // budgeted-accuracy (shed/tau): epsilon-bounded, gated on tau>0
  O_Observe,   // commutative-monoid write (timer): replay/reorder-safe
  Acquire,     // task-acquisition transform (ticket/CLC): a launch-model change
};

inline const char *effectName(Effect e) {
  switch (e) {
  case Effect::E0_Hint: return "E0/hint";
  case Effect::E1_Permute: return "E1/permute";
  case Effect::E2_Budget: return "E2/budget";
  case Effect::O_Observe: return "O/observe";
  case Effect::Acquire: return "acquire";
  }
  return "?";
}

struct Capability {
  const char *name;    // stable id ("pi","shed","policy","timer","pdl","workqueue")
  Effect effect;       // effect type (safety class)
  unsigned minSm;      // arch gate: min sm_NN, 0 = any
  const char *slots;   // runtime slots it reads (informational)
  const char *knob;    // the compile-time env knob that (dis)arms it
  bool disableKnob;    // true: knob is SCHED_NO_<X> (present => OFF); false: SCHED_<X> (present => ON)
  int order;           // pass/emit order, lower first (encodes the invariants)
  const char *tag;     // JIT cache-key fragment when this changes codegen ("" = none)
  const char *contract;// one-line decline/ordering contract
};

// The manifest. Order of ROWS is the pass/emit order (also given explicitly in
// `order` for assertion). Keep names in sync with python/sched_rt.py's mirror.
inline constexpr Capability kManifest[] = {
    {"workqueue", Effect::Acquire, 0, "ctrl(+queue for ticket)",
     "SCHED_WORKQUEUE", /*disableKnob=*/false, 0, "-wq",
     "runs BEFORE weave (clones body -> sched_body); ticket sm<100 needs "
     "queue+ctrl, CLC sm>=100 needs ctrl only; 1D-grid runtime guard; claim "
     "issued AFTER the body (late binding)"},
    {"pi", Effect::E1_Permute, 0, "order,ctrl", "SCHED_NO_INDIRECT",
     /*disableKnob=*/true, 1, "",
     "task = order[ctaid(slot)]; per-launch nctaid clamp + ctrl.order_size "
     "validity keep it BIJECTIVE under scheduler overlap; unarmed => identity"},
    {"shed", Effect::E2_Budget, 0, "ctrl", "SCHED_NO_SHED",
     /*disableKnob=*/true, 2, "-nos",
     "tau: softmax score->-inf or linear value->0 mask via the loop's "
     "CANONICAL IV; declines LOUDLY on unrolled/no-IV loops; emitted BEFORE "
     "prefetch (which invalidates LoopInfo); tau==0 => bit-exact"},
    {"policy", Effect::E0_Hint, 80, "ctrl", "SCHED_NO_POLICY",
     /*disableKnob=*/true, 3, "-nop",
     "prefetch.L2::evict_last (urgent) / discard.global.L2 (polite) at "
     "detected KV-stream sites; declines on FlashInfer cp.async (dormant "
     "there); emitted AFTER shed"},
    {"timer", Effect::O_Observe, 0, "timer,ctrl", "SCHED_NO_TIMER",
     /*disableKnob=*/true, 4, "-ti",
     "clock64 bracket, tid0-gated atomicAdd into timer[task]; ctrl.flags bit0 "
     "= per-step cadence gate; SCHED_TIMER_INDIRECT retargets the channel "
     "(device buffer ~free vs host-mapped zero-touch)"},
    {"pdl", Effect::E0_Hint, 90, "-", "SCHED_PDL", /*disableKnob=*/false, 5, "",
     "griddepcontrol.wait after table reads + launch_dependents at returns; "
     "pure E0 -- no-op unless the launch site opts into programmatic stream "
     "serialization"},
};

inline constexpr unsigned kManifestSize =
    sizeof(kManifest) / sizeof(kManifest[0]);

// The row for a capability name, or nullptr.
inline const Capability *capabilityByName(const char *name) {
  for (const Capability &c : kManifest) {
    const char *a = c.name, *b = name;
    while (*a && *a == *b) { ++a; ++b; }
    if (*a == 0 && *b == 0)
      return &c;
  }
  return nullptr;
}

} // namespace sched

#endif // SCHED_MANIFEST_H
