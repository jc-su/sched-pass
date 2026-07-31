# One game, three maps: the unified model behind every instrument

2026-07-02. The formal companion to `MODEL.md`. Claim: **every PTX instrument
in the probe matrix — CLC, every cache-control form, prefetch, discard, shed,
the timer, and the unbuilt PDL/cluster/TMA levers — is a control on exactly one
of three per-request maps of a single scheduling-and-pebbling game, under one
effect-type discipline, optimized by one two-timescale controller.** Nothing in
the system is a special case; the menu of instructions is the *instruction-set
projection* of one abstract policy space.

Every number cited is measured (A6000 sm_86 / RTX PRO 6000 Blackwell sm_120);
every theory claim is labeled with its honest strength.

---

## 1. The game (math)

A batch step is a tuple

    G = ( D(mu),  H,  W )

- **D(mu)** — the task structure. mu : (request r, position p) -> physical KV
  unit is the address map (eKV's keystone; the object every engine must
  materialize). It induces, per request, a *stream* of unit-reads feeding a
  contraction, and across requests an independent set of tasks. D(mu) is a
  DAG whose nodes are (compute tile, KV unit read) pairs; requests are
  disjoint sub-DAGs (independence is what makes reordering sound — §4, E1).
- **H** — the memory hierarchy: levels ℓ ∈ {REG, SMEM/L1, L2, HBM, (host)}
  with capacities C_ℓ and bandwidths B_ℓ. On the two testbeds: C_L2 = 6 MB
  (GA102) vs 128 MB (GB202) — a fact that alone predicted the sign of one
  lever (§6).
- **W** — the worker pool: CTAs resident on SMs; |W| is set by launch shape
  and occupancy, not by us (CTA→SM placement is the hardware's).

## 2. The policy space (algebra): three maps + one adjoint

All control the system exposes, per request r, is:

    P_r = ( pi,  sigma,  phi )

**pi : claims -> tasks** — *temporal order*. WHEN each unit of work is served.
**sigma : (r, unit, time) -> level set** — *residency*. WHERE each KV unit's
bytes live in H, and with what eviction bias.
**phi : (r, unit) -> {attend, skip}** — *consumption*. HOW MUCH of mu's image
is actually read (eKV's sigma_eff = pi_phi(sigma_nominal), same object).

**Observation is the adjoint, not a fourth map**: it changes nothing, it
evaluates the cost functional and its per-request decomposition (clock64 →
t_r; %smid → spatial trace; the tap → mass). Formally: the levers act on the
*primal* execution; observation reads the *dual* (prices/gradients). This is
why the same pointcut mu carries both — control and measurement attach where
request identity becomes physical.

Every instrument is one cell of (map × level × timescale):

| instrument | map | level | timescale | effect type (§4) |
|---|---|---|---|---|
| `task_order[]` remap | pi | — | per launch | E1 |
| ticket claim (`atomicAdd`) | pi | — | per task | E1 + O |
| **CLC `try_cancel`** | pi | — | per task (hw) | E1 |
| `griddepcontrol` (PDL) — **built** (`SCHED_PDL=1`) | pi | — | per kernel | E0 (hint) |
| `prefetch.global.L2::evict_last` | sigma | L2 | per iteration | E0 |
| `prefetch.L1/L2`, `prefetchu` | sigma | L1/L2 | per iteration | E0 |
| **`discard.global.L2`** | sigma | L2 | per iteration | E0 |
| `createpolicy` + `ld…L2::cache_hint` | sigma | L2 | per load | E0 |
| `ld.cg/.cs/.lu`, `ld.L1::evict_first` | sigma | L1 | per load | E0 |
| `applypriority.L2` | sigma | L2 | per region | E0 |
| `st.cs/.wt` | sigma | L1/L2 | per store | E0 |
| TMA `cp.async.bulk` *(unbuilt)* | sigma | SMEM↔HBM | per tile | E0 |
| cluster + `mapa` DSMEM *(unbuilt)* | sigma | SMEM(remote) | per tile | E0 |
| `nanosleep` throttle | sigma (bandwidth share) | interconnect | per iteration | E0 |
| **shed / tau redirect** | phi | — | per iteration | E2(ε) |
| eKV guard (`importance ≥ tau`) | phi | — | per unit | E2(ε) |
| clock64 → `timer[task]` | adjoint | — | per task | O |
| `%smid`, `redux.sync` | adjoint | — | per task | O |
| CapKV `owner_of[key] ∈ caps` | (isolation plane) | — | per access | E0 (fail-closed) |

The table IS the unification: fifteen-plus instructions, three maps, one
adjoint. New hardware (CLC, PDL, TMA) adds *cells*, never new columns.

## 3. The cost (physics): pebbling traffic composed in (max,+)

Two nested cost laws, both with real theory behind them and honest gaps.

**Inner law — traffic is a pebble game.** For fixed (sigma, phi), the bytes
that must cross each boundary of H are determined by residency decisions over
D(mu): this is the multi-level red-blue pebble game (Hong–Kung). The cache
instruments are precisely pebbling moves: prefetch = place a pebble early;
evict_last = pin a pebble; discard = remove a pebble; cache_hint policy =
a randomized pebble priority; TMA = bulk pebble transfer; phi = delete nodes
(with E2 semantics). Attention decode is streaming — reuse lives in Q tiles,
weights, and *other requests'* small working sets — so the game's content is
mostly: *whose* pebbles occupy the contended level. Theory strength: the
pebble framing gives structural lower bounds and, more usefully here, a
**capacity-regime predicate**: a residency lever has value only when the
competing working set exceeds C_ℓ (see §6, confirmed both directions).

**Outer law — time is tropical.** Per resource k with traffic V_k and
bandwidth B_k, and per worker w with serial work,

    T_step  =  max( max_k V_k / B_k ,  max_{w ∈ W} Σ_{j ∈ pi^{-1}(w)} t_j )

Both maxima and the serial sums are (max,+)-semiring compositions: the
makespan is a tropical product over the schedule, kernels compose in sequence
by tropical multiplication (which is exactly what PDL relaxes: it lets
consecutive kernels' pebbling overlap, turning a tropical product into a
max over a merged DAG). Theory strength: (max,+) is here a *language* that
makes composition and bottleneck structure explicit — we use it for structure,
not for closed-form solutions.

**The coupling correction.** t_j is not a constant: t_j = t_j(a_j; s) where s
is co-runner state (measured: a contended A6000 *inverted* which pi was
optimal). The tractable form is mean-field — t̂_j(a; γ) with a broadcast
congestion multiplier — and it is why the controller is a feedback loop, not a
solver.

## 4. The safety discipline (PL): an effect-type system

Every woven effect has a type; the weave is admissible iff its type is
allowed. This is eKV's effect algebra promoted to a type system:

- **E0 — semantics-invariant effects.** Cache/residency hints, prefetch,
  discard, throttle. The machine may realize or ignore them; program output is
  bit-identical by ISA contract (a hint changes *where bytes wait*, never
  *what is computed*). All sigma instruments are E0. *Soundness: by
  construction (additive weave, untouched demand ops) + ISA semantics.*
- **E1 — permutation effects.** pi edits: reorder/redistribute independent
  tasks. Sound because requests are disjoint sub-DAGs of D(mu) (no cross-task
  data flow inside the kernel) and every task is served exactly once (ticket
  uniqueness; CLC cancel-or-launch exclusivity). *Soundness: measured
  bit-exactness across all pi in every fixture; the invariant "each task
  exactly once" is the proof obligation.*
- **E2(ε) — budgeted accuracy effects.** phi edits: attend fewer units. NOT
  semantics-preserving; admissible only behind an explicit budget with a
  quality contract (the H2O/Quest ε-mass bound — empirical, inherited, and it
  must ship with any tau > 0 policy). Default tau = 0 ⇒ E2 degenerates to
  identity (bit-exact, verified).
- **O — observation effects.** Commutative-monoid writes (atomic add),
  idempotent publishes, pure reads: replay-safe under CUDA graphs, arbitrary
  interleaving-closed.

Mechanism/policy separation (the CapKV lesson): the *pass* fixes the effect
types (what CAN happen); the *tables* choose within them (what DOES happen).
The policy language is deliberately weak — rank, price, budget — so no
expressible policy can exceed its type. A control plane bug can make the
schedule slow; it cannot make it wrong (E0/E1) or unboundedly wrong (E2).

**E2 realization (fixed 2026-07-03).** Shed reuses the loop's CANONICAL
induction variable as the trip counter (an injected counter was the source of
repeated miscompiles) and masks via a DOMINANCE-CHECKED replacement, so the
woven IR is always valid SSA. Two drop semantics by structure: LINEAR
contraction -> mask the loaded VALUE to 0; online SOFTMAX -> mask the SCORE to
-inf (exp(-inf-m)=0), the eKV Gate-0.5 rule, recovered from scalar dataflow
(the score feeds both an fmax and an exp). On a loop with no canonical IV
(fully unrolled/rotated) shed DECLINES LOUDLY -- correct-or-absent, never
wrong. Default on; validated (paged_decode modes G/H: linear + softmax tau=0
bit-exact, tau>0 matches a truncated-attention reference).

## 5. The controller: two timescales, one estimator

    kernel-side (ns, per CTA):   a_r = argmax_{a ∈ A_r} [ q_r·ΔT̂(a) − λ·ΔR̂(a) − H(a) ]
    plane-side  (ms, per step):  t̂_r ← timer;  γ, λ ← congestion;  
                                 pi ← list-schedule(t̂; objective);  tau ← quality budget

The kernel side is a priced table lookup (never an optimizer); the plane side
owns estimation and slow adaptation. Theory strength, stated honestly:
- With fixed, decoupled t_j: LPT gives Graham's (4/3 − 1/3W)·OPT makespan
  bound; SPT minimizes mean completion; EDF handles deadlines. The pi table
  expresses all three — the *objective* is a deployment choice.
- With coupling: no optimality claim; the system is a measurement-driven
  surrogate controller whose value is demonstrated, not proven. The
  contended-GPU inversion is the counterexample that forbids the stronger
  claim — and the reason the adjoint (observation) is load-bearing.

## 6. Predictions the model made, and their confirmations

The test of an abstraction is predicting *sign changes* of levers across
regimes. Three, all confirmed on hardware:

1. **pi pays iff workers ≪ tasks** (queueing: the serial term dominates the
   tropical max). A6000, 16 workers/64 tasks: LPT −32%. Blackwell, grid =
   tasks on 188 SMs: pi and CLC ≈ 0% (±1%) — the hardware scheduler already
   balances; nothing for the serial term to gain.
2. **sigma (bypass) pays iff competing working set > C_L2** (the pebble-game
   capacity predicate). Blackwell, KV 64 MB < 128 MB L2: discard = +0.0%.
   Same binary, KV 512 MB > 128 MB: **−54%**, stable, bit-exact. A6000
   (6 MB L2, always oversubscribed): −55%. One lever, both signs, predicted
   by one inequality.
3. **Coupling is first-order** (the mean-field term is not a refinement).
   Contended A6000: the optimal pi inverted (segregation beat LPT by 23%);
   idle: LPT wins by 32%. No static cost model survives this; a live adjoint
   does.

This is the OSDI-shaped evaluation story: not "our system is fast" but "one
model with three maps predicts when each lever helps, hurts, or vanishes, and
the mechanism makes acting on it safe (E0/E1) or budgeted (E2)."

## 7. What the model says to build next (in its own vocabulary)

1. **PDL / griddepcontrol** — extend pi across the kernel boundary: turn the
   tropical *product* of consecutive kernels (reload; attention) into a max
   over an overlapped DAG. Highest predicted value for continuous batching;
   assembles on sm_90+ per the probe.
2. **Softmax-correct phi** — the `-inf` score mask at LLVM level, so E2 is
   sound on real attention; then replicate eKV's NLL-vs-budget curve.
3. **createpolicy/cache_hint sigma** — replace the binary discard with the
   graded L2-fraction policy (a *continuous* sigma knob, per request).
4. **TMA MOVE tier** — sigma at the SMEM↔HBM boundary; also the path to
   descriptor-carried mu.
5. **The real controller** — today the fixtures hand-drive the tables; the
   plane-side loop (estimator → LPT/λ/tau updates each step) is scripted in
   pieces and needs to become one artifact running against a serving trace.

## 8. Related-abstraction honesty

The pieces are individually classical — Hong–Kung pebbling, Graham list
scheduling, (max,+) composition, Pigouvian pricing, effect systems. The claim
of novelty is NOT any one of them; it is (a) that a *compiler pass* can expose
all three maps per REQUEST from inside a fused kernel that erased request
identity, on stock engines, with a typed safety discipline; and (b) that the
three-map policy space is complete over the actual PTX control surface (the
§2 table — checked against an exhaustive ptxas probe, not a curated subset).

## 9. Robustness of pi under estimation noise (C1: the arming rule, derived)

pi is OPEN-LOOP: LPT ordering is optimal-ish only if the cost estimate ranks
correctly. The closed loop feeds it measured cycles, but estimates are noisy
(cold requests, contention drift). Three statements bound what noise can do,
and together they DERIVE the CLC arming threshold that
`experiments/clc/clc_noise_probe.cu` measured empirically.

Setup: n tiles with true times t_j served by m effective workers (the
resident-wave width); list scheduling in the order pi-hat = LPT on estimates
t-hat_j, with multiplicative noise t-hat_j = t_j * (1 + eps * u_j),
u_j in [-1, 1]. Let rho(eps) = (1+eps)/(1-eps) for eps < 1 (the maximum true-
time ratio that noise can invert: i ranked below j implies
t_i <= rho * t_j).

**(i) Safety floor (Graham).** ANY order -- arbitrary noise, adversarial
pi -- keeps list scheduling's makespan <= (2 - 1/m) * OPT. Noise can cost
pi's refinement, never correctness or the 2-approximation. (This is the
scheduling-theory shadow of the E1 effect type: order changes WHEN, never
WHAT, and never by more than the list-scheduling envelope.)

**(ii) Smoothness.** Makespan(pi-hat) <= Sum t / m + (1 - 1/m) * t_L where
t_L is the true time of the last-STARTED tile (Graham's tail bound). Under
noisy LPT the tile started last has the smallest ESTIMATE among those
remaining, so its TRUE time exceeds the exact-LPT tail tile's by at most
rho(eps):

    Makespan(pi-hat) - Makespan(LPT) <= (1 - 1/m) * (rho(eps) - 1) * t_tail.

Degradation is CONTINUOUS in eps and scales with the tail tile, not the
batch: mild noise costs a mild tail.

**(iii) Separation cliff (the bimodal case = decode's case).** Decode mixes
are near-bimodal (short context vs long context, gap g = t_long / t_short).
A long tile can be ranked below a short tile ONLY if rho(eps) >= g:

    ranking is EXACT for all  eps < (g - 1) / (g + 1).

For the probe's g = 32: threshold eps* = 31/33 = 0.94. MEASURED: recall
stayed 100% (and penalty 0%) through eps = 0.5 and first broke at
eps = 1.0 -- the cliff lands where (iii) puts it. This is why "LPT is
noise-robust" in Finding 5 is not luck; it is the separation of the mix.

**Corollary (the arming rule).** The controller's uncertainty statistic
u ~= E|t-hat - t| / E t = eps/2 for this noise model. pi remains exactly
LPT while eps < (g-1)/(g+1), i.e. while

    u < (g - 1) / (2 (g + 1))      (g=32: u* ~= 0.47; g -> inf: u* -> 0.5).

Below u*, arming CLC buys nothing (pi already optimal; measured ties).
Slightly above u*, (ii) bounds the loss by one tail tile -- and the probe
showed CLC cannot rescue a single late STARTED straggler either (late
binding balances WHO runs a tile, not WHEN it is issued), so arming there
also buys nothing. CLC pays only when ranking is BROADLY scrambled --
empirically recall <~ 75%, i.e. u >> u*. Hence the production default
SCHED_CLC_RESID = 0.75: comfortably above every u* of realistic mixes,
arming only when the estimate is near-uninformative (cold start, phase
change), exactly where the probe measured CLC's -2..-7.5%. The threshold is
now a corollary, not a tuned constant.

Honest scope: (ii) treats t_j as noise-independent (no coupling gamma); under
contention the inversion finding (#6) says re-measure, don't extrapolate --
which is what the EWMA + hysteresis controller does by construction.

## 10. Composition laws for the effect types (C2) and observation cadence (C3)

**C2 — the woven capabilities form a typed monoid under composition.** Write
E0 (semantics-invariant hints), E1 (permutations of disjoint sub-DAGs),
E2(eps) (budgeted-accuracy maps), O (observation writes into a commutative
monoid). For capabilities f, g woven into the same kernel:

    E0 . X     = X . E0 = X          (hints commute with everything: they
                                      touch no architecturally visible state)
    E1 . E1    = E1                  (permutations compose to a permutation;
                                      disjointness is preserved because both
                                      act on the same request partition)
    E2(a) . E2(b) <= E2(a + b)       (budget errors add at worst: each map is
                                      within eps of identity in the quality
                                      metric, and the metric is a norm)
    O . X      = X . O               (commutative-monoid writes commute with
                                      any request-local map; replay-safe by
                                      the monoid laws: reorder + re-add give
                                      the same fold)

**Soundness statement.** Any finite composition of woven capabilities with
all budgets zero (tau = 0) is extensionally the identity on kernel outputs:
E0 and O are identity on outputs by definition of their types; E1 is
identity on the output SET because requests are disjoint sub-DAGs each done
exactly once (the fixture's bit-exact checks are instances); and E2(0) = id
by the gate construction (budget 0 compiles to the untaken branch). This is
the paper-form of the empirical rule "null slots / zero tables => bit-exact
stock behavior," and it is CLOSED UNDER ADDING CAPABILITIES: a new
instrument only needs a type assignment, not a new global proof -- the
manifest (D1) records exactly that assignment.

Two measured caveats the laws make explicit rather than hide: (a) E1's
disjointness premise is a property of the KERNEL (requests share nothing),
checked per kernel family, not assumed -- the acquisition layer's 2D-grid
guard exists because tickets/CLC would silently BREAK the premise on
multi-axis grids; (b) O's monoid is per-row u64 addition -- attribution
correctness (which row) is the tile-binding contract, not the algebra.

**C3 — choosing the probe cadence.** With the device channel, collection is
free and only readout is scheduled; the estimator consumes probe samples
through an EWMA with weight w. A cost drift of relative magnitude d that
persists for k probe periods is tracked to within (1-w)^k * d; the deadband
h suppresses order churn until the predicted makespan gain exceeds h. So the
cadence rule is: probe every N steps such that N * step_time is small
against the workload's mixing time (batch membership half-life), with
w and h jointly satisfying (1-w)^(mix/(N*step)) * d_typ < h -- i.e. the
estimator must forget faster than the workload changes, and the deadband
must absorb what remains. In serving, batch membership changes every few
steps while costs-per-request change slowly (KV grows by one token/step):
N = 8..32 satisfies this with the default w = 0.25, h = 0.02 for any
realistic mix; the failure mode to watch is phase changes (prefill bursts),
which the uncertainty statistic flags and the arming rule (S9) already
routes around.
