# Motivation: the straggler is real, survives split-kv, and is reclaimable

Data collected on RTX PRO 6000 Blackwell (sm_120) against the **real woven
FlashInfer decode kernel** (`test/py/eval_motivation.py`), archived under
`data/`, plotted by `test/py/plot_motivation.py` -> `figures/`. sharegpt-ish
length mix (lognormal page counts, mean~2.5, clamped 1..128 pages of 16
tokens), fp16, one decode step over the batch.

Reproduce:
```
SCHED_PLUGIN=build/libSchedPass.so FLASHINFER_NVCC="python3 python/nvcc_clang_shim.py" \
SCHED_CLANG=clang++-22 SCHED_CUDA_PATH=/usr/local/cuda-12.9 SCHED_ARCH=sm_120a \
SCHED_NO_SHED=1 SCHED_NO_POLICY=1 python test/py/eval_motivation.py
python test/py/plot_motivation.py
```

## The claim chain (Figure `figures/motivation.png`, panels A/B/C)

**A. A fused decode step is a makespan over wildly uneven tiles.**
`data/mot_tilecycles.csv` -- per-tile execution time (woven `clock64` timer)
for one step at 2048 tiles: **p50 = 73 kilocycles, p99 = 591, max = 588**;
the slowest tile is **~8x the median**. The step ends when the last tile
finishes, so this dispersion is exactly the tail a scheduler could reclaim.
The observation is itself woven -- the timer that measures the straggler is
the same lever that would reorder it (the model's observation adjoint).

**B. FlashInfer's own in-kernel balancer, split-kv, does NOT remove it at
serving scale.** `data/mot_dispersion.csv` -- tail dispersion (p99/p50) vs
batch size, with the resident-wave size **R = 940** (occupancy of the woven
kernel) marking the queue boundary. Three regimes, all from real plan data:

| regime | batch (tiles) | split-kv | p99/p50 | why |
|---|---|---|---|---|
| split-active | <= ~440 | ON (adds tiles) | 1.8-2.0 | split flattens, but tiles < R => one wave, order is moot anyway |
| gap | ~440-940 | OFF | 4.8-5.1 | split self-disables, no queue yet: straggler fully exposed, nothing can reorder it (a split-threshold **miscalibration**, reportable upstream) |
| **queued** | **> 940** | OFF | **5.7-6.2** | split off AND tiles > R: tiles queue, and **issue order sets the makespan** |

The planner splits only to create parallelism for small grids; at serving
scale it turns split OFF and the full request dispersion (~6x) lands on the
queue -- precisely the regime where per-request order governs the step time.
split-kv and ordering are complementary by the planner's own logic, never
competing in-regime.

**C. The tail is reclaimable, bit-exact, with no oracle.**
`data/mot_policy.csv` -- step time at 2048 tiles under four issue orders,
all outputs **bit-identical** to identity (pi reorders WHEN, never WHAT):

| order | step time | vs identity |
|---|---|---|
| identity (unordered) | 452 us | -- |
| reversed (adversarial) | 474 us | +4.9% |
| LPT-oracle (true lengths) | 287 us | **-36.6%** |
| LPT from the **woven timer** (closed loop) | 292 us | **-35.4%** |

The closed loop -- one probe step fills the per-tile timer, the next step
issues longest-processing-time-first from the *measured* cycles -- recovers
essentially all of the oracle gain (the timer ranks **88%** of the true
stragglers first) with no knowledge of request lengths. This is the whole
thesis in one measurement: the kernel observes its own imbalance and
reschedules it, bit-exact, from inside the fused launch.

## What the figures do and don't claim (calibration for review)

- **Kernel-level makespan win: demonstrated** (-35% closed-loop, bit-exact,
  Fig C) on the real decode kernel in the queued regime it requires.
- **End-to-end serving overhead: ~zero** (measured separately, `ROADMAP.md`
  B3: enforce ties stock at conc 64 and 2048).
- **End-to-end serving *gain*: not claimed here.** It needs a *stable* batch
  in the queued, attention-dominant regime; under maximum batch churn the
  bijectivity guard correctly retires pi to identity most steps (exactly-once
  preserved), so the E2E gain row is the remaining measurement, its regime
  now exactly specified by Figure B.

## Files
```
data/mot_tilecycles.csv   per-tile cycles at 2048 tiles      (Fig A)
data/mot_dispersion.csv   dispersion vs batch, split, R       (Fig B)
data/mot_policy.csv       step time by issue order, bit-exact (Fig C)
figures/fig1_dispersion.png  fig2_regimes.png  fig3_policy.png  motivation.png
```
