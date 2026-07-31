"""sched_controller.py -- the closed-loop controller artifact (SGLANG.md #4).

The math the serving loop actually runs, in dependency order, as one clean
class -- estimation BEFORE ordering BEFORE any pricing:

  1. ESTIMATION (statistics, not optimization).  Per kernel family f, an
     online least-squares fit  t_hat = alpha_f * kv_len + beta_f  over the
     measured per-request cycles (the woven timer), plus a per-request EWMA
     residual for idiosyncrasy and an EWMA congestion index gamma comparing
     measured vs predicted step makespan.
  2. ORDERING (elementary scheduling theory).  Given t_hat: LPT for makespan,
     SRPT for mean latency, EDF for SLOs. Sorting, not convex programming.
  3. DAMPING (stability under nonstationarity).  Hysteresis: the published
     order changes only when the new order improves the PREDICTED makespan by
     more than a dead-band -- the contended-GPU inversion finding says the
     environment shifts under us, so undamped policy flips oscillate.

Pricing (lambda) stays where THEORY.md puts it: an offline-calibrated scalar
the policy MAY consume, never an online market -- this class exposes gamma so
a later layer can set lambda, and writes nothing it cannot justify.

The class is GPU-free (pure numpy-less Python) and drives any object with the
SchedPlane write interface (set_order / set_row / set_num_tasks / push).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Fit:
    """Incremental least squares for t = alpha*x + beta (one kernel family)."""
    n: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    sxy: float = 0.0

    def add(self, x: float, y: float) -> None:
        self.n += 1
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.sxy += x * y

    def coeffs(self) -> tuple[float, float]:
        """(alpha, beta); degenerate data falls back to proportional."""
        if self.n < 2:
            return (self.sy / self.sx, 0.0) if self.sx > 0 else (1.0, 0.0)
        den = self.n * self.sxx - self.sx * self.sx
        if den <= 0:
            return (self.sy / self.sx, 0.0) if self.sx > 0 else (1.0, 0.0)
        alpha = (self.n * self.sxy - self.sx * self.sy) / den
        return alpha, (self.sy - alpha * self.sx) / self.n


@dataclass
class SchedControlPlane:
    """Per-step estimator + policy for a batch of (rid, kv_len) requests."""

    policy: str = "lpt"          # lpt | srpt | edf
    ewma: float = 0.25           # residual/congestion smoothing
    deadband: float = 0.02       # min predicted-makespan gain to re-order
    _fit: _Fit = field(default_factory=_Fit)
    _resid: dict = field(default_factory=dict)   # rid -> EWMA residual cycles
    _gamma: float = 1.0                          # congestion multiplier
    _last_order: list = field(default_factory=list)

    # -- 1. estimation -------------------------------------------------------
    def observe(self, rids, kv_lens, cycles) -> None:
        """Fold one step's measured per-request cycles into the model."""
        for rid, ln, cy in zip(rids, kv_lens, cycles):
            if cy <= 0:
                continue  # request not measured this step
            self._fit.add(float(ln), float(cy))
            pred = self.predict_one(ln, rid=None)
            r = self._resid.get(rid, 0.0)
            self._resid[rid] = (1 - self.ewma) * r + self.ewma * (cy - pred)

    def observe_step(self, predicted_makespan: float,
                     measured_makespan: float) -> None:
        """Congestion index: measured vs predicted step time (gamma > 1 means
        co-runners are stealing bandwidth -- exposed for a pricing layer).

        INTENTIONALLY UNWIRED SEAM: the serving plugin does not call this yet
        -- gamma feeds the lambda pricing layer (THEORY 'the roof, not the
        foundation'), which is deliberately not built. Keep: it is the
        congestion-observation half of that layer, cheap, and the natural
        integration point when pricing is added (pass the woven timer's
        max-tile cycles as measured_makespan). Removing it would delete the
        gamma capability the model's #6 inversion finding depends on."""
        if predicted_makespan > 0 and measured_makespan > 0:
            g = measured_makespan / predicted_makespan
            self._gamma = (1 - self.ewma) * self._gamma + self.ewma * g

    def predict_one(self, kv_len: float, rid=None) -> float:
        alpha, beta = self._fit.coeffs()
        t = alpha * float(kv_len) + beta
        if rid is not None:
            t += self._resid.get(rid, 0.0)
        return max(t, 0.0)

    @property
    def gamma(self) -> float:
        return self._gamma

    def uncertainty(self, rids, kv_lens) -> float:
        """Relative prediction uncertainty over a batch: mean |EWMA residual|
        / mean predicted cost, in [0, inf). This is the CLC arming signal --
        pi (open-loop LPT) needs the prediction to be right; late-binding
        acquisition does not. Requests never observed contribute a full
        prediction's worth of doubt (their residual is unknown, not zero)."""
        preds, resids = [], []
        for rid, ln in zip(rids, kv_lens):
            p = self.predict_one(ln)
            preds.append(p)
            resids.append(abs(self._resid[rid]) if rid in self._resid else p)
        mp = sum(preds) / len(preds) if preds else 0.0
        return (sum(resids) / len(resids)) / mp if mp > 0 else 1.0

    # -- 2 + 3. ordering with hysteresis -------------------------------------
    def order(self, rids, kv_lens, deadlines=None, elapsed=None) -> list[int]:
        """Slot order (indices into the batch) under the configured policy,
        damped: keep the previous order unless the new one predicts a
        makespan improvement beyond the dead-band."""
        t_hat = [self.predict_one(ln, rid) for rid, ln in zip(rids, kv_lens)]
        n = len(t_hat)
        if self.policy == "lpt":
            new = sorted(range(n), key=lambda i: -t_hat[i])
        elif self.policy == "srpt":
            new = sorted(range(n), key=lambda i: t_hat[i])
        elif self.policy == "edf":
            if deadlines is None:
                raise ValueError("edf needs deadlines")
            el = elapsed or [0.0] * n
            new = sorted(range(n),
                         key=lambda i: deadlines[i] - el[i] - t_hat[i])
        else:
            raise ValueError(f"unknown policy {self.policy!r}")

        if len(self._last_order) == n:
            gain = (self._span(t_hat, self._last_order) -
                    self._span(t_hat, new))
            if gain < self.deadband * max(self._span(t_hat, new), 1e-9):
                return self._last_order  # damped: not worth churning
        self._last_order = new
        return new

    @staticmethod
    def _span(t_hat, order) -> float:
        """Order-DEPENDENT quality scalar for the hysteresis dead-band: total
        (== mean) completion time = sum of prefix sums of the ordered costs.
        LPT/SPT/EDF produce materially different values, so the dead-band fires
        only on a real ranking change. (The previous max(t_hat)+k*1e-9 was
        order-INDEPENDENT -- the max dominates the tiny position bias -- so the
        controller never re-published once _last_order was set; the serving hot
        path sidestepped it, but the class was silently frozen.) Pure makespan
        on R machines would need R; total completion time captures ordering
        quality without it and is monotone in how front-loaded the order is."""
        s = c = 0.0
        for k in range(len(order)):
            c += t_hat[order[k]]
            s += c
        return s

    # -- glue: one serving step ----------------------------------------------
    def step(self, plane, rids, kv_lens, cycles=None,
             deadlines=None, elapsed=None) -> list[int]:
        """Fold measurements, compute the order, write the plane. Returns the
        slot order used (order[k] = batch index served k-th)."""
        if cycles is not None:
            self.observe(rids, kv_lens, cycles)
        order = self.order(rids, kv_lens, deadlines, elapsed)
        plane.set_order(order)
        plane.set_num_tasks(len(order))
        plane.push()
        return order


def _smoke() -> int:
    """GPU-free check: estimator converges, LPT matches oracle, damping holds."""
    import random
    random.seed(0)
    ctl = SchedControlPlane(policy="lpt")

    class FakePlane:
        def set_order(self, o): self.o = list(o)
        def set_num_tasks(self, n): pass
        def push(self): pass
    plane = FakePlane()

    alpha_true, beta_true = 7.5, 120.0
    rids = list(range(64))
    lens = [random.choice([64, 512]) for _ in rids]
    for _ in range(20):  # 20 steps of noisy measurements
        cyc = [alpha_true * ln + beta_true + random.gauss(0, 5) for ln in lens]
        ctl.step(plane, rids, lens, cycles=cyc)

    a, b = ctl._fit.coeffs()
    ok_fit = abs(a - alpha_true) / alpha_true < 0.05
    oracle = sorted(range(len(lens)), key=lambda i: -lens[i])
    got = ctl.order(rids, lens)
    ok_lpt = [lens[i] for i in got] == [lens[i] for i in oracle]
    before = ctl.order(rids, lens)
    lens2 = list(lens)
    lens2[0] += 1  # negligible perturbation: damping must keep the order
    ok_damp = ctl.order(rids, lens2) == before

    for name, ok in [("estimator alpha within 5%", ok_fit),
                     ("LPT order matches oracle by length", ok_lpt),
                     ("hysteresis suppresses negligible churn", ok_damp)]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    fails = [ok_fit, ok_lpt, ok_damp].count(False)
    print("== ALL PASS ==" if fails == 0 else f"== {fails} FAILED ==")
    return fails


if __name__ == "__main__":
    raise SystemExit(_smoke())
