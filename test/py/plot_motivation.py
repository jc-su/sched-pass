"""plot_motivation.py -- the motivation figures from archived data (CPU-only,
reproducible; reads data/mot_*.csv written by eval_motivation.py).

Fig 1  tile-time distribution (ECDF)     -- the straggler exists
Fig 2  dispersion vs batch size + regimes -- split_kv does not remove it
Fig 3  closed-loop pi step time           -- the tail is reclaimable
Plus a combined 1x3 storyline strip.

Palette: dataviz validated categorical set (blue/aqua/yellow/red) on the
light chart surface. Single-series figures carry no legend (the title names
the series); Fig 3's bars are direct-labeled.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a897f"
SURFACE, GRID = "#fcfcfb", "#e7e6e2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans",
})


def rd(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def _recede(ax):
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- Fig 1: tile-time ECDF -------------------------------------------------
def fig_dist(ax):
    rows = rd("mot_tilecycles.csv")
    cyc = sorted(int(r["cycles"]) for r in rows)
    n = len(cyc)
    us = [c / 1000.0 for c in cyc]  # kilocycles for readability
    y = [(i + 1) / n for i in range(n)]
    ax.plot(us, y, color=BLUE, linewidth=2, zorder=3)
    p50 = us[int(0.50 * n)]; p99 = us[int(0.99 * n)]; mx = us[-1]
    for x, lab, yy in ((p50, "p50", 0.50), (p99, "p99", 0.99)):
        ax.plot([x, x], [0, yy], color=MUTED, linewidth=1, ls=(0, (3, 3)),
                zorder=2)
        ax.annotate(f"{lab}={x:.0f}", (x, yy), textcoords="offset points",
                    xytext=(5, -12 if lab == "p99" else 6), color=INK2,
                    fontsize=9)
    ax.annotate(f"max/p50 = {mx / p50:.1f}x", (mx, 1.0),
                textcoords="offset points", xytext=(-8, -6), ha="right",
                color=RED, fontsize=9, fontweight="bold")
    ax.set_xlabel("per-tile time  (kilocycles)")
    ax.set_ylabel("fraction of tiles ≤ x")
    ax.set_title("A  One fused step, many tile times", loc="left",
                 fontweight="bold", fontsize=12)
    ax.set_ylim(0, 1.02); ax.set_xlim(left=0)
    _recede(ax)


# ---- Fig 2: dispersion vs batch, regime map --------------------------------
def fig_regimes(ax):
    rows = rd("mot_dispersion.csv")
    rows.sort(key=lambda r: int(r["tiles"]))
    tiles = [int(r["tiles"]) for r in rows]
    ratio = [float(r["ratio_p99_p50"]) for r in rows]
    split = [int(r["split_active"]) for r in rows]
    R = int(float(rows[0]["R"])) if rows[0]["R"] else 0
    # regime band boundaries in tile-space: split self-disables between the
    # last split-on point and first split-off point; queue begins at R.
    split_edge = None
    for i in range(1, len(split)):
        if split[i - 1] and not split[i]:
            split_edge = (tiles[i - 1] * tiles[i]) ** 0.5  # geo-mean on log x
    lo, hi = tiles[0] * 0.8, tiles[-1] * 1.15
    ymax = max(ratio) * 1.10
    bands = [(lo, split_edge or R, "#eef4fb", "split-kv\nactive"),
             (split_edge or R, R, "#fbf4e6", "gap\n(no queue)"),
             (R, hi, "#fdeeee", "QUEUED\n(order governs)")]
    for x0, x1, col, lab in bands:
        ax.axvspan(x0, x1, color=col, zorder=0)
        ax.annotate(lab, (((x0 * x1) ** 0.5), ymax * 0.98), ha="center",
                    va="top", color=INK2, fontsize=8.5)
    if R:
        ax.axvline(R, color=RED, linewidth=1.1, ls=(0, (4, 3)), zorder=2)
        ax.annotate(f"R = {R}", (R, 1.15), textcoords="offset points",
                    xytext=(4, 0), color=RED, fontsize=9, va="bottom")
    ax.plot(tiles, ratio, color=BLUE, linewidth=2, zorder=3)
    for x, y, s in zip(tiles, ratio, split):
        ax.scatter([x], [y], s=70, zorder=4,
                   facecolor=(AQUA if s else BLUE), edgecolor=SURFACE,
                   linewidth=1.5)
    ax.set_ylim(1.0, ymax)
    ax.set_xlim(lo, hi)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xticks([128, 512, 2048, 8192])
    ax.set_xlabel("batch size  (decode tiles, log)")
    ax.set_ylabel("tail dispersion  (p99 / p50)")
    ax.set_title("B  The straggler survives split-kv", loc="left",
                 fontweight="bold", fontsize=12)
    ax.axhline(1.0, color=GRID, linewidth=1)
    # tiny legend for the marker meaning (2 categories -> legend present)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=AQUA,
               markeredgecolor=SURFACE, markersize=9, label="split-kv on"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
               markeredgecolor=SURFACE, markersize=9, label="split-kv off")],
        loc="lower right", frameon=False, fontsize=9)
    _recede(ax)


# ---- Fig 3: closed-loop pi reclaim -----------------------------------------
def fig_policy(ax):
    rows = {r["policy"]: r for r in rd("mot_policy.csv")}
    order = ["identity", "reversed", "lpt-oracle", "lpt-timer"]
    labels = {"identity": "identity\n(unordered)", "reversed": "reversed\n(adversarial)",
              "lpt-oracle": "LPT-oracle\n(true lengths)",
              "lpt-timer": "LPT from woven\ntimer (closed loop)"}
    vals = [float(rows[k]["norm_vs_identity"]) for k in order]
    colors = [MUTED, YELLOW, "#86b6ef", BLUE]
    y = range(len(order))
    ax.barh(list(y), vals, color=colors, height=0.62, zorder=3,
            edgecolor=SURFACE, linewidth=1.5)
    for i, (k, v) in enumerate(zip(order, vals)):
        delta = (v - 1.0) * 100
        tag = "—" if k == "identity" else f"{delta:+.0f}%"
        ax.annotate(f"{v:.2f}x  {tag}", (v, i), textcoords="offset points",
                    xytext=(6, 0), va="center", color=INK, fontsize=10,
                    fontweight="bold" if k == "lpt-timer" else "normal")
    ax.axvline(1.0, color=MUTED, linewidth=1, ls=(0, (3, 3)), zorder=2)
    ax.set_yticks(list(y)); ax.set_yticklabels([labels[k] for k in order],
                                               fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("step time  (normalized to identity)")
    ax.set_title("C  The tail is reclaimable, bit-exact", loc="left",
                 fontweight="bold", fontsize=12)
    ax.set_xlim(0.55, max(vals) * 1.18)
    rec = rows["lpt-timer"].get("recall", "")
    if rec not in ("", None):
        ax.annotate(f"timer ranks {100*float(rec):.0f}% of true stragglers first",
                    (0.56, len(order) - 0.5), color=INK2, fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def main():
    # individual figures
    for fn, name in ((fig_dist, "fig1_dispersion"),
                     (fig_regimes, "fig2_regimes"),
                     (fig_policy, "fig3_policy")):
        fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
        fn(ax); fig.tight_layout()
        fig.savefig(os.path.join(FIG, name + ".png"))
        plt.close(fig)
    # combined storyline strip
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), dpi=150)
    fig_dist(axes[0]); fig_regimes(axes[1]); fig_policy(axes[2])
    fig.suptitle("Per-request scheduling inside the fused decode kernel: "
                 "the straggler exists (A), survives split-kv (B), and is "
                 "reclaimable bit-exact (C)", fontsize=12, y=1.02, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "motivation.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("figures ->", FIG)
    for f in ("fig1_dispersion", "fig2_regimes", "fig3_policy", "motivation"):
        print("  ", f + ".png")


if __name__ == "__main__":
    raise SystemExit(main())
