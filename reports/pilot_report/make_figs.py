"""pilot 报告配图生成 (matplotlib)。

配色遵循 dataviz 规范 (validated palette):
  series-1 blue #2a78d6 / series-2 orange #eb6834 / series-3 aqua #1baf7a
  ink #0b0b0b, secondary #52514e, muted bar #8f8e89
原则: 单轴 (不同量纲用 small multiples, 不做双轴)、细线 2px、线端直接标注、
文字用墨色而非系列色。

用法: .venv/bin/python reports/pilot_report/make_figs.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS = os.path.join(ROOT, "reports", "pilot_report", "figs")
os.makedirs(FIGS, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#8f8e89"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": SECONDARY,
    "axes.labelcolor": INK,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "axes.grid": True,
    "grid.color": "#e3e2dd",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "lines.solid_capstyle": "round",
})


def read_curve(path):
    steps, train, evals, lrs = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            evals.append(float(row["eval_loss"]))
            lrs.append(float(row["lr"]))
    return steps, train, evals, lrs


PILOT = os.path.join(ROOT, "exp_local/regime_a/pilot-150849/learning_curve.csv")
PROBE_3e4 = os.path.join(ROOT, "exp_local/regime_a/pilot-132632/learning_curve.csv")
PROBE_1e4 = os.path.join(ROOT, "exp_local/regime_a/lrprobe_1e-4-143912/learning_curve.csv")
PROBE_3e5 = os.path.join(ROOT, "exp_local/regime_a/lrprobe_3e-5-141329/learning_curve.csv")

# ---------------------------------------------------------------- fig 1: pilot
s, tr, ev, lr = read_curve(PILOT)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                               gridspec_kw=dict(hspace=0.12, height_ratios=[1.6, 1]))

# panel A: eval loss
ax1.plot(s, ev, color=BLUE, lw=2)
best_i = ev.index(min(ev))
ax1.plot(s[best_i], ev[best_i], "o", color=BLUE, ms=8,
         markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=5)
ax1.annotate(f"best  {ev[best_i]:.2f} @ {s[best_i]}",
             xy=(s[best_i], ev[best_i]), xytext=(s[best_i] + 900, ev[best_i] + 0.03),
             color=INK, fontsize=9)
for x, label, col in [(2500, "warmup end", SECONDARY),
                      (8500, "plateau detected s_p=8500", SECONDARY),
                      (10200, "N = 10200 = ceil(1.2*s_p)", ORANGE)]:
    ax1.axvline(x, color=col, ls="--", lw=1.2, alpha=0.9)
    ax1.text(x + 150, 7.47, label, color=col, fontsize=8.5, rotation=90,
             va="top", ha="left")
# late-phase overfit region
ax1.annotate("", xy=(29000, 7.47), xytext=(21000, 7.31),
             arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1.2))
ax1.text(21000, 7.30, "late-phase overfit (+0.3 nats / 10k steps)", color=SECONDARY, fontsize=8.5)
ax1.set_ylabel("eval loss (nats / masked-span token)")
ax1.set_ylim(6.95, 7.72)
ax1.set_title("Vanilla pilot - lr 3e-5, 30k steps (panel A: eval; panel B: train)",
              color=INK, loc="left", fontsize=11)

# panel B: train loss (noise dominated by sigma sampling; context only)
ax2.plot(s, tr, color=ORANGE, lw=1.2, alpha=0.45)
ax2.set_ylabel("train loss")
ax2.set_xlabel("optimizer step")
ax2.set_xlim(0, 30500)
fig.savefig(os.path.join(FIGS, "fig1_pilot_learning_curve.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------- fig 2: LR probe comparison
fig, ax = plt.subplots(figsize=(8, 4.6))
for path, label, color, end in [
    (PROBE_3e4, "3e-4 (rejected)", ORANGE, "diverging"),
    (PROBE_1e4, "1e-4", AQUA, "wobbly"),
    (PROBE_3e5, "3e-5 (chosen)", BLUE, "monotone"),
]:
    s_, _, ev_, _ = read_curve(path)
    ax.plot(s_, ev_, color=color, lw=2 if end == "monotone" else 1.6,
            alpha=1.0 if end == "monotone" else 0.85)
    ax.text(s_[-1] + 90, ev_[-1], label, color=color, fontsize=9, va="center")
ax.axvline(2500, color=SECONDARY, ls="--", lw=1.2)
ax.text(2560, 8.28, "warmup end (2500)", color=SECONDARY, fontsize=8.5)
ax.set_xlabel("optimizer step")
ax.set_ylabel("eval loss (nats / masked-span token)")
ax.set_xlim(0, 5400)
ax.set_ylim(6.95, 8.35)
ax.set_title("LR calibration probes (3500 steps each; 3e-4 curve from terminated pilot)",
             color=INK, loc="left", fontsize=11)
fig.savefig(os.path.join(FIGS, "fig2_lr_probe_comparison.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------- fig 3: G1 pilot check
metrics = [
    ("masked-token NLL (nats)", 4.2812, 3.8281, "lower is better"),
    ("token accuracy", 0.3223, 0.3619, "higher is better"),
    ("span greedy per-pos hit", 0.4430, 0.4717, "higher is better"),
]
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
for ax, (name, pre, sft, note) in zip(axes, metrics):
    bars = ax.bar(["pretrained", "pilot SFT"], [pre, sft], width=0.55,
                  color=[MUTED, BLUE])
    for b, v in zip(bars, [pre, sft]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (max(pre, sft) * 0.02),
                f"{v:.3f}", ha="center", color=INK, fontsize=9)
    ax.set_title(name, color=INK, fontsize=9.5)
    ax.set_ylim(0, max(pre, sft) * 1.18)
    ax.text(0.5, -0.16, note, transform=ax.transAxes, ha="center",
            color=SECONDARY, fontsize=8)
    ax.tick_params(axis="x", labelrotation=0)
fig.suptitle("G1 pilot check - pretrained vs pilot SFT (step-30000 EMA, frozen manifest N=500)",
             color=INK, fontsize=11, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0.02, 1, 0.93])
fig.savefig(os.path.join(FIGS, "fig3_g1_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("figs saved to", FIGS)
