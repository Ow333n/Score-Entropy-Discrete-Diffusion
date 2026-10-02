"""Stage-4 报告配图生成 (matplotlib, dataviz 规范同 pilot 报告)。

图 1: CPI 轨迹 (pretrained → early/mid/late × 2 seeds), 展示"阶跃+平台"模式
图 2: OrderGap 轨迹 (固定 6 路径口径)
图 3: Stage-4 gate 核心证据 — 每 seed 的 paired Δ 均值 + 95% bootstrap CI (2 面板)
图 4: G1 任务指标轨迹 (NLL / token acc / 贪心命中, small multiples)

用法: .venv/bin/python reports/stage4_report/make_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "reports", "stage4_report", "figs")
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

STEPS = [0, 1020, 5100, 10200]           # 0 = pretrained
STEP_LABELS = ["pretrained", "early 1020", "mid 5100", "late 10200"]
SEEDS = ["s1", "s2"]


def cpi(seed, step):
    if step == 0:
        return json.load(open(os.path.join(R, "pretrained/cpi.json")))["summary"]["cpi_abs"]
    return json.load(open(os.path.join(R, f"vanilla/cpi_{seed}_{step}.json")))["summary"]["cpi_abs"]


def og(seed, step):
    if step == 0:
        return json.load(open(os.path.join(R, "pretrained/order_gap.json")))["order_gap_raw"]
    return json.load(open(os.path.join(R, f"vanilla/og_{seed}_{step}.json")))["order_gap_raw"]


def g1(metric, seed, step):
    if step == 0:
        pre = json.load(open(os.path.join(R, "pretrained/g1_task.json")))
        return {"masked_nll": pre["masked_nll"], "token_acc": pre["token_acc"],
                "greedy_hit": pre["greedy_per_pos_hit_rate"]}[metric]
    d = json.load(open(os.path.join(R, f"vanilla/g1_{seed}_{step}.json")))
    return {"masked_nll": d["masked_nll"], "token_acc": d["token_acc"],
            "greedy_hit": d["greedy_per_pos_hit_rate"]}[metric]


# ------------------------------------------------- fig 1: CPI trajectory
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.axhline(cpi("s1", 0), color=MUTED, ls="--", lw=1.2)
ax.text(100, cpi("s1", 0) + 0.002, f"pretrained CPI_abs = {cpi('s1', 0):.3f}",
        color=SECONDARY, fontsize=8.5)
for seed, color in zip(SEEDS, [BLUE, ORANGE]):
    ys = [cpi(seed, s) for s in STEPS]
    ax.plot(STEPS, ys, color=color, lw=2, marker="o", ms=5,
           markeredgecolor="#fcfcfb", markeredgewidth=1.2)
    ax.text(STEPS[-1] + 80, ys[-1], seed, color=color, fontsize=10, va="center")
ax.annotate("step change completes within warmup\n(early 1020 ≈ late 10200)",
            xy=(1020, cpi("s1", 1020)), xytext=(2600, 0.305),
            color=INK, fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1))
ax.set_xlabel("optimizer step (0 = pretrained)")
ax.set_ylabel("CPI_abs  (E|delta_swap|, nats)")
ax.set_ylim(0.27, 0.335)
ax.set_xticks(STEPS)
ax.set_xticklabels(STEP_LABELS, fontsize=8)
ax.set_title("Fig 1 - CPI trajectory: SFT attenuates incompatibility, effect complete by step 1020",
             color=INK, loc="left", fontsize=10.5)
fig.savefig(os.path.join(FIGS, "fig1_cpi_trajectory.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 2: OrderGap trajectory
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.axhline(og("s1", 0), color=MUTED, ls="--", lw=1.2)
ax.text(100, og("s1", 0) + 0.05, f"pretrained OrderGap = {og('s1', 0):.2f}",
        color=SECONDARY, fontsize=8.5)
for seed, color in zip(SEEDS, [BLUE, ORANGE]):
    ys = [og(seed, s) for s in STEPS]
    ax.plot(STEPS, ys, color=color, lw=2, marker="o", ms=5,
           markeredgecolor="#fcfcfb", markeredgewidth=1.2)
    ax.text(STEPS[-1] + 80, ys[-1], seed, color=color, fontsize=10, va="center")
ax.set_xlabel("optimizer step (0 = pretrained)")
ax.set_ylabel("OrderGap_raw (nats, fixed 6 paths)")
ax.set_ylim(8.3, 10.6)
ax.set_xticks(STEPS)
ax.set_xticklabels(STEP_LABELS, fontsize=8)
ax.set_title("Fig 2 - Teacher-forced OrderGap trajectory (fixed manifest paths)",
             color=INK, loc="left", fontsize=10.5)
fig.savefig(os.path.join(FIGS, "fig2_ordergap_trajectory.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 3: gate evidence
gate = json.load(open(os.path.join(R, "vanilla/stage4_gate.json")))
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
panels = [("d_cpi_abs_mean", "d_cpi_abs_ci", "mean paired Δ|delta| (late vs pretrained)"),
          ("d_og_mean", "d_og_ci", "mean paired ΔOrderGap_raw (late vs pretrained)")]
for ax, (mk, cik, title) in zip(axes, panels):
    for x, (seed, color) in enumerate(zip(SEEDS, [BLUE, ORANGE])):
        m = gate["seeds"][seed][mk]
        lo, hi = gate["seeds"][seed][cik]
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=8, color=color,
                    capsize=6, lw=2, markeredgecolor="#fcfcfb", markeredgewidth=1.2)
        ax.text(x, m + (hi - m) + (hi - lo) * 0.08, f"{m:+.3f}\n[{lo:+.3f}, {hi:+.3f}]",
                ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["s1", "s2"], fontsize=10)
    ax.set_title(title, color=INK, fontsize=9.5)
    ax.set_ylim(min([gate["seeds"][s][cik][0] for s in SEEDS]) * 1.45,
                max([gate["seeds"][s][cik][1] for s in SEEDS]) * 0.55)
fig.suptitle("Fig 3 - Stage-4 gate evidence: paired bootstrap 95% CI (10k resamples, N=500)",
             color=INK, fontsize=11, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(os.path.join(FIGS, "fig3_gate_evidence.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 4: G1 trajectory
metrics = [("masked_nll", "masked-token NLL (nats)", 3.6, 4.4),
           ("token_acc", "token accuracy", 0.30, 0.40),
           ("greedy_hit", "span greedy per-pos hit", 0.40, 0.52)]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
for ax, (mk, title, ylo, yhi) in zip(axes, metrics):
    for seed, color in zip(SEEDS, [BLUE, ORANGE]):
        ys = [g1(mk, seed, s) for s in STEPS]
        ax.plot(STEPS, ys, color=color, lw=2, marker="o", ms=4.5,
                markeredgecolor="#fcfcfb", markeredgewidth=1.2)
    ax.set_xticks(STEPS)
    ax.set_xticklabels(STEP_LABELS, fontsize=7, rotation=15)
    ax.set_title(title, color=INK, fontsize=9.5)
    ax.set_ylim(ylo, yhi)
fig.suptitle("Fig 4 - G1 task metrics trajectory (frozen manifest, EMA weights)",
             color=INK, fontsize=11, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(os.path.join(FIGS, "fig4_g1_trajectory.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print("figs saved to", FIGS)
