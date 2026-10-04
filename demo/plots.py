"""Demo plots（matplotlib，分图，不混量纲）。中文字体：demo/fonts/NotoSansSC-Regular.otf"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "fonts", "NotoSansSC-Regular.otf")
if os.path.exists(_FONT):
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREEN = "#4C72B0", "#DD8452", "#55A868"


def plot_compat_chain(curves):
    chain = curves["compatibility_main_chain"]
    stages = [c["stage"] for c in chain]
    cpi = [c["cpi_abs"] for c in chain]
    og = [c["order_gap"] for c in chain]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(stages, cpi, marker="o", color=BLUE)
    for x, y in zip(stages, cpi):
        ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9)
    ax1.set_title("CPI_abs（E|δ|，current-code harmonized 口径）")
    ax1.set_ylim(min(cpi) - 0.02, max(cpi) + 0.02)
    ax1.grid(alpha=0.3)

    ax2.plot(stages, og, marker="o", color=ORANGE)
    for x, y in zip(stages, og):
        ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9)
    ax2.set_title("OrderGap_raw（current-code harmonized 口径）")
    ax2.set_ylim(min(og) - 0.5, max(og) + 0.5)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_rl_task(curves):
    rl = curves["rl_task"]
    steps = sorted(int(k) for k in rl.keys())
    nll = [rl[str(s)]["nll_raw"] for s in steps]
    sampled = [rl[str(s)]["sampled64_raw"] for s in steps]
    greedy = [rl[str(s)]["greedy_raw"] for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(steps, nll, marker="o", color=BLUE)
    axes[0].set_title("Formal64 NLL（raw）")
    axes[0].set_xlabel("RL step")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, sampled, marker="o", color=ORANGE)
    axes[1].set_title("Formal64 sampled64 reward（raw）")
    axes[1].set_xlabel("RL step")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, greedy, marker="o", color=GREEN)
    axes[2].set_title("Formal64 greedy reward（raw）")
    axes[2].set_xlabel("RL step")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    return fig
