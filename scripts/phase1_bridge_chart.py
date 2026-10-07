"""Phase 1.2 bridge 诊断图（old BF16 vs new FP32 scatter + identity line）。

两个 small-multiple 面板：CPI_abs、CE/NLL；每个点一个 checkpoint，
3 类着色（pretrained / SFT-lr3e-4 / SFT-lr3e-5——palette 前 3 槽位，
参考调色板文档认证的 all-pairs 组合：blue #2a78d6 / orange #eb6834 / aqua #1baf7a；
aqua 低于 3:1 对比 → 图例 + 关键点直标 + 报告全表兜底）。
图只做诊断，不改变 gate。

用法: .venv/bin/python scripts/phase1_bridge_chart.py
输出: results/phase1_bridge/bridge_scatter.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BR = os.path.join(ROOT, "results", "phase1_bridge")

CAT_COLOR = {"pretrained": "#2a78d6", "SFT-lr3e-4": "#eb6834", "SFT-lr3e-5": "#1baf7a"}
CAT_ORDER = ["pretrained", "SFT-lr3e-4", "SFT-lr3e-5"]
LABEL_POINTS = {"pretrained-step0": "pretrained", "v21-L1-2500": "v21-L1", "formal-s1-10200": "formal-s1"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"


def main():
    man = json.load(open(os.path.join(BR, "bridge_manifest.json")))
    an = json.load(open(os.path.join(BR, "bridge_analysis.json")))
    fam_cat = {c["id"]: ("pretrained" if c["family"] == "pretrained"
                         else "SFT-lr3e-4" if c["family"] in ("v2.1-HUL", "v1.2-A")
                         else "SFT-lr3e-5") for c in man["checkpoints"]}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.patch.set_facecolor(SURFACE)
    panels = [("old_cpi", "new_cpi", "CPI_abs"),
              ("old_ce", "new_ce", "CE / NLL (masked)")]
    for ax, (ok, nk, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for cat in CAT_ORDER:
            xs = [r[ok] for r in an["table"] if fam_cat[r["id"]] == cat]
            ys = [r[nk] for r in an["table"] if fam_cat[r["id"]] == cat]
            ax.scatter(xs, ys, s=56, color=CAT_COLOR[cat], label=cat,
                       edgecolors="white", linewidths=0.8, zorder=3)
        lo = min(min(r[ok] for r in an["table"]), min(r[nk] for r in an["table"]))
        hi = max(max(r[ok] for r in an["table"]), max(r[nk] for r in an["table"]))
        pad = (hi - lo) * 0.06
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color=MUTED, linewidth=1.5, linestyle="--", zorder=2)
        ax.set_xlabel(f"old frozen BF16 evaluator — {title}", color=TEXT, fontsize=11)
        ax.set_ylabel(f"new FP32 evaluator — {title}", color=TEXT, fontsize=11)
        ax.set_title(title, color=TEXT, fontsize=12, loc="left", pad=10)
        for r in an["table"]:
            if r["id"] in LABEL_POINTS:
                ax.annotate(LABEL_POINTS[r["id"]], (r[ok], r[nk]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=9, color=MUTED)
        for spine in ax.spines.values():
            spine.set_color("#d8d6cf")
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, color="#e8e6e0", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right",
                   labelcolor=TEXT, handletextpad=0.3)
    fig.suptitle("Phase 1.2 bridge：old BF16 vs new FP32（identity 线 = 两者相同）",
                 color=TEXT, fontsize=13, x=0.03, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(BR, "bridge_scatter.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
