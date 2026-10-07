"""Option B — deterministic dense retrain（用户 2026-10-07 批准；mechanism screening 的
最后一次专门重训，目的：恢复 250–750/1020 早期 CPI 变化窗口的可分析权重）。

run selection：frozen P1 的 2 个 canonical seeds——s1 (1,1,1) / s2 (2,2,2)，
recipe 与 P1 完全一致（lr=3e-5、warmup 2500、batch 32、exact_N 2500 步、
同一 vanilla.py 训练循环、同一 corruption/数据顺序/EMA/优化器）。
**不按历史 CPI effect size 挑 run。**

checkpoint 政策（存储约束 + 用户条款"如果存储允许"）：
- dense schedule（EMA-only 678MB）：50, 100, 150, 200, 250, 300, 350, 400,
  500, 750, 1020, 1500, 2500——比用户建议的 18 点列表少 5 点（350 保留、丢
  600/900/1250/2000），250–750 窗口仍有 250/300/350/400/500/750 六点
- 2500 另存 model（anchor gate 用）；checkpoints-meta 为 EMA-only
- 权重专用保存通过 monkey-patch vanilla.save_ckpt 实现：**不改变训练循环、
  不消耗 RNG、不改任何 frozen 文件**（vanilla.py 本体零改动）
- 分析审核前不删除任何权重

复现门：retrain 的 500/1020/2500 EMA 与 surviving 全量 checkpoint 的 EMA 逐位
比对 + 同 manifest 指标比对（用户 gate 双路径）。

用法（请 tmux 跑，2 seeds ≈ 40min）:
  .venv/bin/python scripts/optionb_dense_retrain.py 1   # s1
  .venv/bin/python scripts/optionb_dense_retrain.py 2   # s2
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import training.vanilla as vanilla

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DENSE_SAVE_AT = [50, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1020, 1500, 2500]
ANCHORS = {500, 1020, 2500}


def save_ckpt_weights_only(path, state):
    """权重专用 checkpoint（不消耗 RNG、不影响训练流）。

    - checkpoints-meta（preemption 快照）：EMA-only
    - anchor steps（500/1020/2500）：EMA + model（复现门）
    - 其余 dense 点：EMA-only
    """
    payload = dict(ema=state["ema"].state_dict(), step=state["step"])
    if "checkpoints-meta" not in path and state["step"] in ANCHORS:
        payload["model"] = state["model"].state_dict()
    torch.save(payload, path)


def main():
    seed = int(sys.argv[1])
    assert seed in (1, 2), "canonical seeds 1/2"
    vanilla.save_ckpt = save_ckpt_weights_only
    overrides = [
        f"training.n_iters=2500",
        f"training.save_at_steps={DENSE_SAVE_AT}",
        "optim.lr=3e-5",
        f"seeds.model_seed={seed}",
        f"seeds.data_order_seed={seed}",
        f"seeds.corruption_seed={seed}",
        f"training.name=optionb-dense-s{seed}",
    ]
    sys.argv = ["training/vanilla.py"] + overrides
    vanilla.main()


if __name__ == "__main__":
    main()
