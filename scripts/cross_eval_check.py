"""交叉评估诊断: formal-s1 与 pilot 的 eval 差异是模型差异还是 eval 腐蚀种子噪声。

假说: 训练期 eval (64 块 × 每块 1 个 t) 被高 t 抽样主导 (dsigma≈999 权重),
不同 run 的 eval 腐蚀种子 (corruption_seed+1000) 不同 → eval 数值高方差,
模型本身可能相近。

方法: 用同一 eval 腐蚀种子分别评估两个模型 (交叉):
  - s1-10200 EMA 模型 × pilot 的 eval 种子 (0+1000)
  - pilot-30000 EMA 模型 × s1 的 eval 种子 (1+1000)
若各自拿到对方的数值 → eval 种子噪声; 若仍各是各的 → 模型真差异。

用法: .venv/bin/python scripts/cross_eval_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose

from model import SEDD
from model.ema import ExponentialMovingAverage
from data import get_dataset
from task_data.corruption import corrupt_span_batch
from training.vanilla import span_task_loss
import graph_lib
import noise_lib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(ROOT, "exp_local/regime_a/formal-vanilla-s1-191414")
PILOT = os.path.join(ROOT, "exp_local/regime_a/pilot-150849")


def load_ema_model(ckpt_path, cfg, device):
    model = SEDD(cfg).to(device).eval()
    loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    ema.load_state_dict(loaded["ema"])
    ema.copy_to(model.parameters())
    return model


def eval_with_seed(model, graph, noise, valid_ds, cfg, device, corr_seed):
    g = torch.Generator(device=device).manual_seed(corr_seed)
    losses_, t_max = [], 0.0
    with torch.no_grad():
        for bi in range(cfg.data.eval_chunks):
            x0 = valid_ds[bi]["input_ids"].to(device)[None]
            x_t, span_mask, sigma, dsigma, t = corrupt_span_batch(
                x0, noise, cfg.data.span_min, cfg.data.span_max, graph.dim - 1, generator=g)
            l = span_task_loss(noise, graph, model, x0, x_t, span_mask, sigma, dsigma, train=False)
            losses_.append(l.item())
            t_max = max(t_max, t.max().item())
    return sum(losses_) / len(losses_), t_max


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    device = torch.device("cuda")
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    valid_ds = get_dataset("wikitext103", "validation", cache_dir="data", block_size=256, num_proc=4)

    m_s1 = load_ema_model(os.path.join(S1, "checkpoint_10200.pth"), cfg, device)
    m_pilot = load_ema_model(os.path.join(PILOT, "checkpoints-meta", "checkpoint.pth"), cfg, device)

    print("=" * 64)
    print("交叉评估 (同一 eval 腐蚀种子下比较两个模型)")
    for name, model in [("s1-10200 EMA", m_s1), ("pilot-30000 EMA", m_pilot)]:
        for corr_seed, label in [(1000, "pilot 的 eval 种子"), (1001, "s1 的 eval 种子")]:
            val, tmax = eval_with_seed(model, graph, noise, valid_ds, cfg, device, corr_seed)
            print(f"  {name:16s} × eval_seed={corr_seed:5d} ({label}): eval={val:.4f}  max_t={tmax:.4f}")
    print("=" * 64)
    print("判读: 若同一模型换 eval 种子后数值大幅变化 → eval 种子噪声主导;")
    print("      若两个模型在同一 eval 种子下差异仍大 → 模型真差异。")


if __name__ == "__main__":
    main()
