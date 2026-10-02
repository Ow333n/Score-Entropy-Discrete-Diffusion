"""256-seq Vanilla 训练内存剖面 (诊断用, 非协议步骤)。

跑 3 个训练步, 在 forward 后 / backward 后 / optimizer step 后打点显存,
最后打印 memory_summary 定位最大分配者。

用法: .venv/bin/python scripts/profile_mem.py [--batch 32]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose

from data import get_dataset
from task_data.corruption import corrupt_span_batch
from training.vanilla import init_model_from_pretrained, span_task_loss
from model.ema import ExponentialMovingAverage
import losses
import graph_lib
import noise_lib


def snap(tag):
    torch.cuda.synchronize()
    print(f"[mem] {tag:24s}: allocated {torch.cuda.max_memory_allocated()/1e9:.2f}GB "
          f"reserved {torch.cuda.memory_reserved()/1e9:.2f}GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256", overrides=[f"training.batch_size={args.batch}"])

    device = torch.device("cuda")
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    model, _, _ = init_model_from_pretrained(cfg, device, cfg.model_path)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
    optimizer = losses.get_optimizer(cfg, model.parameters())
    scaler = torch.cuda.amp.GradScaler()

    ds = get_dataset("wikitext103", "train", cache_dir="data", block_size=256, num_proc=4)
    batch = torch.stack([ds[i]["input_ids"] for i in range(args.batch)]).to(device)
    print(f"batch: {batch.shape}")

    torch.cuda.reset_peak_memory_stats()
    snap("before loop")

    corr_g = torch.Generator(device=device).manual_seed(0)
    for step in range(3):
        x_t, span_mask, sigma, dsigma, _ = corrupt_span_batch(
            batch, noise, cfg.data.span_min, cfg.data.span_max, graph.dim - 1, generator=corr_g)
        snap(f"step {step} after corrupt")
        loss = span_task_loss(noise, graph, model, batch, x_t, span_mask, sigma, dsigma, train=True)
        snap(f"step {step} after forward")
        scaler.scale(loss).backward()
        snap(f"step {step} after backward")
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model.parameters())
        optimizer.zero_grad()
        snap(f"step {step} after optimizer")

    print("\n--- memory_summary (top allocations) ---")
    print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()
