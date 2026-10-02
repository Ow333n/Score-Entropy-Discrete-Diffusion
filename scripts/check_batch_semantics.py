"""G0 §4.1: batch semantics 运行时核实。

协议要求打印并冻结实际语义 (不再把 "batch 4 / accum 2 = eff 8" 当未验证事实):
  config.training.batch_size / actual DataLoader batch.shape[0] / training.accum /
  ngpus / optimizer_step_count / sequences_per_optimizer_step
以及 B_effective = actual_micro × accum × ngpus 的运行时验证。

用法: .venv/bin/python scripts/check_batch_semantics.py
"""
import os
import sys
from itertools import chain

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from hydra import initialize, compose

import data
import losses
import graph_lib
import noise_lib
from model import SEDD
from model.ema import ExponentialMovingAverage


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=[
            "ngpus=1",
            "training.batch_size=4",
            "training.accum=2",
            "training.n_iters=5",
            "data.train=wikitext103",   # 默认 openwebtext 的裸 repo id 会触发 hf-hub URI 报错 (Day 3 已知)
            "data.valid=wikitext103",
        ])

    # 与 train.py 的 run_multiprocess.setup() 一致: 单卡单进程组
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=0, world_size=1)

    device = torch.device("cuda")
    graph = graph_lib.get_graph(cfg, device)
    model = SEDD(cfg).to(device)
    noise = noise_lib.get_noise(cfg).to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
    optimizer = losses.get_optimizer(cfg, chain(model.parameters(), noise.parameters()))
    scaler = torch.cuda.amp.GradScaler()
    state = dict(optimizer=optimizer, scaler=scaler, model=model, noise=noise, ema=ema, step=0)

    train_ds, eval_ds = data.get_dataloaders(cfg)
    train_iter = iter(train_ds)
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)

    torch.cuda.reset_peak_memory_stats()

    # 跑 4 个 micro-batch (accum=2 → 预期 2 个 optimizer step)
    for k in range(4):
        batch = next(train_iter)["input_ids"].to(device)
        if k == 0:
            actual_batch = batch.shape[0]
        train_step_fn(state, batch)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print("=" * 60)
    print("G0 §4.1 batch semantics 运行时核实")
    print(f"  config.training.batch_size   = {cfg.training.batch_size}")
    print(f"  actual DataLoader batch[0]   = {actual_batch}")
    print(f"  training.accum               = {cfg.training.accum}")
    print(f"  ngpus                        = {cfg.ngpus}")
    print(f"  optimizer_step_count (4 micro) = {state['step']}")
    print(f"  sequences_per_optimizer_step = {actual_batch} x {cfg.training.accum} x {cfg.ngpus} "
          f"= {actual_batch * cfg.training.accum * cfg.ngpus}")
    print(f"  B_effective                  = {actual_batch * cfg.training.accum * cfg.ngpus}")
    print(f"  峰值显存                      = {peak_gb:.2f} GB")
    print("=" * 60)

    # 运行时断言: optimizer step 数 = micro 数 / accum
    assert state["step"] == 4 // cfg.training.accum, "batch semantics 与预期不符!"
    print("✅ 语义核实通过: 4 micro-batches → 2 optimizer steps (accum=2)")


if __name__ == "__main__":
    main()
