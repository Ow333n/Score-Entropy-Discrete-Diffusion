"""Mechanism pilot 训练入口（protocol v1.2 §9，staged execution 2026-10-06）。

Stage P1: A1, B1, A2, B2（primary causal comparison 先行）
Stage P2: C1, D1, C2, D2（directional auxiliary，P1 review 后启动）

用法:
  .venv/bin/python training/pilot_mechanism.py mechanism.policy=A mechanism.replicate=1 \
      training.n_iters=2500 training.name=mechpilot-A1

设计要点（协议 v1.2）:
  - 共享 schedule 五元组 (sample_id, span_start, span_len, σ, K) 由 SharedSchedule 流生成；
    policy 只改变 mask 位置选择（A 均匀 K 子集 / B 固定 π_L 秩序 / C 最右 / D 最左）
  - 损失 = frozen recipe 同口径：Absorbing.score_entropy × dσ 加权 × masked-span 均值 × batch 均值
  - checkpoint 容器 = {"ema": state_dict}（frozen evaluator 的 weights="ema" 路径直接可读）；
    2500 另存 {"model": ...}；step0 全库共享一份
  - 硬停守卫（不临时修参数继续跑）：
    ① frozen 文件被改动（git diff）→ RuntimeError
    ② 非有限 loss（NaN/Inf）→ RuntimeError
    ③ 结束 schedule SHA-256 != preflight 参考 → RuntimeError
    ④ OOM / 任何异常 → 进程直接失败退出（无自动重试）
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose

from model import SEDD
from model.ema import ExponentialMovingAverage
from model import utils as mutils
from data import get_dataset
from training.vanilla import init_model_from_pretrained, build_dataloader
import losses
import graph_lib
import noise_lib
from task_data.policy_corruption import (SharedSchedule, draw_pi_L, make_generator,
                                        select_mask, stream_seed)

PROTOCOL_VERSION = "v1.2"
RECIPE_VERSION = "mechanism_pilot_v1"
FROZEN_FILES = ["losses.py", "graph_lib.py", "noise_lib.py", "data.py",
                "training/vanilla.py", "task_data/corruption.py", "model/"]
SAVE_STEPS = (500, 1020, 2500)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_frozen_clean():
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"] + FROZEN_FILES,
                       cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError("frozen 文件被改动，硬停（禁止临时修改继续跑）")


def apply_masks(x0, masks, mask_token):
    x_t = x0.clone()
    span_mask = torch.zeros_like(x0, dtype=torch.bool)
    for b, sel in enumerate(masks):
        if sel.numel():
            x_t[b, sel.to(x_t.device)] = mask_token
            span_mask[b, sel.to(span_mask.device)] = True
    return x_t, span_mask


def span_task_loss(graph, model, x0, x_t, span_mask, sigma, dsigma):
    """与 frozen vanilla.py span_task_loss 逐项同口径（SE × dσ × masked-span 均值）。"""
    log_score_fn = mutils.get_score_fn(model, train=True, sampling=False)
    log_score = log_score_fn(x_t, sigma)
    se = graph.score_entropy(log_score, sigma[:, None], x_t, x0)
    support = span_mask & (x_t == graph.dim - 1)
    n_support = support.sum(-1)
    weighted = (dsigma[:, None] * se) * support
    per_seq = weighted.sum(-1) / n_support.clamp(min=1)
    return per_seq.mean()


def main():
    overrides = sys.argv[1:]
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256", overrides=overrides)

    policy = cfg.mechanism.policy
    r = int(cfg.mechanism.replicate)
    assert policy in ("A", "B", "C", "D"), policy
    assert cfg.training.n_iters == 2500, "pilot 步数固定 2500（协议 §3）"

    check_frozen_clean()
    device = torch.device("cuda")
    torch.cuda.manual_seed(stream_seed("dropout", r))

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg)              # CPU 实例，供 schedule 流使用
    torch.manual_seed(cfg.seeds.model_seed)
    score_model, missing, unexpected = init_model_from_pretrained(cfg, device, cfg.model_path)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)
    optimizer = losses.get_optimizer(cfg, score_model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    optimize_fn = losses.optimization_manager(cfg)

    work_dir = os.path.join(cfg.work_dir, cfg.training.name + "-" + time.strftime("%H%M%S"))
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, "train.log")

    def log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        print(msg)

    log(f"work_dir: {work_dir}")
    log(f"protocol_version={PROTOCOL_VERSION} recipe_version={RECIPE_VERSION}")
    log(f"policy={policy} replicate={r} n_iters={cfg.training.n_iters} "
        f"seq_len={cfg.data.seq_len} span=[{cfg.data.span_min},{cfg.data.span_max}] "
        f"batch={cfg.training.batch_size} accum={cfg.training.accum}")
    log(f"stream seeds: data_order={stream_seed('data_order', r)} sigma={stream_seed('sigma', r)} "
        f"span={stream_seed('span', r)} k={stream_seed('k', r)} "
        f"policy_a={stream_seed('policy_a', r)} policy_b={stream_seed('policy_b', r)} "
        f"dropout={stream_seed('dropout', r)}")

    # step0 共享 artifact（每 run 首次创建一次；frozen evaluator 可直接读）
    shared_dir = os.path.join(cfg.work_dir, "mechpilot_shared")
    os.makedirs(shared_dir, exist_ok=True)
    step0_path = os.path.join(shared_dir, "checkpoint_0000.pth")
    if not os.path.exists(step0_path):
        torch.save(dict(model=score_model.state_dict(), ema=ema.state_dict()), step0_path)
        log(f"step0 共享 artifact 已创建: {step0_path}")

    micro_batch = cfg.training.batch_size // (cfg.ngpus * cfg.training.accum)
    train_ds = get_dataset("wikitext103", "train", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)
    train_loader = build_dataloader(train_ds, micro_batch, stream_seed("data_order", r))

    sched = SharedSchedule(r, cfg.data.seq_len, cfg.data.span_min, cfg.data.span_max,
                           noise=noise)
    g_a = make_generator("policy_a", r)
    pi_L = (draw_pi_L(make_generator("policy_b", r), cfg.data.seq_len)
            if policy == "B" else None)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    K_hist = torch.zeros(cfg.data.seq_len + 1, dtype=torch.long)
    sigma_hist = torch.zeros(50)
    step = 0
    accum_iter = 0
    total_loss = 0.0

    while step < cfg.training.n_iters:
        batch = next(iter(train_loader))["input_ids"].to(device)
        sigma, dsigma, span_len, span_start, K = sched.draw(micro_batch)
        # 直方图在 CPU 上累加（sigma/K 此时仍为 CPU 张量）
        K_hist += torch.bincount(K, minlength=cfg.data.seq_len + 1)
        sigma_hist += torch.bincount(
            sigma.mul(49).long().clamp(0, 49), minlength=50)

        masks = select_mask(span_start, span_len, K, policy, cfg.data.seq_len,
                            pi_L=pi_L, generator=g_a)
        x_t, span_mask = apply_masks(batch, masks, graph.dim - 1)
        sigma = sigma.to(device)
        dsigma = dsigma.to(device)

        loss = span_task_loss(graph, score_model, batch, x_t, span_mask,
                              sigma, dsigma) / cfg.training.accum
        if not torch.isfinite(loss):
            raise RuntimeError(f"非有限 loss={loss.item()} at step {step}，硬停")
        scaler.scale(loss).backward()
        accum_iter += 1
        total_loss += loss.item()

        if accum_iter == cfg.training.accum:
            accum_iter = 0
            step += 1
            optimize_fn(optimizer, scaler, score_model.parameters(), step=step)
            ema.update(score_model.parameters())
            optimizer.zero_grad()

            if step % 100 == 0 or step == 1:
                el = time.time() - t0
                log(f"step {step:6d}: train_loss={total_loss:.5f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} {step/el:.1f} steps/s "
                    f"vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            total_loss = 0.0

            if step in SAVE_STEPS:
                ema_path = os.path.join(work_dir, f"checkpoint_{step}.pth")
                torch.save(dict(ema=ema.state_dict()), ema_path)
                log(f"EMA-only checkpoint: {ema_path}")
            if step == cfg.training.n_iters:
                raw_path = os.path.join(work_dir, "checkpoint_2500_raw.pth")
                torch.save(dict(model=score_model.state_dict()), raw_path)
                log(f"raw weights: {raw_path}")

    # 结束守卫：schedule SHA-256 与 preflight 参考一致
    ref_path = os.path.join(ROOT, "results", "mechanism_pilot_dryrun",
                            "dryrun_summary.json")
    ref = json.load(open(ref_path))
    ref_hash = ref["replicates"][str(r)]["schedule_sha256"]["A"]
    if sched.hexdigest() != ref_hash:
        raise RuntimeError(
            f"schedule hash mismatch: got {sched.hexdigest()[:16]}… "
            f"expected {ref_hash[:16]}…，硬停")

    elapsed = time.time() - t0
    metadata = dict(
        protocol_version=PROTOCOL_VERSION,
        recipe_version=RECIPE_VERSION,
        policy=policy, replicate=r,
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        schedule_sha256=sched.hexdigest(),
        preflight_reference=ref_path,
        n_iters=step,
        steps_per_second=step / max(elapsed, 1e-6),
        peak_vram_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        peak_vram_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
        K_hist=K_hist.tolist(),
        sigma_hist=sigma_hist.tolist(),
        stream_seeds={k: stream_seed(k, r) for k in
                      ("data_order", "sigma", "span", "k", "policy_a", "policy_b", "dropout")},
    )
    with open(os.path.join(work_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"完成: {step} optimizer steps in {elapsed:.0f}s ({metadata['steps_per_second']:.1f} steps/s)")
    log(f"schedule sha256: {sched.hexdigest()[:16]}…（与 preflight 一致 ✓）")
    log(f"metadata: {os.path.join(work_dir, 'run_metadata.json')}")


if __name__ == "__main__":
    main()
