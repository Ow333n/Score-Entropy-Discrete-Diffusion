"""Regime A Vanilla span-infilling 条件 SFT (protocol §7/§9/§3)。

同一 recipe 代码路径覆盖三种模式:
  - smoke (§6.7): n_iters=300, 校准 256-seq 实际吞吐/显存
  - Vanilla pilot (§9.2): plateau 检测确定总预算 N (§3.1, 参数已冻结在 protocol)
  - formal (§3): 显式 analysis checkpoints + §27 协议字段记录

Batch 语义 (G0 §4.1 已核实): config.training.batch_size 是 effective batch,
micro = batch // (ngpus*accum), optimizer step 吃 accum 个 micro。

Task loss (§7.3 + recipe 设计选择, 记录于 regime_a_protocol.yaml):
  - 只计算 span 内当前 mask 位置: (dsigma * SE) 在 masked-span 支撑集上按序列取均值
  - 目标: mean over batch。reduction 选择已冻结: mean_over_masked_span

用法:
  smoke: .venv/bin/python training/vanilla.py training.n_iters=300 training.name=smoke256
  pilot: .venv/bin/python training/vanilla.py training.n_iters=30000 training.name=pilot
"""
import csv
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
from task_data.corruption import corrupt_span_batch, masked_span_positions, EPS
import losses

PROTOCOL_VERSION = "v4.2"
RECIPE_VERSION = "vanilla_v1"


def build_dataloader(ds, micro_batch, data_order_seed):
    """num_workers=0 → 确定性 DataLoader (protocol §3.11A: worker_seed_rule=N/A)。"""
    g = torch.Generator().manual_seed(data_order_seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=micro_batch, shuffle=True, generator=g,
        num_workers=0, pin_memory=True, drop_last=True,
    )


def init_model_from_pretrained(cfg, device, model_path):
    """SEDD(cfg) (dropout=0.0) + 从 HF checkpoint 迁移权重 (strict=False)。

    HF 配置的 dropout 是训练期参数, 不影响权重形状; strict load 应全命中。
    """
    score_model = SEDD(cfg).to(device)
    hf_model = SEDD.from_pretrained(model_path)
    missing, unexpected = score_model.load_state_dict(hf_model.state_dict(), strict=False)
    del hf_model
    return score_model, missing, unexpected


def span_task_loss(noise, graph, model, x0, x_t, span_mask, sigma, dsigma, train):
    """§7.3: (dsigma * SE) 在 span 内 mask 支撑集上按序列均值 → batch 均值。"""
    log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
    log_score = log_score_fn(x_t, sigma)
    se = graph.score_entropy(log_score, sigma[:, None], x_t, x0)      # [B, L]

    support = masked_span_positions(x_t, span_mask, graph.dim - 1)   # [B, L]
    n_support = support.sum(-1)                                      # [B]

    weighted = (dsigma[:, None] * se) * support
    per_seq = weighted.sum(-1) / n_support.clamp(min=1)              # 无 mask 支撑的序列不贡献
    return per_seq.mean()


def run_eval(state, noise, graph, eval_ds, cfg, device):
    """确定性 eval: 固定 64 个验证块 + 每次 eval 重置同一种子 → 跨 checkpoint 可比。"""
    model = state["model"]
    ema = state["ema"]
    g = torch.Generator(device=device).manual_seed(cfg.seeds.corruption_seed + 1000)

    eval_loss, n = 0.0, 0
    with torch.no_grad():
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        for bi in range(cfg.data.eval_chunks):
            x0 = eval_ds[bi]["input_ids"].to(device)[None]            # [1, L]
            x_t, span_mask, sigma, dsigma, _ = corrupt_span_batch(
                x0, noise, cfg.data.span_min, cfg.data.span_max,
                graph.dim - 1, generator=g)
            l = span_task_loss(noise, graph, model, x0, x_t, span_mask, sigma, dsigma, train=False)
            eval_loss += l.item()
            n += 1
        ema.restore(model.parameters())
    return eval_loss / n


def save_ckpt(path, state):
    torch.save(dict(
        model=state["model"].state_dict(),
        ema=state["ema"].state_dict(),
        optimizer=state["optimizer"].state_dict(),
        scaler=state["scaler"].state_dict(),
        step=state["step"],
    ), path)


def restore_ckpt(path, state, device):
    loaded = torch.load(path, map_location=device, weights_only=False)
    state["model"].load_state_dict(loaded["model"], strict=False)
    state["ema"].load_state_dict(loaded["ema"])
    state["optimizer"].load_state_dict(loaded["optimizer"])
    state["scaler"].load_state_dict(loaded["scaler"])
    state["step"] = loaded["step"]
    return state


def main():
    # hydra 风格命令行 override: training.n_iters=30000 training.name=pilot
    overrides = sys.argv[1:]
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256", overrides=overrides)

    device = torch.device("cuda")
    graph_ = __import__("graph_lib")
    noise_lib = __import__("noise_lib")

    work_dir = os.path.join(cfg.work_dir, cfg.training.name + "-" + time.strftime("%H%M%S"))
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, "train.log")
    curve_path = os.path.join(work_dir, "learning_curve.csv")

    def log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        print(msg)

    log(f"work_dir: {work_dir}")
    log(f"protocol_version={PROTOCOL_VERSION} recipe_version={RECIPE_VERSION}")
    log(f"cfg: batch_size={cfg.training.batch_size} accum={cfg.training.accum} ngpus={cfg.ngpus} "
        f"n_iters={cfg.training.n_iters} seq_len={cfg.data.seq_len} span=[{cfg.data.span_min},{cfg.data.span_max}] "
        f"save_at={cfg.training.save_at_steps}")
    log(f"seeds: model={cfg.seeds.model_seed} data_order={cfg.seeds.data_order_seed} "
        f"corruption={cfg.seeds.corruption_seed}")

    # --- 模型 / 图 / 噪声 / 优化状态 ---
    torch.manual_seed(cfg.seeds.model_seed)
    graph = graph_.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model, missing, unexpected = init_model_from_pretrained(cfg, device, cfg.model_path)
    log(f"权重迁移: missing={len(missing)} unexpected={len(unexpected)} (strict=False)")
    n_params = sum(p.numel() for p in score_model.parameters())
    log(f"参数量: {n_params}")

    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)
    optimizer = losses.get_optimizer(cfg, score_model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    state = dict(model=score_model, ema=ema, optimizer=optimizer, scaler=scaler, step=0,
                 noise=noise, graph=graph)
    optimize_fn = losses.optimization_manager(cfg)

    if cfg.load_dir is not None:
        state = restore_ckpt(os.path.join(cfg.load_dir, "checkpoints-meta", "checkpoint.pth"),
                             state, device)
        log(f"resume: 从 {cfg.load_dir} 恢复到 step {state['step']}")

    # --- 数据 ---
    train_ds = get_dataset("wikitext103", "train", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)
    valid_ds = get_dataset("wikitext103", "validation", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)
    micro_batch = cfg.training.batch_size // (cfg.ngpus * cfg.training.accum)
    assert cfg.training.batch_size % (cfg.ngpus * cfg.training.accum) == 0, \
        f"batch_size {cfg.training.batch_size} 不能被 ngpus*accum 整除"
    train_loader = build_dataloader(train_ds, micro_batch, cfg.seeds.data_order_seed)
    log(f"batch semantics: config={cfg.training.batch_size} micro={micro_batch} "
        f"accum={cfg.training.accum} ngpus={cfg.ngpus} eff={micro_batch * cfg.training.accum * cfg.ngpus}")
    log(f"train chunks={len(train_ds)} valid chunks={len(valid_ds)}")

    # --- 训练循环 ---
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    corr_g = torch.Generator(device=device).manual_seed(cfg.seeds.corruption_seed)
    accum_iter = 0
    total_loss = 0.0
    curve_rows = []
    plateau_history = []
    plateau_detected_at = None

    with open(curve_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "eval_loss", "lr"])

    # v4.2 语义修正: n_iters = 精确 optimizer steps (v4.1 的 +1 惯例已审计记录为偏离)
    while state["step"] < cfg.training.n_iters:
        step = state["step"]
        batch = next(iter(train_loader))["input_ids"].to(device)
        x_t, span_mask, sigma, dsigma, _ = corrupt_span_batch(
            batch, noise, cfg.data.span_min, cfg.data.span_max, graph.dim - 1, generator=corr_g)

        loss = span_task_loss(noise, graph, score_model, batch, x_t, span_mask,
                              sigma, dsigma, train=True) / cfg.training.accum
        scaler.scale(loss).backward()
        accum_iter += 1
        total_loss += loss.item()

        if accum_iter == cfg.training.accum:
            accum_iter = 0
            state["step"] += 1
            optimize_fn(optimizer, scaler, score_model.parameters(), step=state["step"])
            ema.update(score_model.parameters())
            optimizer.zero_grad()
            loss_val = total_loss
            total_loss = 0.0

            if state["step"] % cfg.training.log_freq == 0 or state["step"] == 1:
                el = time.time() - t_start
                vram = torch.cuda.max_memory_allocated() / 1e9
                log(f"step {state['step']:6d}: train_loss={loss_val:.5f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} {state['step']/el:.1f} steps/s "
                    f"vram_peak={vram:.2f}GB")

            if state["step"] % cfg.training.eval_freq == 0:
                eval_loss = run_eval(state, noise, graph, valid_ds, cfg, device)
                curve_rows.append([state["step"], loss_val, eval_loss,
                                   optimizer.param_groups[0]["lr"]])
                with open(curve_path, "a", newline="") as f:
                    csv.writer(f).writerow([state["step"], loss_val, eval_loss,
                                            optimizer.param_groups[0]["lr"]])
                log(f"step {state['step']:6d}: eval_loss={eval_loss:.5f}")

                # plateau 检测 (§3.1): 连续 K 次 eval 相对改善 < eps
                plateau_history.append(eval_loss)
                if len(plateau_history) > cfg.training.pilot.plateau_window:
                    window = plateau_history[-cfg.training.pilot.plateau_window - 1:]
                    imps = [(a - b) / abs(a) for a, b in zip(window[:-1], window[1:])]
                    if all(imp < cfg.training.pilot.eps_rel for imp in imps) and plateau_detected_at is None:
                        plateau_detected_at = state["step"]
                        s_p = plateau_detected_at
                        N_hat = min(cfg.training.pilot.n_pilot_max,
                                    int(torch.ceil(torch.tensor(cfg.training.pilot.alpha * s_p))))
                        log(f"PLATEAU detected at step {s_p} (K={cfg.training.pilot.plateau_window}, "
                            f"eps={cfg.training.pilot.eps_rel}) → N = min(N_max, ceil(alpha*s_p)) = {N_hat}")

            if state["step"] in cfg.training.save_at_steps:
                path = os.path.join(work_dir, f"checkpoint_{state['step']}.pth")
                save_ckpt(path, state)
                log(f"analysis checkpoint: {path}")

            if state["step"] % cfg.training.snapshot_freq_for_preemption == 0:
                os.makedirs(os.path.join(work_dir, "checkpoints-meta"), exist_ok=True)
                save_ckpt(os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth"), state)

    # --- 协议字段记录 (§27) ---
    elapsed = time.time() - t_start
    n_steps = state["step"]
    eff_batch = micro_batch * cfg.training.accum * cfg.ngpus
    metadata = dict(
        protocol_version=PROTOCOL_VERSION,
        recipe_version=RECIPE_VERSION,
        loop_semantics="exact_N",   # v4.2 §3.1: n_iters = 精确 optimizer steps (v4.1 的 +1 偏离已审计)
        name=cfg.training.name,
        model_path=cfg.model_path,
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip() or "unknown",
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
        model_seed=cfg.seeds.model_seed,
        data_order_seed=cfg.seeds.data_order_seed,
        corruption_seed=cfg.seeds.corruption_seed,
        dataloader_worker_seed_rule="N/A (num_workers=0, §3.11A)",
        config_batch=cfg.training.batch_size,
        actual_micro_batch=micro_batch,
        grad_accum=cfg.training.accum,
        effective_batch=eff_batch,
        seq_len=cfg.data.seq_len,
        tokens_per_optimizer_step=eff_batch * cfg.data.seq_len,
        unique_training_sequences_seen=n_steps * eff_batch,
        unique_source_tokens_seen=n_steps * eff_batch * cfg.data.seq_len,
        model_forward_tokens=(n_steps * cfg.training.accum + cfg.training.eval_freq > 0) *
                             (n_steps * cfg.training.accum) * micro_batch * cfg.data.seq_len,
        optimizer_steps=n_steps,
        optimizer_steps_per_second=n_steps / max(elapsed, 1e-6),
        tokens_per_second=n_steps * eff_batch * cfg.data.seq_len / max(elapsed, 1e-6),
        wall_clock_seconds=elapsed,
        peak_vram_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        peak_vram_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
        plateau_detected_at=plateau_detected_at,
        N_hat=min(cfg.training.pilot.n_pilot_max,
                  int(torch.ceil(torch.tensor(cfg.training.pilot.alpha * plateau_detected_at))))
        if plateau_detected_at else cfg.training.pilot.n_pilot_max,
        N_pilot_max=cfg.training.pilot.n_pilot_max,
    )
    with open(os.path.join(work_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"\n完成: {n_steps} optimizer steps in {elapsed:.0f}s "
        f"({metadata['optimizer_steps_per_second']:.1f} steps/s, "
        f"{metadata['tokens_per_second']:.0f} tokens/s)")
    log(f"峰值显存: allocated {metadata['peak_vram_allocated_gb']:.2f}GB / "
        f"reserved {metadata['peak_vram_reserved_gb']:.2f}GB")
    if plateau_detected_at:
        log(f"plateau: step {plateau_detected_at} → N_hat = {metadata['N_hat']} (α={cfg.training.pilot.alpha})")
    else:
        log(f"未检测到 plateau → N = N_pilot_max = {cfg.training.pilot.n_pilot_max} (§3.1)")
    log(f"metadata: {os.path.join(work_dir, 'run_metadata.json')}")


if __name__ == "__main__":
    main()
