"""v2.1 complementary-exposure pilot 训练入口
（protocol mechanism_complementary_exposure_v2_1 §10/§14 步骤 6，2026-10-06）。

U/H/L × 2 replicates = 6 runs；每 run：
- 共享 schedule（σ/dσ/span/K/phase/singleton-coin，同 replicate 内三 policy 逐位一致）
- policy 只决定 mask 子集选择（frozen pair map / odd-m bank；H/L/U 见 v21_policy.py）
- 损失 = frozen recipe 同口径：Absorbing.score_entropy × dσ 加权 × masked-span 均值
  × batch 均值（training objective 零改动）
- checkpoint 容器 = {"ema": state_dict, "step": ...}（v1.2 同款；frozen evaluator 的
  weights="ema" 路径直接可读）；2500 另存 {"model": ...}；step0 全库共享一份
- 训练期 logging（1F）：loss 轨迹、grad norm（每 100 步 + rolling variance 近似）、
  每 checkpoint 固定 eval 的 masked-NLL/acc（64 块、冻结 corruption realization、
  EMA 权重；诊断性，正式任务指标由 P7 eval_task 给出）
- G3a/§7.6 诊断全程流式记录（exact-K 断言、E_comp、C̄_treated、L/R、marginal、
  degenerate K 表、7 项 confound diagnostics）
- 硬停守卫（不临时修参数继续跑）：
  ① frozen 文件被改动（git diff）→ RuntimeError
  ② 非有限 loss（NaN/Inf）→ RuntimeError
  ③ 协议 v2.1 sha256 不符 → RuntimeError
  ④ frozen maps sha256 不符（load_maps_verified）→ AssertionError
  ⑤ 结束 schedule digest / E_comp / C̄ 与 dryrun 参考不一致 → RuntimeError
  ⑥ OOM / 任何异常 → 进程直接失败退出（无自动重试）

用法:
  .venv/bin/python training/pilot_v21.py mechanism_v21.policy=H \
      mechanism_v21.replicate=1 training.n_iters=2500 training.name=v21pilot-H1

recipe（与 v1.2 实际执行一致，记录于 run_metadata）：lr=config.optim.lr=3e-4、
warmup 2500、batch 32 eff、accum 1、EMA 0.9999、dropout 0、span [10,50]、seq 256。
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
from task_data.corruption import corrupt_span_batch, masked_span_positions
from task_data.policy_corruption import stream_seed as v12_stream_seed
from task_data.v21_policy import (MaskDiagnostics, V21Schedule, closed_form_e_comp,
                                  load_maps_verified, make_generator, select_mask_v21,
                                  stream_seed)
from compatibility.posterior import clean_log_probs

PROTOCOL_VERSION = "v2.1"
RECIPE_VERSION = "v21_pilot_v1"
PROTOCOL_SHA256 = "af312d864d41d8f67da343d55a8dee8a29a91c37dea622e6b2c4b22ef441fc61"
FROZEN_FILES = ["losses.py", "graph_lib.py", "noise_lib.py", "data.py",
                "training/vanilla.py", "task_data/corruption.py",
                "task_data/policy_corruption.py", "model/",
                "configs/vanilla_256.yaml",
                "task_data/v21_policy.py", "evaluation/eval_cpi_pairs.py",
                "evaluation/eval_order_gap_subset.py"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_STREAM_NAMES = {
    "U": ("u_uniform",),
    "H": ("h_pair", "h_orient"),
    "L": ("l_pair", "l_orient"),
}


def check_frozen_clean():
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"] + FROZEN_FILES,
                       cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError("frozen 文件被改动，硬停（禁止临时修改继续跑）")


def check_protocol_hash():
    import hashlib
    digest = hashlib.sha256(open(os.path.join(
        ROOT, "protocol/mechanism_complementary_exposure_v2_1.md"), "rb").read()).hexdigest()
    if digest != PROTOCOL_SHA256:
        raise RuntimeError(f"协议 v2.1 sha256 不符: {digest[:16]}… 硬停")


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


def run_fixed_eval(state, noise, graph, valid_ds, cfg, device):
    """1F 诊断性固定 eval：64 块、冻结 corruption realization（seed+1000）、EMA 权重。

    返回 (eval_loss, masked_nll, token_acc)。正式任务指标由 P7 eval_task 给出。
    """
    model, ema = state["model"], state["ema"]
    g = torch.Generator(device=device).manual_seed(cfg.seeds.corruption_seed + 1000)
    eval_loss, nll_sum, acc_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        log_score_fn = mutils.get_score_fn(model, train=False, sampling=False)
        for bi in range(cfg.data.eval_chunks):
            x0 = valid_ds[bi]["input_ids"].to(device)[None]
            x_t, span_mask, sigma, dsigma, _ = corrupt_span_batch(
                x0, noise, cfg.data.span_min, cfg.data.span_max, graph.dim - 1, generator=g)
            log_score = log_score_fn(x_t, sigma)
            se = graph.score_entropy(log_score, sigma[:, None], x_t, x0)
            support = masked_span_positions(x_t, span_mask, graph.dim - 1)
            n_support = support.sum(-1)
            weighted = (dsigma[:, None] * se) * support
            per_seq = weighted.sum(-1) / n_support.clamp(min=1)
            eval_loss += per_seq.mean().item()
            logp = clean_log_probs(log_score, graph.dim)
            pos = support[0].nonzero().squeeze(1)
            if pos.numel():
                gold = x0[0, pos]
                nll_sum += (-logp[0, pos, gold]).mean().item()
                acc_sum += (logp[0, pos].argmax(-1) == gold).float().mean().item()
                n += 1
        ema.restore(model.parameters())
    return eval_loss / cfg.data.eval_chunks, nll_sum / max(n, 1), acc_sum / max(n, 1)


def grad_norm_of(model):
    return sum(p.grad.detach().float().pow(2).sum()
               for p in model.parameters() if p.grad is not None).sqrt().item()


def preclip_grad_norm_of(model, scale):
    """pre-clip grad norm：backward 后 grads 为 scaled 值，除以 scaler scale 精确还原
    （loss 已断言 finite → 无 inf 通道，除法逐位精确）。"""
    return grad_norm_of(model) / scale


def main():
    overrides = sys.argv[1:]
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256", overrides=overrides)

    policy = cfg.mechanism_v21.policy
    r = int(cfg.mechanism_v21.replicate)
    assert policy in ("U", "H", "L"), policy
    assert cfg.training.accum == 1 and cfg.training.batch_size == 32, "recipe 固定 batch 32/accum 1"
    save_steps = tuple(int(s) for s in cfg.mechanism_v21.save_steps)
    assert cfg.training.n_iters in (100, 2500), "仅支持 smoke(100) 与 formal(2500) 两种步数"
    if cfg.training.n_iters == 2500:
        assert save_steps == (500, 1020, 2500), "formal save_steps 固定 [500,1020,2500]（§10）"

    check_frozen_clean()
    check_protocol_hash()
    maps = load_maps_verified(os.path.join(ROOT, cfg.mechanism_v21.maps_dir))

    device = torch.device("cuda")
    torch.cuda.manual_seed(v12_stream_seed("dropout", r))

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg)
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
        f"batch={cfg.training.batch_size} accum={cfg.training.accum} "
        f"lr={cfg.optim.lr} warmup={cfg.optim.warmup} ema={cfg.training.ema}")
    log(f"maps: {cfg.mechanism_v21.maps_dir} manifest_sha={maps['file_sha256']['maps_manifest.json'][:16]}…")

    # step0 共享 artifact（与 v1.2 共用同一 pretrained init；frozen evaluator 可直接读）
    shared_dir = os.path.join(cfg.work_dir, "mechpilot_shared")
    step0_path = os.path.join(shared_dir, "checkpoint_0000.pth")
    assert os.path.exists(step0_path), f"step0 共享 artifact 不存在: {step0_path}"
    log(f"step0 共享 artifact 复用: {step0_path}")

    micro_batch = cfg.training.batch_size // (cfg.ngpus * cfg.training.accum)
    train_ds = get_dataset("wikitext103", "train", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)
    valid_ds = get_dataset("wikitext103", "validation", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)
    train_loader = build_dataloader(train_ds, micro_batch, v12_stream_seed("data_order", r))

    sched = V21Schedule(r, cfg.data.seq_len, cfg.data.span_min, cfg.data.span_max,
                        noise=noise)
    gens = {n: make_generator(n, r) for n in POLICY_STREAM_NAMES[policy]}
    rep_maps = maps[f"rep{r}"]

    def pair_map_for(m):
        return rep_maps["even" if m % 2 == 0 else "odd"][str(m)]

    diag = MaskDiagnostics()
    grad_norms = []
    fixed_evals = []

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    K_hist = torch.zeros(cfg.data.seq_len + 1, dtype=torch.long)
    sigma_hist = torch.zeros(50)
    step = 0
    accum_iter = 0
    total_loss = 0.0
    e_comp_closed_sum = 0.0
    n_items_total = 0

    while step < cfg.training.n_iters:
        batch = next(iter(train_loader))["input_ids"].to(device)
        sigma, dsigma, span_len, span_start, K, phase, coin = sched.draw(micro_batch)
        K_hist += torch.bincount(K, minlength=cfg.data.seq_len + 1)
        sigma_hist += torch.bincount(sigma.mul(49).long().clamp(0, 49), minlength=50)

        masks = []
        for j in range(micro_batch):
            m = int(span_len[j].item())
            k = int(K[j].item())
            s0 = int(span_start[j].item())
            pm = pair_map_for(m)
            c = float(coin[j].item()) if m % 2 == 1 else None
            sel = select_mask_v21(s0, m, k, policy, cfg.data.seq_len, pm,
                                  phase=int(phase[j].item()), coin=c, gens=gens)
            sel_t = torch.tensor(sel, dtype=torch.long)
            if sel_t.numel() != k:
                diag.exact_k_violations += 1
            masks.append(sel_t)
            diag.update({p - s0 for p in sel}, m, k, float(sigma[j].item()), pm,
                        int(phase[j].item()) if m % 2 == 1 else None)
            e_comp_closed_sum += closed_form_e_comp(m, k, policy)
            n_items_total += 1

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
            gn_pre = preclip_grad_norm_of(score_model, scaler.get_scale())  # pre-clip（1F）
            optimize_fn(optimizer, scaler, score_model.parameters(), step=step)
            gn = grad_norm_of(score_model)          # post-clip（optimize_fn 内已 unscale + clip）
            grad_norms.append(gn)
            ema.update(score_model.parameters())
            optimizer.zero_grad()

            if step % 100 == 0 or step == 1:
                el = time.time() - t0
                gn_var = (torch.tensor(grad_norms[-100:]).var(unbiased=True).item()
                          if len(grad_norms) >= 2 else 0.0)
                log(f"step {step:6d}: train_loss={total_loss:.5f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"grad_norm_pre={gn_pre:.4f} grad_norm_post={gn:.4f} "
                    f"grad_norm_var(100)={gn_var:.2e} {step/el:.1f} steps/s "
                    f"vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            total_loss = 0.0

            if step in save_steps:
                ema_path = os.path.join(work_dir, f"checkpoint_{step}.pth")
                torch.save(dict(ema=ema.state_dict(), step=step), ema_path)
                log(f"EMA-only checkpoint: {ema_path}")
                el = time.time() - t0
                ev_loss, ev_nll, ev_acc = run_fixed_eval(
                    dict(model=score_model, ema=ema), noise, graph, valid_ds, cfg, device)
                fixed_evals.append(dict(step=step, eval_loss=ev_loss,
                                        masked_nll=ev_nll, token_acc=ev_acc))
                log(f"step {step:6d}: fixed_eval loss={ev_loss:.4f} "
                    f"masked_nll={ev_nll:.4f} token_acc={ev_acc:.4f} "
                    f"({el:.0f}s elapsed)")
            if step == cfg.training.n_iters:
                raw_path = os.path.join(work_dir, f"checkpoint_{step}_raw.pth")
                torch.save(dict(model=score_model.state_dict(), step=step), raw_path)
                log(f"raw weights: {raw_path}")

    # --- 结束守卫：与 dryrun 参考比对（训练/dryrun 同流同 pattern → 应逐位一致）---
    ref_path = os.path.join(ROOT, "results", "mechanism_pilot_v21_dryrun",
                            "dryrun_summary.json")
    ref = json.load(open(ref_path))["replicates"][str(r)]
    n_done = step
    if n_done == 2500:
        ref_digest = ref["shared_v21_digest"]
        ref_e = ref["e_comp_measured"][policy]
        ref_c = ref["cbar_treated"][policy]
    elif str(n_done) in ref["prefixes"]:
        ref_digest = ref["prefixes"][str(n_done)]["shared_digest"]
        ref_e = ref["prefixes"][str(n_done)]["e_comp"][policy]
        ref_c = ref["prefixes"][str(n_done)]["cbar_treated"][policy]
    else:
        raise RuntimeError(f"无 {n_done} 步的 dryrun 参考，硬停")
    if sched.hexdigest() != ref_digest:
        raise RuntimeError(
            f"v21 schedule digest mismatch: got {sched.hexdigest()[:16]}… "
            f"expected {ref_digest[:16]}…，硬停")
    out_diag = diag.finalize(e_comp_closed=e_comp_closed_sum / n_items_total)
    if abs(out_diag["e_comp"]["overall"] - ref_e) > 1e-9:
        raise RuntimeError(
            f"E_comp 与 dryrun 参考不一致: "
            f"{out_diag['e_comp']['overall']:.9f} vs {ref_e:.9f}，硬停")
    if abs(out_diag["covariance"]["treated_edges"]["cbar"] - ref_c) > 1e-9:
        raise RuntimeError(
            f"C̄_treated 与 dryrun 参考不一致: "
            f"{out_diag['covariance']['treated_edges']['cbar']:.9f} vs {ref_c:.9f}，硬停")

    elapsed = time.time() - t0
    metadata = dict(
        protocol_version=PROTOCOL_VERSION,
        recipe_version=RECIPE_VERSION,
        protocol_sha256=PROTOCOL_SHA256,
        policy=policy, replicate=r,
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        maps_manifest_sha256=maps["file_sha256"]["maps_manifest.json"],
        dryrun_reference=ref_path,
        schedule_sha256=sched.hexdigest(),
        shared_quad_digest=sched.sched.hexdigest(),
        n_iters=step,
        steps_per_second=step / max(elapsed, 1e-6),
        peak_vram_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        peak_vram_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
        K_hist=K_hist.tolist(),
        sigma_hist=sigma_hist.tolist(),
        grad_norms=grad_norms,
        grad_norm_rolling_window=100,
        fixed_evals=fixed_evals,
        lr=cfg.optim.lr,
        warmup=cfg.optim.warmup,
        stream_seeds={k: stream_seed(k, r) for k in
                      ("h_pair", "h_orient", "l_pair", "l_orient",
                       "u_uniform", "singleton_coin")},
        shared_stream_seeds={k: v12_stream_seed(k, r) for k in
                             ("data_order", "sigma", "span", "k")},
        diagnostics=out_diag,
    )
    with open(os.path.join(work_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"完成: {step} optimizer steps in {elapsed:.0f}s "
        f"({metadata['steps_per_second']:.1f} steps/s)")
    log(f"schedule digest: {sched.hexdigest()[:16]}…（与 dryrun 一致 ✓）")
    log(f"E_comp={out_diag['e_comp']['overall']:.4f}（闭式 "
        f"{out_diag['e_comp']['closed_form']:.4f}，dryrun 一致 ✓）")
    log(f"C̄_treated={out_diag['covariance']['treated_edges']['cbar']:+.4f} "
        f"（dryrun 一致 ✓）")
    log(f"metadata: {os.path.join(work_dir, 'run_metadata.json')}")


if __name__ == "__main__":
    main()
