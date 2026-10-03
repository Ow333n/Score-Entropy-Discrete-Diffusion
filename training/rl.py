"""RL-1 simplified REINFORCE smoke (plan v0.2 §21 + review 定案)。

纯 on-policy / G=4 / mean-centered group advantage / K=1 / uniform timestep /
无 replay、无 ratio、无 clipping、无 critic。

用法:
  .venv/bin/python training/rl.py rl.n_steps=150 rl.lr=3e-6 rl.name=rl1-smoke
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

import graph_lib
import noise_lib
from model import utils as mutils
from model.ema import ExponentialMovingAverage
from rl import loader, reward as rw
from rl import transition as T
from rl.rollout import rollout_chunk
from rl.eval_formal import build_eval_pools, task_eval_point, save_eval_point

PROTOCOL_VERSION = "v1.0-rl1"


def save_snapshot(work_dir, model, ema, step, full=False, optimizer=None, scaler=None):
    """§31: eval snapshot ({model, ema, step}) 或 full checkpoint (含 optimizer/scaler)。
    写入 .tmp + rename 防半截文件。"""
    if full:
        snap = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    ema=dict(decay=ema.decay, num_updates=ema.num_updates,
                             shadow_params=[s.detach().cpu() for s in ema.shadow_params]),
                    optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step)
        name = f"checkpoint_step{step}.pth"
    else:
        snap = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    ema=dict(decay=ema.decay, num_updates=ema.num_updates,
                             shadow_params=[s.detach().cpu() for s in ema.shadow_params]),
                    step=step)
        name = f"eval_snapshot_step{step}.pth"
    path = os.path.join(work_dir, name)
    tmp = path + ".tmp"
    torch.save(snap, tmp)
    os.replace(tmp, path)
    return path


def main():
    overrides = sys.argv[1:]
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256", overrides=overrides)

    device = torch.device("cuda")
    torch.manual_seed(cfg.seeds.model_seed)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    D = cfg.tokens + 1
    MASK = cfg.tokens

    work_dir = os.path.join(cfg.work_dir, cfg.rl.name + "-" + time.strftime("%H%M%S"))
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, "train.log")
    def log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        print(msg)

    log(f"work_dir: {work_dir}")
    log(f"protocol={PROTOCOL_VERSION} lr={cfg.rl.lr} n_steps={cfg.rl.n_steps} "
        f"G={cfg.rl.g} P={cfg.rl.prompts_per_step} pool={cfg.rl.pool} "
        f"rollout_steps={cfg.rl.rollout_steps} seed={cfg.seeds.model_seed}")
    log(f"allocator_env={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')} "
        f"git_head={subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or 'unknown'}")

    # RL init = SFT s1 EMA-10200 (RL-G0 已验证逐位一致)
    model, info = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    log(f"RL init: {cfg.rl.init_dir}/{cfg.rl.init_ckpt} step={info['step']} "
        f"ema_num_updates={info['ema_num_updates']}")
    model.train()
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)  # fresh shadow, n=0
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.rl.lr,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scaler = torch.cuda.amp.GradScaler()

    # 数据: frozen manifest 前 pool 个样本
    import json as _json
    recs = [_json.loads(l) for l in open(cfg.rl.manifest)][: cfg.rl.pool]
    x0_pool = torch.stack([torch.tensor(r["x0"]) for r in recs]).to(device)
    xt0_pool = torch.stack([torch.tensor(r["initial_state"]) for r in recs]).to(device)
    sig0_pool = torch.tensor([r["sigma"] for r in recs], device=device)
    span_pool = torch.zeros(cfg.rl.pool, cfg.data.seq_len, dtype=torch.bool, device=device)
    m0_pool = torch.zeros(cfg.rl.pool, cfg.data.seq_len, dtype=torch.bool, device=device)
    for k, r in enumerate(recs):
        span_pool[k, r["span_start"]:r["span_end"]] = True
        if r["initial_masked_positions"]:
            m0_pool[k, r["initial_masked_positions"]] = True
    log(f"prompt pool: {cfg.rl.pool} 样本 (|M0| mean={m0_pool.sum(-1).float().mean().item():.1f})")

    # formal fixed eval subset (v1.0 §35): manifest 索引 64–127, 64 samples,
    # 独立于 training pool 0–63; IDs 固定并落盘
    eval_pools, eval_ids = build_eval_pools(cfg, device, start=64, n=64)
    with open(os.path.join(work_dir, "eval_subset_ids.json"), "w") as f:
        json.dump(dict(indices=list(range(64, 128)), ids=eval_ids), f, indent=2)

    # 有效数据集 (eval loss 监控)
    from data import get_dataset
    valid_ds = get_dataset("wikitext103", "validation", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)

    sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
    train_score_fn = mutils.get_score_fn(model, train=True, sampling=False)

    rng = torch.Generator().manual_seed(cfg.rl.seed)
    stats = dict(zero_var_groups=0, n_groups=0, nan_rollouts=0)
    curve_rows = []
    t0 = time.time()

    # --- formal step0 task evaluation (v1.0 §32/§35: 在任何 RL update 之前) ---
    log("step0: formal task evaluation (64-sample subset, RAW/EMA, sampled+greedy, frozen NLL)")
    model.eval()
    res0 = task_eval_point(model, ema, sampling_score_fn, graph, noise, valid_ds,
                           eval_pools, eval_ids, cfg, device, tag="step0")
    save_eval_point(work_dir, res0)
    model.train()
    log(f"step0: nll RAW={res0['nll_raw']:.4f} EMA={res0['nll_ema']:.4f} "
        f"sampled RAW={res0['sampled_raw_mean']:.4f} EMA={res0['sampled_ema_mean']:.4f} "
        f"greedy RAW={res0['greedy_raw_mean']:.4f} EMA={res0['greedy_ema_mean']:.4f}")

    # --- formal step0 CPI / OrderGap (v1.0 §36 + gate 4 严格顺序; step0 raw==EMA) ---
    log("step0: CPI / OrderGap evaluation (frozen manifest, weights=ema; step0 raw==EMA)")
    import subprocess as _sp
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for script, out in [("evaluation/eval_cpi.py", "step0_cpi.json"),
                        ("evaluation/eval_order_gap.py", "step0_order_gap.json")]:
        log(f"step0: running {script} → {out}")
        _sp.run([sys.executable, os.path.join(root, script),
                 "--model_path", cfg.rl.init_dir, "--ckpt", cfg.rl.init_ckpt,
                 "--weights", "ema", "--chunk", "4",
                 "--out", os.path.join(work_dir, out)], check=True, cwd=root)
        log(f"step0: {out} done")
    log("step0 evaluation complete; optimizer step 1 next")

    for step in range(1, cfg.rl.n_steps + 1):
        # --- rollout (no_grad) ---
        idx = torch.randperm(cfg.rl.pool, generator=rng)[: cfg.rl.prompts_per_step]
        x0 = x0_pool[idx].repeat_interleave(cfg.rl.g, dim=0)
        xt0 = xt0_pool[idx].repeat_interleave(cfg.rl.g, dim=0)
        sig0 = sig0_pool[idx].repeat_interleave(cfg.rl.g)
        m0 = m0_pool[idx].repeat_interleave(cfg.rl.g, dim=0)
        model.eval()
        torch.cuda.empty_cache()          # WDDM 8GB 碎片敏感: 训练图释放后清缓存再 rollout
        trajs = []
        with torch.no_grad():
            for lo in range(0, x0.shape[0], cfg.rl.chunk):
                trs, gst = rollout_chunk(model, sampling_score_fn, graph, noise,
                                         x0[lo:lo + cfg.rl.chunk], xt0[lo:lo + cfg.rl.chunk],
                                         sig0[lo:lo + cfg.rl.chunk],
                                         cfg.rl.rollout_steps, seed=cfg.rl.seed + step * 1000 + lo,
                                         mask_token=MASK)
                trajs += trs
                for k in gst:
                    stats.setdefault(k, 0)
                    stats[k] += gst[k]
        rewards = torch.stack([rw.m0_reward(x0[b:b + 1], trajs[b]["final_x"].unsqueeze(0).to(device),
                                            m0[b:b + 1]) for b in range(x0.shape[0])]).squeeze(1)
        A = T.mean_centered_advantage(rewards.view(-1, cfg.rl.g)).view(-1)
        adv_mean, adv_std, adv_mean_abs = A.mean().item(), A.std().item(), A.abs().mean().item()
        zvg = int((A.view(-1, cfg.rl.g).abs().sum(dim=1) == 0).sum())
        stats["zero_var_groups"] += zvg
        stats["n_groups"] += cfg.rl.prompts_per_step

        # --- recompute logπ @J + PG update ---
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        n_pos = 0
        for b in range(x0.shape[0]):
            tj = trajs[b]
            xJ = tj["x_J"].to(device)[None]
            sigJ = torch.full((1, 1), tj["sigma_J"], device=device)
            dJ = torch.full((1, 1), tj["dsigma_J"], device=device)
            log_score = train_score_fn(xJ, sigJ.squeeze(-1))       # log score
            s = log_score.exp()
            w = T.transition_weights(graph, xJ, s, dJ)
            ok, gstats = T.validity_check(w)
            if not ok:
                stats["nan_rollouts"] += 1
                continue
            logpi = T.policy_log_probs(w)
            mask = tj["mask_J"].to(device)
            lp = T.log_probs_for_actions(logpi, tj["action"].to(device)[None]).squeeze(0)
            # masked_select: 非 MASK 位置 lp = -inf (w=0), -inf*0(False) = nan 会污染梯度
            loss = -A[b].detach() * lp.masked_select(mask).sum() / cfg.rl.g
            scaler.scale(loss).backward()
            total_loss += loss.detach().item()
            n_pos += int(mask.sum())
        if n_pos > 0:
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model.parameters())
        else:
            gnorm = 0.0
        del trajs
        torch.cuda.empty_cache()

        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            alloc = torch.cuda.memory_allocated() / 1e9
            resv = torch.cuda.memory_reserved() / 1e9
            max_alloc = torch.cuda.max_memory_allocated() / 1e9
            max_resv = torch.cuda.max_memory_reserved() / 1e9
            free_b, _ = torch.cuda.mem_get_info()
            log(f"step {step:4d}: reward={rewards.mean().item():.4f}±{rewards.std().item():.4f} "
                f"zvg={stats['zero_var_groups']}/{stats['n_groups']} "
                f"A_mean={adv_mean:+.5f} A_std={adv_std:.5f} A_mean_abs={adv_mean_abs:.5f} "
                f"grad_norm={gnorm:.2f} loss={total_loss:.4f} "
                f"{step / el:.3f} steps/s alloc={alloc:.2f}GB resv={resv:.2f}GB "
                f"peak_alloc={max_alloc:.2f}GB peak_resv={max_resv:.2f}GB free={free_b / 1e9:.2f}GB")
            curve_rows.append([step, rewards.mean().item(), rewards.std().item(), gnorm, total_loss,
                               adv_mean, adv_std, adv_mean_abs, alloc, resv, max_alloc, max_resv,
                               free_b / 1e9])

        if step in (50, 100, 250, 500):
            # v1.0 §31/§32: 评估点 0/50/100/250/500; 先落 eval snapshot 供 post-run CPI/OrderGap
            model.eval()
            save_snapshot(work_dir, model, ema, step, full=False)
            res = task_eval_point(model, ema, sampling_score_fn, graph, noise, valid_ds,
                                  eval_pools, eval_ids, cfg, device, tag=f"step{step}")
            save_eval_point(work_dir, res)
            model.train()
            log(f"step {step:4d}: formal eval nll RAW={res['nll_raw']:.4f} EMA={res['nll_ema']:.4f} "
                f"sampled64 RAW={res['sampled_raw_mean']:.4f} EMA={res['sampled_ema_mean']:.4f} "
                f"greedy RAW={res['greedy_raw_mean']:.4f} EMA={res['greedy_ema_mean']:.4f}")

        if step % 25 == 0:
            # §31: atomic rolling checkpoint_latest（每 25 步覆盖）
            p = save_snapshot(work_dir, model, ema, step, full=True,
                              optimizer=optimizer, scaler=scaler)
            os.replace(p, os.path.join(work_dir, "checkpoint_latest.pth"))

    # --- §31: step500 full checkpoint + rolling latest ---
    import shutil
    final_full = save_snapshot(work_dir, model, ema, cfg.rl.n_steps, full=True,
                               optimizer=optimizer, scaler=scaler)
    shutil.copyfile(final_full, os.path.join(work_dir, "checkpoint_latest.pth"))
    json.dump(dict(protocol=PROTOCOL_VERSION, stats=stats, n_steps=cfg.rl.n_steps,
                   final_reward_mean=float(rewards.mean()), init=info,
                   allocator_env=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
                   git_head=subprocess.run(["git", "rev-parse", "HEAD"],
                                           capture_output=True, text=True).stdout.strip() or "unknown"),
              open(os.path.join(work_dir, "run_metadata.json"), "w"), indent=2)
    with open(os.path.join(work_dir, "learning_curve.csv"), "w", newline="") as f:
        csv.writer(f).writerow(["step", "reward_mean", "reward_std", "grad_norm", "loss",
                                "adv_mean", "adv_std", "adv_mean_abs", "alloc_gb", "resv_gb",
                                "peak_alloc_gb", "peak_resv_gb", "free_gb"])
        csv.writer(f).writerows(curve_rows)
    log(f"完成: {cfg.rl.n_steps} RL steps in {time.time() - t0:.0f}s")
    log(f"metadata: {os.path.join(work_dir, 'run_metadata.json')}")


if __name__ == "__main__":
    main()
