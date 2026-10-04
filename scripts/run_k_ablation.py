"""K=1 vs K=4 diagnostic ablation runner（protocol/rl_k_ablation_protocol.md）。

与 formal RL-1 的唯一 scientific 差异 = K；实现差异（记录在案）：
  - J timestep 采样用独立 j_rng（seed = rl.seed + step*10000 + chunk_offset），
    与 action-sampling 全局 RNG 隔离 → K=1 与 K=4 的 rollout action stream 逐位相同，
    K=1 的 J = K=4 的第 1 个 J（同 seed 同 randperm 首元素）。
  - K=4: uniform without replacement from safe-region support（σ≥0.05）。
  - sequential backward：每 J 独立 graph，loss_j = −A·Σlp/(G·K)，backward 后即释，
    全部完成后 optimizer.step 一次（数学上 = 4 个 timestep estimator 的平均梯度）。
  - 轻量存储：rolling checkpoint（成功后删除）+ 最终 raw/EMA snapshot（eval 用）。

用法:
  .venv/bin/python scripts/run_k_ablation.py --k 4 --steps 5  --name smoke-k4
  .venv/bin/python scripts/run_k_ablation.py --k 1 --steps 100 --name k1-100
"""
import argparse
import csv
import json
import os
import re
import resource
import shutil
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
from rl.eval_formal import build_eval_pools, task_eval_point, save_eval_point
from catsample import sample_categorical

SIGMA_GATE_MIN = 0.05
EPS = 1e-3


def rollout_chunk_k(model, sampling_score_fn, graph, noise, x0, x_t0, sigma0, steps,
                    seed, mask_token, K, j_rng):
    """rollout_chunk 的 diagnostic 变体：j_rng 隔离 + K 个 J 保存点。"""
    torch.manual_seed(seed)          # action sampling 全局流（与 formal 相同来源）
    device = x_t0.device
    B = x_t0.shape[0]
    t0 = ((1 - (-sigma0).exp()) / (1 - EPS)).clamp(min=EPS + 1e-6)
    ts_mat = EPS + (t0[:, None] - EPS) * (1 - (torch.arange(1, steps + 1, device=device) / steps)[None, :])
    sig_mat = noise.total_noise(ts_mat)
    valid = sig_mat >= SIGMA_GATE_MIN
    Js = []
    support_sizes = []
    for b in range(B):
        v = valid[b].nonzero().squeeze(-1)
        nv = int(v.numel())
        support_sizes.append(nv)
        kk = min(K, nv)
        perm = torch.randperm(nv, generator=j_rng)   # uniform without replacement（专用 j_rng）
        Js.append(v[perm[:kk]].tolist())
    x = x_t0.clone()
    t_cur = t0.clone()
    saved = [dict(js=[None] * K, final_x=None) for _ in range(B)]
    gate_stats = dict(soft_neg_elements=0, soft_steps=0)
    with torch.no_grad():
        for i in range(steps):
            t_next = EPS + (t0 - EPS) * (1 - (i + 1) / steps)
            sigma_cur = noise.total_noise(t_cur)
            sigma_next = noise.total_noise(t_next)
            dsigma = (sigma_cur - sigma_next)[:, None]
            s = sampling_score_fn(x, sigma_cur.squeeze(-1))
            stag = T.staggered_score_fn(s, dsigma)
            trans = graph.transp_transition(x, dsigma)
            w = stag * trans
            ok, stats = T.validity_check(w)
            if not ok:
                if (sigma_cur >= SIGMA_GATE_MIN).all():
                    raise RuntimeError(
                        f"rollout validity gate 硬失败 step={i}: {stats} | σ={float(sigma_cur)}")
                gate_stats["soft_neg_elements"] += stats["n_neg"]
                gate_stats["soft_steps"] += 1
            x_next = sample_categorical(w)
            for b in range(B):
                for j, jj in enumerate(Js[b]):
                    if jj == i:
                        saved[b]["js"][j] = dict(
                            x_J=x[b].cpu(), action=x_next[b].cpu(),
                            mask_J=(x[b] == mask_token).cpu(),
                            sigma_J=float(sigma_cur[b].item()),
                            dsigma_J=float(dsigma[b, 0].item()), J=jj)
            x = x_next
            t_cur = t_next
        s = sampling_score_fn(x, sigma_next.squeeze(-1))
        stag = T.staggered_score_fn(s, sigma_next[:, None])
        probs = stag * graph.transp_transition(x, sigma_next[:, None])
        probs = probs[..., :-1]
        x = sample_categorical(probs)
    for b in range(B):
        saved[b]["final_x"] = x[b].cpu()
    return saved, gate_stats, support_sizes


def meminfo():
    m = open("/proc/meminfo").read()
    total = int(re.search(r"MemTotal:\s+(\d+)", m).group(1))
    avail = int(re.search(r"MemAvailable:\s+(\d+)", m).group(1))
    sw_total = int(re.search(r"SwapTotal:\s+(\d+)", m).group(1))
    sw_free = int(re.search(r"SwapFree:\s+(\d+)", m).group(1))
    return dict(avail_gb=avail / 1e6, total_gb=total / 1e6,
                swap_used_gb=(sw_total - sw_free) / 1e6)


def check_ram_stop(hist, log):
    if len(hist) < 1:
        return False
    m = hist[-1]
    if m["swap_used_gb"] > 1.5:
        log(f"RAM STOP: swap used {m['swap_used_gb']:.2f}GiB > 1.5GiB")
        return True
    if len(hist) >= 3 and all(h["avail_gb"] < 2.5 for h in hist[-3:]):
        log(f"RAM STOP: MemAvailable < 2.5GiB 连续 3 个采样点 ({hist[-3]['avail_gb']:.2f}GiB)")
        return True
    return False


def check_disk_stop(log):
    free = shutil.disk_usage("/mnt/d").free / 1e9
    if free < 10.0:
        log(f"DISK STOP: /mnt/d free {free:.2f}GB < 10GB")
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--work-dir", type=str, default="exp_local/regime_a/rl-k-ablation")
    args = ap.parse_args()
    K = args.k

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    device = torch.device("cuda")
    torch.manual_seed(cfg.seeds.model_seed)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    MASK = cfg.tokens

    work_dir = os.path.join(args.work_dir, args.name)
    os.makedirs(work_dir, exist_ok=True)
    log_path = os.path.join(work_dir, "train.log")
    def log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        print(msg)

    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or "unknown"
    log(f"K-ablation runner: K={K} steps={args.steps} name={args.name}")
    log(f"protocol=rl_k_ablation_protocol.md lr={cfg.rl.lr} G={cfg.rl.g} P={cfg.rl.prompts_per_step} "
        f"rollout_steps={cfg.rl.rollout_steps} seed={cfg.seeds.model_seed}")
    log(f"allocator_env={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')} git_head={git_head}")
    log(f"j_rng seed schedule: rl.seed + step*10000 + chunk_offset（独立 generator，"
        f"action RNG = torch.manual_seed(seed) 同 formal 来源）")
    log(f"K semantics: uniform without replacement from safe support; "
        f"loss_j = -A·Σlp/(G·K); sequential backward; 单次 optimizer.step")

    model, info = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    log(f"RL init: step={info['step']} ema_num_updates={info['ema_num_updates']}")
    model.train()
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.rl.lr,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scaler = torch.cuda.amp.GradScaler()

    recs = [json.loads(l) for l in open(cfg.rl.manifest)][: cfg.rl.pool]
    x0_pool = torch.stack([torch.tensor(r["x0"]) for r in recs]).to(device)
    xt0_pool = torch.stack([torch.tensor(r["initial_state"]) for r in recs]).to(device)
    sig0_pool = torch.tensor([r["sigma"] for r in recs], device=device)
    m0_pool = torch.zeros(cfg.rl.pool, cfg.data.seq_len, dtype=torch.bool, device=device)
    for k, r in enumerate(recs):
        if r["initial_masked_positions"]:
            m0_pool[k, r["initial_masked_positions"]] = True

    eval_pools = None
    eval_ids = None
    from data import get_dataset
    valid_ds = get_dataset("wikitext103", "validation", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)
    sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
    train_score_fn = mutils.get_score_fn(model, train=True, sampling=False)

    rng = torch.Generator().manual_seed(cfg.rl.seed)
    stats = dict(zero_var_groups=0, n_groups=0, nan_rollouts=0, soft_neg_elements=0, soft_steps=0)
    curve_rows = []
    ram_hist = []
    t0 = time.time()

    def do_eval(tag):
        nonlocal eval_pools, eval_ids
        if eval_pools is None:
            x0_e, xt0_e, sig0_e, m0_e, eval_ids = build_eval_pools(cfg, device, start=64, n=64)
            eval_pools = (x0_e, xt0_e, sig0_e, m0_e)
        model.eval()
        res = task_eval_point(model, ema, sampling_score_fn, graph, noise, valid_ds,
                              eval_pools, eval_ids, cfg, device, tag=tag)
        save_eval_point(work_dir, res)
        model.train()
        log(f"{tag}: nll RAW={res['nll_raw']:.4f} EMA={res['nll_ema']:.4f} "
            f"sampled RAW={res['sampled_raw_mean']:.4f} greedy RAW={res['greedy_raw_mean']:.4f}")

    # step0 eval（任何 update 之前）
    if args.steps >= 50:
        log("step0: formal64 evaluation")
        do_eval("step0")

    init_params = [p.detach().clone() for p in model.parameters()]
    n_steps_done = 0
    failure = None
    try:
        for step in range(1, args.steps + 1):
            # --- rollout (no_grad) ---
            idx = torch.randperm(cfg.rl.pool, generator=rng)[: cfg.rl.prompts_per_step]
            x0 = x0_pool[idx].repeat_interleave(cfg.rl.g, dim=0)
            xt0 = xt0_pool[idx].repeat_interleave(cfg.rl.g, dim=0)
            sig0 = sig0_pool[idx].repeat_interleave(cfg.rl.g)
            m0 = m0_pool[idx].repeat_interleave(cfg.rl.g, dim=0)
            model.eval()
            torch.cuda.empty_cache()
            trajs = []
            supports = []
            with torch.no_grad():
                for lo in range(0, x0.shape[0], cfg.rl.chunk):
                    j_rng = torch.Generator().manual_seed(
                        cfg.rl.seed + step * 10000 + lo)
                    trs, gst, sup = rollout_chunk_k(
                        model, sampling_score_fn, graph, noise,
                        x0[lo:lo + cfg.rl.chunk], xt0[lo:lo + cfg.rl.chunk],
                        sig0[lo:lo + cfg.rl.chunk], cfg.rl.rollout_steps,
                        seed=cfg.rl.seed + step * 1000 + lo,
                        mask_token=MASK, K=K, j_rng=j_rng)
                    trajs += trs
                    supports += sup
                    for k in gst:
                        stats[k] = stats.get(k, 0) + gst[k]
            rewards = torch.stack([rw.m0_reward(x0[b:b + 1], trajs[b]["final_x"].unsqueeze(0).to(device),
                                                m0[b:b + 1]) for b in range(x0.shape[0])]).squeeze(1)
            A = T.mean_centered_advantage(rewards.view(-1, cfg.rl.g)).view(-1)
            adv_mean, adv_std, adv_mean_abs = A.mean().item(), A.std().item(), A.abs().mean().item()
            zvg = int((A.view(-1, cfg.rl.g).abs().sum(dim=1) == 0).sum())
            stats["zero_var_groups"] += zvg
            stats["n_groups"] += cfg.rl.prompts_per_step
            support_frac = sum(supports) / (len(supports) * cfg.rl.rollout_steps)

            # --- sequential recompute (每 J 独立 graph) + backward ---
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            n_pos = 0
            for b in range(x0.shape[0]):
                tjb = trajs[b]
                for j in range(K):
                    tj = tjb["js"][j]
                    if tj is None:
                        raise RuntimeError(f"J save point missing: b={b} j={j} step={step}")
                    xJ = tj["x_J"].to(device)[None]
                    sigJ = torch.full((1, 1), tj["sigma_J"], device=device)
                    dJ = torch.full((1, 1), tj["dsigma_J"], device=device)
                    log_score = train_score_fn(xJ, sigJ.squeeze(-1))
                    s = log_score.exp()
                    w = T.transition_weights(graph, xJ, s, dJ)
                    ok, gstats = T.validity_check(w)
                    if not ok:
                        stats["nan_rollouts"] += 1
                        continue
                    logpi = T.policy_log_probs(w)
                    mask = tj["mask_J"].to(device)
                    lp = T.log_probs_for_actions(logpi, tj["action"].to(device)[None]).squeeze(0)
                    loss = -A[b].detach() * lp.masked_select(mask).sum() / (cfg.rl.g * K)
                    scaler.scale(loss).backward()
                    total_loss += loss.detach().item()
                    n_pos += int(mask.sum())
                    del log_score, s, w, logpi, lp, loss
            if n_pos > 0:
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                if not (torch.isfinite(torch.tensor(gnorm))):
                    raise RuntimeError(f"non-finite grad norm at step {step}")
                scaler.step(optimizer)
                scaler.update()
                ema.update(model.parameters())
            else:
                gnorm = 0.0
            del trajs
            torch.cuda.empty_cache()

            # --- RAM / disk STOP 检查 ---
            m = meminfo()
            ram_hist.append(m)
            if check_ram_stop(ram_hist, log) or check_disk_stop(log):
                failure = "ram_or_disk_stop"
                break

            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                log(f"step {step:4d}: reward={rewards.mean().item():.4f}±{rewards.std().item():.4f} "
                    f"zvg={stats['zero_var_groups']}/{stats['n_groups']} "
                    f"A_std={adv_std:.4f} A_mean_abs={adv_mean_abs:.4f} "
                    f"grad_norm={gnorm:.2f} loss={total_loss:.4f} "
                    f"{step / el:.3f} steps/s "
                    f"J_support={support_frac:.3f} trunc={(1 - support_frac):.3f} "
                    f"peak_alloc={torch.cuda.max_memory_allocated() / 1e9:.2f}GB "
                    f"peak_resv={torch.cuda.max_memory_reserved() / 1e9:.2f}GB "
                    f"RAM_avail={m['avail_gb']:.1f}GiB swap={m['swap_used_gb']:.2f}GiB rss={rss_gb:.2f}GB")
                curve_rows.append([step, rewards.mean().item(), rewards.std().item(), gnorm,
                                   total_loss, adv_std, adv_mean_abs, support_frac,
                                   m["avail_gb"], m["swap_used_gb"]])
            if step % 25 == 0:
                snap = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            ema=dict(decay=ema.decay, num_updates=ema.num_updates,
                                     shadow_params=[s.detach().cpu() for s in ema.shadow_params]),
                            optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step)
                tmp = os.path.join(work_dir, "checkpoint_latest.pth.tmp")
                torch.save(snap, tmp)
                os.replace(tmp, os.path.join(work_dir, "checkpoint_latest.pth"))

            if step in (50, 100):
                log(f"step {step}: formal64 evaluation")
                do_eval(f"step{step}")
            n_steps_done = step
    except torch.cuda.OutOfMemoryError as e:
        failure = "oom"
        log(f"FAILURE OOM: {e}")
    except Exception as e:
        failure = str(e)[:200]
        log(f"FAILURE: {type(e).__name__}: {e}")

    # --- 收尾 ---
    max_diff = max((p.detach() - ip).abs().max().item()
                   for p, ip in zip(model.parameters(), init_params))
    meta = dict(K=K, steps_requested=args.steps, steps_done=n_steps_done,
                failure=failure, lr=cfg.rl.lr, G=cfg.rl.g, P=cfg.rl.prompts_per_step,
                rollout_steps=cfg.rl.rollout_steps, seed=cfg.seeds.model_seed,
                j_rng_schedule="rl.seed + step*10000 + chunk_offset",
                K_sampling="uniform without replacement from safe support (sigma>=0.05)",
                loss_norm="loss_j = -A*sum(lp)/(G*K), sequential backward, single optimizer.step",
                stats=stats, final_max_param_diff=float(max_diff),
                peak_alloc_gb=torch.cuda.max_memory_allocated() / 1e9,
                peak_resv_gb=torch.cuda.max_memory_reserved() / 1e9,
                ram_final=ram_hist[-1] if ram_hist else None,
                git_head=git_head,
                allocator_env=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"))
    json.dump(meta, open(os.path.join(work_dir, "run_metadata.json"), "w"), indent=2)
    with open(os.path.join(work_dir, "learning_curve.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["step", "reward_mean", "reward_std", "grad_norm", "loss",
                       "adv_std", "adv_mean_abs", "j_support_frac", "ram_avail_gb", "swap_used_gb"])
        wcsv.writerows(curve_rows)
    if failure is None and n_steps_done == args.steps:
        # final lightweight raw/EMA snapshot（compatibility eval 用）
        snap = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    ema=dict(decay=ema.decay, num_updates=ema.num_updates,
                             shadow_params=[s.detach().cpu() for s in ema.shadow_params]),
                    step=n_steps_done)
        tmp = os.path.join(work_dir, f"eval_snapshot_step{n_steps_done}.pth.tmp")
        torch.save(snap, tmp)
        os.replace(tmp, os.path.join(work_dir, f"eval_snapshot_step{n_steps_done}.pth"))
        # 删除 rolling checkpoint（成功后不留巨型临时文件）
        rp = os.path.join(work_dir, "checkpoint_latest.pth")
        if os.path.exists(rp):
            os.remove(rp)
        log(f"RUN_COMPLETE: {n_steps_done}/{args.steps} steps, "
            f"max_param_diff={max_diff:.2e}, failure={failure}")
    else:
        log(f"RUN_FAILED: done={n_steps_done}/{args.steps} failure={failure}")
        json.dump(dict(failure=failure, steps_done=n_steps_done),
                  open(os.path.join(work_dir, "failure.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
