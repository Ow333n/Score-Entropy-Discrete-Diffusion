"""RL-1 smoke 后审计 (用户 A 段指令, 纯 eval/offline, 不重训):

A1. step0 fixed eval —— 与 smoke 完全相同口径: RL init (EMA-10200) 上跑
    固定 8 prompts 的 sampled rollout reward + 64-chunk NLL
A2. step0 greedy —— NEW audit-only 代码 (现有 eval 无 RL reverse 路径的
    greedy): rollout_chunk 的 argmax 变体 (除 argmax vs gumbel 外逐行一致,
    同 seed → 同 J), 报告 greedy M0 reconstruction reward
A3. sigma diagnostics 审计:
    (a) 解析: 每 prompt 的 128-step σ 网格 (loglinear noise), σ<0.05 步数/比例,
        J 合法支撑 (=σ≥0.05 步数), 被截断 objective support
    (b) 实证: instrumented rollout (rollout_chunk 复制+记录, 算法逐行一致),
        fixed 8 prompts × 3 seed 组, 记录 soft-region 负权重的 timestep/σ 分布
A4. model drift: ||θ150−θ0||/||θ0|| (raw 与 EMA 两种), θ0=EMA-10200 init
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose

import graph_lib
import noise_lib
from model import utils as mutils
from model.ema import ExponentialMovingAverage
from rl import loader, reward as rw
from rl import transition as T
from catsample import sample_categorical

EPS = 1e-3
SIGMA_GATE_MIN = 0.05
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_pools(cfg, device):
    recs = [json.loads(l) for l in open(cfg.rl.manifest)][: cfg.rl.pool]
    x0_pool = torch.stack([torch.tensor(r["x0"]) for r in recs]).to(device)
    xt0_pool = torch.stack([torch.tensor(r["initial_state"]) for r in recs]).to(device)
    sig0_pool = torch.tensor([r["sigma"] for r in recs], device=device)
    m0_pool = torch.zeros(cfg.rl.pool, cfg.data.seq_len, dtype=torch.bool, device=device)
    for k, r in enumerate(recs):
        if r["initial_masked_positions"]:
            m0_pool[k, r["initial_masked_positions"]] = True
    return x0_pool, xt0_pool, sig0_pool, m0_pool


# --- A2/A3b: rollout_chunk 的 audit 变体 (逐行一致, 仅加记录 / 换 argmax) ---
def rollout_chunk_audit(model, sampling_score_fn, graph, noise, x0, x_t0, sigma0, steps,
                        seed, mask_token=50257, eps=1e-3, greedy=False, record_sigma=False):
    torch.manual_seed(seed)
    device = x0.device
    B = x0.shape[0]
    t0 = ((1 - (-sigma0).exp()) / (1 - eps)).clamp(min=eps + 1e-6)
    ts_mat = eps + (t0[:, None] - eps) * (1 - (torch.arange(1, steps + 1, device=device) / steps)[None, :])
    sig_mat = noise.total_noise(ts_mat)
    valid = sig_mat >= SIGMA_GATE_MIN
    J = torch.empty(B, dtype=torch.long, device=device)
    for b in range(B):
        v = valid[b].nonzero().squeeze(-1)
        J[b] = v[torch.randint(0, len(v), (1,), device=device)[0]]
    x = x_t0.clone()
    t_cur = t0.clone()
    saved = [None] * B
    gate_stats = dict(soft_neg_elements=0, soft_steps=0)
    sigma_records = []
    with torch.no_grad():
        for i in range(steps):
            t_next = eps + (t0 - eps) * (1 - (i + 1) / steps)
            sigma_cur = noise.total_noise(t_cur)
            sigma_next = noise.total_noise(t_next)
            dsigma = (sigma_cur - sigma_next)[:, None]
            s = sampling_score_fn(x, sigma_cur.squeeze(-1))
            stag = T.staggered_score_fn(s, dsigma)
            trans = graph.transp_transition(x, dsigma)
            w = stag * trans
            ok, stats = T.validity_check(w)
            if record_sigma:
                sigma_records.append(dict(
                    i=i, sigma_min=float(sigma_cur.min()), sigma_max=float(sigma_cur.max()),
                    n_soft=int((sigma_cur < SIGMA_GATE_MIN).sum()),
                    ok=bool(ok), n_neg=int(stats["n_neg"]),
                    n_nonfinite=int(stats["n_nonfinite"]), rel_neg_mass=float(stats["rel_neg_mass"])))
            if not ok:
                if (sigma_cur >= SIGMA_GATE_MIN).all():
                    sigma_vals = sigma_cur.squeeze(-1).tolist()
                    raise RuntimeError(
                        f"rollout validity gate 硬失败 step={i}: {stats} | "
                        f"σ range=[{min(sigma_vals):.4f},{max(sigma_vals):.4f}] "
                        f"dσ={float(dsigma[0, 0]):.4f}")
                else:
                    gate_stats["soft_neg_elements"] += stats["n_neg"]
                    gate_stats["soft_steps"] += 1
            if greedy:
                x_next = w.argmax(dim=-1)
            else:
                x_next = sample_categorical(w)
            hit = J == i
            if hit.any():
                logpi = T.policy_log_probs(w)
                lp = T.log_probs_for_actions(logpi, x_next)
                for b in range(B):
                    if hit[b]:
                        saved[b] = dict(
                            x_J=x[b].cpu(), x_next=x_next[b].cpu(), action=x_next[b].cpu(),
                            old_logpi=lp[b].cpu(), mask_J=(x[b] == mask_token).cpu(),
                            sigma_J=float(sigma_cur[b].item()),
                            dsigma_J=float(dsigma[b, 0].item()), J=int(J[b].item()))
            x = x_next
            t_cur = t_next
        s = sampling_score_fn(x, sigma_next.squeeze(-1))
        stag = T.staggered_score_fn(s, sigma_next[:, None])
        probs = stag * graph.transp_transition(x, sigma_next[:, None])
        probs = probs[..., :-1]
        if greedy:
            x = probs.argmax(dim=-1)
        else:
            x = sample_categorical(probs)
    for b in range(B):
        assert saved[b] is not None
        saved[b]["final_x"] = x[b].cpu()
    return saved, gate_stats, (sigma_records if record_sigma else None)


def fixed_eval8(model, sampling_score_fn, graph, noise, pools, eval_idx, cfg, greedy=False,
                record_sigma=False, seed_offsets=(0,)):
    """smoke fixed-eval 同口径: 8 prompts, chunks of 4, seed=rl.seed+424242+lo。"""
    x0_pool, xt0_pool, sig0_pool, m0_pool = pools
    rewards = []
    gate_stats = dict(soft_neg_elements=0, soft_steps=0)
    sigma_records = []
    for off in seed_offsets:
        for lo in range(0, 8, 4):
            trs, gst, srec = rollout_chunk_audit(
                model, sampling_score_fn, graph, noise,
                x0_pool[eval_idx[lo:lo + 4]], xt0_pool[eval_idx[lo:lo + 4]],
                sig0_pool[eval_idx[lo:lo + 4]], cfg.rl.rollout_steps,
                seed=cfg.rl.seed + 424242 + lo + off, mask_token=cfg.tokens,
                greedy=greedy, record_sigma=record_sigma)
            for b in range(4):
                r = rw.m0_reward(x0_pool[eval_idx[lo + b:lo + b + 1]],
                                 trs[b]["final_x"].unsqueeze(0).to(x0_pool.device),
                                 m0_pool[eval_idx[lo + b:lo + b + 1]])
                rewards.append(float(r))
            for k in gst:
                gate_stats[k] += gst[k]
            if srec:
                sigma_records += srec
    return rewards, gate_stats, sigma_records


def run_nll_eval(model, noise, graph, eval_ds, cfg, device, gen_seed):
    """training/rl.py run_eval 同口径 (64 块, span_task_loss)。"""
    from task_data.corruption import corrupt_span_batch
    from training.vanilla import span_task_loss
    g = torch.Generator(device=device).manual_seed(gen_seed)
    total, n = 0.0, 0
    with torch.no_grad():
        for bi in range(cfg.data.eval_chunks):
            x0 = eval_ds[bi]["input_ids"].to(device)[None]
            x_t, span_mask, sigma, dsigma, _ = corrupt_span_batch(
                x0, noise, cfg.data.span_min, cfg.data.span_max, graph.dim - 1, generator=g)
            total += span_task_loss(noise, graph, model, x0, x_t, span_mask,
                                    sigma, dsigma, train=False).item()
            n += 1
    return total / n


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    device = torch.device("cuda")
    torch.manual_seed(cfg.seeds.model_seed)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    D = cfg.tokens + 1

    model, info = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    model.eval()
    print(f"[A0] RL init loaded: step={info['step']} ema_num_updates={info['ema_num_updates']} "
          f"mismatches={info['shadow_vs_model_mismatches']}")
    sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
    pools = build_pools(cfg, device)
    x0_pool, xt0_pool, sig0_pool, m0_pool = pools
    eval_idx = torch.tensor(list(range(8)), device=device)

    # ---- A3a: 解析 sigma 网格 (loglinear noise, 逐 prompt) ----
    print("\n[A3a] 解析 sigma 网格 (128 步, σ(t)=-ln(1-0.999·t)):")
    n_soft_counts, first_soft = [], []
    for k in range(cfg.rl.pool):
        t0 = ((1 - (-sig0_pool[k]).exp()) / (1 - EPS)).clamp(min=EPS + 1e-6)
        ts = EPS + (t0 - EPS) * (1 - torch.arange(1, cfg.rl.rollout_steps + 1, device=device) / cfg.rl.rollout_steps)
        sig = noise.total_noise(ts)
        n_soft = int((sig < SIGMA_GATE_MIN).sum())
        n_soft_counts.append(n_soft)
        first_soft.append(int((sig < SIGMA_GATE_MIN).nonzero()[0]) if n_soft else 128)
    n_soft_counts = torch.tensor(n_soft_counts)
    first_soft = torch.tensor(first_soft)
    print(f"  pool σ0: min={sig0_pool.min():.4f} median={sig0_pool.median():.4f} max={sig0_pool.max():.4f}")
    print(f"  128 步中 σ<0.05 步数 (64 prompts): min={n_soft_counts.min()} median={n_soft_counts.median()} "
          f"mean={n_soft_counts.float().mean():.1f} max={n_soft_counts.max()}")
    print(f"  soft-region 比例 (mean): {n_soft_counts.float().mean() / cfg.rl.rollout_steps * 100:.1f}%")
    print(f"  首个 σ<0.05 步 index: min={first_soft.min()} median={first_soft.median()} max={first_soft.max()}")
    print(f"  → J 合法支撑 (=σ≥0.05 步数) mean: {cfg.rl.rollout_steps - n_soft_counts.float().mean():.1f}/128 "
          f"({100 * (1 - n_soft_counts.float().mean() / 128):.1f}%)")
    print(f"  → PG objective 被截断 support (σ<0.05 步不能当 J) mean: {n_soft_counts.float().mean():.1f}/128 "
          f"({100 * n_soft_counts.float().mean() / 128:.1f}%)")
    n8 = int(n_soft_counts[:8].sum())
    print(f"  fixed 8 子集: σ<0.05 步数合计 {n8} (of 8×128=1024, {100 * n8 / 1024:.1f}%)")

    # ---- A1: step0 fixed eval (sampled, smoke 完全同口径) ----
    print("\n[A1] step0 fixed eval (sampled rollout, smoke 同 seed 同代码):")
    rewards, gst, _ = fixed_eval8(model, sampling_score_fn, graph, noise, pools, eval_idx, cfg)
    print(f"  step0 fixed reward (8 prompts): mean={sum(rewards) / len(rewards):.4f} "
          f"per-prompt={[round(r, 4) for r in rewards]}")
    print(f"  gate stats: {gst}")

    # ---- A1: step0 NLL (run_eval 同口径; step0 raw=EMA, fresh EMA shadow = 当前权重) ----
    from data import get_dataset
    valid_ds = get_dataset("wikitext103", "validation", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)
    nll = run_nll_eval(model, noise, graph, valid_ds, cfg, device,
                       gen_seed=cfg.seeds.corruption_seed + 1000)
    print(f"  step0 NLL (64-chunk, raw=EMA @step0): {nll:.4f}")

    # ---- A2: step0 greedy (NEW audit-only, argmax 变体, 同 seed → 同 J) ----
    print("\n[A2] step0 greedy (argmax 变体, audit-only 新代码, 同 seed 同 J):")
    g_rewards, g_gst, _ = fixed_eval8(model, sampling_score_fn, graph, noise, pools, eval_idx, cfg,
                                      greedy=True)
    print(f"  step0 greedy M0 reward (8 prompts): mean={sum(g_rewards) / len(g_rewards):.4f} "
          f"per-prompt={[round(r, 4) for r in g_rewards]}")
    print(f"  sampled vs greedy (same J): sampled={sum(rewards) / len(rewards):.4f} "
          f"greedy={sum(g_rewards) / len(g_rewards):.4f}")

    # ---- A3b: instrumented rollout, fixed 8 × 3 seed 组, soft-region 负权重定位 ----
    print("\n[A3b] instrumented rollout (fixed 8 × 3 seed 组, 6 chunk-rollouts, 3072 prompt-steps):")
    _, _, srecs = fixed_eval8(model, sampling_score_fn, graph, noise, pools, eval_idx, cfg,
                              record_sigma=True, seed_offsets=(0, 10000, 20000))
    n_chunk_steps = len(srecs)
    n_prompt_steps = n_chunk_steps * 4
    n_soft_region = sum(r["n_soft"] for r in srecs)
    negs = [r for r in srecs if r["n_neg"] > 0]
    print(f"  total chunk-steps: {n_chunk_steps} (= {n_prompt_steps} prompt-steps, B=4/chunk)")
    print(f"  σ<0.05 region prompt-steps: {n_soft_region} ({100 * n_soft_region / n_prompt_steps:.1f}%)")
    print(f"  chunk-steps with negative elements: {len(negs)} "
          f"({100 * len(negs) / n_chunk_steps:.3f}%)")
    print(f"  total negative elements: {sum(r['n_neg'] for r in negs)}")
    for r in negs[:20]:
        print(f"    i={r['i']:3d} σ∈[{r['sigma_min']:.4f},{r['sigma_max']:.4f}] "
              f"n_neg={r['n_neg']} rel_neg_mass={r['rel_neg_mass']:.2e}")

    # ---- A4: drift ||θ150−θ0||/||θ0|| (raw & EMA) ----
    print("\n[A4] model drift (θ0 = EMA-10200 init):")
    snap = torch.load(os.path.join(cfg.work_dir, "rl1-smoke-022319", "eval_snapshot.pth"),
                      map_location="cpu", weights_only=False)
    # 用 loader 标准路径: 新建模型并加载 init EMA → θ0; 再灌 snapshot raw → θ150_raw
    m0_model, _ = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    theta0 = [p.detach().float() for p in m0_model.parameters()]
    m_raw, _ = loader.load_rl_init(cfg.rl.init_dir, cfg.rl.init_ckpt, device)
    m_raw.load_state_dict(snap["model"], strict=False)
    theta150_raw = [p.detach().float() for p in m_raw.parameters()]
    ema150 = ExponentialMovingAverage(m_raw.parameters(), decay=0.9999)
    ema150.load_state_dict(snap["ema"])
    theta150_ema = [s.detach().float().to(device) for s in ema150.shadow_params]

    def rel_drift(a, b):
        num, den = 0.0, 0.0
        for pa, pb in zip(a, b):
            num += ((pa - pb) ** 2).sum().item()
            den += (pb ** 2).sum().item()
        return (num ** 0.5) / (den ** 0.5)

    print(f"  ||θ150_raw − θ0|| / ||θ0|| = {rel_drift(theta150_raw, theta0):.6f}")
    print(f"  ||θ150_ema − θ0|| / ||θ0|| = {rel_drift(theta150_ema, theta0):.6f}")
    print("\nAUDIT_DONE")


if __name__ == "__main__":
    main()
