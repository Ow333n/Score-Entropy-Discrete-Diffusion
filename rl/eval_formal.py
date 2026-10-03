"""Formal fixed task eval（protocol v1.0 §35，eval-only 代码，不改 training objective/sampling）。

- 64-sample 独立子集：frozen manifest `regime_a_eval_v1.jsonl` 索引 **64–127**（training pool = 0–63）
- 每评估点完全复用：相同 samples / corruption / decoding protocol
- sampled + greedy（argmax 变体，与 rollout_chunk 逐行同源、同 seed 同 J）× RAW + EMA
- NLL：raw/ema 共用冻结 corruption realization（corruption_seed+1000 流，独立 generator 对象配对）
- seed schedule（固定）：chunk seed = cfg.rl.seed + 424242 + chunk 起点索引（64, 68, …, 124），
  chunk size 4 → 16 chunks；与历史 eval8（索引 0–7，seed 424242+0/4）不冲突
- eval 不改 model weights/state：ema store/copy/restore 平衡；rollout 无 autograd；
  score fn train=False 内部置 model.eval() → 调用方在 eval 后必须自行 model.train()
"""
import json
import os

import torch

from rl import transition as T
from rl import reward as rw
from catsample import sample_categorical

FORMAL_EVAL_IDX = list(range(64, 128))   # 64 samples, 与 training pool 0–63 零 overlap
SIGMA_GATE_MIN = 0.05
EPS = 1e-3


def build_eval_pools(cfg, device, start=64, n=64):
    """manifest 索引 [start, start+n) 的 eval pools（x0/xt0/sig0/m0 + 行 sha256）。"""
    lines = open(cfg.rl.manifest).readlines()
    recs = [json.loads(l) for l in lines[start:start + n]]
    assert len(recs) == n, f"manifest 只有 {len(lines)} 行, 取不到 {start}:{start + n}"
    ids = []
    for l, r in zip(lines[start:start + n], recs):
        import hashlib
        ids.append(dict(index=r.get("index", start + len(ids)), sha256=hashlib.sha256(l.encode()).hexdigest()))
    x0 = torch.stack([torch.tensor(r["x0"]) for r in recs]).to(device)
    xt0 = torch.stack([torch.tensor(r["initial_state"]) for r in recs]).to(device)
    sig0 = torch.tensor([r["sigma"] for r in recs], device=device)
    m0 = torch.zeros(n, cfg.data.seq_len, dtype=torch.bool, device=device)
    for k, r in enumerate(recs):
        if r["initial_masked_positions"]:
            m0[k, r["initial_masked_positions"]] = True
    return x0, xt0, sig0, m0, ids


def greedy_rollout_chunk(model, sampling_score_fn, graph, noise, x0, x_t0, sigma0, steps,
                         seed, mask_token=50257, eps=1e-3):
    """rollout_chunk 的 argmax 变体（v1.0 §35 greedy supplementary metric）。

    与 rl/rollout.py rollout_chunk 逐行同源，唯一差异：sample_categorical → argmax。
    同 seed → 同 J；greedy 输出无采样噪声、deterministic。
    """
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
            x_next = w.argmax(dim=-1)                     # greedy（唯一与 rollout_chunk 不同的行）
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
        x = probs.argmax(dim=-1)                          # greedy denoiser
    for b in range(B):
        assert saved[b] is not None
        saved[b]["final_x"] = x[b].cpu()
    return saved, gate_stats


def _reward_pass(model, sampling_score_fn, graph, noise, pools, cfg, greedy):
    """64 samples, chunks of 4, seed = rl.seed + 424242 + chunk 起点。"""
    from rl.rollout import rollout_chunk
    x0, xt0, sig0, m0 = pools
    n = x0.shape[0]
    per = []
    gst = dict(soft_neg_elements=0, soft_steps=0)
    for lo in range(0, n, 4):
        chunk_fn = greedy_rollout_chunk if greedy else rollout_chunk
        trs, g = chunk_fn(model, sampling_score_fn, graph, noise,
                          x0[lo:lo + 4], xt0[lo:lo + 4], sig0[lo:lo + 4],
                          cfg.rl.rollout_steps, seed=cfg.rl.seed + 424242 + 64 + lo,
                          mask_token=cfg.tokens)
        for b in range(min(4, n - lo)):
            r = rw.m0_reward(x0[lo + b:lo + b + 1],
                             trs[b]["final_x"].unsqueeze(0).to(x0.device),
                             m0[lo + b:lo + b + 1])
            per.append(float(r))
        for k in g:
            gst[k] += g[k]
    return per, gst


def nll_paired(model, ema, noise, graph, eval_ds, cfg, device):
    """64-chunk NLL: RAW + EMA，共用冻结 corruption realization（seed+1000 流，配对）。

    与 smoke/probe 的 run_eval 逐位同口径（EMA 用 seed+1000 首次使用）。
    """
    from task_data.corruption import corrupt_span_batch
    from training.vanilla import span_task_loss
    g_ema = torch.Generator(device=device).manual_seed(cfg.seeds.corruption_seed + 1000)
    g_raw = torch.Generator(device=device).manual_seed(cfg.seeds.corruption_seed + 1000)

    def nll_pass(gen):
        total, n = 0.0, 0
        with torch.no_grad():
            for bi in range(cfg.data.eval_chunks):
                x0 = eval_ds[bi]["input_ids"].to(device)[None]
                x_t, span_mask, sigma, dsigma, _ = corrupt_span_batch(
                    x0, noise, cfg.data.span_min, cfg.data.span_max, graph.dim - 1, generator=gen)
                total += span_task_loss(noise, graph, model, x0, x_t, span_mask,
                                        sigma, dsigma, train=False).item()
                n += 1
        return total / n

    raw_nll = nll_pass(g_raw)
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    ema_nll = nll_pass(g_ema)
    ema.restore(model.parameters())
    return raw_nll, ema_nll


def task_eval_point(model, ema, sampling_score_fn, graph, noise, valid_ds, pools,
                    eval_ids, cfg, device, tag):
    """一个评估点的完整 formal task eval（§35 全部指标）。

    RAW = PRIMARY, EMA = SECONDARY；sampled + greedy；NLL 冻结 corruption realization。
    结束状态：model 持有 RAW 权重；model.training 可能被 score fn 置为 eval →
    调用方必须自行 model.train()。
    """
    torch.cuda.empty_cache()
    res = dict(tag=tag, n_samples=len(eval_ids), eval_indices=list(range(64, 128)),
               eval_ids=eval_ids)

    # RAW（primary）
    per_s, gst_s = _reward_pass(model, sampling_score_fn, graph, noise, pools, cfg, greedy=False)
    per_g, gst_g = _reward_pass(model, sampling_score_fn, graph, noise, pools, cfg, greedy=True)
    res.update(sampled_raw_mean=float(sum(per_s) / len(per_s)), sampled_raw_per=per_s,
               greedy_raw_mean=float(sum(per_g) / len(per_g)), greedy_raw_per=per_g,
               gate_sampled_raw=gst_s, gate_greedy_raw=gst_g)

    # EMA（secondary）
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    per_s_e, gst_s_e = _reward_pass(model, sampling_score_fn, graph, noise, pools, cfg, greedy=False)
    per_g_e, gst_g_e = _reward_pass(model, sampling_score_fn, graph, noise, pools, cfg, greedy=True)
    ema.restore(model.parameters())
    res.update(sampled_ema_mean=float(sum(per_s_e) / len(per_s_e)), sampled_ema_per=per_s_e,
               greedy_ema_mean=float(sum(per_g_e) / len(per_g_e)), greedy_ema_per=per_g_e,
               gate_sampled_ema=gst_s_e, gate_greedy_ema=gst_g_e)

    # NLL（raw + ema，冻结 corruption realization）
    nll_raw, nll_ema = nll_paired(model, ema, noise, graph, valid_ds, cfg, device)
    res.update(nll_raw=nll_raw, nll_ema=nll_ema,
               corruption_realization=f"corruption_seed+1000 ({cfg.seeds.corruption_seed + 1000}, frozen)")
    return res


def save_eval_point(run_dir, res):
    path = os.path.join(run_dir, f"eval_point_{res['tag']}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    return path
