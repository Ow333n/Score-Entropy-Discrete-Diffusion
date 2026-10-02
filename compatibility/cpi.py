"""Swap defect δ 与 Data CPI 评估 (protocol §4.4 / §5)。

数学定义:
  状态 C = 当前可见上下文, i,j 为两个 [MASK] 位置, a,b 为 ground-truth token。
  delta_swap = log p(a|C) + log p(b|C,a) - log p(b|C) - log p(a|C,b)
  CPI_abs = E|delta|, CPI_RMS = sqrt(E delta^2)

协议约束:
  - quartet (C, C+a, C+b) 共享完全相同的 sigma (§4.3)
  - 3 次 forward, 全部 no_grad (评估路径)
  - 大张量用完即 del: 3 个状态 logits+logp 同活会把 8GB 顶爆 (smoke 实测 8.13GB 峰值)
"""
import math

import torch
from compatibility.posterior import clean_log_probs


def delta_swap(score_fn, x_t, i, j, a, b, sigma, D):
    """3 次 forward 计算 swap 残差 (单样本或 batch)。

    Args:
        score_fn: score_fn(x_t, sigma) -> [B, L, D]
        x_t: [B, L], i,j 位置必须是 MASK
        i, j: [B] 位置索引; a, b: [B] ground-truth token; sigma: [B] 或标量
    Returns dict: delta, log_p_a_C, log_p_b_C, log_p_b_Ca, log_p_a_Cb, n_forwards=3
    """
    with torch.no_grad():
        s_C = score_fn(x_t, sigma)

        x_Ca = x_t.clone()
        x_Ca[torch.arange(x_t.shape[0]), i] = a
        s_Ca = score_fn(x_Ca, sigma)

        x_Cb = x_t.clone()
        x_Cb[torch.arange(x_t.shape[0]), j] = b
        s_Cb = score_fn(x_Cb, sigma)

    B = x_t.shape[0]
    idx = torch.arange(B)

    logp_C = clean_log_probs(s_C, D)
    logp_Ca = clean_log_probs(s_Ca, D)
    logp_Cb = clean_log_probs(s_Cb, D)

    log_p_a_C = logp_C[idx, i, a]
    log_p_b_C = logp_C[idx, j, b]
    log_p_b_Ca = logp_Ca[idx, j, b]
    log_p_a_Cb = logp_Cb[idx, i, a]

    delta = log_p_a_C + log_p_b_Ca - log_p_b_C - log_p_a_Cb

    return dict(
        delta=delta,
        log_p_a_C=log_p_a_C,
        log_p_b_C=log_p_b_C,
        log_p_b_Ca=log_p_b_Ca,
        log_p_a_Cb=log_p_a_Cb,
        n_forwards=3,
    )


def evaluate_delta_swap_batch(score_fn, samples, D, chunk=8):
    """对一组样本批量计算 delta_swap + local CE + token accuracy。

    Args:
        samples: list of dict, 每项含
            x_t: [L] 当前状态 (i,j 位置为 MASK)
            i, j: int 位置索引
            a, b: int ground-truth token
            x0: [L] 干净序列 (local CE / token acc 用)
            sigma: float (quartet 共享, §4.3)
        chunk: 每次 forward 的样本数 (8GB 安全值)

    Returns dict:
        delta:    [N] delta_swap
        local_ce: [N] 状态 C 上所有 mask 位置的 -log p_hat(x0[pos]|C) 均值
        token_acc: [N] 状态 C 上 mask 位置 argmax p_hat == gold 的比例
        n_masked: [N]
        sigma:    [N]
    """
    N = len(samples)
    deltas, ces, accs, n_masked, sigmas = [], [], [], [], []

    with torch.no_grad():
        for start in range(0, N, chunk):
            chunk_samples = samples[start:start + chunk]
            c = len(chunk_samples)
            device = chunk_samples[0]["x_t"].device

            x_t = torch.stack([s["x_t"] for s in chunk_samples]).to(device)     # [c, L]
            x_Ca = x_t.clone()
            x_Cb = x_t.clone()
            i = torch.tensor([s["i"] for s in chunk_samples], device=device)
            j = torch.tensor([s["j"] for s in chunk_samples], device=device)
            a = torch.tensor([s["a"] for s in chunk_samples], device=device)
            b = torch.tensor([s["b"] for s in chunk_samples], device=device)
            sigma_b = torch.tensor([s["sigma"] for s in chunk_samples],
                                   device=device, dtype=torch.float32)

            x_Ca[torch.arange(c), i] = a
            x_Cb[torch.arange(c), j] = b

            idx = torch.arange(c)
            mask = x_t == D - 1

            # 显存纪律: 每个大张量 (logits/logp ≈ 0.8GB at chunk=8) 用完立刻 del。
            s_C = score_fn(x_t, sigma_b)
            logp_C = clean_log_probs(s_C, D)
            pa_C = logp_C[idx, i, a]
            pb_C = logp_C[idx, j, b]
            for k in range(c):
                pos = mask[k].nonzero().squeeze(1)
                if pos.numel() == 0:
                    ces.append(torch.tensor(float("nan")))
                    accs.append(torch.tensor(float("nan")))
                else:
                    gold = chunk_samples[k]["x0"][pos].to(device)
                    ce = -logp_C[k, pos, gold].mean()
                    acc = (logp_C[k, pos].argmax(-1) == gold).float().mean()
                    ces.append(ce.cpu())
                    accs.append(acc.cpu())
                n_masked.append(pos.numel())
            del s_C, logp_C

            s_Ca = score_fn(x_Ca, sigma_b)
            logp_Ca = clean_log_probs(s_Ca, D)
            pb_Ca = logp_Ca[idx, j, b]
            del s_Ca, logp_Ca

            s_Cb = score_fn(x_Cb, sigma_b)
            logp_Cb = clean_log_probs(s_Cb, D)
            pa_Cb = logp_Cb[idx, i, a]
            del s_Cb, logp_Cb

            delta = pa_C + pb_Ca - pb_C - pa_Cb
            deltas.append(delta.cpu())

            sigmas.extend([s["sigma"] for s in chunk_samples])

    return dict(
        delta=torch.cat(deltas),
        local_ce=torch.stack(ces),
        token_acc=torch.stack(accs),
        n_masked=torch.tensor(n_masked),
        sigma=torch.tensor(sigmas),
    )


def summarize(results, name=""):
    """protocol §5.4 的报告指标: CPI_abs, CPI_RMS, quantiles, signed mean, local CE, token acc。

    不确定度口径: SEM = per-sample SD / sqrt(N)。paired 对比 (同一 manifest、不同
    checkpoint) 用逐样本差做 paired bootstrap, 不比较两个独立均值 (protocol §3.17)。
    """
    delta = results["delta"]
    ce = results["local_ce"][torch.isfinite(results["local_ce"])]
    acc = results["token_acc"][torch.isfinite(results["token_acc"])]
    N = delta.numel()
    ad = delta.abs()

    def sem(v):
        return v.std(unbiased=True).item() / math.sqrt(v.numel())

    def quantile(v, q):
        return torch.quantile(v.float(), q).item()

    return dict(
        name=name,
        n_samples=N,
        cpi_abs=ad.mean().item(),
        cpi_abs_sem=sem(ad),
        cpi_rms=(delta ** 2).mean().sqrt().item(),
        delta_mean=delta.mean().item(),
        delta_sem=sem(delta),
        delta_median=delta.median().item(),
        delta_abs_p90=quantile(ad, 0.90),
        delta_abs_p99=quantile(ad, 0.99),
        delta_sd=delta.std(unbiased=True).item(),
        local_ce=ce.mean().item(),
        local_ce_sem=sem(ce),
        token_acc=acc.mean().item(),
        token_acc_sem=sem(acc),
    )


def summarize_buckets(results, n_buckets=4, by="sigma"):
    """按冻结分桶变量汇总 CPI (protocol §5.4: sigma / mask ratio / span len / pair distance ...)"""
    vals = results[by]
    order = vals.argsort()
    delta = results["delta"]
    edges = torch.linspace(0, vals.numel(), n_buckets + 1).long()
    rows = []
    for k in range(n_buckets):
        sel = order[edges[k]:edges[k + 1]]
        if sel.numel() == 0:
            continue
        rows.append(dict(
            bucket=k,
            n=sel.numel(),
            lo=vals[sel].min().item(),
            hi=vals[sel].max().item(),
            cpi_abs=delta[sel].abs().mean().item(),
            delta_mean=delta[sel].mean().item(),
        ))
    return rows
