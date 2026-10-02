"""严格逐 token 揭示 + Teacher-Forced path likelihood (protocol §4.8 / §6.1)。

与 sampling.py 的 pc_sampler 不同: 后者是 rate-matrix 跳变, 一步可揭示多个 token。
这里每步只选中 1 个 mask 位置并揭示 1 个 token —— 用于隔离顺序效应
(protocol §18: "begin with strict one-token-at-a-time reveal")。

两种模式 (protocol §4.8 都要求验证):
  - teacher_forced: 策略选位置, 揭示 gold token。路径概率
        Q_pi(x) = sum_{k=1..m} log p_theta(x_{i_k} | C, x_{i_1..i_{k-1}})   (§3.7/§6.1)
    只对初始 mask 集合 M_0 求和, 初始可见的 target 位置不参与。
  - sampled: 策略选位置, 从 p_hat 采样/贪心 token (protocol §3.9: greedy, T=1.0 主结果;
    T=0.5/1.5 仅 sensitivity)。

顺序策略 (§6.3): l2r / r2l / fixed random / confidence-first (+ reverse-confidence)。
confidence tie-breaking (§3.8): p_max 完全相同时选 sequence index 最小者, 记录 tie 数。
fixed path: 从 manifest 传入的预生成路径 [steps, B], 所有 checkpoint 共享 (§3.10)。

不变式断言 (§4.8): 每步 mask 数恰好 -1、只有选中位置变化、选中前该位置是 MASK。
违反即实现 bug, 直接抛错, 不静默继续。
"""
import torch

from compatibility.posterior import clean_log_probs

ORDERS = ("l2r", "r2l", "random", "confidence", "reverse-confidence")


def _pick_positions(logp, mask, order, generator=None):
    """在当前 mask 集合上按顺序策略为每个序列选一个位置。

    Returns (pos [B], n_ties [B]):
        pos: 选中位置, 无 mask 的序列为 -1
        n_ties: confidence 类策略的并列数 (p_max 相同的 mask 位置数 - 1), 其他策略为 0
    """
    B, L = mask.shape
    n_mask = mask.sum(1)

    if order == "l2r":
        pos = torch.where(mask, torch.arange(L, device=mask.device)[None].expand(B, -1),
                          torch.full_like(mask, L, dtype=torch.long)).min(1).values
        ties = torch.zeros(B, dtype=torch.long)
    elif order == "r2l":
        pos = torch.where(mask, torch.arange(L, device=mask.device)[None].expand(B, -1),
                          torch.full_like(mask, -1, dtype=torch.long)).max(1).values
        ties = torch.zeros(B, dtype=torch.long)
    elif order == "random":
        probs = mask.float() / n_mask.clamp(min=1)[:, None]
        # 无 mask 的序列给一个 dummy 概率, 避免 multinomial 全零报错 (后面统一置 -1)
        probs[n_mask == 0, -1] = 1.0
        pos = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        ties = torch.zeros(B, dtype=torch.long)
    elif order in ("confidence", "reverse-confidence"):
        p_max = logp.max(-1).values                                    # [B, L]
        fill = float("inf") if order == "reverse-confidence" else float("-inf")
        p_max = p_max.masked_fill(~mask, fill)
        pos = p_max.argmin(1) if order == "reverse-confidence" else p_max.argmax(1)
        # §3.8 tie counting: 与选中位置 p_max 相同值的 mask 位置数 - 1
        top = p_max.gather(1, pos[:, None]).squeeze(1)                 # [B]
        ties = ((p_max == top[:, None]) & mask).sum(1) - 1
        ties = ties.clamp(min=0)
    else:
        raise ValueError(f"未知顺序策略: {order}, 可选 {ORDERS}")

    pos = torch.where(n_mask > 0, pos, torch.full_like(pos, -1))
    ties = torch.where(n_mask > 0, ties, torch.zeros_like(ties))
    return pos, ties


def _apply_temperature(lp, temperature):
    """T!=1 时对干净词表 log-probs 做温度缩放 (protocol §3.9 sensitivity)。"""
    if temperature == 1.0:
        return lp
    lp_t = lp / temperature
    return lp_t - torch.logsumexp(lp_t, dim=-1, keepdim=True)


def strict_reveal(score_fn, x, gold, sigma, D, order="random", mode="sampled",
                  sample="greedy", temperature=1.0, fixed_path=None, generator=None):
    """严格一次揭示一个 token 的解码循环。

    Args:
        score_fn: score_fn(x_t, sigma) -> [B, L, D]
        x: [B, L] 初始状态 (含 MASK 位置)
        gold: [B, L] 干净序列 (mode="teacher_forced" 必需; sampled 模式可传 None)
        sigma: 标量, 每一步的评估噪声 (RADD 下 p_hat 与 sigma 无关是模型性质,
               固定 sigma 去除时间调度对顺序比较的混淆)
        order: l2r | r2l | random | confidence | reverse-confidence
        mode: "teacher_forced" (揭示 gold) | "sampled" (采样/贪心 token)
        sample: "greedy" | "categorical" (sampled 模式)
        temperature: 采样温度 (greedy 不受影响)
        fixed_path: [steps, B] 预生成路径 (manifest §3.10), 给定后忽略 order

    Returns (x, info):
        x: [B, L] 完全揭示后的状态 (TF 模式下 = 初始可见 + gold 揭示)
        info: steps, pos_history, tok_history, logp_history, n_mask_history,
              path_logp (Q_pi, TF 模式), n_forwards, tie_count, tie_fraction,
              invariant_ok=True (违反不变式会直接抛 RuntimeError)
    """
    assert mode in ("teacher_forced", "sampled")
    if mode == "teacher_forced":
        assert gold is not None, "teacher_forced 模式必须提供 gold"

    x = x.clone()
    B, L = x.shape
    device = x.device
    mask = x == D - 1
    n_mask = mask.sum(1)
    total_masked = n_mask.sum().item()
    sigma_b = sigma * torch.ones(B, device=device)

    # fixed path 长度校验: 必须等于初始 mask 数
    if fixed_path is not None:
        assert fixed_path.shape[0] == total_masked and fixed_path.shape[1] == B, \
            f"fixed_path 形状 {fixed_path.shape} != ({total_masked}, {B})"

    pos_history, tok_history, logp_history = [], [], []
    n_mask_history = [n_mask.clone().cpu()]
    tie_total = 0
    active_steps = 0

    with torch.no_grad():
        for step in range(total_masked):
            logits = score_fn(x, sigma_b)
            logp = clean_log_probs(logits, D)
            logp = _apply_temperature(logp, temperature)

            if fixed_path is not None:
                pos = fixed_path[step].to(device)
                ties = torch.zeros(B, dtype=torch.long)
            else:
                pos, ties = _pick_positions(logp, mask, order, generator)
            tie_total += ties.sum().item()
            active_steps += (pos >= 0).sum().item()

            # 选择 token: TF 揭示 gold, sampled 按 p_hat
            tok = torch.full((B,), -1, dtype=torch.long, device=device)
            logp_tok = torch.zeros(B, device=device, dtype=logp.dtype)
            for b in range(B):
                if pos[b] < 0:
                    continue
                if mode == "teacher_forced":
                    t = gold[b, pos[b]]
                elif sample == "greedy":
                    t = logp[b, pos[b]].argmax()
                elif sample == "categorical":
                    t = torch.multinomial(logp[b, pos[b]].exp(), 1, generator=generator).squeeze(0)
                else:
                    raise ValueError(f"未知采样方式: {sample}")
                tok[b] = t
                logp_tok[b] = logp[b, pos[b], t]

            # §4.8 不变式: ① 选中前是 MASK ② 只有选中位置变化 ③ mask 数恰好 -1
            prev = x.clone()
            active = pos >= 0
            assert (x[active, pos[active]] == D - 1).all(), "invariant violation: 选中位置不是 MASK"
            x[active, pos[active]] = tok[active]
            changed = (x != prev)
            assert (changed.sum(1) <= 1).all(), "invariant violation: 一步改了多个位置"
            new_mask = x == D - 1
            assert ((n_mask - new_mask.sum(1))[active] == 1).all(), "invariant violation: mask 数不是 -1"

            mask = new_mask
            n_mask = mask.sum(1)
            pos_history.append(pos.cpu())
            tok_history.append(tok.cpu())
            logp_history.append(logp_tok.cpu())
            n_mask_history.append(n_mask.cpu())

    pos_hist = torch.stack(pos_history)
    tok_hist = torch.stack(tok_history)
    logp_hist = torch.stack(logp_history)
    path_logp = logp_hist.sum(0)   # TF 模式 = Q_pi(x) (§6.1)

    return x, dict(
        steps=total_masked,
        pos_history=pos_hist,
        tok_history=tok_hist,
        logp_history=logp_hist,
        n_mask_history=torch.stack(n_mask_history),
        path_logp=path_logp,
        n_forwards=total_masked,
        tie_count=tie_total,
        tie_fraction=tie_total / max(active_steps, 1),
    )


def evaluate_path_Qs(score_fn, samples, D, chunk=16):
    """TF 路径概率 Q_pi 的批量评估 (§6.1)。

    Args:
        samples: list of dict(x_t [L], x0 [L], sigma float, path [m] 揭示位置序列)
        chunk: 每次 forward 的最大样本数 (不同样本的 path 长度不同, ragged 处理)

    Returns Q: [N] 每样本的路径 log 概率和 (不改变输入状态)。
    """
    N = len(samples)
    device = samples[0]["x_t"].device

    states = [s["x_t"].clone() for s in samples]
    gold = [s["x0"] for s in samples]
    paths = [s["path"] for s in samples]
    next_idx = [0] * N
    Q = torch.zeros(N, device=device)
    remaining = list(range(N))

    with torch.no_grad():
        while remaining:
            chunk_ids = remaining[:chunk]
            c = len(chunk_ids)

            xs = torch.stack([states[k] for k in chunk_ids])          # [c, L]
            pos = torch.tensor([paths[k][next_idx[k]] for k in chunk_ids], device=device)
            sigma_b = torch.tensor([samples[k]["sigma"] for k in chunk_ids],
                                   device=device, dtype=torch.float32)

            logp = clean_log_probs(score_fn(xs, sigma_b), D)          # [c, L, D-1]
            idx = torch.arange(c, device=device)
            gtok = torch.tensor([gold[k][paths[k][next_idx[k]]] for k in chunk_ids], device=device)
            Q[chunk_ids] += logp[idx, pos, gtok]

            # 更新状态并推进
            xs[idx, pos] = gtok
            for local, k in enumerate(chunk_ids):
                states[k] = xs[local].clone()
                next_idx[k] += 1

            remaining = [k for k in remaining if next_idx[k] < len(paths[k])]

    return Q
