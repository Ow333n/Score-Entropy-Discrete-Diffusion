"""Phase 2 block decoder（frozen Phase 2 协议 §2/§3/§4，2026-10-07）。

- block size {1,2,4,8}；block=1 = sequential reference
- 位置选择：confidence-first（用**同一轮** forward 的 argmax 概率排序，零额外 forward）/
  random（seed 0/1/2，per-round randperm）/ fixed（l2r 块序）
- block 内独立并行更新（同轮内各 token 条件于同一上下文 C，互不见对方的揭示）
- 每轮重新 forward（全部 mask 位置同批打分）
- greedy 为主（deterministic）；sampled 为 secondary（temperature=1.0，frozen seed）
- stopping：span 内全部揭示
- **NFE accounting（协议 §3）**：NFE_total = NFE_predict + NFE_select 两栏分列；
  每轮 1 次 forward 计入 NFE_predict；选择复用同轮分数 → NFE_select=0（运行时断言）；
  任何 lookahead 型独立选择 forward 必须逐次计入 NFE_select（本实现无 lookahead 模式，
  由单测验证计数函数）
- endpoints：token recovery accuracy、exact-span recovery（primary）；
  pseudo-NLL、reveal 记录（path divergence / sequential disagreement 在分析层计算）

用法: 作为库模块 import；smoke 见 scripts/phase2_preflight_smoke.py
"""
import torch
from compatibility.posterior import clean_log_probs
from evaluation.eval_diag_fp32 import fp32_forward

D = 50258


def block_decode(model, x_t, x0, sigma, span, block_size, select="confidence",
                 mode="greedy", seed=0, temperature=1.0, nfe=None):
    """对单样本执行 block decoding。

    Args:
        model: fp32 镜像 forward 兼容模型（eval 模式）
        x_t: [L] 初始状态（span 内部分位置为 MASK）
        x0: [L] ground truth；span: (start, end)
        block_size: {1,2,4,8}
        select: "confidence" | "random" | "fixed"
        mode: "greedy" | "sampled"（sampled: temperature=1.0、seed 冻结）
        nfe: dict 计数器（NFE accounting 注入；缺省新建）

    Returns dict: endpoints + per-round 记录 + nfe 三栏。
    """
    if nfe is None:
        nfe = dict(predict=0, select=0)
    assert block_size in (1, 2, 4, 8)
    assert select in ("confidence", "random", "fixed")
    assert mode in ("greedy", "sampled")

    state = x_t.clone()
    s0, s1 = span
    all_masked = [p for p in (state == D - 1).nonzero().squeeze(1).tolist()
                  if s0 <= p < s1]
    masked = list(all_masked)                     # 待揭示集合（initially masked）
    rng = torch.Generator(device=state.device).manual_seed(seed)

    rounds = []
    total_logp = 0.0
    while masked:
        nfe["predict"] += 1                          # 本轮 forward
        logp = clean_log_probs(
            fp32_forward(model, state[None], torch.tensor([sigma], device=state.device)),
            D)[0].double()                           # [L, D-1] fp32 提取 → fp64
        # 位置选择（复用本轮 logp → NFE_select += 0，断言）
        if select == "confidence":
            conf = logp[masked].max(-1).values
            order = torch.argsort(conf, descending=True)
            block = [masked[i] for i in order[:block_size].tolist()]
        elif select == "random":
            perm = torch.randperm(len(masked), generator=rng)
            block = [masked[i] for i in perm[:block_size].tolist()]
        else:  # fixed：l2r 块序
            block = masked[:block_size]
        assert nfe["select"] == 0, "本实现的选择路径零额外 forward；任何 lookahead 必须计入 NFE_select"
        # block 内独立更新（同上下文 C）
        placed = []
        for pos in block:
            if mode == "greedy":
                tok = int(logp[pos].argmax().item())
            else:
                tok = int(torch.multinomial(
                    (logp[pos] / temperature).exp(), 1, generator=rng).item())
            total_logp += float(logp[pos, tok].item())
            state[pos] = tok
            placed.append((pos, tok))
        rounds.append(dict(round=len(rounds) + 1, positions=block,
                           placed=placed, n_remaining=len(masked) - len(block)))
        masked = [p for p in masked if p not in block]

    span_pos = list(range(s0, s1))
    # primary 口径：揭示位置（initially masked）的 GT 命中（可见 token 的平凡命中不混入）
    rev_hits = (state[all_masked] == x0[all_masked]).sum().item()
    whole_hits = (state[span_pos] == x0[span_pos]).sum().item()
    nfe["total"] = nfe["predict"] + nfe["select"]
    return dict(
        block_size=block_size, select=select, mode=mode, seed=seed,
        token_recovery=rev_hits / len(all_masked) if all_masked else float("nan"),
        exact_revealed=float(rev_hits == len(all_masked)) if all_masked else float("nan"),
        whole_span_recovery=whole_hits / len(span_pos),
        exact_span=float(whole_hits == len(span_pos)),
        n_revealed=len(all_masked),
        pseudo_nll=total_logp,
        n_rounds=len(rounds),
        nfe=dict(nfe),
        final_state=state.clone(),
        rounds=rounds,
    )
