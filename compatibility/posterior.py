"""归一化条件后验提取 (protocol §4.2, RADD 桥)。

absorbing 扩散的 concrete score 对 mask 位置满足
    s_{i,v}(t) = log r(t) + log p_hat(v | C) + const
其中 r(t) = 1/(e^{sigma}-1) 是只依赖时间的标量 (由 graph_lib.Absorbing.score_entropy
的最优解 exp(score[v]) = ratio * p_v 推出)。对干净词表做 softmax 后所有加性常数
(时间标量、词表常数) 全部消掉:
    p_hat(v | C) = softmax_{v in clean vocab}(score[i, :D-1])

G0 验证要求 (§4.2): ① normalization ② MASK exclusion ③ time scalar cancellation
④ repeated deterministic evaluation consistency。
"""
import torch
import torch.nn.functional as F


def clean_log_probs(score, D):
    """score: [..., D] log-score → [..., D-1] log p_hat(v | C), 只对干净词表 softmax。

    排除 MASK 条目是必须的: 模型 forward 里 scatter 把 x_t 自身 token 的 score 置 0,
    mask 位置处 score[..., D-1] = 0 而不是有效 logit; 若把它纳入 softmax 会污染 p_hat。
    非 mask 位置的输出无意义, 调用方必须用 x_t == D-1 掩掉。
    """
    return F.log_softmax(score[..., :D - 1], dim=-1)


def check_deterministic(score_fn, x_t, sigma):
    """G0 §4.2-④: 同输入重复评估必须逐位一致 (eval 模式下的确定性)。"""
    with torch.no_grad():
        a = score_fn(x_t, sigma)
        b = score_fn(x_t, sigma)
    return (a == b).all().item(), (a - b).abs().max().item()
