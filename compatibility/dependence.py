"""Phase 2 conditional-dependence proxies（frozen Phase 2 协议 §1C 定义，2026-10-07）。

三个预注册指标（全部来自同一个 3-forward delta-swap quartet，零额外 forward）：
- delta（compatibility / curl，frozen 定义）：δ = p̂(a|C) + p̂(b|C,a) − p̂(b|C) − p̂(a|C,b)
- D_ab（secondary GT-conditioned dependence，协议 C.3）：
  D_ab = 0.5·[log p̂(b|C,a) − log p̂(b|C) + log p̂(a|C,b) − log p̂(a|C)]
  = 0.5·(PMI_a→b + PMI_b→a)；δ = PMI_a→b − PMI_b→a（代数关系，empirically corr≈0.02）
- JS_dep（primary distribution-level dependence，协议 C.2）：
  JS_dep(i,j) = 0.5·[JSD(P_j^a ‖ Q_j) + JSD(P_i^b ‖ Q_i)]
  P_j^a = p̂(·|C,a)|_j、Q_j = p̂(·|C)|_j（干净词表归一化分布，clean_log_probs 同源）；
  JSD(P‖Q) = 0.5·KL(P‖M) + 0.5·KL(Q‖M)，M = (P+Q)/2
  性质：对称、∈[0, ln2]、JS=0 ⇒ δ=0（δ=0 ⇏ JS=0）——严格强于 δ
- 命名纪律：一律称 conditional-dependence proxy；**不是 exact total correlation**。

数值口径：全部 fp64 计算；log 域实现（P(v)=exp(lp[v]−lse) 归一）避免下溢。
本模块不触碰任何 frozen 文件。
"""
import torch


def jsd_from_logp(lp_P, lp_Q):
    """JSD(P‖Q) = 0.5·KL(P‖M) + 0.5·KL(Q‖M)，M=(P+Q)/2；log 域稳定实现（fp64）。

    lp_P / lp_Q: 两个归一化分布的 log 概率向量（同维度）。
    """
    lp_P = lp_P.double()
    lp_Q = lp_Q.double()
    P = (lp_P - lp_P.max()).exp()
    Q = (lp_Q - lp_Q.max()).exp()
    P = P / P.sum()
    Q = Q / Q.sum()
    M = 0.5 * (P + Q)

    def kl(A, B):
        return torch.where(A > 0, A * (A.log() - B.log()), torch.zeros_like(A)).sum()
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def pointwise_terms(logp_C, logp_Ca, logp_Cb, i, j, a, b):
    """GT-token pointwise 项（fp64）。

    返回 dict：delta、d_ab、pmi_ab（=log p̂(b|C,a)−log p̂(b|C)）、
    pmi_ba（=log p̂(a|C,b)−log p̂(a|C)）、四 logp 原始值。
    """
    lp = logp_C.double()
    lp_a = logp_Ca.double()
    lp_b = logp_Cb.double()
    pa_C = lp[i, a]
    pb_C = lp[j, b]
    pb_Ca = lp_a[j, b]
    pa_Cb = lp_b[i, a]
    pmi_ab = pb_Ca - pb_C
    pmi_ba = pa_Cb - pa_C
    delta = pa_C + pb_Ca - pb_C - pa_Cb   # frozen δ 定义；= pmi_ab − pmi_ba（代数恒等）
    d_ab = 0.5 * (pmi_ab + pmi_ba)
    return dict(delta=delta, d_ab=d_ab, pmi_ab=pmi_ab, pmi_ba=pmi_ba,
                log_p_a_C=pa_C, log_p_b_C=pb_C, log_p_b_Ca=pb_Ca, log_p_a_Cb=pa_Cb)


def js_dep(logp_C, logp_Ca, logp_Cb, i, j):
    """distribution-level JS dependence（协议 C.2；位置 i/j 的完整分布）。

    返回 dict：js_dep、js_ab（reveal a 对位置 j 的 JSD）、js_ba（reveal b 对位置 i）。
    """
    lp = logp_C.double()
    lp_a = logp_Ca.double()
    lp_b = logp_Cb.double()
    js_ab = jsd_from_logp(lp_a[j], lp[j])     # P_j^a vs Q_j
    js_ba = jsd_from_logp(lp_b[i], lp[i])     # P_i^b vs Q_i
    return dict(js_dep=0.5 * (js_ab + js_ba), js_ab=js_ab, js_ba=js_ba)
