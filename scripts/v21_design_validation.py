"""v2.1 complementary-exposure design validation（设计期证明/枚举，只读，无训练）。

用途：protocol v2.1 DRAFT 的 identifiability preflight 数值验证：
- even m：frozen perfect matching 下 U/H/L 的 exact 分布、TV、marginals、E_comp、C̄_treated、L/R 趋势
- odd m：balanced persistent matching bank（循环相位）下同样的 exact 校验
- large-m：even m 的 TV 闭式公式数值；odd/even 大 m 的 Monte Carlo marginal 检查

所有概率用 Fraction 精确计算。本脚本不实现任何训练机制，只是设计证明。
用法: .venv/bin/python scripts/v21_design_validation.py
"""
import math
from fractions import Fraction as F
from itertools import combinations

import numpy as np


def comb(n, k):
    return math.comb(n, k)


def tv(p, q):
    """exact TV(p,q) = 1/2 Σ|p(M)-q(M)| over union of supports."""
    keys = set(p) | set(q)
    return F(1, 2) * sum(abs(p.get(k, F(0)) - q.get(k, F(0))) for k in keys)


def marginals(dist, m):
    out = []
    for i in range(m):
        out.append(sum(p for M, p in dist.items() if i in M))
    return out


def e_comp(dist, pairs, n_pairs=None):
    """E[#exactly-one-masked pairs] / #active pairs."""
    n = n_pairs or len(pairs)
    tot = F(0)
    for M, p in dist.items():
        d = sum(1 for (i, j) in pairs if (i in M) != (j in M))
        tot += p * F(d, n)
    return tot


def cbar_treated(dist, pairs, m, K):
    """mean over treated pairs of [P(both masked in pair) - (K/m)^2]."""
    km = F(K, m)
    vals = []
    for (i, j) in pairs:
        pboth = sum(p for M, p in dist.items() if i in M and j in M)
        vals.append(pboth - km * km)
    return sum(vals, F(0)) / len(vals)


def lr_trend(dist, pairs):
    """mean over treated pairs of [P(left masked) - P(right masked)]（应恒为 0）。"""
    vals = []
    for (i, j) in pairs:
        pl = sum(p for M, p in dist.items() if i in M)
        pr = sum(p for M, p in dist.items() if j in M)
        vals.append(pl - pr)
    return sum(vals, F(0)) / len(vals)


def even_distributions(m, K, pairs):
    """even m：单 frozen perfect matching。返回 U/H/L 分布 + 诊断。"""
    n = len(pairs)
    C = comb(m, K)
    b_max = min(K, m - K)
    a = (K - b_max) // 2
    b_min = K % 2
    aL = (K - b_min) // 2
    dist_U, dist_H, dist_L = {}, {}, {}
    for s in combinations(range(m), K):
        S = set(s)
        key = tuple(sorted(s))
        dist_U[key] = F(1, C)
        pattern = []
        for (i, j) in pairs:
            mi, mj = i in S, j in S
            pattern.append("d" if mi != mj else ("b" if mi else "v"))
        nd, nb = pattern.count("d"), pattern.count("b")
        if nd == b_max and nb == a:
            dist_H[key] = F(1, comb(n, b_max)) * F(1, 2) ** b_max * F(1, comb(n - b_max, a))
        if nd == b_min and nb == aL:
            dist_L[key] = F(1, comb(n, b_min)) * F(1, 2) ** b_min * F(1, comb(n - b_min, aL))
    return dist_U, dist_H, dist_L


def report_even(m, K, pairs, verbose=True):
    dist_U, dist_H, dist_L = even_distributions(m, K, pairs)
    if verbose:
        print(f"\n=== even m={m}, K={K}, matching={pairs} ===")
        print(f"b_max={min(K, m - K)} a={(K - min(K, m - K)) // 2} | b_min={K % 2} aL={(K - (K % 2)) // 2}")
        allM = sorted(set(dist_U) | set(dist_H) | set(dist_L))
        print(f"{'M':>14} {'P_U':>8} {'P_H':>8} {'P_L':>8}")
        for M in allM:
            print(f"{str(M):>14} {str(dist_U.get(M, F(0))):>8} {str(dist_H.get(M, F(0))):>8} {str(dist_L.get(M, F(0))):>8}")
        print(f"marginals U: {[str(x) for x in marginals(dist_U, m)]}")
        print(f"marginals H: {[str(x) for x in marginals(dist_H, m)]}")
        print(f"marginals L: {[str(x) for x in marginals(dist_L, m)]}")
    tv_hu = tv(dist_H, dist_U)
    tv_lu = tv(dist_L, dist_U)
    tv_hl = tv(dist_H, dist_L)
    eh = e_comp(dist_H, pairs)
    eu = e_comp(dist_U, pairs)
    el = e_comp(dist_L, pairs)
    ch = cbar_treated(dist_H, pairs, m, K)
    cu = cbar_treated(dist_U, pairs, m, K)
    cl = cbar_treated(dist_L, pairs, m, K)
    lr = (lr_trend(dist_U, pairs), lr_trend(dist_H, pairs), lr_trend(dist_L, pairs))
    print(f"TV(H,U)={tv_hu}  TV(L,U)={tv_lu}  TV(H,L)={tv_hl}")
    print(f"E_comp: H={eh} U={eu} L={el}  (H>U>L: {eh > eu > el})")
    print(f"C̄_treated: H={ch} U={cu} L={cl}  (H<U<L: {ch < cu < cl})")
    print(f"L/R trend (U,H,L): {lr} (全部 0: {all(x == 0 for x in lr)})")
    return dict(tv_hu=tv_hu, tv_lu=tv_lu, tv_hl=tv_hl, eh=eh, eu=eu, el=el,
                ch=ch, cu=cu, cl=cl)


def odd_bank_distributions(m, K, verbose=True):
    """odd m：balanced bank（循环相位，singleton 轮换），singleton 硬币 s~Bern(K/m) 共享流。

    phase t: singleton = t, pairs = (t+1,t+2),(t+3,t+4),... (mod m)。
    P(M) = (1/m) Σ_t Σ_s P(s) P(M|t,s)，U 与 phase/s 无关。
    """
    n = (m - 1) // 2
    q = F(K, m)
    C = comb(m, K)
    dist_U = {tuple(sorted(s)): F(1, C) for s in combinations(range(m), K)}
    dist_H, dist_L = {}, {}
    phase_pairs = {}  # t -> pairs
    for t in range(m):
        seq = [(t + 1 + k) % m for k in range(m - 1)]
        pairs = [(seq[2 * c], seq[2 * c + 1]) for c in range(n)]
        phase_pairs[t] = pairs
        for s_val in (0, 1):
            Kp = K - s_val  # pairs 内 mask 数
            if Kp < 0 or Kp > m - 1:
                continue  # K=0 ⇒ s=1 概率 0；K=m ⇒ s=0 概率 0
            for hp in ("H", "L"):
                if hp == "H":
                    b = min(Kp, (m - 1) - Kp)
                else:
                    b = Kp % 2
                a = (Kp - b) // 2
                base = F(1, comb(n, b)) * F(1, 2) ** b * F(1, comb(n - b, a))
                for sub in combinations(range(m), K):
                    S = set(sub)
                    if (t in S) != bool(s_val):
                        continue
                    pat = []
                    for (i, j) in pairs:
                        mi, mj = i in S, j in S
                        pat.append("d" if mi != mj else ("b" if mi else "v"))
                    if pat.count("d") == b and pat.count("b") == a:
                        w = (q if s_val else (1 - q)) * F(1, m) * base
                        d = dist_H if hp == "H" else dist_L
                        d[tuple(sorted(sub))] = d.get(tuple(sorted(sub)), F(0)) + w
    return dist_U, dist_H, dist_L, phase_pairs


def odd_closed_forms(m, K):
    """odd m 的闭式诊断（条件在 phase/s，E_comp 与 C̄_treated 与枚举口径一致）。

    singleton s~Bern(K/m)（共享流）。active pairs n=(m-1)/2。
    - E_comp_H = E[b(s)]/n, b(s)=min(K-s, m-1-K+s)
    - E_comp_L = E[(K-s) mod 2]/n
    - E_comp_U = 2K(m-K)/(m(m-1))（任一对 discordant 概率）
    - C̄_treated = E[a(s)]/n - (K/m)^2（H: a=(K-s-b)/2；L: aL=(K-s-bL)/2）
    - C̄_treated_U = K(K-1)/(m(m-1)) - (K/m)^2
    """
    n = (m - 1) // 2
    q = F(K, m)
    km = F(K, m)
    # H
    b0 = min(K, (m - 1) - K)  # s=0
    b1 = min(K - 1, (m - 1) - (K - 1)) if K >= 1 else 0  # s=1
    a0 = (K - b0) // 2
    a1 = (K - 1 - b1) // 2 if K >= 1 else 0
    eh = ((1 - q) * F(b0, n) + q * F(b1, n)) if K >= 1 else F(0)
    ch = ((1 - q) * F(a0, n) + q * F(a1, n)) - km * km if K >= 1 else F(0)
    # L
    bL0 = K % 2
    bL1 = (K - 1) % 2 if K >= 1 else 0
    aL0 = (K - bL0) // 2
    aL1 = (K - 1 - bL1) // 2 if K >= 1 else 0
    el = ((1 - q) * F(bL0, n) + q * F(bL1, n)) if K >= 1 else F(0)
    cl = ((1 - q) * F(aL0, n) + q * F(aL1, n)) - km * km if K >= 1 else F(0)
    eu = F(2 * K * (m - K), m * (m - 1))
    cu = F(K * (K - 1), m * (m - 1)) - km * km
    return dict(eh=eh, eu=eu, el=el, ch=ch, cu=cu, cl=cl)


def report_odd(m, K, verbose=True):
    dist_U, dist_H, dist_L, phase_pairs = odd_bank_distributions(m, K)
    if verbose:
        print(f"\n=== odd m={m}, K={K}, bank={len(phase_pairs)} phases ===")
        print(f"marginals U: {[str(x) for x in marginals(dist_U, m)]}")
        print(f"marginals H: {[str(x) for x in marginals(dist_H, m)]}")
        print(f"marginals L: {[str(x) for x in marginals(dist_L, m)]}")
    tv_hu = tv(dist_H, dist_U)
    tv_lu = tv(dist_L, dist_U)
    tv_hl = tv(dist_H, dist_L)
    cf = odd_closed_forms(m, K)
    eh, eu, el = cf["eh"], cf["eu"], cf["el"]
    ch, cu, cl = cf["ch"], cf["cu"], cf["cl"]
    print(f"TV(H,U)={tv_hu}  TV(L,U)={tv_lu}  TV(H,L)={tv_hl}")
    print(f"E_comp: H={eh} U={eu} L={el}  (H>U>L: {eh > eu > el})")
    print(f"C̄_treated: H={ch} U={cu} L={cl}  (H<U<L: {ch < cu < cl})")
    return dict(tv_hu=tv_hu, tv_lu=tv_lu, tv_hl=tv_hl, eh=eh, eu=eu, el=el,
                ch=ch, cu=cu, cl=cl)


def closed_form_tv_even(m, K):
    """even m 的 TV 闭式：TV(H,U) = 1 - |H|/C(m,K) 等。"""
    n = m // 2
    C = comb(m, K)
    b_max, a = min(K, m - K), (K - min(K, m - K)) // 2
    b_min, aL = K % 2, (K - (K % 2)) // 2
    size_h = comb(n, b_max) * (2 ** b_max) * comb(n - b_max, a)
    size_l = comb(n, b_min) * (2 ** b_min) * comb(n - b_min, aL)
    tv_hu = 1 - F(size_h, C)
    tv_lu = 1 - F(size_l, C)
    tv_hl = F(1) if b_max != b_min else F(0)
    return tv_hu, tv_lu, tv_hl


def mc_marginals_even(m, K, n_items=200000, seed=0):
    rng = np.random.default_rng(seed)
    n = m // 2
    pairs = [(2 * c, 2 * c + 1) for c in range(n)]
    counts = np.zeros(m)
    for _ in range(n_items):
        # H
        b_max = min(K, m - K)
        a = (K - b_max) // 2
        M = set()
        idx = rng.choice(n, size=b_max, replace=False)
        for c in idx:
            l, r = pairs[c]
            M.add(l if rng.integers(0, 2) else r)
        idx2 = rng.choice(n, size=a, replace=False)
        for c in idx2:
            M.update(pairs[c])
        for i in M:
            counts[i] += 1
    return counts / n_items, F(K, m)


def mc_marginals_odd(m, K, n_items=200000, seed=0):
    rng = np.random.default_rng(seed)
    n = (m - 1) // 2
    q = K / m
    counts = np.zeros(m)
    for it in range(n_items):
        t = it % m
        seq = [(t + 1 + k) % m for k in range(m - 1)]
        pairs = [(seq[2 * c], seq[2 * c + 1]) for c in range(n)]
        s = int(rng.random() < q)
        Kp = K - s
        b = min(Kp, (m - 1) - Kp)
        a = (Kp - b) // 2
        M = set()
        if s:
            M.add(t)
        idx = rng.choice(n, size=b, replace=False)
        for c in idx:
            l, r = pairs[c]
            M.add(l if rng.integers(0, 2) else r)
        idx2 = rng.choice(n, size=a, replace=False)
        for c in idx2:
            M.update(pairs[c])
        for i in M:
            counts[i] += 1
    return counts / n_items, F(K, m)


if __name__ == "__main__":
    # F. m=4,K=2（frozen matching (1,2),(3,4)）
    report_even(4, 2, [(0, 1), (2, 3)])
    # G. even small cases
    pairs6 = [(0, 1), (2, 3), (4, 5)]
    for K in (2, 3, 4):
        report_even(6, K, pairs6)
    # odd bank small cases
    for K in (2, 3):
        report_odd(5, K)
    # H. large-m even TV 闭式
    print("\n=== even m 大 case TV 闭式 ===")
    for m, K in ((10, 3), (10, 5), (20, 7), (50, 10), (50, 25), (50, 40)):
        t_hu, t_lu, t_hl = closed_form_tv_even(m, K)
        print(f"m={m:>3} K={K:>3}: TV(H,U)={str(t_hu):>24} TV(L,U)={str(t_lu):>24} TV(H,L)={t_hl}")
    # MC large-case marginal 检查
    print("\n=== MC marginal 检查（H policy）===")
    for m, K in ((50, 25), (49, 24)):
        fn = mc_marginals_even if m % 2 == 0 else mc_marginals_odd
        rates, target = fn(m, K)
        dev = float(max(abs(F(rates[i]).limit_denominator(10**8) - target) for i in range(m)))
        print(f"m={m} K={K}: max|rate_i - K/m| = {dev:.5f}  目标 K/m={target}")

    # 全范围 sweep：pilot 的 m∈[10,50]，K∈[1,m-1]，E_comp 与 C̄_treated 排序闭式验证
    print("\n=== 全范围闭式 sweep（m=10..50, K=1..m-1）===")
    def even_cf(m, K):
        n = m // 2
        km = F(K, m)
        b = min(K, m - K)
        a = (K - b) // 2
        bL = K % 2
        aL = (K - bL) // 2
        return dict(eh=F(b, n), eu=F(2 * K * (m - K), m * (m - 1)), el=F(bL, n),
                    ch=F(a, n) - km * km,
                    cu=F(K * (K - 1), m * (m - 1)) - km * km,
                    cl=F(aL, n) - km * km)

    for parity, name, cf in ((0, "even", even_cf), (1, "odd", odd_closed_forms)):
        strict_e, strict_c, equal_e, viol = 0, 0, [], []
        for m in range(10, 51):
            if m % 2 != parity:
                continue
            for K in range(1, m):
                r = cf(m, K)
                if r["eh"] > r["eu"] > r["el"]:
                    strict_e += 1
                elif r["eh"] == r["eu"] == r["el"]:
                    equal_e.append((m, K))
                else:
                    viol.append((m, K, r["eh"], r["eu"], r["el"]))
                if r["ch"] < r["cu"] < r["cl"]:
                    strict_c += 1
        total = sum(m - 1 for m in range(10, 51) if m % 2 == parity)
        print(f"{name} m∈[10,50]: E_comp 严格 H>U>L: {strict_e}/{total}；"
              f"C̄ 严格 H<U<L: {strict_c}/{total}；全相等(退化) K: {equal_e}；违反: {viol}")

    # degenerate K = {0, 1, m-1, m}：三 policy 分布应完全相同（TV=0、E_comp/C̄ 相等）
    print("\n=== degenerate K∈{0,1,m-1,m} 校验（H=U=L 同分布）===")
    for m, pairs, fn in ((4, [(0, 1), (2, 3)], "even"),
                          (6, [(0, 1), (2, 3), (4, 5)], "even"),
                          (5, None, "odd")):
        for K in (0, 1, m - 1, m):
            if fn == "even":
                dist_U, dist_H, dist_L = even_distributions(m, K, pairs)
            else:
                dist_U, dist_H, dist_L, _ = odd_bank_distributions(m, K, verbose=False)
            t_hu = tv(dist_H, dist_U)
            t_lu = tv(dist_L, dist_U)
            t_hl = tv(dist_H, dist_L)
            ok = (t_hu == 0 and t_lu == 0 and t_hl == 0)
            print(f"m={m} K={K}: TV(H,U)={t_hu} TV(L,U)={t_lu} TV(H,L)={t_hl} "
                  f"→ 同分布: {ok}")
            assert ok
