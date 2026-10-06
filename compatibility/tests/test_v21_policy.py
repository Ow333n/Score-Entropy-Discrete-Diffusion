"""v2.1 complementary-exposure policy 单元测试（protocol v2.1 §14 步骤 6 + preflight P5，
CPU、无模型）。

覆盖 preflight 验收清单：
  A. exact K：每 sample |M| == K
  B. shared schedule：U/H/L 同 replicate 共享 sample/span/σ/K（+phase/coin）逐位一致
  C. pair-map hash：训练 map / heldout map / odd-m bank 固定 hash（sidecar 校验 + 再生成一致）
  D. heldout disjointness：heldout edges 与 rep1/rep2 training treated edges 交集 = 0
  E. position marginal：P(i masked) ≈ K/m（MC）
  F. non-degenerate K（2≤K≤m−2）：E_H > E_U > E_L
  G. degenerate K（{0,1,m−1,m}）：三 policy 允许完全相同，不构成 FAIL
  H. treated covariance：C̄_H < C̄_U < C̄_L
  I. no left/right bias
  J. odd-m singleton balance（bank 轮换性质 + odd marginal parity）
  K. mask-set distribution：small exact case TV > 0（m=4 K=2 support 分离）
  L. RNG isolation：policy RNG 互不影响、不动全局 RNG；evaluator 无 RNG 依赖
"""
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from task_data import v21_policy as v21
from task_data.v21_policy import (MaskDiagnostics, V21Schedule, closed_form_cbar_active,
                                  closed_form_e_comp, load_maps_verified, select_mask_v21)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS_DIR = os.path.join(ROOT, "exp_local", "regime_a", "mechpilot_v21_maps")
L = 256


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def _make_gens(policy, r=1):
    names = {"U": ("u_uniform",), "H": ("h_pair", "h_orient"),
             "L": ("l_pair", "l_orient")}[policy]
    return {n: v21.make_generator(n, r) for n in names}


def _item(pair_map, policy, r=1, m=20, K=8, s0=40, phase=None, coin=None, gens=None):
    gens = gens or _make_gens(policy, r)
    return select_mask_v21(s0, m, K, policy, L, pair_map, phase=phase, coin=coin, gens=gens)


def _even_map(m=20, seed=12345):
    """synthetic even map（与生成脚本同构）。"""
    pi = torch.randperm(m, generator=torch.Generator().manual_seed(seed)).tolist()
    pairs = [[pi[2 * c], pi[2 * c + 1]] for c in range(m // 2)]
    return dict(pi=pi, pairs=pairs, treated=pairs)


def _odd_map(m=11, seed=12345):
    pi = torch.randperm(m, generator=torch.Generator().manual_seed(seed)).tolist()
    phases = {}
    for t in range(m):
        seq = [(t + 1 + kk) % m for kk in range(m - 1)]
        phases[str(t)] = dict(
            singleton=pi[t],
            pairs=[[pi[seq[2 * c]], pi[seq[2 * c + 1]]] for c in range((m - 1) // 2)])
    treated = [[pi[i], pi[(i + 1) % m]] for i in range(m)]
    return dict(pi=pi, phases=phases, treated=treated)


# --------------------------------------------------------------------------
# A. exact K
# --------------------------------------------------------------------------

def test_a_exact_k_even():
    pm = _even_map()
    for policy in ("U", "H", "L"):
        gens = _make_gens(policy, 1)
        for m, K in [(20, 0), (20, 1), (20, 8), (20, 19), (20, 20), (10, 5), (10, 3)]:
            entry = dict(pairs=[[2 * c, 2 * c + 1] for c in range(m // 2)], treated=[])
            entry["treated"] = entry["pairs"]
            for _ in range(200):
                sel = select_mask_v21(40, m, K, policy, L, entry, gens=gens)
                assert len(sel) == K, f"{policy} m={m} K={K}: |M|={len(sel)}"


def test_a_exact_k_odd():
    for m in (11, 21, 49):
        pm = _odd_map(m)
        for policy in ("U", "H", "L"):
            gens = _make_gens(policy, 2)
            for K in (0, 1, 2, m // 2, m - 1, m):
                for _ in range(100):
                    phase = 0
                    u = torch.rand(1, generator=torch.Generator().manual_seed(7)).item()
                    sel = select_mask_v21(40, m, K, policy, L, pm, phase=phase,
                                          coin=u, gens=gens)
                    assert len(sel) == K, f"{policy} m={m} K={K}: |M|={len(sel)}"
                    assert all(40 <= p < 40 + m for p in sel)


# --------------------------------------------------------------------------
# B. shared schedule 逐位一致
# --------------------------------------------------------------------------

def test_b_shared_schedule_identical():
    s1 = V21Schedule(1, L, 10, 50)
    s2 = V21Schedule(1, L, 10, 50)
    s3 = V21Schedule(2, L, 10, 50)
    for _ in range(64):
        a = s1.draw(32)
        b = s2.draw(32)
        for x, y in zip(a, b):
            assert torch.equal(x, y), "同 replicate 两次重放应逐位一致"
    assert s1.hexdigest() == s2.hexdigest()
    assert s1.hexdigest() != s3.hexdigest(), "不同 replicate digest 应不同"
    # odd m 的 phase/coin 确定性
    s4 = V21Schedule(1, L, 10, 50)
    s5 = V21Schedule(1, L, 10, 50)
    a = s4.draw(128)
    b = s5.draw(128)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_b_policy_shared_streams_untouched():
    """policy 选择不改变共享流：同 replicate 内 U/H/L 的 σ/span/K/phase/coin 一致。"""
    scheds = [V21Schedule(1, L, 10, 50) for _ in range(3)]
    draws = [s.draw(64) for s in scheds]
    for i in range(1, 3):
        for x, y in zip(draws[0], draws[i]):
            assert torch.equal(x, y)
    # U/H/L 执行后共享 digest 仍一致（policy 只消耗自己的流）
    for pol, gens in (("U", _make_gens("U", 1)), ("H", _make_gens("H", 1)),
                      ("L", _make_gens("L", 1))):
        s = V21Schedule(1, L, 10, 50)
        sigma, dsigma, m_t, s0_t, K, phase, coin = s.draw(200)
        for i in range(200):
            m = int(m_t[i].item())
            pm = _even_map(m) if m % 2 == 0 else _odd_map(m)
            select_mask_v21(int(s0_t[i].item()), m, int(K[i].item()), pol, L, pm,
                            phase=int(phase[i].item()),
                            coin=float(coin[i].item()) if m % 2 else None,
                            gens=gens)
        # 共享流消耗与 policy 无关 → digest 应等于同 pattern 的纯重放
        # （torch.randint CPU 按 batch 形状消耗流，故比较必须同 draw pattern：
        #  训练与 dryrun 均为每步 draw(micro_batch) 一次，模式一致）
        s_pure = V21Schedule(1, L, 10, 50)
        s_pure.draw(200)
        assert s.hexdigest() == s_pure.hexdigest(), \
            f"policy {pol} 改变了共享流 digest"


# --------------------------------------------------------------------------
# C. pair-map hash 固定 + D. heldout disjointness + J. singleton 轮换
# --------------------------------------------------------------------------

def test_c_d_j_map_integrity():
    maps = load_maps_verified(MAPS_DIR)
    man = maps["manifest"]
    for fname in man["files"]:
        path = os.path.join(MAPS_DIR, fname)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        sidecar = open(path + ".sha256").read().strip()
        assert digest == sidecar == man["file_sha256"][fname], fname
        # 再加载一次 hash 稳定
        assert maps["file_sha256"][fname] == digest
    # D: heldout 与 rep1/rep2 treated 边集不重叠（逐 m）
    for r in (1, 2):
        rep = maps[f"rep{r}"]
        for parity in ("even", "odd"):
            for m, entry in rep[parity].items():
                assert entry["treated"], m
    heldout = maps["heldout"]
    for parity in ("even", "odd"):
        for m, he in heldout[parity].items():
            he_set = {frozenset(e) for e in he["pairs"]}
            for r in (1, 2):
                tr = maps[f"rep{r}"][parity][m]["treated"]
                assert not (he_set & {frozenset(e) for e in tr}), f"heldout m={m} rep{r} 重叠"
            assert heldout["disjointness"][m]["ok"]
    # J: odd-m singleton 轮换性质（每位置恰 1 次 singleton）
    for r in (1, 2):
        for m, entry in maps[f"rep{r}"]["odd"].items():
            mi = int(m)
            s = [entry["phases"][str(t)]["singleton"] for t in range(mi)]
            assert sorted(s) == list(range(mi)), f"rep{r} m={m} singleton 轮换坏"
    # even matching 完美覆盖
    for r in (1, 2):
        for m, entry in maps[f"rep{r}"]["even"].items():
            covered = sorted(p for e in entry["pairs"] for p in e)
            assert covered == list(range(int(m))), f"rep{r} m={m} matching 坏"


def test_c_map_regeneration_stable():
    """再生成同 seed 的 map 与落盘文件逐位一致（frozen 性质）。"""
    import scripts.v21_generate_maps as gm
    maps = load_maps_verified(MAPS_DIR)
    for r in (1, 2):
        rep = maps[f"rep{r}"]
        for parity in ("even", "odd"):
            for m, entry in rep[parity].items():
                mi = int(m)
                seed = gm.MAP_SEED_BASE_REP + 100 * r + mi
                new = (gm.make_even_map(mi, seed) if parity == "even"
                       else gm.make_odd_bank(mi, seed))
                assert new == entry, f"rep{r} m={m} 再生成不一致"


# --------------------------------------------------------------------------
# E. position marginal ≈ K/m（MC，固定 m,K）
# --------------------------------------------------------------------------

def _marginal_mc(policy, pm, m, K, n_items=20000, r=1, seed_extra=0):
    counts = torch.zeros(m, dtype=torch.long)
    gens = _make_gens(policy, r)
    coin_gen = torch.Generator().manual_seed(9)
    for _ in range(n_items):
        phase = None
        coin = None
        if m % 2 == 1:
            phase = 0
            coin = torch.rand(1, generator=coin_gen).item()
        sel = select_mask_v21(0, m, K, policy, L, pm, phase=phase, coin=coin, gens=gens)
        for p in sel:
            counts[p] += 1
    return counts / n_items


def test_e_marginal_parity():
    for m, K in [(20, 8), (50, 25), (11, 4), (49, 24)]:
        pm = _even_map(m) if m % 2 == 0 else _odd_map(m)
        for policy in ("U", "H", "L"):
            rate = _marginal_mc(policy, pm, m, K, n_items=12000)
            target = K / m
            se = math.sqrt(target * (1 - target) / 12000)
            dev = float((rate - target).abs().max())
            assert dev < 5 * se + 1e-3, \
                f"{policy} m={m} K={K}: max|rate−K/m|={dev:.5f} (5σ={5*se:.5f})"


# --------------------------------------------------------------------------
# F. E_comp 排序 + H. covariance 排序（non-degenerate K）
# --------------------------------------------------------------------------

def _e_comp_mc(policy, pm, m, K, n_items=20000, r=1):
    total = 0.0
    gens = _make_gens(policy, r)
    active = pm["pairs"] if m % 2 == 0 else pm["phases"]["0"]["pairs"]
    coin_gen = torch.Generator().manual_seed(9)
    for _ in range(n_items):
        phase, coin = None, None
        if m % 2 == 1:
            phase = 0
            coin = torch.rand(1, generator=coin_gen).item()
        sel = set(select_mask_v21(0, m, K, policy, L, pm, phase=phase, coin=coin, gens=gens))
        total += sum(1 for (i, j) in active if (i in sel) != (j in sel)) / len(active)
    return total / n_items


def _cbar_mc(policy, pm, m, K, n_items=20000, r=1):
    """treated edges 上的 C̄（主口径 §6）。"""
    s_both = 0
    n_edges = 0
    gens = _make_gens(policy, r)
    treated = pm["treated"]
    coin_gen = torch.Generator().manual_seed(9)
    for _ in range(n_items):
        phase, coin = None, None
        if m % 2 == 1:
            phase = 0
            coin = torch.rand(1, generator=coin_gen).item()
        sel = set(select_mask_v21(0, m, K, policy, L, pm, phase=phase, coin=coin, gens=gens))
        for (i, j) in treated:
            s_both += float((i in sel) and (j in sel))
            n_edges += 1
    return s_both / n_edges - (K / m) ** 2


def test_f_h_ordering():
    for m, K in [(20, 8), (16, 6), (21, 10)]:
        pm = _even_map(m) if m % 2 == 0 else _odd_map(m)
        eU = _e_comp_mc("U", pm, m, K)
        eH = _e_comp_mc("H", pm, m, K)
        eL = _e_comp_mc("L", pm, m, K)
        assert eH > eU > eL, f"m={m} K={K}: E_comp H={eH:.4f} U={eU:.4f} L={eL:.4f}"
        # 与闭式一致（容差 3σ_MC）
        for pol, e_mc in (("U", eU), ("H", eH), ("L", eL)):
            cf = closed_form_e_comp(m, K, pol)
            se = math.sqrt(e_mc * (1 - e_mc) / 20000)
            assert abs(e_mc - cf) < 3 * se + 1e-3, \
                f"{pol} m={m} K={K}: E_comp mc={e_mc:.4f} closed={cf:.4f}"
        cU = _cbar_mc("U", pm, m, K)
        cH = _cbar_mc("H", pm, m, K)
        cL = _cbar_mc("L", pm, m, K)
        assert cH < cU < cL, f"m={m} K={K}: C̄ H={cH:.4f} U={cU:.4f} L={cL:.4f}"
        for pol, c_mc in (("U", cU), ("H", cH), ("L", cL)):
            cf = closed_form_cbar_active(m, K, pol)
            # even m：treated edges == active pairs → 与闭式严格可比
            if m % 2 == 0:
                assert abs(c_mc - cf) < 3e-3, \
                    f"{pol} m={m} K={K}: C̄ mc={c_mc:.4f} closed={cf:.4f}"


# --------------------------------------------------------------------------
# G. degenerate K ∈ {0,1,m−1,m}：三 policy 同分布（不构成 FAIL）
# --------------------------------------------------------------------------

def test_g_degenerate_k_same_distribution():
    pm4 = dict(pairs=[[0, 1], [2, 3]], treated=[[0, 1], [2, 3]])
    for K in (0, 1, 3, 4):
        supports = []
        for policy in ("U", "H", "L"):
            gens = _make_gens(policy, 1)
            seen = set()
            for _ in range(2000):
                sel = select_mask_v21(0, 4, K, policy, L, pm4, gens=gens)
                seen.add(tuple(sorted(sel)))
            supports.append(seen)
        # 三 policy support 相同（m=4 全子集空间 = {空},{单点},{三点},{全集}）
        assert supports[0] == supports[1] == supports[2], f"K={K}: supports 不同"
        # 与均匀分布一致：K=1 → 4 个单点全出现；K=3 → 4 个三点全出现
        assert len(supports[0]) == (4 if K in (1, 3) else 1), f"K={K}: support={supports[0]}"
    # odd m degenerate
    pm11 = _odd_map(11)
    for K in (0, 1, 10, 11):
        supports = []
        for policy in ("U", "H", "L"):
            gens = _make_gens(policy, 2)
            seen = set()
            coin_gen = torch.Generator().manual_seed(3)
            for _ in range(300):
                phase = 0
                coin = torch.rand(1, generator=coin_gen).item()
                sel = select_mask_v21(0, 11, K, policy, L, pm11, phase=phase,
                                      coin=coin, gens=gens)
                seen.add(tuple(sorted(sel)))
            supports.append(seen)
        assert supports[0] == supports[1] == supports[2], f"odd K={K}: supports 不同"


# --------------------------------------------------------------------------
# I. no L/R bias
# --------------------------------------------------------------------------

def test_i_no_lr_bias():
    for m, K in [(20, 8), (21, 10)]:
        pm = _even_map(m) if m % 2 == 0 else _odd_map(m)
        active = pm["pairs"] if m % 2 == 0 else pm["phases"]["0"]["pairs"]
        for policy in ("U", "H", "L"):
            gens = _make_gens(policy, 1)
            left_only = right_only = 0
            coin_gen = torch.Generator().manual_seed(9)
            for _ in range(20000):
                phase, coin = None, None
                if m % 2 == 1:
                    phase = 0
                    coin = torch.rand(1, generator=coin_gen).item()
                sel = set(select_mask_v21(0, m, K, policy, L, pm, phase=phase,
                                          coin=coin, gens=gens))
                for (i, j) in active:
                    mi, mj = (i in sel), (j in sel)
                    if mi != mj:
                        left_only += 1 if mi else 0
                        right_only += 0 if mi else 1
            n = left_only + right_only
            z = (left_only - right_only) / max(math.sqrt(n), 1)
            assert abs(z) < 5, f"{policy} m={m} K={K}: L/R z={z:+.2f}"


# --------------------------------------------------------------------------
# J. odd-m singleton balance（mask 率）
# --------------------------------------------------------------------------

def test_j_odd_singleton_balance():
    m, K = 21, 10
    pm = _odd_map(m)
    for policy in ("U", "H", "L"):
        counts = torch.zeros(m, dtype=torch.long)
        gens = _make_gens(policy, 1)
        coin_gen = torch.Generator().manual_seed(9)
        for _ in range(12000):
            phase = 0
            coin = torch.rand(1, generator=coin_gen).item()
            sel = select_mask_v21(0, m, K, policy, L, pm, phase=phase, coin=coin,
                                  gens=gens)
            for p in sel:
                counts[p] += 1
        rate = counts / 12000
        target = K / m
        se = math.sqrt(target * (1 - target) / 12000)
        assert float((rate - target).abs().max()) < 5 * se + 1e-3, \
            f"{policy} odd singleton balance: max dev={float((rate-target).abs().max()):.5f}"


# --------------------------------------------------------------------------
# K. mask-set distribution：small exact case TV > 0（m=4 K=2）
# --------------------------------------------------------------------------

def test_k_mask_distribution_tv_positive():
    pm4 = dict(pairs=[[0, 1], [2, 3]], treated=[[0, 1], [2, 3]])
    cross = {(0, 2), (0, 3), (1, 2), (1, 3)}   # H support（§7.1）
    co = {(0, 1), (2, 3)}                      # L support
    for policy, expected in (("H", cross), ("L", co), ("U", cross | co)):
        gens = _make_gens(policy, 1)
        seen = set()
        for _ in range(4000):
            sel = select_mask_v21(0, 4, 2, policy, L, pm4, gens=gens)
            seen.add(tuple(sorted(sel)))
        assert seen == expected, f"{policy}: support={seen} 期望={expected}"
    # support 分离 → TV(H,L) = 1 > 0（§7.1）
    assert not (cross & co)


# --------------------------------------------------------------------------
# L. RNG isolation
# --------------------------------------------------------------------------

def test_l_rng_isolation():
    # 1) policy 流互不影响
    h_gens = _make_gens("H", 1)
    l_gens = _make_gens("L", 1)
    u_gens = _make_gens("U", 1)
    pm = _even_map()
    l_state = l_gens["l_pair"].get_state()
    u_state = u_gens["u_uniform"].get_state()
    for _ in range(200):
        select_mask_v21(0, 20, 8, "H", L, pm, gens=h_gens)
    assert torch.equal(l_gens["l_pair"].get_state(), l_state), "H 消耗污染了 L 流"
    assert torch.equal(u_gens["u_uniform"].get_state(), u_state), "H 消耗污染了 U 流"
    # 2) 不触碰全局 RNG（CPU 与 CUDA 状态快照）
    cpu_state = torch.get_rng_state()
    for _ in range(100):
        select_mask_v21(0, 20, 8, "U", L, pm, gens=u_gens)
    assert torch.equal(torch.get_rng_state(), cpu_state), "policy 采样动了全局 CPU RNG"
    # 3) evaluator 无 RNG 依赖：V21Schedule 也不动全局 RNG
    cpu_state = torch.get_rng_state()
    s = V21Schedule(1, L, 10, 50)
    for _ in range(10):
        s.draw(8)
    assert torch.equal(torch.get_rng_state(), cpu_state), "V21Schedule 动了全局 RNG"


# --------------------------------------------------------------------------
# 诊断累加器 sanity（§7.6 七项产出非空、E_comp/C̄ 口径正确）
# --------------------------------------------------------------------------

def test_diagnostics_accumulator():
    pm = _even_map()
    diag = MaskDiagnostics()
    gens = _make_gens("H", 1)
    m, K = 20, 8
    for _ in range(200):
        sel = set(select_mask_v21(0, m, K, "H", L, pm, gens=gens))
        diag.update(sel, m, K, 1.5, pm, None)
    out = diag.finalize()
    assert out["exact_k_violations"] == 0
    assert out["n_items"] == 200
    c = out["confound"]
    assert c["mask_run_length"]["n_runs"] > 0
    assert c["visible_run_length"]["n_runs"] > 0
    assert c["transition_count_mean"] > 0
    assert (c["pair_distance"]["buckets"]["near"] + c["pair_distance"]["buckets"]["medium"]
            + c["pair_distance"]["buckets"]["far"]) == 200 * 10
    assert c["context_visible"]["mean"] == m - K
    assert c["nt_cov"]["n_pairs"] == 200 * (m * (m - 1) // 2 - m // 2)
    assert out["e_comp"]["overall"] > 0.79  # H even m=20 K=8：b=8,a=0 → E_comp=8/10=0.8 恒成立
    assert abs(out["e_comp"]["overall"] - closed_form_e_comp(m, K, "H")) < 0.01
    assert abs(out["covariance"]["treated_edges"]["cbar"] - closed_form_cbar_active(m, K, "H")) < 0.02
