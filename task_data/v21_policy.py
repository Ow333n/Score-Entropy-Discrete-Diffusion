"""v2.1 complementary-exposure mask policies（protocol mechanism_complementary_exposure_v2_1 §4/§6/§8）。

纯 CPU、无模型依赖。U/H/L 三 policy 直接操纵 E_comp（exactly-one-masked pair 频率），
其余（sample/span/σ/K/单位置 marginal）共享且不变；pair 结构 frozen/persistent
（每 replicate × m 生成一次、落盘、sha256、训练加载不重生成——§3.1）。

- even m：frozen perfect matching（random permutation π 连续配对）；
  H b=min(K,m−K) 个 discordant、L b=K mod 2 个 discordant、U 均匀 K 子集（§4）
- odd m：balanced persistent matching bank（π 旋转 matching、singleton 轮换）（§6）：
  phase = (o_span + s) mod m，o_span = (span_start + span_len) mod m、s≡0
  （实现解释：每 item 即其 span 的唯一 item；phase 由共享 span 流确定、
  确定性、无额外 RNG——记录为冻结实现细节）；
  singleton 硬币 s = (u < K/m)，u 来自 8500+r 共享流（U 消耗但不用该值）；
  pairs 内需 K−s 个 mask（0 ≤ K−s ≤ m−1 断言）
- 流种子（协议 §10）：共享四元流 1000+r…4000+r（沿用 v1.2 约定，streams 逐位相同）；
  policy 流 H pair-choice 8001+r / H orient 8101+r / L pair-choice 8201+r /
  L orient 8301+r / U uniform 8401+r / singleton coin（仅 odd m 消耗）8501+r；
  dropout 7001+r（cuda.manual_seed）。与 v1.2 policy 流（5000/6000）及
  evaluator seed 域不相交（§9 修订 #3）。
- 本模块不触碰任何 frozen 代码路径（policy_corruption.py / corruption.py /
  losses.py / graph_lib.py 零改动）。
"""
import hashlib
import json
import os

import numpy as np
import torch

from task_data.policy_corruption import SharedSchedule

POLICIES = ("U", "H", "L")

STREAM_SEEDS = dict(
    h_pair=8001,
    h_orient=8101,
    l_pair=8201,
    l_orient=8301,
    u_uniform=8401,
    singleton_coin=8501,
)

# confound #4 pair-distance 分桶（冻结规则，preflight 记录）
PAIR_DIST_BUCKETS = dict(near=(1, 4), medium=(5, 15), far=(16, 49))
# confound #7 非 treated 协方差的直方图（[-1, 1] 400 bins，冻结）
NT_COV_BINS = 400
NT_COV_LO, NT_COV_HI = -1.0, 1.0
# E_comp by K/m 分桶（冻结）
K_RATIO_BUCKET_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)


def stream_seed(name, replicate):
    return STREAM_SEEDS[name] + replicate


def make_generator(name, replicate, device="cpu"):
    return torch.Generator(device=device).manual_seed(stream_seed(name, replicate))


def phase_for_item(span_start, span_len):
    """odd m 的 phase：o_span = (span_start + span_len) mod m，phase = (o_span + 0) mod m。

    确定性、由共享 span 流确定、无 RNG（§6；实现解释冻结于本 docstring）。
    """
    m = int(span_len)
    return int((int(span_start) + int(span_len)) % m)


def singleton_coin(u, K, m):
    """s = (u < K/m)，u ~ U(0,1) 共享流（§6 修订 #1；K=0 ⇒ s≡0、K=m ⇒ s≡1）。"""
    return 1 if u < (K / m) else 0


def _randperm(n, generator):
    return torch.randperm(n, generator=generator).tolist()


def _rand_float(generator):
    return torch.rand(1, generator=generator).item()


def select_mask_v21(span_start, span_len, K, policy, L, pair_map, phase=None,
                    coin=None, gens=None):
    """返回单 item 的绝对 mask 位置（list[int]），|M| == K 恒成立（G3a-1）。

    Args:
        span_start, span_len, K: 共享四元流值（int）；L: 序列总长
        policy: "U" | "H" | "L"
        pair_map: 该 m 的 map 条目（even: {"pairs": [(i,j),...], "treated": [...]}；
                  odd: {"phases": {str(t): {"singleton": s, "pairs": [...]}}, "treated": [...]}）
        phase: odd m 的 phase（even 忽略）；coin: odd m 的共享硬币 u（even 忽略）
        gens: policy 专属 generator dict（只消耗本 policy 的流；G3a/L 隔离）

    RNG 消耗（确定性、per item）：
      U: 1 × randperm(m)
      H: 1 × randperm(n)（discordant 前 b 个 + both-masked 后 a 个）+ b × fair coin
      L(b=0): 1 × randperm(n)；L(b=1): 1 × randperm(n) + 1 × coin + 1 × randperm(n−1)
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    m = int(span_len)
    s0 = int(span_start)
    k = int(K)
    assert 0 <= k <= m

    if m % 2 == 0:
        pairs = pair_map["pairs"]
        n = len(pairs)
        assert n == m // 2

        if policy == "U":
            sel = _randperm(m, gens["u_uniform"])[:k]
            return [s0 + i for i in sel]

        if policy == "H":
            b, a = min(k, m - k), (k - min(k, m - k)) // 2
            perm = _randperm(n, gens["h_pair"])
            disc, both = perm[:b], perm[b:b + a]
            masked = []
            for idx in disc:
                i, j = pairs[idx]
                masked.append(i if _rand_float(gens["h_orient"]) < 0.5 else j)
            for idx in both:
                masked.extend(pairs[idx])
        else:  # L
            b, a = k % 2, (k - (k % 2)) // 2
            masked = []
            if b:
                perm = _randperm(n, gens["l_pair"])
                disc = perm[:1]
                i, j = pairs[disc[0]]
                masked.append(i if _rand_float(gens["l_orient"]) < 0.5 else j)
                rem = [x for x in range(n) if x not in disc]
                both = [rem[x] for x in _randperm(n - 1, gens["l_pair"])[:a]]
            else:
                perm = _randperm(n, gens["l_pair"])
                both = perm[:a]
            for idx in both:
                masked.extend(pairs[idx])

        out = [s0 + i for i in masked]
        assert len(out) == k, f"{policy} even m={m} K={k}: |M|={len(out)}"
        return out

    # ---- odd m：bank phase + singleton 硬币 ----
    assert phase is not None and coin is not None
    ph = pair_map["phases"][str(int(phase))]
    pairs, singleton = ph["pairs"], int(ph["singleton"])
    n = len(pairs)
    assert n == (m - 1) // 2

    if policy == "U":
        sel = _randperm(m, gens["u_uniform"])[:k]
        return [s0 + i for i in sel]

    s = singleton_coin(float(coin), k, m)
    kp = k - s
    assert 0 <= kp <= m - 1, f"odd m={m} K={k} s={s}: K−s={kp} 越界"  # §5.1 可行性断言

    if policy == "H":
        b, a = min(kp, (m - 1) - kp), (kp - min(kp, (m - 1) - kp)) // 2
        perm = _randperm(n, gens["h_pair"])
        disc, both = perm[:b], perm[b:b + a]
        masked = []
        for idx in disc:
            i, j = pairs[idx]
            masked.append(i if _rand_float(gens["h_orient"]) < 0.5 else j)
        for idx in both:
            masked.extend(pairs[idx])
        if s:
            masked.append(singleton)
    else:  # L
        b, a = kp % 2, (kp - (kp % 2)) // 2
        masked = []
        if b:
            perm = _randperm(n, gens["l_pair"])
            disc = perm[:1]
            i, j = pairs[disc[0]]
            masked.append(i if _rand_float(gens["l_orient"]) < 0.5 else j)
            rem = [x for x in range(n) if x not in disc]
            both = [rem[x] for x in _randperm(n - 1, gens["l_pair"])[:a]]
        else:
            perm = _randperm(n, gens["l_pair"])
            both = perm[:a]
        for idx in both:
            masked.extend(pairs[idx])
        if s:
            masked.append(singleton)

    out = [s0 + i for i in masked]
    assert len(out) == k, f"{policy} odd m={m} K={k} s={s}: |M|={len(out)}"
    return out


class V21Schedule:
    """v2.1 共享 schedule：SharedSchedule 四元流（沿用 v1.2 种子）+ odd-m phase +
    singleton 硬币流。U/H/L 同 replicate 逐位一致（§3.3 原则 3、G3a-2）。

    digest（G3a-2 schedule digest）：σ/dσ/span_len/span_start/K/phase/coin
    逐 item 全入 sha256；训练结束与 dryrun 参考比对。
    同时保留 SharedSchedule 自身 digest（与 v1.2 同口径，可对拍 v1.2 dryrun 参考）。
    """

    def __init__(self, replicate, L, span_min, span_max, eps=1e-3, noise=None):
        self.r = replicate
        self.L = L
        self.span_min = span_min
        self.span_max = span_max
        self.sched = SharedSchedule(replicate, L, span_min, span_max, eps, noise)
        self.g_coin = make_generator("singleton_coin", replicate)
        self._digest = hashlib.sha256()

    def draw(self, n_items):
        sigma, dsigma, span_len, span_start, K = self.sched.draw(n_items)
        phase = torch.empty(n_items, dtype=torch.long)
        coin = torch.empty(n_items, dtype=torch.float32)
        for i in range(n_items):
            m = int(span_len[i].item())
            phase[i] = phase_for_item(int(span_start[i].item()), m)
            if m % 2 == 1:
                coin[i] = torch.rand(1, generator=self.g_coin).item()
            else:
                coin[i] = -1.0  # even m 不消耗硬币流（sentinel，digest 用）
        for arr in (sigma, dsigma, span_len, span_start, K, phase, coin):
            self._digest.update(arr.numpy().tobytes())
        return sigma, dsigma, span_len, span_start, K, phase, coin

    def hexdigest(self):
        return self._digest.hexdigest()


def load_maps_verified(maps_dir):
    """加载 maps_manifest.json + 全部 map 文件，逐文件 sha256 校验（G3a-7）。

    Returns dict: {"rep1": ..., "rep2": ..., "heldout": ..., "manifest": ...,
                   "file_sha256": {fname: digest}}
    """
    man_path = os.path.join(maps_dir, "maps_manifest.json")
    man = json.load(open(man_path))
    out = {"manifest": man, "file_sha256": {}}
    out["file_sha256"]["maps_manifest.json"] = hashlib.sha256(
        open(man_path, "rb").read()).hexdigest()
    for fname in man["files"]:
        path = os.path.join(maps_dir, fname)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        expected = open(path + ".sha256").read().strip()
        assert digest == expected, f"map 文件被改动: {fname}"
        out["file_sha256"][fname] = digest
        if fname.startswith("maps_rep"):
            out["rep" + fname[len("maps_rep")]] = json.load(open(path))
        elif fname.startswith("maps_heldout"):
            out["heldout"] = json.load(open(path))
    return out


# --------------------------------------------------------------------------
# G3a / §7.6 流式诊断累加器
# --------------------------------------------------------------------------

class MaskDiagnostics:
    """§7.6 七项 confound diagnostics + G3a 实测量的流式累加器（CPU，逐 item）。

    更新口径：
    - active pairs：even → matching 全部 n 对；odd → 当前 phase 的 n 对（§8）
    - treated edges：even → matching 边；odd → bank 的 m 条 cycle 边（§6）
    - E_comp 只统计 active pairs；C̄_treated 主口径统计 treated edges、
      补充口径统计 active pairs（对应 §7.4 闭式）
    """

    def __init__(self):
        self.n_items = 0
        self.exact_k_violations = 0
        self.kp_range_violations = 0
        # 1/2/3: mask run-length / visible run-length / transition count
        self.mask_run_hist = {}
        self.vis_run_hist = {}
        self.transition_total = 0
        # 4: treated pair distance
        self.pair_dist_hist = {}
        self.pair_dist_buckets = dict(near=0, medium=0, far=0)
        # 5: L/R（active pairs，左/右各 exactly-one-masked 计数）+ 相对位置 mask 率
        self.lr_left_only = 0
        self.lr_right_only = 0
        self.rel_pos_mask = torch.zeros(64, dtype=torch.long)
        self.rel_pos_n = torch.zeros(64, dtype=torch.long)
        # 6: context-visible-count（m−K）
        self.ctx_vis_hist = {}
        # 7: 非 treated pair 协方差直方图
        self.nt_cov_hist = torch.zeros(NT_COV_BINS, dtype=torch.float64)
        self.nt_cov_n = 0
        # E_comp（§8）：overall + per-item（σ 分桶用）+ K/m 分桶
        self.e_comp_sum = 0.0
        self.e_comp_by_kr = [0.0, 0.0, 0.0, 0.0]
        self.e_comp_by_kr_n = [0, 0, 0, 0]
        self.e_comp_items = []
        self.e_comp_sigma = []
        # C̄_treated（主：treated edges；补：active pairs）
        self.treated_both_sum = 0
        self.treated_edge_n = 0
        self.active_both_sum = 0
        self.active_edge_n = 0
        self.km2_sum = 0.0
        # 单位置 marginal（绝对位置，G3a-3 用相对位置表）
        self.pos_counts = torch.zeros(256, dtype=torch.long)
        self.pos_n = torch.zeros(256, dtype=torch.long)
        # degenerate K ∈ {0,1,m−1,m} 表（G3a-5 修订 #2）
        self.degenerate_counts = dict(k0=0, k1=0, km1=0, km=0)
        self.degenerate_e_sum = 0.0
        self.degenerate_e_items = 0
        # odd-m singleton 平衡（G3a/J）：每位置 singleton 出现计数
        self.singleton_counts = torch.zeros(64, dtype=torch.long)
        self.singleton_n = 0

    # -- 工具 -------------------------------------------------------------
    @staticmethod
    def _active_pairs(pair_map, m, phase):
        if m % 2 == 0:
            return pair_map["pairs"]
        return pair_map["phases"][str(int(phase))]["pairs"]

    @staticmethod
    def _runs(sorted_positions):
        """连续 run 长度列表（sorted ints）。"""
        out = []
        for p in sorted_positions:
            if out and p == out[-1][1]:
                out[-1] = (out[-1][0], p + 1)
            else:
                out.append((p, p + 1))
        return [e - s for s, e in out]

    # -- 主入口 -----------------------------------------------------------
    def update(self, mask_local, m, K, sigma, pair_map, phase):
        """mask_local: 该 item span 内被 mask 的 local 位置（iterable of int）。"""
        self.n_items += 1
        k = int(K)
        if len(mask_local) != k:
            self.exact_k_violations += 1
        M = set(int(x) for x in mask_local)
        km = k / m
        self.km2_sum += km * km

        # confound 1/2: run-length（masked / visible）
        for length in self._runs(sorted(M)):
            self.mask_run_hist[length] = self.mask_run_hist.get(length, 0) + 1
        visible = [i for i in range(m) if i not in M]
        for length in self._runs(visible):
            self.vis_run_hist[length] = self.vis_run_hist.get(length, 0) + 1
        # confound 3: transition count
        self.transition_total += sum(
            1 for i in range(m - 1) if (i in M) != ((i + 1) in M))

        # confound 4: treated pair distance（policy 无关，由 map 决定；仍按 policy 记录以验共享）
        for (i, j) in pair_map["treated"]:
            d = abs(int(j) - int(i))
            self.pair_dist_hist[d] = self.pair_dist_hist.get(d, 0) + 1
            for name, (lo, hi) in PAIR_DIST_BUCKETS.items():
                if lo <= d <= hi:
                    self.pair_dist_buckets[name] += 1

        # confound 5: L/R + 相对位置 mask 率
        active = self._active_pairs(pair_map, m, phase)
        for (i, j) in active:
            mi, mj = (i in M), (j in M)
            if mi != mj:
                if mi:
                    self.lr_left_only += 1
                else:
                    self.lr_right_only += 1
        for d in range(m):
            self.rel_pos_n[d] += 1
            if d in M:
                self.rel_pos_mask[d] += 1

        # confound 6: context-visible-count
        self.ctx_vis_hist[m - k] = self.ctx_vis_hist.get(m - k, 0) + 1

        # confound 7: 非 treated 边协方差直方图
        treated_set = {frozenset((int(i), int(j))) for (i, j) in pair_map["treated"]}
        v = np.zeros(m, dtype=bool)
        for i in M:
            v[i] = True
        nt_cov = []
        for i in range(m):
            for j in range(i + 1, m):
                if frozenset((i, j)) in treated_set:
                    continue
                nt_cov.append(float(v[i] and v[j]) - km * km)
        if nt_cov:
            self.nt_cov_hist += torch.histc(
                torch.tensor(nt_cov), bins=NT_COV_BINS, min=NT_COV_LO, max=NT_COV_HI)
            self.nt_cov_n += len(nt_cov)

        # E_comp（active pairs，§8）
        n_active = len(active)
        if n_active > 0:
            e = sum(1 for (i, j) in active if (i in M) != (j in M)) / n_active
            self.e_comp_sum += e
            self.e_comp_items.append(e)
            self.e_comp_sigma.append(float(sigma))
            kr = km
            for b in range(4):
                if K_RATIO_BUCKET_EDGES[b] <= kr <= K_RATIO_BUCKET_EDGES[b + 1]:
                    self.e_comp_by_kr[b] += e
                    self.e_comp_by_kr_n[b] += 1
        # degenerate K 表（§7.3：K∈{0,1,m−1,m}）
        if k == 0:
            self.degenerate_counts["k0"] += 1
        elif k == 1:
            self.degenerate_counts["k1"] += 1
        elif k == m - 1:
            self.degenerate_counts["km1"] += 1
        elif k == m:
            self.degenerate_counts["km"] += 1
        if k in (0, 1, m - 1, m):
            self.degenerate_e_items += 1
            if n_active > 0:
                self.degenerate_e_sum += sum(
                    1 for (i, j) in active if (i in M) != (j in M)) / n_active

        # C̄_treated：treated edges（主）+ active pairs（补）
        for (i, j) in pair_map["treated"]:
            self.treated_both_sum += float((i in M) and (j in M))
            self.treated_edge_n += 1
        for (i, j) in active:
            self.active_both_sum += float((i in M) and (j in M))
            self.active_edge_n += 1

        # 绝对位置 marginal（mask 率）
        for p in M:
            self.pos_counts[p] += 1

        # odd-m singleton 平衡：phase 的 singleton 位置计数
        if m % 2 == 1:
            self.singleton_counts[int(pair_map["phases"][str(int(phase))]["singleton"])] += 1
            self.singleton_n += 1

    # -- 汇总 -------------------------------------------------------------
    def _hist_summary(self, hist):
        if not hist:
            return dict(mean=None, counts={})
        total = sum(hist.values())
        mean = sum(k * v for k, v in hist.items()) / total
        return dict(mean=mean, n_runs=total, counts={str(k): v for k, v in sorted(hist.items())})

    def finalize(self, e_comp_closed=None):
        """e_comp_closed: optional per-policy closed-form overall E_comp（dryrun 对拍）。"""
        km2_mean = self.km2_sum / max(self.n_items, 1)
        out = dict(
            n_items=self.n_items,
            exact_k_violations=self.exact_k_violations,
            kp_range_violations=self.kp_range_violations,
            confound=dict(
                mask_run_length=self._hist_summary(self.mask_run_hist),
                visible_run_length=self._hist_summary(self.vis_run_hist),
                transition_count_mean=self.transition_total / max(self.n_items, 1),
                pair_distance=dict(
                    counts={str(k): v for k, v in sorted(self.pair_dist_hist.items())},
                    buckets=self.pair_dist_buckets,
                ),
                lr_counts=dict(left_only=self.lr_left_only, right_only=self.lr_right_only),
                rel_pos_mask_rate=[float(self.rel_pos_mask[i].item() / max(self.rel_pos_n[i].item(), 1))
                                   for i in range(50)],
                context_visible=self._hist_summary(self.ctx_vis_hist),
                nt_cov=dict(
                    bins=NT_COV_BINS, lo=NT_COV_LO, hi=NT_COV_HI,
                    hist=[float(x) for x in self.nt_cov_hist.tolist()],
                    n_pairs=self.nt_cov_n,
                ),
            ),
            e_comp=dict(
                overall=self.e_comp_sum / max(self.n_items, 1),
                closed_form=e_comp_closed,
                by_k_ratio=[dict(mean=self.e_comp_by_kr[b] / max(self.e_comp_by_kr_n[b], 1),
                                 n=self.e_comp_by_kr_n[b]) for b in range(4)],
                by_sigma_quantiles=None,  # 由 e_comp_items/e_comp_sigma 离线分桶（见下）
                per_item=dict(items=[float(x) for x in self.e_comp_items],
                              sigma=[float(x) for x in self.e_comp_sigma]),
            ),
            covariance=dict(
                treated_edges=dict(
                    mean_both=self.treated_both_sum / max(self.treated_edge_n, 1),
                    n_edges=self.treated_edge_n,
                    cbar=(self.treated_both_sum / max(self.treated_edge_n, 1)) - km2_mean,
                ),
                active_pairs=dict(
                    mean_both=self.active_both_sum / max(self.active_edge_n, 1),
                    n_edges=self.active_edge_n,
                    cbar=(self.active_both_sum / max(self.active_edge_n, 1)) - km2_mean,
                ),
                km2_mean=km2_mean,
            ),
            degenerate_k=dict(
                counts=self.degenerate_counts,
                e_comp_over_degenerate=(
                    self.degenerate_e_sum / max(self.degenerate_e_items, 1)),
            ),
            singleton=dict(
                counts=[int(x) for x in self.singleton_counts.tolist()],
                n=self.singleton_n,
            ),
        )
        return out


# --------------------------------------------------------------------------
# 闭式参考（§8 / §7.4，dryrun 对拍用）
# --------------------------------------------------------------------------

def closed_form_e_comp(m, K, policy):
    """per-item 闭式 E_comp（给定 m, K）。even: H=b_max/n, L=(K mod 2)/n, U=2K(m−K)/(m(m−1))。
    odd: H=E_s[min(K−s, m−1−K+s)]/n 等。"""
    k = int(K)
    if m % 2 == 0:
        n = m // 2
        if policy == "U":
            return 2 * k * (m - k) / (m * (m - 1))
        b = min(k, m - k) if policy == "H" else (k % 2)
        return b / n
    n = (m - 1) // 2
    if policy == "U":
        return 2 * k * (m - k) / (m * (m - 1))
    q = k / m
    vals = []
    for s in (0, 1):
        p = (1 - q) if s == 0 else q
        if p == 0:
            continue
        kp = k - s
        if not (0 <= kp <= m - 1):
            continue
        b = min(kp, m - 1 - kp) if policy == "H" else (kp % 2)
        vals.append(p * b)
    return sum(vals) / n


def closed_form_cbar_active(m, K, policy):
    """active-pair C̄ 闭式（§7.4）。"""
    k = int(K)
    km = k / m
    if m % 2 == 0:
        n = m // 2
        if policy == "U":
            a = k * (k - 1) / (m * (m - 1)) * n  # E[both-masked active pairs]
        else:
            b = min(k, m - k) if policy == "H" else (k % 2)
            a = (k - b) / 2
        return a / n - km * km
    n = (m - 1) // 2
    if policy == "U":
        a = k * (k - 1) / (m * (m - 1)) * n
        return a / n - km * km
    q = k / m
    ea = 0.0
    for s in (0, 1):
        p = (1 - q) if s == 0 else q
        if p == 0:
            continue
        kp = k - s
        if not (0 <= kp <= m - 1):
            continue
        b = min(kp, m - 1 - kp) if policy == "H" else (kp % 2)
        a = (kp - b) / 2
        ea += p * a
    return ea / n - km * km
