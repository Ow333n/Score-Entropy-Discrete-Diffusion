"""v2.1 mechanism pilot dry-run / preflight（CPU，无模型；protocol v2.1 §14 步骤 4 / P3）。

重放 2 replicates × 3 policies 的完整 2500-step 共享流（与训练同 pattern：
每步 draw(micro_batch)），逐 policy 执行 mask 选择路径，验证：

1. exact-K：每 item |M| == K（0 违反）；odd-m K−s ∈ [0, m−1] 可行性断言
2. 共享性：同 replicate 内 U/H/L 的 v21 schedule digest（σ/dσ/span/K/phase/coin）
   逐位一致；且与 v1.2 dryrun 的四元组 digest 一致（共享四元流沿用 v1.2 约定）
3. E_comp 实测 vs 闭式（overall + K/m 桶 + σ 分桶）；non-degenerate K 严格 H>U>L；
   degenerate K∈{0,1,m−1,m} 单独报告（三 policy 应相等）
4. C̄_treated 实测排序（treated edges 主口径 + active pairs 补口径）
5. 单位置 marginal parity（相对位置 mask 率表 ≈ 经验 mean(K/m)）、L/R ≈ 0
6. §7.6 七项 confound diagnostics 逐 policy 产出
7. singleton 硬币流统计（odd m：消耗次数、s=1 率 ≈ mean(K/m)）

输出 results/mechanism_pilot_v21_dryrun/：
  dryrun_summary.json（训练 end-guard 参考：per-replicate v21 digest +
  per-policy E_comp/C̄ 参考值）+ diagnostics_<policy>_r<r>.json（全量诊断）

用法: .venv/bin/python scripts/v21_dryrun.py
"""
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import initialize, compose

from data import get_dataset
from task_data.policy_corruption import make_generator as _v12_make_generator
from task_data.v21_policy import (MaskDiagnostics, V21Schedule, closed_form_e_comp,
                                  load_maps_verified, select_mask_v21)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "mechanism_pilot_v21_dryrun")
POLICIES = ("U", "H", "L")
N_STEPS = 2500
PREFIX_STEPS = (100, 500, 1000, 1500, 2000, 2500)   # smoke/正式 run 的 end-guard 参考

POLICY_STREAM_NAMES = {
    "U": ("u_uniform",),
    "H": ("h_pair", "h_orient"),
    "L": ("l_pair", "l_orient"),
}


def replay(r, policy, micro_batch, N_chunks, cfg, maps):
    """重放 replicate r 的共享流并执行 policy 的 mask 路径。

    与训练严格同 consumption pattern：每步 draw(micro_batch)、每步 randperm(N_chunks)。
    返回 (five_tuple_digest, v21_digest, diag_final, misc)。
    """
    from task_data.v21_policy import make_generator
    g_data = _v12_make_generator("data_order", r)
    sched = V21Schedule(r, cfg.data.seq_len, cfg.data.span_min, cfg.data.span_max, noise=None)
    gens = {n: make_generator(n, r) for n in POLICY_STREAM_NAMES[policy]}
    rep_maps = maps[f"rep{r}"]

    diag = MaskDiagnostics()
    five_tuple = hashlib.sha256()
    coin_draws = 0
    coin_ones = 0
    cf_sum = 0.0     # E_comp 闭式（逐 item 按 realized (m,K) 平均）
    prefixes = {}    # step -> (digest, e_comp, cbar_trt)（smoke/正式 run 的 end-guard 参考）

    def pair_map_for(m):
        return rep_maps["even" if m % 2 == 0 else "odd"][str(m)]

    n_done = 0
    for _ in range(N_STEPS):
        sample_ids = torch.randperm(N_chunks, generator=g_data)[:micro_batch]
        sigma, dsigma, span_len, span_start, K, phase, coin = sched.draw(micro_batch)
        five_tuple.update(sample_ids.numpy().tobytes())
        for arr in (sigma, dsigma, span_len, span_start, K, phase, coin):
            five_tuple.update(arr.numpy().tobytes())
        for j in range(micro_batch):
            m = int(span_len[j].item())
            k = int(K[j].item())
            s0 = int(span_start[j].item())
            pm = pair_map_for(m)
            c = float(coin[j].item()) if m % 2 == 1 else None
            if m % 2 == 1:
                coin_draws += 1
                coin_ones += 1 if c < k / m else 0
            sel = select_mask_v21(s0, m, k, policy, cfg.data.seq_len, pm,
                                  phase=int(phase[j].item()), coin=c, gens=gens)
            if len(sel) != k:
                diag.exact_k_violations += 1
            diag.update({p - s0 for p in sel}, m, k, float(sigma[j].item()), pm,
                        int(phase[j].item()) if m % 2 == 1 else None)
            cf_sum += closed_form_e_comp(m, k, policy)
            n_done += 1
        if _ + 1 in PREFIX_STEPS:
            prefixes[str(_ + 1)] = dict(
                digest=sched.hexdigest(),
                e_comp=diag.e_comp_sum / n_done,
                cbar_treated=(diag.treated_both_sum / diag.treated_edge_n
                              - diag.km2_sum / n_done),
            )

    n_items = N_STEPS * micro_batch
    return five_tuple.hexdigest(), sched.hexdigest(), sched.sched.hexdigest(), \
        diag, prefixes, dict(coin_draws=coin_draws, coin_ones=coin_ones,
                             e_comp_closed_overall=cf_sum / n_items)


def main():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")

    micro_batch = cfg.training.batch_size // (cfg.ngpus * cfg.training.accum)
    assert micro_batch == 32, micro_batch
    train_ds = get_dataset("wikitext103", "train", cache_dir=cfg.data.cache_dir,
                           block_size=cfg.data.seq_len, num_proc=4)
    N_chunks = len(train_ds)
    maps = load_maps_verified(os.path.join(ROOT, cfg.mechanism_v21.maps_dir))

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"重放规模：{N_STEPS} 步 × {micro_batch} items = {N_STEPS * micro_batch} items/run"
          f"（× 3 policy × 2 replicate）")

    summary = dict(protocol="mechanism_complementary_exposure_v2_1",
                   preflight="2026-10-06", n_steps=N_STEPS, micro_batch=micro_batch,
                   n_items=N_STEPS * micro_batch, N_train_chunks=N_chunks,
                   seq_len=cfg.data.seq_len, span_min=cfg.data.span_min,
                   span_max=cfg.data.span_max, replicates={})

    for r in (1, 2):
        v21_digests, five_digests, e_comp_meas, e_comp_closed = {}, {}, {}, {}
        cbar_trt, cbar_act = {}, {}
        prefixes_by_policy = {}
        for policy in POLICIES:
            five, v21d, quad, diag, prefixes, misc = replay(
                r, policy, micro_batch, N_chunks, cfg, maps)
            v21_digests[policy] = v21d
            five_digests[policy] = five
            prefixes_by_policy[policy] = prefixes
            out = diag.finalize(e_comp_closed=misc["e_comp_closed_overall"])
            out["shared_v21_digest"] = v21d
            out["shared_quad_digest"] = quad
            out["coin_stream"] = dict(draws=misc["coin_draws"],
                                      s1_rate=misc["coin_ones"] / max(misc["coin_draws"], 1))
            with open(os.path.join(OUT_DIR, f"diagnostics_{policy}_r{r}.json"), "w") as f:
                json.dump(out, f, indent=2)
            e_comp_meas[policy] = out["e_comp"]["overall"]
            e_comp_closed[policy] = out["e_comp"]["closed_form"]
            cbar_trt[policy] = out["covariance"]["treated_edges"]["cbar"]
            cbar_act[policy] = out["covariance"]["active_pairs"]["cbar"]
            print(f"r={r} {policy}: v21_digest={v21d[:16]}… five={five[:16]}… "
                  f"E_comp={e_comp_meas[policy]:.4f} (闭式 {e_comp_closed[policy]:.4f}) "
                  f"C̄_trt={cbar_trt[policy]:+.4f} C̄_act={cbar_act[policy]:+.4f} "
                  f"exactK_viol={out['exact_k_violations']} "
                  f"coin_s1={out['coin_stream']['s1_rate']:.4f}")

        shared_ok = len(set(v21_digests.values())) == 1
        e_order = e_comp_meas["H"] > e_comp_meas["U"] > e_comp_meas["L"]
        c_order = cbar_trt["H"] < cbar_trt["U"] < cbar_trt["L"]
        c_order_act = cbar_act["H"] < cbar_act["U"] < cbar_act["L"]
        # E_comp 实测 vs 闭式：even-m item 逐 item 精确相等；odd-m item 的实测用
        # 硬币实际值、闭式为 E_s 期望 → 差 ~sd/√n ≈ 0.003；U 为统计量。
        # 统一容差 0.01（even 精确性已由 unit tests test_f 覆盖）。
        e_cf_ok = all(abs(e_comp_meas[p] - e_comp_closed[p]) < 0.01 for p in POLICIES)
        print(f"r={r}: 共享 v21 digest U==H==L: {shared_ok} | "
              f"E_comp H>U>L: {e_order} ({e_comp_meas}) | 实测==闭式: {e_cf_ok} | "
              f"C̄_trt H<U<L: {c_order} ({cbar_trt}) | C̄_act H<U<L: {c_order_act}")

        # v1.2 四元组 digest 连续性检查（共享四元流沿用 v1.2 约定）
        v12_ref = json.load(open(os.path.join(
            ROOT, "results", "mechanism_pilot_dryrun", "dryrun_summary.json")))
        v12_quad = v12_ref["replicates"][str(r)]["shared_schedule_sha256"]
        quad_ok = all(
            json.load(open(os.path.join(OUT_DIR, f"diagnostics_{p}_r{r}.json")))
            ["shared_quad_digest"] == v12_quad for p in POLICIES)
        print(f"r={r}: 四元组 digest 与 v1.2 dryrun 一致（共享流沿用约定）: {quad_ok}")

        # prefix 参考（smoke/正式 run 的 end-guard）：digest 为共享（取 U），E_comp/C̄ 按 policy
        prefixes = {}
        for s in (str(x) for x in PREFIX_STEPS):
            prefixes[s] = dict(
                shared_digest=prefixes_by_policy["U"][s]["digest"],
                e_comp={p: round(prefixes_by_policy[p][s]["e_comp"], 9) for p in POLICIES},
                cbar_treated={p: round(prefixes_by_policy[p][s]["cbar_treated"], 9)
                              for p in POLICIES},
            )

        summary["replicates"][str(r)] = dict(
            shared_v21_digest=v21_digests["U"],
            v21_digests_equal=shared_ok,
            five_tuple_digests_equal=len(set(five_digests.values())) == 1,
            quadruple_digest_matches_v12=quad_ok,
            e_comp_measured={k: round(v, 9) for k, v in e_comp_meas.items()},
            e_comp_closed={k: round(v, 9) for k, v in e_comp_closed.items()},
            e_comp_measured_equals_closed=e_cf_ok,
            cbar_treated={k: round(v, 9) for k, v in cbar_trt.items()},
            cbar_active={k: round(v, 9) for k, v in cbar_act.items()},
            e_comp_ordering_ok=e_order,
            cbar_ordering_ok=c_order,
            cbar_active_ordering_ok=c_order_act,
            prefixes=prefixes,
        )

    with open(os.path.join(OUT_DIR, "dryrun_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary: {os.path.join(OUT_DIR, 'dryrun_summary.json')}")


if __name__ == "__main__":
    main()
