"""Stage-4 stop/go gate (protocol §11 + regime_a_protocol.yaml 预注册判据)。

预注册判据 (结果前冻结, 2026-10-01):
  - primary = EMA 权重, late checkpoint (10200)
  - ΔCPI stable: 每 seed 独立, paired per-sample Δ|δ| 的 bootstrap 95% CI (10k) 不含 0,
    且两个 seed 符号一致
  - ΔOrderGap stable: 同上, 用 OrderGap_raw
  - co-movement: 每 seed 独立, sign(CPI_late−CPI_early)==sign(OG_late−OG_early)
    且两者的 paired bootstrap CI 均不含 0; 两个 seed 判断一致
  - gate: A_HARD_STOP (ΔCPI 不 stable) / B_DIAGNOSTIC_ONLY (stable 但联动不成立)
    / C_PROCEED (stable 且联动成立)
  - seed 方向相反 → unstable, 不加 seed

用法: .venv/bin/python evaluation/stage4_gate.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
N_BOOT = 10_000
SEEDS = ["s1", "s2"]


def load(path):
    with open(path) as f:
        return json.load(f)


def paired_bootstrap_ci(diff, n_boot=N_BOOT, seed=0):
    """paired bootstrap 95% CI of mean(diff), 重采样样本索引。"""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = diff[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), means


def main():
    pre_cpi = load(os.path.join(R, "pretrained/cpi.json"))
    pre_og = load(os.path.join(R, "pretrained/order_gap.json"))
    pre_delta_abs = np.asarray(pre_cpi["per_sample"]["delta_abs"])
    pre_og_raw = np.asarray(pre_og["per_sample"]["order_gap_raw"])
    n = len(pre_delta_abs)

    print("=" * 68)
    print("Stage-4 stop/go gate (protocol §11, 预注册判据)")
    print(f"样本数 N={n}; pretrained 基线: CPI_abs={pre_cpi['summary']['cpi_abs']:.4f}, "
          f"OrderGap_raw={pre_og['order_gap_raw']:.3f}")

    results = {}
    for seed in SEEDS:
        late_cpi = load(os.path.join(R, f"vanilla/cpi_{seed}_10200.json"))
        late_og = load(os.path.join(R, f"vanilla/og_{seed}_10200.json"))
        early_cpi = load(os.path.join(R, f"vanilla/cpi_{seed}_1020.json"))
        early_og = load(os.path.join(R, f"vanilla/og_{seed}_1020.json"))

        # paired per-sample 差 (manifest 同一顺序, 按索引对齐)
        d_abs = np.asarray(late_cpi["per_sample"]["delta_abs"]) - pre_delta_abs
        d_og = np.asarray(late_og["per_sample"]["order_gap_raw"]) - pre_og_raw
        # co-movement 分量 (early→late 轨迹)
        cpi_el = (np.asarray(late_cpi["per_sample"]["delta_abs"])
                  - np.asarray(early_cpi["per_sample"]["delta_abs"]))
        og_el = (np.asarray(late_og["per_sample"]["order_gap_raw"])
                 - np.asarray(early_og["per_sample"]["order_gap_raw"]))

        lo, hi, _ = paired_bootstrap_ci(d_abs)
        lo_og, hi_og, _ = paired_bootstrap_ci(d_og)
        lo_c, hi_c, _ = paired_bootstrap_ci(cpi_el)
        lo_o, hi_o, _ = paired_bootstrap_ci(og_el)

        cpi_stable = (lo < 0 < hi) is False and np.sign(d_abs.mean()) != 0
        og_stable = (lo_og < 0 < hi_og) is False and np.sign(d_og.mean()) != 0
        co_move = (np.sign(cpi_el.mean()) == np.sign(og_el.mean())
                   and (lo_c < 0 < hi_c) is False and (lo_o < 0 < hi_o) is False
                   and cpi_el.mean() != 0 and og_el.mean() != 0)

        results[seed] = dict(
            cpi_late=late_cpi["summary"]["cpi_abs"],
            cpi_early=early_cpi["summary"]["cpi_abs"],
            og_late=late_og["order_gap_raw"],
            og_early=early_og["order_gap_raw"],
            d_cpi_abs_mean=float(d_abs.mean()), d_cpi_abs_ci=[lo, hi],
            d_og_mean=float(d_og.mean()), d_og_ci=[lo_og, hi_og],
            cpi_early_late_mean=float(cpi_el.mean()), cpi_early_late_ci=[lo_c, hi_c],
            og_early_late_mean=float(og_el.mean()), og_early_late_ci=[lo_o, hi_o],
            cpi_stable=bool(cpi_stable), og_stable=bool(og_stable),
            co_movement=bool(co_move),
        )
        print(f"\n[{seed}] CPI late={late_cpi['summary']['cpi_abs']:.4f} "
              f"(early={early_cpi['summary']['cpi_abs']:.4f}, pre={pre_cpi['summary']['cpi_abs']:.4f})")
        print(f"  Δ|δ|_abs mean={d_abs.mean():+.4f}  CI95=[{lo:+.4f}, {hi:+.4f}] → "
              f"CPI stable={cpi_stable}")
        print(f"  OG late={late_og['order_gap_raw']:.3f} (early={early_og['order_gap_raw']:.3f}, "
              f"pre={pre_og['order_gap_raw']:.3f}); ΔOG mean={d_og.mean():+.3f} "
              f"CI95=[{lo_og:+.3f}, {hi_og:+.3f}] → OG stable={og_stable}")
        print(f"  co-movement: sign(ΔCPI_el)={np.sign(cpi_el.mean()):+.0f} "
              f"sign(ΔOG_el)={np.sign(og_el.mean()):+.0f} CI(ΔCPI_el)=[{lo_c:+.4f},{hi_c:+.4f}] "
              f"CI(ΔOG_el)=[{lo_o:+.3f},{hi_o:+.3f}] → {co_move}")

    # gate 判定
    s1, s2 = results["s1"], results["s2"]
    cpi_stable_both = s1["cpi_stable"] and s2["cpi_stable"] and \
        np.sign(s1["d_cpi_abs_mean"]) == np.sign(s2["d_cpi_abs_mean"])
    og_stable_both = s1["og_stable"] and s2["og_stable"] and \
        np.sign(s1["d_og_mean"]) == np.sign(s2["d_og_mean"])
    co_move_both = s1["co_movement"] and s2["co_movement"]

    unstable = (not cpi_stable_both and (s1["cpi_stable"] or s2["cpi_stable"])) or \
               (not og_stable_both and (s1["og_stable"] or s2["og_stable"])) or \
               (not co_move_both and (s1["co_movement"] or s2["co_movement"]))

    if not cpi_stable_both:
        verdict = "A_HARD_STOP"
    elif not co_move_both:
        verdict = "B_DIAGNOSTIC_ONLY"
    else:
        verdict = "C_PROCEED"
    if unstable:
        verdict += " + UNSTABLE(seed 分歧)"

    print("\n" + "=" * 68)
    print(f"GATE VERDICT: {verdict}")
    if "A_HARD_STOP" in verdict:
        print("→ 停止 Swap 方法路线, 转诊断论文 (protocol §11 Outcome A/E)")
    elif "B_DIAGNOSTIC_ONLY" in verdict:
        print("→ compatibility dynamics 是现象, 机制声明弱化 (§11 Outcome B)")
    else:
        print("→ 进入 PAPL/Swap 路线 (§11 Outcome C)")

    with open(os.path.join(R, "vanilla/stage4_gate.json"), "w") as f:
        json.dump(dict(verdict=verdict, seeds=results), f, indent=2)
    print("结果: results/vanilla/stage4_gate.json")


if __name__ == "__main__":
    main()
