"""Option B dense trajectory 分析（用户 2026-10-07 §6/§7 gate，只读）。

对每个 seed：13 个 dense checkpoint + step0 的完整 metric trajectory，
窗口：0–250 / 250–750 / 750–1020 / 1020–2500（active = 250–750，plateau = 1020–2500）。
输出：per-checkpoint trajectory、相邻 Δ、窗口 slope、bootstrap CI、两 seed 方向一致性。

预注册 gate（用户 §7）：
- Case A：CPI 下降与 CE 改善同窗、结构量（JS_dep/D_ab）无可复现 transition
  → generic conditional estimation improvement = 最简解释；项目转 empirical framing
- Case B：非 δ 派生结构量在两 seed 方向一致 + 集中于 active window + 1020 后
  plateau + 不能明显由 CE 单独解释 → candidate mechanism（才允许 Phase 2C 设计）
  注（2026-10-08 用户裁定）："不能被 CE/NLL 单独解释" 判据未实现前，脚本只输出
  B_candidate_unadjudicated，不输出正式 Case B；最终 Case 由人工 adjudication 定。
- Case C：无复现结构信号 → mechanism-screening HARD STOP

注意：PMI_forward−PMI_reverse≡δ（恒等），PMI 收敛不作独立 evidence；
JS/D_ab 对 |δ| 的 CV 预测力仍只是 secondary descriptive。

用法: .venv/bin/python scripts/optionb_dense_analysis.py
输出: results/phase2/optionb_dense_analysis.json
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "phase2")
PAIRS = os.path.join(OUT, "dense_pairs")
STEPS = [50, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1020, 1500, 2500]
WINDOWS = [(0, 250), (250, 750), (750, 1020), (1020, 2500)]
N_BOOT = 10000
BOOT_SEED = 0


def j(path):
    with open(path) as f:
        return json.load(f)


def boot_ci(x, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_checkpoint(cid):
    ps = j(os.path.join(PAIRS, f"{cid}.json"))["per_sample"]
    d = np.array(ps["delta"])
    dab = np.array(ps["d_ab"])
    pmi_f = dab + d / 2
    pmi_r = dab - d / 2
    asym = np.abs(d) / (np.abs(pmi_f) + np.abs(pmi_r) + 1e-9)
    ad = np.abs(d)
    return dict(
        delta_abs=dict(mean=ad.mean(), ci=list(boot_ci(ad))),
        delta_mean=d.mean(),
        js_dep=dict(mean=np.array(ps["js_dep"]).mean(),
                    ci=list(boot_ci(np.array(ps["js_dep"])))),
        d_ab=dict(mean=dab.mean(), ci=list(boot_ci(dab))),
        pmi_forward=dict(mean=pmi_f.mean(), ci=list(boot_ci(pmi_f))),
        pmi_reverse=dict(mean=pmi_r.mean(), ci=list(boot_ci(pmi_r))),
        ce=dict(mean=np.array(ps["local_ce"]).mean(),
               ci=list(boot_ci(np.array(ps["local_ce"])))),
        a_sym=dict(mean=asym.mean()),
    )


def slope(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2 or np.ptp(xs) == 0:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def main():
    step0 = per_checkpoint("s0-step0") if os.path.exists(
        os.path.join(PAIRS, "s0-step0.json")) else None
    # step0 复用 pretrained pairs（若 dense_pairs 无，从 phase2/pairs 读）
    if step0 is None:
        ps = j(os.path.join(ROOT, "results", "phase2", "pairs", "pretrained-step0.json"))["per_sample"]
        d = np.array(ps["delta"])
        dab = np.array(ps["d_ab"])
        pmi_f = dab + d / 2
        pmi_r = dab - d / 2
        ad = np.abs(d)
        step0 = dict(
            delta_abs=dict(mean=ad.mean(), ci=list(boot_ci(ad))),
            delta_mean=d.mean(),
            js_dep=dict(mean=np.array(ps["js_dep"]).mean(),
                        ci=list(boot_ci(np.array(ps["js_dep"])))),
            d_ab=dict(mean=dab.mean(), ci=list(boot_ci(dab))),
            pmi_forward=dict(mean=pmi_f.mean(), ci=list(boot_ci(pmi_f))),
            pmi_reverse=dict(mean=pmi_r.mean(), ci=list(boot_ci(pmi_r))),
            ce=dict(mean=np.array(ps["local_ce"]).mean(),
                   ci=list(boot_ci(np.array(ps["local_ce"])))),
            a_sym=dict(mean=(np.abs(d) / (np.abs(pmi_f) + np.abs(pmi_r) + 1e-9)).mean()),
        )

    traj = {"s1": {"0": step0}, "s2": {"0": step0}}
    for seed in (1, 2):
        for st in STEPS:
            traj[f"s{seed}"][str(st)] = per_checkpoint(f"s{seed}-{st}")

    metrics = ("delta_abs", "js_dep", "d_ab", "pmi_forward", "pmi_reverse", "ce", "a_sym")
    window_means = {}
    window_slopes = {}
    consistency = {}
    for (lo, hi) in WINDOWS:
        wname = f"{lo}–{hi}"
        xs = [s for s in [0] + STEPS if lo < s <= hi] if lo == 0 else \
             [s for s in STEPS if lo < s <= hi]
        if lo == 0:
            xs = [0] + [s for s in STEPS if s <= hi]
        wm, ws = {}, {}
        for seed in (1, 2):
            t = traj[f"s{seed}"]
            for k in metrics:
                vals = [t[str(s)][k]["mean"] for s in xs if str(s) in t]
                wm.setdefault(k, {}).setdefault(f"s{seed}", []).append(
                    (vals[-1] - vals[0]) if len(vals) >= 2 else None)
                ws.setdefault(k, {}).setdefault(f"s{seed}", []).append(
                    slope([s for s in xs if str(s) in t][:len(vals)], vals))
        window_means[wname] = {k: {s: v[0] for s, v in d.items()} for k, d in wm.items()}
        window_slopes[wname] = {k: {s: v[0] for s, v in d.items()} for k, d in ws.items()}
        consistency[wname] = {
            k: {s: np.sign(v) if v is not None else 0
                for s, v in d.items()}
            for k, d in window_means[wname].items()}

    # 两 seed 方向一致性（每窗口每 metric）
    agree = {}
    for wname in window_means:
        agree[wname] = {}
        for k in metrics:
            s1 = window_means[wname][k].get("s1")
            s2 = window_means[wname][k].get("s2")
            if s1 is None or s2 is None:
                agree[wname][k] = None
            else:
                agree[wname][k] = bool(np.sign(s1) == np.sign(s2) != 0)

    # 相邻 Δ（per seed，主要窗口）
    deltas = {}
    for seed in (1, 2):
        t = traj[f"s{seed}"]
        seq = [0] + STEPS
        rows = []
        for a, b in zip(seq[:-1], seq[1:]):
            if str(a) not in t or str(b) not in t:
                continue
            rows.append({k: t[str(b)][k]["mean"] - t[str(a)][k]["mean"] for k in metrics})
        deltas[f"s{seed}"] = rows

    # ---- Case 判定（用户 §7 gate）----
    aw = window_means["250–750"]
    pl = window_means["1020–2500"]
    early = window_means["0–250"]

    def m(k, seed):
        return aw[k].get(seed)

    cpi_active_down = all(m("delta_abs", s) is not None and m("delta_abs", s) < 0
                          for s in ("s1", "s2"))
    ce_active_down = all(m("ce", s) is not None and m("ce", s) < 0 for s in ("s1", "s2"))
    js_active = all(m("js_dep", s) is not None for s in ("s1", "s2"))
    js_consistent = agree["250–750"]["js_dep"] if agree["250–750"].get("js_dep") is not None else False
    dab_active = all(m("d_ab", s) is not None for s in ("s1", "s2"))
    dab_consistent = agree["250–750"]["d_ab"] if agree["250–750"].get("d_ab") is not None else False
    plateau = all(abs(pl["delta_abs"].get(s) or 0) < 0.5 * abs(m("delta_abs", s) or 1e9)
                  for s in ("s1", "s2"))
    structural_win = (js_consistent and abs(m("js_dep", "s1")) > 0.001) or \
                     (dab_consistent and abs(m("d_ab", "s1")) > 0.001)

    # 2026-10-08 用户裁定：§7 Case B 完整标准含 "不能被 CE/NLL 单独解释"。
    # 该判据（CE-adjusted independence）尚未实现 → 脚本禁止输出正式 Case B，
    # 只能输出 B_candidate_unadjudicated；最终 Case 由人工 adjudication 给出
    # （本 run 的人工裁定：results/phase2/optionb_case_adjudication.md）。
    case_b_full_criteria_met = False
    case_b_missing_criterion = "independence_from_ce"

    if cpi_active_down and ce_active_down and not structural_win:
        case = ("A（generic estimation）：CPI 下降与 CE 改善同窗（250–750），"
                "JS_dep/D_ab 无两 seed 一致的可复现 transition → 最简解释 = "
                "generic conditional estimation improvement；项目转 empirical/diagnostic framing")
        case_label = "A"
    elif cpi_active_down and structural_win:
        case = ("B_candidate_unadjudicated：非 δ 派生结构量方向一致 + 集中于 active window，"
                "但 CE-adjusted independence 未检验（case_b_full_criteria_met=false, "
                "missing_criterion=independence_from_ce）→ 不得视为正式 Case B；"
                "最终 Case 以人工 adjudication 为准")
        case_label = "B_candidate_unadjudicated"
    else:
        case = ("C（no replicated structural signal）：mechanism-screening HARD STOP；"
                "保留 'partial-reveal SFT reproducibly improves CPI/OrderGap，但当前 "
                "tested mechanisms 均不能解释该变化' 的 empirical result")
        case_label = "C"

    out = dict(
        note="Option B dense trajectory 分析（gate 见用户 2026-10-07 §7）",
        trajectory={s: {k: {kk: (vv["mean"] if isinstance(vv, dict) else vv)
                            for kk, vv in v.items()}
                        for k, v in t.items()}
                    for s, t in traj.items()},
        window_means=window_means,
        window_slopes=window_slopes,
        two_seed_agreement=agree,
        adjacent_deltas=deltas,
        case=case,
        case_label=case_label,
        case_b_full_criteria_met=case_b_full_criteria_met,
        case_b_missing_criterion=case_b_missing_criterion,
        case_components=dict(
            cpi_active_down=cpi_active_down, ce_active_down=ce_active_down,
            js_consistent=js_consistent, dab_consistent=dab_consistent,
            plateau=plateau),
        caveats=[
            "PMI_forward−PMI_reverse≡δ（恒等）——PMI 收敛不作独立 mechanism evidence",
            "JS/D_ab 的 CV 预测力仍只是 secondary descriptive evidence",
            "0–250 窗口的 '0' 点用 pretrained step0（全体 run 共享 init）",
        ])
    with open(os.path.join(OUT, "optionb_dense_analysis.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("=" * 78)
    print("Option B dense trajectory 分析")
    for (lo, hi) in WINDOWS:
        w = window_means[f"{lo}–{hi}"]
        print(f"  window {lo}–{hi}: " + ", ".join(
            f"{k}: s1={w[k].get('s1') and round(w[k]['s1'],4)}/s2={w[k].get('s2') and round(w[k]['s2'],4)}"
            for k in metrics))
    print(f"  两 seed 方向一致（250–750）: "
          f"|δ|↓={agree['250–750']['delta_abs']} CE↓={agree['250–750']['ce']} "
          f"JS={agree['250–750']['js_dep']} D_ab={agree['250–750']['d_ab']}")
    print(f"\n  CASE: {case}")
    print(f"\nsaved: {os.path.join(OUT, 'optionb_dense_analysis.json')}")


if __name__ == "__main__":
    main()
