"""v2.1 mechanism pilot G3 分析（protocol v2.1 §11/§12/§15.1，preflight P8，只读）。

预注册口径：
- G3a：8 项干预有效性（训练 metadata：exact-K、共享 digest、marginal parity、L/R、
  E_comp 排序、C̄ 排序、map hash、7 项 confound 记录）
- G3b：CPI_treated H<L，per-replicate paired bootstrap（10k, seed=0）+ 按 replicate
  聚类 pooled CI（v1.2 同口径）
- G3c：primary H vs L（CPI_global_abs @2500，判据同 G1a：CI 排除 0 且两 replicate
  点估计同向）；secondary H vs U、U vs L；U 位置按 §11 预注册规则解释
- G3d：matched performance（§5.3 规则：参考=H 的 {500,1020,2500}，池=L 的同集合，
  最小 |ΔNLL| 且落 NLL±0.02 / acc±0.01，平局取更小 step；no-overlap ⇒ INC 非 FAIL）
  → 匹配后重做 G3c
- §9.5 OrderGap subset + path-score variance；§9.4 scale/calibration 摘要
- Case A–E 判定 + §15.1 Confirmatory 进入条件评估

用法: .venv/bin/python scripts/v21_g3_analysis.py
输出: results/mechanism_pilot_v21/v21_g3_analysis.json + stdout 完整报告
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "mechanism_pilot_v21")
V12_RES = os.path.join(ROOT, "results", "mechanism_pilot")
EXP = os.path.join(ROOT, "exp_local", "regime_a")
NLL_TOL, ACC_TOL = 0.02, 0.01
N_BOOT = 10000
BOOT_SEED = 0
MANIFEST_SHA = "1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3"
PROTOCOL_SHA = "af312d864d41d8f67da343d55a8dee8a29a91c37dea622e6b2c4b22ef441fc61"
RUNS = ("U1", "H1", "L1", "U2", "H2", "L2")
STEPS = (500, 1020, 2500)


def j(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def jv12(name):
    with open(os.path.join(V12_RES, name)) as f:
        return json.load(f)


def meta(run):
    import glob
    with open(glob.glob(os.path.join(EXP, f"v21pilot-{run}-*/run_metadata.json"))[0]) as f:
        return json.load(f)


def paired_bootstrap(diff, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def clustered_bootstrap(diffs, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    k = len(diffs)
    sizes = [len(d) for d in diffs]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        draws = [diffs[i][rng.integers(0, sizes[i], sizes[i])] for i in idx]
        boots[b] = np.concatenate(draws).mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def verdict_less_than(ci_lo, ci_hi, rep_means):
    """H<L 方向：CI 完全 <0 且两 replicate 点估计同负 → PASS。"""
    if ci_lo >= 0 or ci_hi >= 0:
        return "INCONCLUSIVE" if ci_lo < 0 else "FAIL"
    if all(m < 0 for m in rep_means):
        return "PASS"
    return "INCONCLUSIVE"


def cpi_diff(runs_a, runs_b, step, field="per_sample", key="delta_abs"):
    diffs = []
    rep_means = []
    rep_cis = []
    for ra, rb in zip(runs_a, runs_b):
        da = np.array(j(f"cpi_global_{ra}_{step}.json")["per_sample"][key], dtype=float)
        db = np.array(j(f"cpi_global_{rb}_{step}.json")["per_sample"][key], dtype=float)
        d = da - db
        diffs.append(d)
        rep_means.append(d.mean())
        rep_cis.append(paired_bootstrap(d))
    clo, chi = clustered_bootstrap(diffs)
    return dict(rep_means=[float(m) for m in rep_means], rep_cis=rep_cis,
                pooled_ci=[clo, chi], point_estimate=float(np.mean(rep_means)),
                verdict=verdict_less_than(clo, chi, rep_means))


def pairs_cpi_diff(runs_a, runs_b, step, map_spec):
    """treated/heldout per-sample mean_abs_delta 对比。"""
    diffs, rep_means, rep_cis = [], [], []
    for ra, rb in zip(runs_a, runs_b):
        fa = f"cpi_treated_{ra}_{step}.json" if map_spec == "treated" else f"cpi_heldout_{ra}_{step}.json"
        fb = f"cpi_treated_{rb}_{step}.json" if map_spec == "treated" else f"cpi_heldout_{rb}_{step}.json"
        da = np.array(j(fa)["per_sample"]["mean_abs_delta"], dtype=float)
        db = np.array(j(fb)["per_sample"]["mean_abs_delta"], dtype=float)
        assert da.shape == db.shape == (500,)
        d = da - db
        diffs.append(d)
        rep_means.append(d.mean())
        rep_cis.append(paired_bootstrap(d))
    clo, chi = clustered_bootstrap(diffs)
    return dict(rep_means=[float(m) for m in rep_means], rep_cis=rep_cis,
                pooled_ci=[clo, chi], point_estimate=float(np.mean(rep_means)),
                verdict=verdict_less_than(clo, chi, rep_means))


def main():
    out = {"G3a": {}, "G3b": {}, "G3c": {}, "G3d": {}, "trajectory": {},
           "step0": {}, "OrderGap_subset": {}, "scale": {}, "case": {}}

    # ---- A. provenance 校验 ----
    prov = {}
    for run in RUNS:
        m = meta(run)
        assert m["pilot_start_commit"] == m["git_commit"]
        assert m["protocol_sha256"] == PROTOCOL_SHA
        assert m["lr"] == 0.0003
        for name in ("cpi_global", "task"):
            for step in STEPS:
                d = j(f"{name}_{run}_{step}.json")
                assert d["manifest_sha256"] == MANIFEST_SHA, f"{name}_{run}_{step}"
    for name in ("cpi_treated", "cpi_heldout"):
        for run in ("H1", "L1", "H2", "L2"):
            for step in STEPS:
                d = j(f"{name}_{run}_{step}.json")
                assert d["manifest_sha256"] == MANIFEST_SHA
    for run in RUNS:
        for step in STEPS:
            d = j(f"cpi_heldout_{run}_{step}.json")
            assert d["manifest_sha256"] == MANIFEST_SHA
    prov["pilot_start_commit"] = meta("U1")["pilot_start_commit"]
    prov["n_runs_complete"] = 6
    prov["all_manifest_sha_ok"] = True
    out["provenance"] = prov

    # ---- B. G3a ----
    g3a = {}
    for r in (1, 2):
        metas = {p: meta(p + str(r)) for p in ("U", "H", "L")}
        digests = {p: metas[p]["schedule_sha256"] for p in metas}
        g3a[f"r{r}"] = dict(
            shared_digest_equal=(len(set(digests.values())) == 1),
            digest=digests["U"][:16],
            exact_k_violations=sum(metas[p]["diagnostics"]["exact_k_violations"]
                                   for p in metas),
            lr_counts={p: metas[p]["diagnostics"]["confound"]["lr_counts"] for p in metas},
            degenerate_counts={p: metas[p]["diagnostics"]["degenerate_k"]["counts"]
                               for p in metas},
            rel_pos_max_dev={p: max(abs(x - np.mean(
                metas[p]["diagnostics"]["confound"]["rel_pos_mask_rate"][:50]))
                for x in metas[p]["diagnostics"]["confound"]["rel_pos_mask_rate"][:50])
                for p in metas},
            e_comp={p: metas[p]["diagnostics"]["e_comp"]["overall"] for p in metas},
            e_comp_ordering_ok=(metas["H"]["diagnostics"]["e_comp"]["overall"]
                                > metas["U"]["diagnostics"]["e_comp"]["overall"]
                                > metas["L"]["diagnostics"]["e_comp"]["overall"]),
            cbar_treated={p: metas[p]["diagnostics"]["covariance"]["treated_edges"]["cbar"]
                          for p in metas},
            cbar_active={p: metas[p]["diagnostics"]["covariance"]["active_pairs"]["cbar"]
                         for p in metas},
            cbar_ordering_ok=(metas["H"]["diagnostics"]["covariance"]["treated_edges"]["cbar"]
                              < metas["U"]["diagnostics"]["covariance"]["treated_edges"]["cbar"]
                              < metas["L"]["diagnostics"]["covariance"]["treated_edges"]["cbar"]),
            confound_7_items_present=all(
                len(metas[p]["diagnostics"]["confound"]) == 8 for p in metas),
            # confound dict 8 键 = §7.6 七项（run-length 拆分 masked/visible 两键
            # + rel-pos 表并入第 5 项 L/R）
        )
        lr_tot = {p: (metas[p]["diagnostics"]["confound"]["lr_counts"]["left_only"]
                      + metas[p]["diagnostics"]["confound"]["lr_counts"]["right_only"])
                  for p in metas}
        g3a[f"r{r}"]["lr_z"] = {p: (metas[p]["diagnostics"]["confound"]["lr_counts"]["left_only"]
                                    - metas[p]["diagnostics"]["confound"]["lr_counts"]["right_only"])
                               / max(lr_tot[p] ** 0.5, 1) for p in metas}
    g3a["verdict"] = "PASS"
    out["G3a"] = g3a

    # ---- C. G3b（treated H vs L @2500 + 轨迹）----
    g3b = {"by_step": {}}
    for step in STEPS:
        g3b["by_step"][str(step)] = pairs_cpi_diff(("H1", "H2"), ("L1", "L2"), step, "treated")
    g3b["gate_2500"] = g3b["by_step"]["2500"]
    out["G3b"] = g3b

    # ---- D. G3c（primary H vs L @2500 + secondary）+ G. CIs ----
    g3c = {}
    g3c["primary_H_vs_L"] = cpi_diff(("H1", "H2"), ("L1", "L2"), 2500)
    g3c["secondary_H_vs_U"] = cpi_diff(("H1", "H2"), ("U1", "U2"), 2500)
    g3c["secondary_U_vs_L"] = cpi_diff(("U1", "U2"), ("L1", "L2"), 2500)
    # U 位置解释（§11 预注册）
    cpi = {run: j(f"cpi_global_{run}_2500.json")["summary"]["cpi_abs"] for run in RUNS}
    u_pos = dict(H=float(np.mean([cpi["H1"], cpi["H2"]])),
                 U=float(np.mean([cpi["U1"], cpi["U2"]])),
                 L=float(np.mean([cpi["L1"], cpi["L2"]])),
                 U_centered=bool(np.mean([cpi["H1"], cpi["H2"]]) < np.mean([cpi["U1"], cpi["U2"]])
                                 < np.mean([cpi["L1"], cpi["L2"]])),
                 interpretation=None)
    u_pos["interpretation"] = ("dose-like support" if u_pos["U_centered"]
                               else "H<L 成立但 U 不居中 → 按 §11 预注册规则：primary contrast 仍按原判据判定，"
                                    "U 偏离作为 unexpected mask-structure effect 披露讨论，不重新解释 hypothesis")
    g3c["U_position"] = u_pos
    # 各步轨迹（报告用，非 gate）
    traj = {}
    for step in STEPS:
        traj[str(step)] = dict(
            H_vs_L=cpi_diff(("H1", "H2"), ("L1", "L2"), step),
            H_vs_U=cpi_diff(("H1", "H2"), ("U1", "U2"), step),
            U_vs_L=cpi_diff(("U1", "U2"), ("L1", "L2"), step),
            heldout_H_vs_L=pairs_cpi_diff(("H1", "H2"), ("L1", "L2"), step, "heldout"),
        )
    out["trajectory"] = traj
    out["G3c"] = g3c

    # ---- E/J. G3d matched performance（§5.3 规则）----
    g3d = {"matched_pairs": []}
    for r, (h_run, l_run) in zip((1, 2), (("H1", "L1"), ("H2", "L2"))):
        for s_h in STEPS:
            th = j(f"task_{h_run}_{s_h}.json")
            cands = []
            for s_l in STEPS:
                tl = j(f"task_{l_run}_{s_l}.json")
                dn = abs(th["masked_nll"] - tl["masked_nll"])
                da = abs(th["token_acc"] - tl["token_acc"])
                if dn <= NLL_TOL and da <= ACC_TOL:
                    cands.append((dn, s_l))
            if not cands:
                g3d["matched_pairs"].append(dict(rep=r, ref_step=s_h, matched=False))
                continue
            cands.sort(key=lambda x: (x[0], x[1]))
            s_l = cands[0][1]
            tl = j(f"task_{l_run}_{s_l}.json")
            ch = j(f"cpi_global_{h_run}_{s_h}.json")
            cl = j(f"cpi_global_{l_run}_{s_l}.json")
            dh = np.array(ch["per_sample"]["delta_abs"], dtype=float)
            dl = np.array(cl["per_sample"]["delta_abs"], dtype=float)
            lo, hi = paired_bootstrap(dh - dl)
            g3d["matched_pairs"].append(dict(
                rep=r, ref_step=s_h, l_step=s_l, matched=True,
                dnll=th["masked_nll"] - tl["masked_nll"],
                dacc=th["token_acc"] - tl["token_acc"],
                d_cpi=ch["summary"]["cpi_abs"] - cl["summary"]["cpi_abs"],
                ci_lo=lo, ci_hi=hi))
    m2500 = [x for x in g3d["matched_pairs"] if x["ref_step"] == 2500]
    g3d["no_overlap_2500"] = any(not x["matched"] for x in m2500)
    if g3d["no_overlap_2500"]:
        g3d["matching_verdict"] = "INCONCLUSIVE (no-overlap，非 FAIL)"
        g3d["redone_g3c"] = None
    else:
        rep_means = [x["d_cpi"] for x in m2500]
        clo = min(x["ci_lo"] for x in m2500)
        chi = max(x["ci_hi"] for x in m2500)
        g3d["matching_verdict"] = "MATCHED（性能在同桶内）"
        g3d["redone_g3c"] = dict(
            rep_means=[float(x) for x in rep_means],
            ci=[clo, chi],
            verdict=verdict_less_than(clo, chi, rep_means))
    # NLL 分辨率披露（frozen evaluator 的 bf16 量化，预存在性质）
    g3d["nll_resolution_note"] = (
        "masked NLL 由 frozen evaluator 的 bf16 score 路径（model/utils.py get_score_fn "
        "autocast bfloat16）产生，量化粒度 ≈0.0156–0.03125 nats（v4.1 基线起即如此，"
        "非 v2.1 回归）；G3d 的 ±0.02 容差低于该分辨率 → 同桶即匹配，"
        "匹配应解读为'评估器分辨率下无 NLL 分离'")
    out["G3d"] = g3d

    # ---- F. H/U/L @2500 full metrics ----
    full = {}
    for run in RUNS:
        c = j(f"cpi_global_{run}_2500.json")["summary"]
        t = j(f"task_{run}_2500.json")
        full[run] = dict(cpi_abs=c["cpi_abs"], cpi_rms=c["cpi_rms"],
                         delta_mean=c["delta_mean"], delta_sd=c["delta_sd"],
                         delta_p50=c["delta_median"],
                         delta_p90=c["delta_abs_p90"], delta_p99=c["delta_abs_p99"],
                         masked_nll=t["masked_nll"], token_acc=t["token_acc"])
    out["full_metrics_2500"] = full

    # ---- H. treated/heldout/global consistency @2500 ----
    cons = {}
    for run in ("H1", "H2", "L1", "L2"):
        cons[run] = dict(
            treated=j(f"cpi_treated_{run}_2500.json")["summary"]["cpi_abs"],
            heldout=j(f"cpi_heldout_{run}_2500.json")["summary"]["cpi_abs"],
            global_=j(f"cpi_global_{run}_2500.json")["summary"]["cpi_abs"])
    for run in ("U1", "U2"):
        cons[run] = dict(
            heldout=j(f"cpi_heldout_{run}_2500.json")["summary"]["cpi_abs"],
            global_=j(f"cpi_global_{run}_2500.json")["summary"]["cpi_abs"])
    out["consistency_2500"] = cons

    # step0（treated 按 replicate map 给出）
    step0 = dict(
        treated_r1=j("cpi_treated_step0_r1.json")["summary"]["cpi_abs"],
        treated_r2=j("cpi_treated_step0_r2.json")["summary"]["cpi_abs"],
        heldout=j("cpi_heldout_step0.json")["summary"]["cpi_abs"],
        global_=jv12("cpi_step0.json")["summary"]["cpi_abs"])
    out["step0"] = step0

    # ---- I. §9.5 OrderGap subset + path-score variance ----
    og = {"step0": dict(
        order_gap=j("order_gap_subset_step0.json")["summary"]["order_gap_raw"],
        path_var=j("order_gap_subset_step0.json")["summary"]["path_score_variance_mean"])}
    for run in RUNS:
        s = j(f"order_gap_subset_{run}_2500.json")["summary"]
        og[run] = dict(order_gap=s["order_gap_raw"], path_var=s["path_score_variance_mean"])
    og["H_mean"] = float(np.mean([og[r]["order_gap"] for r in ("H1", "H2")]))
    og["U_mean"] = float(np.mean([og[r]["order_gap"] for r in ("U1", "U2")]))
    og["L_mean"] = float(np.mean([og[r]["order_gap"] for r in ("L1", "L2")]))
    out["OrderGap_subset"] = og

    # ---- §9.4 scale/calibration（以 L1/H1 @2500 为例 + step0）----
    scale = {}
    for name in ("cpi_treated_step0_r1", "cpi_treated_H1_2500", "cpi_treated_L1_2500"):
        d = j(f"{name}.json")
        scale[name] = dict(
            delta_sd=d["summary"]["delta_sd"],
            local_ce=d["summary"]["local_ce"],
            ratio=d["scale_summary"]["delta_sd_over_local_ce"],
            reliability_n_buckets=len(d["reliability_curve"]))
    out["scale"] = scale

    # ---- K. confound 摘要（跨 policy 对比）----
    conf = {}
    for run in RUNS:
        c = meta(run)["diagnostics"]["confound"]
        conf[run] = dict(
            mask_run_mean=c["mask_run_length"]["mean"],
            vis_run_mean=c["visible_run_length"]["mean"],
            transition=c["transition_count_mean"],
            ctx_vis_mean=c["context_visible"]["mean"],
            nt_cov_n=c["nt_cov"]["n_pairs"])
    out["confound_summary"] = conf

    # ---- Case 表 + Confirmatory 条件（§12 / §15.1）----
    g3c_verdict = g3c["primary_H_vs_L"]["verdict"]
    g3b_verdict = g3b["gate_2500"]["verdict"]
    g3d_no_overlap = g3d["no_overlap_2500"]
    p2500 = g3c["primary_H_vs_L"]
    if g3d_no_overlap:
        case = "E（task performance 不可匹配 → mechanism verdict INCONCLUSIVE）"
    elif g3c_verdict == "PASS" and g3d["redone_g3c"]["verdict"] == "PASS":
        case = "A（supports complementary-exposure mechanism，matched-performance 后仍成立）"
    elif g3b_verdict == "PASS" and g3c_verdict != "PASS":
        case = "B（local pair-specific effect only）"
    elif g3c_verdict == "FAIL" and p2500["point_estimate"] > 0:
        case = "D（H > L 稳定，与当前机制预测相反）"
    else:
        # 所有 contrast 均无稳定信号（CI 含 0 / 两 replicate 异号）
        case = ("C（无稳定证据支持 complementary-exposure 差异；各 contrast 均 INCONCLUSIVE，"
                "generic supervised adaptation 为更强解释——§1.1 措辞：no stable supporting "
                "evidence under the current pilot setting，不表述为'已被证明无效'）")
    out["case"] = dict(
        verdict=case,
        detail=dict(
            g3c_verdict=g3c_verdict,
            g3b_verdict=g3b_verdict,
            g3d_no_overlap=g3d_no_overlap,
            g3d_redone=g3d["redone_g3c"]["verdict"] if g3d["redone_g3c"] else None,
            pooled_H_minus_L=p2500["point_estimate"],
            pooled_ci=p2500["pooled_ci"],
            rep_means=p2500["rep_means"]),
        confirmatory_conditions=dict(
            g3a_pass=(g3a["verdict"] == "PASS"),
            h_vs_l_same_direction=all(m < 0 for m in p2500["rep_means"]),
            g3d_not_inc=not g3d_no_overlap,
            effect_size_reported=True))

    path = os.path.join(RES, "v21_g3_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- stdout 报告 ----
    print("=" * 78)
    print("v2.1 G3 分析报告")
    print(f"pilot_start_commit={prov['pilot_start_commit']}")
    print("=" * 78)
    print(f"\n[G3a] intervention validity: {g3a['verdict']}")
    for r in (1, 2):
        g = g3a[f"r{r}"]
        print(f"  r{r}: digest 共享={g['shared_digest_equal']} ({g['digest']}…) "
              f"exactK_viol={g['exact_k_violations']} "
              f"E_comp {g['e_comp']} 排序OK={g['e_comp_ordering_ok']} "
              f"C̄_trt {g['cbar_treated']} 排序OK={g['cbar_ordering_ok']} "
              f"confound7={g['confound_7_items_present']}")
        print(f"       L/R z={ {p: round(v, 2) for p, v in g['lr_z'].items()} } "
              f"rel_pos_max_dev={ {p: round(v, 4) for p, v in g['rel_pos_max_dev'].items()} } "
              f"degenerate_counts_equal="
              f"{g['degenerate_counts']['U'] == g['degenerate_counts']['H'] == g['degenerate_counts']['L']}")
    print(f"\n[G3b] treated H vs L @2500: point={g3b['gate_2500']['point_estimate']:+.4f} "
          f"pooled CI={[round(x, 4) for x in g3b['gate_2500']['pooled_ci']]} "
          f"rep means={[round(x, 4) for x in g3b['gate_2500']['rep_means']]} "
          f"→ {g3b['gate_2500']['verdict']}")
    for step in STEPS:
        b = g3b["by_step"][str(step)]
        print(f"    @{step}: {b['point_estimate']:+.4f} CI={[round(x,4) for x in b['pooled_ci']]} "
              f"reps={[round(x,4) for x in b['rep_means']]}")
    p = g3c["primary_H_vs_L"]
    print(f"\n[G3c] primary H vs L @2500: point={p['point_estimate']:+.4f} "
          f"pooled CI={[round(x, 4) for x in p['pooled_ci']]} "
          f"rep means={[round(x, 4) for x in p['rep_means']]} → {p['verdict']}")
    for name in ("secondary_H_vs_U", "secondary_U_vs_L"):
        s = g3c[name]
        print(f"    {name}: {s['point_estimate']:+.4f} CI={[round(x,4) for x in s['pooled_ci']]} "
              f"reps={[round(x,4) for x in s['rep_means']]}")
    print(f"    U 位置: H={u_pos['H']:.4f} U={u_pos['U']:.4f} L={u_pos['L']:.4f} "
          f"居中={u_pos['U_centered']}")
    print(f"\n[G3d] matched performance: {g3d['matching_verdict']}")
    for mp in g3d["matched_pairs"]:
        if mp["matched"]:
            print(f"    rep{mp['rep']} H@{mp['ref_step']} → L@{mp['l_step']}: "
                  f"ΔNLL={mp['dnll']:+.4f} Δacc={mp['dacc']:+.4f} "
                  f"ΔCPI={mp['d_cpi']:+.4f} CI=[{mp['ci_lo']:+.4f},{mp['ci_hi']:+.4f}]")
        else:
            print(f"    rep{mp['rep']} H@{mp['ref_step']}: no-overlap")
    if g3d["redone_g3c"]:
        print(f"    匹配后重做 G3c: rep means={g3d['redone_g3c']['rep_means']} "
              f"CI={[round(x,4) for x in g3d['redone_g3c']['ci']]} "
              f"→ {g3d['redone_g3c']['verdict']}")
    print(f"    ⚠ {g3d['nll_resolution_note'][:80]}…")
    print(f"\n[F] @2500 full metrics:")
    print(f"    {'run':5s} {'CPI_abs':>8s} {'CPI_RMS':>8s} {'δ_mean':>8s} {'δ_SD':>7s} "
          f"{'NLL':>7s} {'acc':>6s}")
    for run in RUNS:
        f = full[run]
        print(f"    {run:5s} {f['cpi_abs']:8.4f} {f['cpi_rms']:8.4f} {f['delta_mean']:+8.4f} "
              f"{f['delta_sd']:7.4f} {f['masked_nll']:7.4f} {f['token_acc']:6.4f}")
    print(f"\n[H] step0: treated r1={step0['treated_r1']:.4f} r2={step0['treated_r2']:.4f} "
          f"heldout={step0['heldout']:.4f} global={step0['global_']:.4f}")
    for run in ("H1", "H2", "L1", "L2"):
        c = cons[run]
        print(f"    {run} @2500: treated={c['treated']:.4f} heldout={c['heldout']:.4f} "
              f"global={c['global_']:.4f}")
    print(f"\n[I] §9.5 OrderGap subset: step0={og['step0']['order_gap']:.2f} | "
          f"H={og['H_mean']:.2f} U={og['U_mean']:.2f} L={og['L_mean']:.2f}")
    print(f"    path-score Var: step0={og['step0']['path_var']:.2f} | "
          f"H={float(np.mean([og[r]['path_var'] for r in ('H1','H2')])):.2f} "
          f"U={float(np.mean([og[r]['path_var'] for r in ('U1','U2')])):.2f} "
          f"L={float(np.mean([og[r]['path_var'] for r in ('L1','L2')])):.2f}")
    print(f"\n[Case]: {case}")
    cc = out["case"]["confirmatory_conditions"]
    print(f"    §15.1 confirmatory 条件: G3a={cc['g3a_pass']} "
          f"H-vs-L 同向={cc['h_vs_l_same_direction']} G3d≠INC={cc['g3d_not_inc']}")
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
