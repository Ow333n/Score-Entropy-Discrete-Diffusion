"""P1 G1b matched-performance 分析（protocol v1.2 §5.3 预注册规则，只读）。

规则：
- primary = masked-span NLL ±0.02；secondary = token acc ±0.01
- 对 A 的每个参考 checkpoint（500/1020/2500），在 B 的池 {500,1020,2500}
  中选 |ΔNLL| 最小且落入双容差的 checkpoint；平局取 step 更小者
- 无候选 → 该参考点 no-overlap → G1b INCONCLUSIVE（非 FAIL）
- 匹配对选定后：重算 ΔCPI 与 per-replicate paired bootstrap CI（10k）

用法: .venv/bin/python scripts/p1_g1b_analysis.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "mechanism_pilot")
NLL_TOL, ACC_TOL = 0.02, 0.01
N_BOOT = 10000


def j(n):
    with open(os.path.join(RES, n)) as f:
        return json.load(f)


def paired_bootstrap(diff, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    out = {"replicates": {}, "G1b_overall": None}
    for r, (a_run, b_run) in zip((1, 2), (("A1", "B1"), ("A2", "B2"))):
        print(f"=== replicate {r} ({a_run} vs {b_run}) ===")
        rep = {"matched_pairs": []}
        for s_a in (500, 1020, 2500):
            ta = j(f"task_{a_run}_{s_a}.json")
            cands = []
            for s_b in (500, 1020, 2500):
                tb = j(f"task_{b_run}_{s_b}.json")
                dn = abs(ta["masked_nll"] - tb["masked_nll"])
                da = abs(ta["token_acc"] - tb["token_acc"])
                if dn <= NLL_TOL and da <= ACC_TOL:
                    cands.append((dn, s_b))
            if not cands:
                print(f"  A@{s_a}: no-overlap（NLL_A={ta['masked_nll']:.4f}，池内无 ±{NLL_TOL} 匹配）→ INCONCLUSIVE")
                rep["matched_pairs"].append(dict(ref_step=s_a, matched=False))
                continue
            cands.sort(key=lambda x: (x[0], x[1]))     # 最小 |ΔNLL|，平局更小 step
            s_b = cands[0][1]
            tb = j(f"task_{b_run}_{s_b}.json")
            ca = j(f"cpi_{a_run}_{s_a}.json")
            cb = j(f"cpi_{b_run}_{s_b}.json")
            da = np.array(ca["per_sample"]["delta_abs"])
            db = np.array(cb["per_sample"]["delta_abs"])
            d_cpi = ca["summary"]["cpi_abs"] - cb["summary"]["cpi_abs"]
            lo, hi = paired_bootstrap(da - db)
            print(f"  A@{s_a} → B@{s_b}: ΔNLL={ta['masked_nll']-tb['masked_nll']:+.4f} "
                  f"Δacc={ta['token_acc']-tb['token_acc']:+.4f} "
                  f"ΔCPI={d_cpi:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
            rep["matched_pairs"].append(dict(
                ref_step=s_a, matched=True, b_step=s_b,
                dnll=ta["masked_nll"] - tb["masked_nll"],
                dacc=ta["token_acc"] - tb["token_acc"],
                d_cpi=d_cpi, ci_lo=lo, ci_hi=hi))
        # G1b 判定（gate step 2500）
        m2500 = next(x for x in rep["matched_pairs"] if x["ref_step"] == 2500)
        if not m2500["matched"]:
            verdict = "INCONCLUSIVE"
        else:
            verdict = ("PASS" if m2500["ci_hi"] < 0
                       else "FAIL" if m2500["ci_lo"] > 0
                       else "INCONCLUSIVE")
        rep["G1b_rep_verdict_2500"] = verdict
        print(f"  G1b @2500 ({r}): {verdict}")
        out["replicates"][str(r)] = rep
    v1, v2 = (out["replicates"]["1"]["G1b_rep_verdict_2500"],
              out["replicates"]["2"]["G1b_rep_verdict_2500"])
    out["G1b_overall"] = ("PASS" if v1 == v2 == "PASS"
                          else "FAIL" if v1 == v2 == "FAIL"
                          else "INCONCLUSIVE")
    print(f"\nG1b overall: {out['G1b_overall']}")
    path = os.path.join(RES, "p1_g1b_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
