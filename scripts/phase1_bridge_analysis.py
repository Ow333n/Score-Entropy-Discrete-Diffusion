"""Phase 1.2 bridge 分析（bridge_manifest FROZEN + 协议 §4 预注册口径，只读）。

预注册分析 A–E + BRIDGE-A/B/C/D 判定：
  A. attenuation direction preservation：pretrained→SFT 全部对 + within-run 相邻对的
     sign(ΔCPI_old) == sign(ΔCPI_fp32)
  B. checkpoint ranking preservation：Spearman + Kendall（CPI_abs / CPI_RMS / CE）
  C. absolute evaluator shift：FP32−BF16 的 mean/median/range
  D. per-sample agreement：|δ| / signed δ / local CE 的 Pearson/Spearman/paired diff
  E. BF16 CE quantization impact：unique 值数 / 网格间距 / old-vs-new rank 保存
  图：old vs new 散点（CPI_abs、CE 两面板 + identity line，3 类着色：
  pretrained / SFT lr=3e-4 / SFT lr=3e-5）

用法: .venv/bin/python scripts/phase1_bridge_analysis.py
输出: results/phase1_bridge/bridge_analysis.json + bridge_scatter.png
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BR = os.path.join(ROOT, "results", "phase1_bridge")
MANIFEST_SHA_BM = "986dcf1d0c79e7ff6fd21f9d00cf782ce81707556d62f3ed2b0425ea98d6587b"

FAMILY_CAT = {   # 3 类（scatter 3-series cap）
    "pretrained": "pretrained",
    "p1-v4.2": "SFT-lr3e-5",
    "formal-v4.1": "SFT-lr3e-5",
    "v2.1-HUL": "SFT-lr3e-4",
    "v1.2-A": "SFT-lr3e-4",
}
CAT_COLOR = {"pretrained": "#2a78d6", "SFT-lr3e-4": "#eb6834", "SFT-lr3e-5": "#1baf7a"}


def j(name):
    with open(os.path.join(BR, name)) as f:
        return json.load(f)


def spearman(a, b):
    """Pearson on ranks（无 scipy 依赖）。"""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def kendall(a, b):
    """tau-b（无 scipy 依赖；n 小，O(n²) 可接受）。"""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for jj in range(i + 1, n):
            da, db = a[i] - a[jj], b[i] - b[jj]
            if da * db > 0:
                conc += 1
            elif da * db < 0:
                disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else 0.0


def spearman_kendall(x, y):
    return (spearman(x, y), None, kendall(x, y))


def main():
    man = j("bridge_manifest.json")
    digest = hashlib.sha256(open(os.path.join(BR, "bridge_manifest.json"), "rb").read()).hexdigest()
    assert digest == MANIFEST_SHA_BM, "bridge manifest 被改动"

    ids = [c["id"] for c in man["checkpoints"]]
    families = {c["id"]: c["family"] for c in man["checkpoints"]}
    steps = {c["id"]: c["step"] for c in man["checkpoints"]}

    old, new = {}, {}
    for cid in ids:
        o = j(f"old_{cid}.json")
        n = j(f"new_{cid}.json")
        assert o["manifest_sha256"] == n["manifest_sha256"] == \
            "1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3"
        assert len(o["per_sample"]["delta"]) == 500 and len(n["per_sample"]["delta"]) == 500
        old[cid], new[cid] = o, n

    # ---- 完整表 + C. absolute shifts ----
    table = []
    shifts = {k: [] for k in ("cpi_abs", "cpi_rms", "delta_mean", "local_ce")}
    for cid in ids:
        so, sn = old[cid]["summary"], new[cid]["summary"]
        row = dict(id=cid, family=families[cid], step=steps[cid],
                   old_cpi=so["cpi_abs"], new_cpi=sn["cpi_abs"],
                   old_rms=so["cpi_rms"], new_rms=sn["cpi_rms"],
                   old_dm=so["delta_mean"], new_dm=sn["delta_mean"],
                   old_ce=so["local_ce"], new_ce=sn["local_ce"],
                   old_acc=so["token_acc"], new_acc=sn["token_acc"])
        for k in shifts:
            row[f"d_{k}"] = sn[k] - so[k]
            shifts[k].append(sn[k] - so[k])
        table.append(row)

    def msr(vals):
        return dict(mean=float(np.mean(vals)), median=float(np.median(vals)),
                    lo=float(np.min(vals)), hi=float(np.max(vals)))

    # ---- A. attenuation direction ----
    pairs = []
    for cid in ids:
        if cid != "pretrained-step0":
            pairs.append(("pretrained-step0", cid, "pretrained→SFT"))
    for (a, b, tag) in (("v21-U1-1020", "v21-U1-2500", "v21-U1 1020→2500"),
                        ("v21-H1-1020", "v21-H1-2500", "v21-H1 1020→2500"),
                        ("v21-L1-1020", "v21-L1-2500", "v21-L1 1020→2500"),
                        ("v12-A1-1020", "v12-A1-2500", "v12-A1 1020→2500"),
                        ("formal-s1-1020", "formal-s1-10200", "formal-s1 1020→10200"),
                        ("formal-s2-1020", "formal-s2-10200", "formal-s2 1020→10200")):
        pairs.append((a, b, tag))
    direction = []
    for a, b, tag in pairs:
        do = old[b]["summary"]["cpi_abs"] - old[a]["summary"]["cpi_abs"]
        dn = new[b]["summary"]["cpi_abs"] - new[a]["summary"]["cpi_abs"]
        direction.append(dict(pair=tag, d_old=do, d_new=dn,
                              same_sign=bool(np.sign(do) == np.sign(dn) != 0),
                              d_zero=bool(abs(do) < 1e-9 or abs(dn) < 1e-9)))
    agree = sum(1 for d in direction if d["same_sign"])
    n_nz = sum(1 for d in direction if not d["d_zero"])
    core = [d for d in direction if d["pair"] == "pretrained→SFT"]
    core_agree = sum(1 for d in core if d["same_sign"])

    # ---- B. ranking preservation ----
    def ranks(key_old, key_new):
        x = [next(r[f"old_{key_old}"] for r in table if r["id"] == cid) for cid in ids]
        y = [next(r[f"new_{key_new}"] for r in table if r["id"] == cid) for cid in ids]
        return spearman_kendall(x, y)
    rank = dict(
        cpi=dict(spearman=ranks("cpi", "cpi")[0], kendall=ranks("cpi", "cpi")[2]),
        rms=dict(spearman=ranks("rms", "rms")[0], kendall=ranks("rms", "rms")[2]),
        ce=dict(spearman=ranks("ce", "ce")[0], kendall=ranks("ce", "ce")[2]))

    # ---- D. per-sample agreement（pooled 7500 + per-checkpoint 范围）----
    ps_old_abs = np.concatenate([np.array(old[c]["per_sample"]["delta_abs"]) for c in ids])
    ps_new_abs = np.concatenate([np.array(new[c]["per_sample"]["delta_abs"]) for c in ids])
    ps_old_d = np.concatenate([np.array(old[c]["per_sample"]["delta"]) for c in ids])
    ps_new_d = np.concatenate([np.array(new[c]["per_sample"]["delta"]) for c in ids])
    ps_old_ce = np.concatenate([np.array(old[c]["per_sample"]["local_ce"]) for c in ids])
    ps_new_ce = np.concatenate([np.array(new[c]["per_sample"]["local_ce"]) for c in ids])
    def pairwise_agree(a, b):
        return dict(pearson=float(np.corrcoef(a, b)[0, 1]),
                    spearman=spearman_kendall(a, b)[0],
                    diff_mean=float((b - a).mean()), diff_sd=float((b - a).std()))
    per_sample = dict(
        abs_delta=pairwise_agree(ps_old_abs, ps_new_abs),
        signed_delta=pairwise_agree(ps_old_d, ps_new_d),
        local_ce=pairwise_agree(ps_old_ce, ps_new_ce))
    per_ckpt_pearson = []
    for cid in ids:
        for key, k_new, k_old in (("abs_delta", "delta_abs", "delta_abs"),
                                  ("signed_delta", "delta", "delta"),
                                  ("local_ce", "local_ce", "local_ce")):
            per_ckpt_pearson.append(dict(
                id=cid, key=key,
                pearson=float(np.corrcoef(np.array(new[cid]["per_sample"][k_new]),
                                          np.array(old[cid]["per_sample"][k_old]))[0, 1])))
    per_sample["per_checkpoint_pearson_range"] = {
        k: dict(lo=float(min(x["pearson"] for x in per_ckpt_pearson if x["key"] == k)),
                hi=float(max(x["pearson"] for x in per_ckpt_pearson if x["key"] == k)))
        for k in ("abs_delta", "signed_delta", "local_ce")}

    # ---- E. BF16 CE quantization impact ----
    uniq_old = len(np.unique(np.round(ps_old_ce, 9)))
    uniq_new = len(np.unique(np.round(ps_new_ce, 9)))
    pos_diffs = np.diff(np.sort(np.unique(np.round(ps_old_ce, 9))))
    pos_diffs = pos_diffs[pos_diffs > 0]
    spacing = float(np.median(pos_diffs)) if len(pos_diffs) else None
    quant = dict(n_values=len(ps_old_ce), unique_old=uniq_old, unique_new=uniq_new,
                 old_median_spacing=spacing,
                 old_spacing_hist=[float(x) for x in np.unique(np.round(pos_diffs, 6))[:10]],
                 ce_rank_spearman_pooled=pairwise_agree(ps_old_ce, ps_new_ce)["spearman"])

    # ---- Case 判定（协议 §4，FROZEN）----
    # direction_ok = 核心 pretrained→SFT 全部对同号（BRIDGE-C 的判据只针对核心方向，
    # 不混入 within-run 小效应对）
    direction_ok = core_agree == len(core)
    cpi_rank_high = rank["cpi"]["spearman"] >= 0.9
    per_sample_high = per_sample["abs_delta"]["pearson"] >= 0.9
    ce_rank_changed = rank["ce"]["spearman"] < 0.7
    if not direction_ok:
        case = "BRIDGE-C（HARD STOP：核心 pretrained→SFT attenuation direction 翻转/消失）"
    elif direction_ok and cpi_rank_high and per_sample_high and not ce_rank_changed:
        case = "BRIDGE-A（历史 compatibility 结论定性稳健；后续精细 dynamics 用 FP32）"
    elif direction_ok and ce_rank_changed:
        case = "BRIDGE-D（CPI 稳健但 CE ranking 明显变化：旧 BF16 NLL 不适合 dynamics；"
        case += "Phase 1 CE/NLL 必须用 FP32；compatibility observation 保留）"
    else:
        case = "BRIDGE-B（总体方向保存但部分 ranking/小效应改变：大效应结论保留；"
        case += "小 differential effects 不跨 evaluator 重解释；Phase 1 统一 FP32）"

    out = dict(
        bridge_manifest_sha256=digest,
        n_checkpoints=len(ids),
        table=table,
        direction=dict(pairs=direction, agree=agree, n_nonzero=n_nz,
                       core_agree=core_agree, core_pairs=len(core)),
        ranking=rank,
        absolute_shifts={k: msr(v) for k, v in shifts.items()},
        per_sample_agreement=per_sample,
        ce_quantization=quant,
        case=case,
        case_operationalization=dict(
            direction_ok=direction_ok, cpi_spearman=rank["cpi"]["spearman"],
            cpi_rank_high_threshold=0.9, per_sample_pearson=per_sample["abs_delta"]["pearson"],
            per_sample_high_threshold=0.9, ce_spearman=rank["ce"]["spearman"],
            ce_changed_threshold=0.7))
    with open(os.path.join(BR, "bridge_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 78)
    print("Phase 1.2 Bridge 分析")
    print(f"  A. 方向保存: {agree}/{n_nz} 对同号；核心 pretrained→SFT {core_agree}/{len(core)}")
    print(f"  B. ranking: CPI Spearman={rank['cpi']['spearman']:.4f} Kendall={rank['cpi']['kendall']:.4f} | "
          f"RMS S={rank['rms']['spearman']:.4f} | CE S={rank['ce']['spearman']:.4f}")
    print(f"  C. shift (FP32−BF16): CPI {msr(shifts['cpi_abs'])} | CE {msr(shifts['local_ce'])}")
    print(f"  D. per-sample: |δ| pearson={per_sample['abs_delta']['pearson']:.4f} | "
          f"signed δ {per_sample['signed_delta']['pearson']:.4f} | "
          f"CE {per_sample['local_ce']['pearson']:.4f}")
    print(f"  E. CE 量化: old unique={uniq_old} new unique={uniq_new} spacing={spacing}")
    print(f"\n  CASE: {case}")
    print(f"\n  完整表（前 5 行）:")
    for r in table[:5]:
        print(f"    {r['id']:18s} old_cpi={r['old_cpi']:.4f} new_cpi={r['new_cpi']:.4f} "
              f"d_cpi={r['d_cpi_abs']:+.4f} old_ce={r['old_ce']:.4f} new_ce={r['new_ce']:.4f}")
    print(f"\nsaved: {os.path.join(BR, 'bridge_analysis.json')}")
    return out


if __name__ == "__main__":
    main()
