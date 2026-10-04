"""Phase B: Demo example selection（数据驱动 + curated，规则透明落盘）。

两套样本：
  A. Representative（9 个）：固定分位规则 + 最近邻选取，tie-break 按 manifest index 升序
       - SFT local_ce  p10 / p50 / p90 最近（3）
       - Pretrained delta_abs p10 / p50 / p90 最近（3）
       - SFT token_acc p80(easy) / p50(medium) / p20(hard) 最近（3）
  B. Case Studies（6 个，明确标记 curated，不代表统计总体）：
       cs1 强 SFT compatibility attenuation:  max(pret.delta_abs − sft.delta_abs)
       cs2 强 SFT task improvement:           max(sft.token_acc − pret.token_acc)
       cs3 unusual reveal-order sensitivity:  max(sft.delta_abs)（与前两项去重）
       cs4 obvious failure case:              min(sft.token_acc)
       cs5 RL approximately unchanged:        fixed64 内 min|step500.sampled − step0.sampled|（reward ∈ (0,1)）
       cs6 semantic-equivalent mismatch (candidate): fixed64 内 sampled500 ∈ (0,1) 且 |M0|≤30 中
             reward 最高者；semantic 判定待 Phase C 导出文本后人工核实

输出 demo_assets/examples.json；所有统计值标注 provenance 文件。
只读数据 + HF tokenizer 解码，无 GPU。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from transformers import GPT2TokenizerFast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "manifests/regime_a_eval_v1.jsonl")
PRET_CPI = os.path.join(ROOT, "results/pretrained/cpi.json")
SFT_CPI = os.path.join(ROOT, "results/vanilla/cpi_s1_10200.json")
RL_EVAL0 = os.path.join(ROOT, "exp_local/regime_a/rlpilot-185545/eval_point_step0.json")
RL_EVAL500 = os.path.join(ROOT, "exp_local/regime_a/rlpilot-185545/eval_point_step500.json")
OUT = os.path.join(ROOT, "demo_assets/examples.json")

MASK = 50257
tok = GPT2TokenizerFast.from_pretrained("gpt2")


def decode(ids, mask_positions=None):
    """ids → 展示文本；mask_positions 处显示 [MASK]。"""
    out = []
    mask_set = set(mask_positions or [])
    for p, t in enumerate(ids):
        if p in mask_set:
            out.append("[MASK]")
        else:
            out.append(tok.decode([t]))
    return "".join(out).replace("Ġ", " ").strip()


def nearest_to_percentile(vals, indices, p, used):
    target = float(np.percentile(vals, p))
    order = sorted(range(len(vals)), key=lambda k: (abs(vals[k] - target), indices[k]))
    for k in order:
        if indices[k] not in used:
            return indices[k], float(vals[k]), target
    raise RuntimeError("no candidate left")


def main():
    recs = [json.loads(l) for l in open(MANIFEST)]
    n = len(recs)
    pret = json.load(open(PRET_CPI))["per_sample"]
    sft = json.load(open(SFT_CPI))["per_sample"]
    rl0 = json.load(open(RL_EVAL0))
    rl500 = json.load(open(RL_EVAL500))
    assert len(pret["local_ce"]) == n and len(sft["local_ce"]) == n

    pret_ce = np.array(pret["local_ce"])
    pret_da = np.array(pret["delta_abs"])
    pret_acc = np.array(pret["token_acc"])
    sft_ce = np.array(sft["local_ce"])
    sft_da = np.array(sft["delta_abs"])
    sft_acc = np.array(sft["token_acc"])
    indices = np.arange(n)
    used = set()

    sel = []

    def add(idx, cat, stype, reason, stats, extra=None):
        r = recs[idx]
        item = dict(
            sample_id=f"s{idx:03d}",
            manifest_index=int(idx),
            category=cat,
            selection_type=stype,
            selection_reason=reason,
            gt_text=decode(r["x0"]),
            initial_state_text=decode(r["initial_state"], r["initial_masked_positions"]),
            span_start=r["span_start"], span_end=r["span_end"],
            m0_positions=r["initial_masked_positions"],
            sigma=float(r["sigma"]),
            i=r["i"], j=r["j"], a=r["a"], b=r["b"],
            stats=stats,
            provenance=dict(
                manifest="manifests/regime_a_eval_v1.jsonl",
                pretrained="results/pretrained/cpi.json",
                sft="results/vanilla/cpi_s1_10200.json",
                rl="exp_local/regime_a/rlpilot-185545/eval_point_step{0,500}.json"))
        if extra:
            item.update(extra)
        sel.append(item)
        used.add(idx)

    # --- A. Representative（分位最近邻规则） ---
    for label, p in [("sft local_ce p10", 10), ("sft local_ce p50", 50), ("sft local_ce p90", 90)]:
        idx, v, t = nearest_to_percentile(sft_ce, indices, p, used)
        add(idx, "representative", "quantile_nearest",
            f"{label}: sample value {v:.4f} closest to p{p} target {t:.4f}; tie-break index ascending",
            dict(local_ce_sft=float(v)))
    for label, p in [("pretrained delta_abs p10", 10), ("pretrained delta_abs p50", 50),
                     ("pretrained delta_abs p90", 90)]:
        idx, v, t = nearest_to_percentile(pret_da, indices, p, used)
        add(idx, "representative", "quantile_nearest",
            f"{label}: sample value {v:.4f} closest to p{p} target {t:.4f}; tie-break index ascending",
            dict(delta_abs_pretrained=float(v)))
    for label, p in [("sft token_acc p80 (easy)", 80), ("sft token_acc p50 (medium)", 50),
                     ("sft token_acc p20 (hard)", 20)]:
        idx, v, t = nearest_to_percentile(sft_acc, indices, p, used)
        add(idx, "representative", "quantile_nearest",
            f"{label}: sample value {v:.4f} closest to p{p} target {t:.4f}; tie-break index ascending",
            dict(token_acc_sft=float(v)))

    # --- B. Case Studies ---
    d_atten = pret_da - sft_da
    for cs_idx in np.argsort(-d_atten):
        if int(cs_idx) not in used:
            add(int(cs_idx), "case_study", "curated",
                "cs1 strong SFT compatibility attenuation: max(pretrained.delta_abs − sft.delta_abs)="
                f"{d_atten[cs_idx]:.4f}",
                dict(delta_abs_pretrained=float(pret_da[cs_idx]), delta_abs_sft=float(sft_da[cs_idx])))
            break
    d_task = sft_acc - pret_acc
    for cs_idx in np.argsort(-d_task):
        if int(cs_idx) not in used:
            add(int(cs_idx), "case_study", "curated",
                "cs2 strong SFT task improvement: max(sft.token_acc − pretrained.token_acc)="
                f"{d_task[cs_idx]:.4f}",
                dict(token_acc_pretrained=float(pret_acc[cs_idx]), token_acc_sft=float(sft_acc[cs_idx])))
            break
    for cs_idx in np.argsort(-sft_da):
        if int(cs_idx) not in used:
            add(int(cs_idx), "case_study", "curated",
                f"cs3 unusual reveal-order sensitivity: max(sft.delta_abs)={sft_da[cs_idx]:.4f}",
                dict(delta_abs_sft=float(sft_da[cs_idx])))
            break
    for cs_idx in np.argsort(sft_acc):
        if int(cs_idx) not in used:
            add(int(cs_idx), "case_study", "curated",
                f"cs4 obvious failure case: min(sft.token_acc)={sft_acc[cs_idx]:.4f}",
                dict(token_acc_sft=float(sft_acc[cs_idx])))
            break
    # cs5: RL approximately unchanged（fixed64 = manifest 64–127）
    sr0 = {64 + k: rl0["sampled_raw_per"][k] for k in range(64)}
    sr500 = {64 + k: rl500["sampled_raw_per"][k] for k in range(64)}
    cand5 = [k for k in sr0 if 0 < sr0[k] < 1 and 0 < sr500[k] < 1]
    cs5 = min(cand5, key=lambda k: (abs(sr500[k] - sr0[k]), k))
    add(cs5, "case_study", "curated",
        f"cs5 RL approximately unchanged: min|step500.sampled−step0.sampled|="
        f"{abs(sr500[cs5]-sr0[cs5]):.4f} (fixed64, rewards in (0,1), tie-break index ascending)",
        dict(rl_sampled_step0=float(sr0[cs5]), rl_sampled_step500=float(sr500[cs5])))
    # cs6: semantic-equivalent mismatch candidate（Phase C 导出文本后人工核实）
    cand6 = [k for k in sr500 if 0 < sr500[k] < 1
             and len(recs[k]["initial_masked_positions"]) <= 30]
    if cand6:
        cs6 = max(cand6, key=lambda k: (sr500[k], -k))
        add(cs6, "case_study", "curated",
            f"cs6 semantic-equivalent token mismatch (CANDIDATE): fixed64, sampled500={sr500[cs6]:.4f} "
            f"∈(0,1), |M0|={len(recs[cs6]['initial_masked_positions'])}≤30, 最高 reward 候选",
            dict(rl_sampled_step500=float(sr500[cs6]), rl_greedy_step500=float(rl500["greedy_raw_per"][cs6 - 64])),
            extra=dict(semantic_status="candidate_pending_export_verification"))
    else:
        sel.append(dict(semantic_case="no clear semantic-equivalent mismatch found",
                        note="no candidate met the fixed64 rule; verified textually after export if any"))

    out = dict(
        export="Phase B example selection",
        rule_version="v1",
        rules=dict(
            representative="quantile_nearest on sft local_ce p10/50/90, pretrained delta_abs p10/50/90, "
                           "sft token_acc p80/50/20; tie-break manifest index ascending; dedup keeps first rule",
            case_studies="curated, labeled 'Illustrative case study', NOT statistically representative"),
        n_total=len(sel),
        examples=sel)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"saved {OUT}: {len(sel)} examples "
          f"({sum(1 for s in sel if s.get('category')=='representative')} representative, "
          f"{sum(1 for s in sel if s.get('category')=='case_study')} case studies)")
    for s in sel:
        if "manifest_index" in s:
            print(f"  {s['sample_id']} [{s['category']:>14}] {s['selection_reason'][:80]}")


if __name__ == "__main__":
    main()
