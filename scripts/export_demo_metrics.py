"""Phase C3: Demo harmonized metric assets + formal curves（纯 JSON 组装，无 GPU）。

按用户裁定（方案 1）：Demo 主链 compatibility 数值统一使用 current frozen
evaluation code 的复算结果。不覆盖/删除任何历史结果文件。

产出:
  demo_assets/metrics/harmonized/{pretrained,sft,rl500}_{cpi,og}_current.json
    —— 每个含 source/provenance/current value/historical value/note + per_sample 副本
  demo_assets/metrics/formal_curves.json —— Tab 4 绘图数据（三阶段链 + RL 任务曲线 + LR probe）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARM = os.path.join(ROOT, "demo_assets/metrics/harmonized")
GIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

SOURCES = {
    # (asset, source file, stage, kind, historical file)
    "pretrained_cpi_current.json": (
        "/tmp/compat_diag/cpi_pretrained_current.json", "pretrained", "cpi",
        os.path.join(ROOT, "results/pretrained/cpi.json")),
    "pretrained_og_current.json": (
        "/tmp/compat_diag/og_pretrained_current.json", "pretrained", "og",
        os.path.join(ROOT, "results/pretrained/order_gap.json")),
    "sft_cpi_current.json": (
        os.path.join(ROOT, "exp_local/regime_a/rlpilot-185545/step0_cpi.json"), "sft", "cpi",
        os.path.join(ROOT, "results/vanilla/cpi_s1_10200.json")),
    "sft_og_current.json": (
        os.path.join(ROOT, "exp_local/regime_a/rlpilot-185545/step0_order_gap.json"), "sft", "og",
        os.path.join(ROOT, "results/vanilla/og_s1_10200.json")),
    "rl500_cpi_current.json": (
        os.path.join(ROOT, "results/rl_pilot/cpi_step500_raw.json"), "rl500", "cpi", None),
    "rl500_og_current.json": (
        os.path.join(ROOT, "results/rl_pilot/og_step500_raw.json"), "rl500", "og", None),
}

PROVENANCE_NOTE_ZH = ("Demo 主链 compatibility 数值统一由当前冻结评估代码复算，保证三阶段完全同口径。"
                      "项目早期 formal 结果由当时未提交的评估代码生成，无法 bit-level 复现；"
                      "历史记录已保留，聚合差异很小，不改变主要定性结论。")
PROVENANCE_NOTE_EN = ("Demo compatibility values are recomputed with the current frozen evaluation "
                      "harness for a consistent Pretrained→SFT→RL comparison. Historical early-run "
                      "aggregates are preserved separately; their exact evaluation implementation was "
                      "not committed and is therefore not bit-reproducible. The small aggregate "
                      "discrepancies do not alter the qualitative conclusions.")

CKPT_SOURCE = {
    "pretrained": "louaaron/sedd-small (HF)",
    "sft": "exp_local/regime_a/formal-vanilla-s1-191414/checkpoint_10200.pth EMA",
    "rl500": "exp_local/regime_a/rlpilot-185545/checkpoint_step500.pth RAW",
}


def main():
    os.makedirs(HARM, exist_ok=True)
    manifest_sha = subprocess.run(
        ["sha256sum", os.path.join(ROOT, "manifests/regime_a_eval_v1.jsonl")],
        capture_output=True, text=True).stdout.split()[0]

    for asset, (src, stage, kind, hist) in SOURCES.items():
        cur = json.load(open(src))
        out = dict(
            asset=asset,
            stage=stage, kind=kind,
            evaluation_git_head=GIT,
            evaluation_code_path="evaluation/eval_cpi.py" if kind == "cpi"
                                 else "evaluation/eval_order_gap.py",
            source_checkpoint=CKPT_SOURCE[stage],
            source_manifest="manifests/regime_a_eval_v1.jsonl",
            manifest_sha256=manifest_sha,
            source_result_file=os.path.relpath(src, ROOT),
            current=dict(
                cpi_abs=cur["summary"]["cpi_abs"], cpi_rms=cur["summary"]["cpi_rms"],
                delta_mean=cur["summary"]["delta_mean"]) if kind == "cpi" else dict(
                order_gap_raw=cur["order_gap_raw"],
                order_gap_per_token=cur["order_gap_per_token"]),
            historical=(dict(
                cpi_abs=json.load(open(hist))["summary"]["cpi_abs"],
                cpi_rms=json.load(open(hist))["summary"]["cpi_rms"]) if hist else None)
            if kind == "cpi" else (dict(
                order_gap_raw=json.load(open(hist))["order_gap_raw"],
                order_gap_per_token=json.load(open(hist))["order_gap_per_token"]) if hist else None),
            per_sample=cur.get("per_sample"),
            provenance_note_zh=PROVENANCE_NOTE_ZH,
            provenance_note_en=PROVENANCE_NOTE_EN,
        )
        path = os.path.join(HARM, asset)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"harmonized: {asset} (source={os.path.relpath(src, ROOT)})")

    # --- formal_curves.json（Tab 4 绘图） ---
    rl_eval = {}
    for tag in ["step0", "step50", "step100", "step250", "step500"]:
        d = json.load(open(os.path.join(
            ROOT, "exp_local/regime_a/rlpilot-185545", f"eval_point_{tag}.json")))
        rl_eval[int(tag[4:])] = dict(
            nll_raw=d["nll_raw"], nll_ema=d["nll_ema"],
            sampled64_raw=d["sampled_raw_mean"], sampled64_ema=d["sampled_ema_mean"],
            greedy_raw=d["greedy_raw_mean"], greedy_ema=d["greedy_ema_mean"])
    cpi_chain = [
        dict(stage="Pretrained", cpi_abs=0.3301, order_gap=10.2246),
        dict(stage="SFT (s1-10200 EMA)", cpi_abs=0.2871, order_gap=8.7475),
        dict(stage="RL-500 RAW", cpi_abs=0.2852, order_gap=8.7442),
    ]
    curves = dict(
        compatibility_main_chain=cpi_chain,
        rl_task=rl_eval,
        lr_probe=json.load(open(os.path.join(ROOT, "reports/lr_probe_summary.json"))),
        provenance=dict(rule="Demo 主链使用 current-code harmonized 值；历史值见 errata 与 harmonized assets",
                        errata="protocol/errata_v4.2_eval_provenance.md"))
    p = os.path.join(ROOT, "demo_assets/metrics/formal_curves.json")
    with open(p, "w") as f:
        json.dump(curves, f, indent=2, ensure_ascii=False)
    print(f"formal_curves: {p}")


if __name__ == "__main__":
    main()
