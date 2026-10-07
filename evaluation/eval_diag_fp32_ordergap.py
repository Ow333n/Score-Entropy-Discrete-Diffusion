"""Phase 1 FP32 finite-path OrderGap（DIAG_PROTOCOL_VERSION=v1，precision 同 Level C）。

- frozen §9.5 64 样本子集（exp_local/regime_a/mechpilot_v21_maps/
  order_gap_subset_indices.json + sha256 sidecar 校验）
- frozen 6 路径（manifest paths：l2r/r2l/random_0/1/2/confidence）——
  **finite-path OrderGap**（明确不声称枚举 m!）
- score_fn = fp32 镜像 forward（复用 evaluation/eval_diag_fp32.py 的
  fp32_forward，不改任何 frozen 文件）；logp 提取 fp32（compatibility/order_gap.py
  的 evaluate_path_Qs + clean_log_probs 同口径）；聚合 fp64
- 输出：OrderGap_raw / per-token / path-score variance + per-sample + provenance

用法:
  .venv/bin/python evaluation/eval_diag_fp32_ordergap.py --model_path DIR \
      --ckpt checkpoint_2500.pth --weights ema --tag v21-H1-2500 \
      --out results/phase1_diag/ordergap/v21-H1-2500.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

import compatibility.order_gap as og
from evaluation.eval_diag_fp32 import fp32_forward, load_model, MANIFEST, MANIFEST_SHA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 50258
DIAG_PROTOCOL_VERSION = "v1"
PRECISION_MODE = "true_fp32_mirrored_forward"
SUBSET = os.path.join(ROOT, "exp_local/regime_a/mechpilot_v21_maps",
                      "order_gap_subset_indices.json")


def run(model, step, param_dtypes, model_path, ckpt, weights, chunk, tag, out):
    """核心计算 + payload + 落盘（供 standalone main() 与合并执行器共用）。

    数学与输出 schema 与 standalone 逐字段一致（模型由调用方加载）。
    """
    subset = json.load(open(SUBSET))
    sub_digest = hashlib.sha256(open(SUBSET, "rb").read()).hexdigest()
    assert sub_digest == open(SUBSET + ".sha256").read().strip(), "subset 被改动"
    indices = subset["indices"]

    proto = OmegaConf.load(os.path.join(ROOT, "protocol/regime_a_protocol.yaml"))
    path_types = list(proto.eval_manifest.path_types)

    score_fn = lambda x, sigma: fp32_forward(model, x, sigma)

    records = [json.loads(line) for line in open(MANIFEST)]
    records = [records[k] for k in indices]

    torch.cuda.reset_peak_memory_stats()
    Q_by_path = {}
    for ptype in path_types:
        samples = [dict(x_t=torch.tensor(r["initial_state"], device="cuda"),
                        x0=torch.tensor(r["x0"], device="cuda"),
                        sigma=r["sigma"], path=r["paths"][ptype])
                   for r in records]
        Q = og.evaluate_path_Qs(score_fn, samples, D=D, chunk=chunk)
        Q_by_path[ptype] = Q.cpu().double()
    peak = torch.cuda.max_memory_allocated() / 1e9

    Qs = torch.stack([Q_by_path[p] for p in path_types], dim=0)
    m = torch.tensor([len(r["paths"]["l2r"]) for r in records], dtype=torch.float64)
    order_gap = Qs.max(0).values - Qs.min(0).values
    og_per_token = order_gap / m
    path_var = Qs.var(0, unbiased=True)

    def sem(v):
        return v.std(unbiased=True).item() / v.numel() ** 0.5

    payload = dict(
        tag=tag,
        diag_protocol_version=DIAG_PROTOCOL_VERSION,
        precision_mode=PRECISION_MODE,
        evaluator_file=os.path.relpath(os.path.abspath(__file__), ROOT),
        evaluator_sha256=hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        model_path=model_path, ckpt=ckpt, weights=weights, step=step,
        model_param_dtypes=param_dtypes,
        compute_dtype="fp32（镜像 forward）", aggregation_dtype="fp64",
        manifest_file=os.path.relpath(MANIFEST, ROOT), manifest_sha256=MANIFEST_SHA,
        subset_file=os.path.relpath(SUBSET, ROOT), subset_sha256=sub_digest,
        n_samples=len(records), n_paths=len(path_types), path_types=path_types,
        finite_path_note="finite-path OrderGap：固定 6 条冻结路径，不声称枚举 m! 排列",
        summary=dict(
            order_gap_raw=order_gap.mean().item(),
            order_gap_raw_sem=sem(order_gap),
            order_gap_per_token=og_per_token.mean().item(),
            order_gap_per_token_sem=sem(og_per_token),
            path_score_variance_mean=path_var.mean().item(),
            path_score_variance_sem=sem(path_var),
            mean_m=m.mean().item(),
            Q_by_path={p: dict(mean=Q_by_path[p].mean().item(), sem=sem(Q_by_path[p]))
                       for p in path_types}),
        per_sample=dict(
            manifest_index=indices,
            order_gap_raw=[float(x) for x in order_gap.tolist()],
            order_gap_per_token=[float(x) for x in og_per_token.tolist()],
            path_score_variance=[float(x) for x in path_var.tolist()],
            Q_by_path={p: [float(x) for x in Q_by_path[p].tolist()] for p in path_types}),
        peak_vram_gb=peak)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    s = payload["summary"]
    print(f"FP32 finite-path OrderGap（{tag}）: OG_raw={s['order_gap_raw']:.4f} ± "
          f"{s['order_gap_raw_sem']:.4f} | per-token={s['order_gap_per_token']:.4f} | "
          f"path-var={s['path_score_variance_mean']:.4f} | VRAM {peak:.2f}GB")
    print(f"结果: {out}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--weights", default="ema", choices=["ema", "raw"])
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--tag", default="diag-fp32-og")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    digest_man = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    assert digest_man == MANIFEST_SHA, "manifest sha 变化，硬停"
    model, step, param_dtypes = load_model(args.model_path, args.ckpt, args.weights)
    run(model, step, param_dtypes, args.model_path, args.ckpt, args.weights,
        args.chunk, args.tag, args.out)


if __name__ == "__main__":
    main()
