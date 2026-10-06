"""v2.1 §9.5 step2500 专属 frozen 子集 OrderGap + path-score variance
（protocol mechanism_complementary_exposure_v2_1 §9.5；secondary，不入 G3c primary）。

- 冻结 64 索引子集（exp_local/regime_a/mechpilot_v21_maps/order_gap_subset_indices.json
  + .sha256，与 maps 同期冻结；H/U/L、rep1/rep2 完全共享）
- frozen 6 路径（manifest paths，同 eval_order_gap 口径）；复用
  compatibility/order_gap 的 evaluate_path_Qs（无 RNG：manifest + 固定路径确定）
- 指标：OrderGap_raw / per-token（同 frozen evaluator 口径）+ path-score variance
  （per sample 6 条冻结路径 Q 的方差，跨样本平均）
- 只在 step2500（+step0）上跑，不跑全 checkpoint；只读，不修改任何冻结资产

用法:
  .venv/bin/python evaluation/eval_order_gap_subset.py --model_path DIR \
      --ckpt checkpoint_2500.pth --weights ema --tag v21-og-H1-2500 \
      --out results/mechanism_pilot_v21/order_gap_subset_H1_2500.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hydra import initialize, compose
from omegaconf import OmegaConf

from model import SEDD
from model.ema import ExponentialMovingAverage
from model import utils as mutils
import compatibility.order_gap as og

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def verify_manifest(manifest_path):
    digest = hashlib.sha256(open(manifest_path, "rb").read()).hexdigest()
    assert digest == open(manifest_path + ".sha256").read().strip(), \
        "manifest SHA-256 不匹配! (§5A 冻结规则)"
    return digest


def load_subset_indices(subset_path):
    digest = hashlib.sha256(open(subset_path, "rb").read()).hexdigest()
    assert digest == open(subset_path + ".sha256").read().strip(), \
        "subset 索引文件被改动（§9.5 冻结规则）"
    subset = json.load(open(subset_path))
    return subset["indices"], digest


def load_model_for_eval(model_path, device, ckpt=None, weights="ema"):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    model = SEDD(cfg).to(device).eval()
    ckpt_path = os.path.join(model_path, ckpt) if ckpt else os.path.join(
        model_path, "checkpoints-meta", "checkpoint.pth")
    if os.path.exists(ckpt_path):
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        if weights == "ema":
            ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
            ema.load_state_dict(loaded["ema"])
            ema.copy_to(model.parameters())
        elif weights == "raw":
            model.load_state_dict(loaded["model"], strict=False)
        else:
            raise ValueError(f"weights 必须是 ema|raw, 得到 {weights}")
        step = loaded["step"]
        source = f"local-{weights} ({model_path})"
    else:
        hf = SEDD.from_pretrained(model_path)
        model.load_state_dict(hf.state_dict(), strict=False)
        del hf
        step = None
        source = model_path
    return model, step, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=os.path.join(ROOT, "protocol/regime_a_protocol.yaml"))
    parser.add_argument("--subset", default=os.path.join(
        ROOT, "exp_local/regime_a/mechpilot_v21_maps/order_gap_subset_indices.json"))
    parser.add_argument("--model_path", default="louaaron/sedd-small")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--weights", default="ema", choices=["ema", "raw"])
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--tag", default="v21-og-subset")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    proto = OmegaConf.load(args.protocol)
    manifest_path = os.path.join(ROOT, proto.eval_manifest.file)
    digest = verify_manifest(manifest_path)
    indices, subset_digest = load_subset_indices(args.subset)
    path_types = list(proto.eval_manifest.path_types)

    device = torch.device("cuda")
    model, step, source = load_model_for_eval(args.model_path, device, args.ckpt, args.weights)
    score_fn = mutils.get_score_fn(model, train=False, sampling=False)
    print(f"模型: {source}" + (f" (step {step})" if step else ""))
    print(f"subset: n={len(indices)} sha={subset_digest[:16]}…")

    records_all = [json.loads(line) for line in open(manifest_path)]
    records = [records_all[i] for i in indices]

    torch.cuda.reset_peak_memory_stats()
    Q_by_path = {}
    for ptype in path_types:
        samples = [dict(x_t=torch.tensor(r["initial_state"], device=device),
                        x0=torch.tensor(r["x0"], device=device),
                        sigma=r["sigma"],
                        path=r["paths"][ptype])
                   for r in records]
        Q = og.evaluate_path_Qs(score_fn, samples, D=50258, chunk=args.chunk)
        Q_by_path[ptype] = Q.cpu()
        print(f"  {ptype:12s}: Q mean {Q.mean().item():8.3f} ± "
              f"{Q.std().item() / Q.numel() ** 0.5:.3f} (nats, n={Q.numel()})")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    Qs_fixed = torch.stack([Q_by_path[p] for p in path_types], dim=0)   # [6, N]
    m = torch.tensor([len(r["paths"]["l2r"]) for r in records], dtype=torch.float32)
    order_gap = Qs_fixed.max(0).values - Qs_fixed.min(0).values
    og_per_token = order_gap / m
    path_score_var = Qs_fixed.var(0, unbiased=True)                     # §9.5 path-score variance

    def sem(v):
        return v.std(unbiased=True).item() / v.numel() ** 0.5

    payload = dict(
        tag=args.tag,
        model_path=args.model_path,
        ckpt=args.ckpt,
        weights=args.weights,
        step=step,
        manifest_file=proto.eval_manifest.file,
        manifest_sha256=digest,
        subset_file=os.path.basename(args.subset),
        subset_sha256=subset_digest,
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip() or "unknown",
        protocol_version="v2.1",
        n_samples=len(records),
        summary=dict(
            order_gap_raw=order_gap.mean().item(),
            order_gap_raw_sem=sem(order_gap),
            order_gap_per_token=og_per_token.mean().item(),
            order_gap_per_token_sem=sem(og_per_token),
            path_score_variance_mean=path_score_var.mean().item(),
            path_score_variance_sem=sem(path_score_var),
            mean_m=m.mean().item(),
            Q_by_path={p: dict(mean=Q_by_path[p].mean().item(), sem=sem(Q_by_path[p]))
                       for p in path_types},
        ),
        peak_vram_gb=peak_gb,
        per_sample=dict(
            order_gap_raw=[float(x) for x in order_gap.tolist()],
            order_gap_per_token=[float(x) for x in og_per_token.tolist()],
            path_score_variance=[float(x) for x in path_score_var.tolist()],
            manifest_index=indices,
            Q_by_path={p: [float(x) for x in Q_by_path[p].tolist()] for p in path_types},
        ),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    s = payload["summary"]
    print("=" * 60)
    print(f"v2.1 §9.5 OrderGap subset ({args.tag})")
    print(f"  OrderGap_raw:     {s['order_gap_raw']:.4f} ± {s['order_gap_raw_sem']:.4f} nats")
    print(f"  OrderGap/token:   {s['order_gap_per_token']:.4f} ± "
          f"{s['order_gap_per_token_sem']:.4f}")
    print(f"  path-score Var:   {s['path_score_variance_mean']:.4f} ± "
          f"{s['path_score_variance_sem']:.4f}")
    print(f"结果: {args.out}")


if __name__ == "__main__":
    main()
