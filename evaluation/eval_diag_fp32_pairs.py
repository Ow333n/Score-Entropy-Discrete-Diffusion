"""Phase 2 predictor 提取（DIAG_PROTOCOL_VERSION=v1，precision 同 Level C）。

per-sample（frozen manifest 的 (i,j,a,b) pair）从同一 3-forward quartet 提取：
- compatibility：δ、|δ|（frozen δ 定义，fp32 提取 fp64 计算）
- estimation error：CE/NLL（mask 位置 GT NLL 均值，FP32——**不用旧 BF16 CE**）
- dependence：JS_dep（primary distribution-level，协议 C.2）、D_ab 与两方向 PMI
  （secondary GT-conditioned，协议 C.3）
- 伴生：token acc、entropy、confidence

score_fn = fp32 镜像 forward（复用 frozen eval_diag_fp32 的 fp32_forward，不改
任何 frozen 文件）；聚合 fp64。

用法:
  .venv/bin/python evaluation/eval_diag_fp32_pairs.py --model_path DIR \
      --ckpt checkpoint_2500.pth --weights ema --tag v21-H1-2500 \
      --out results/phase2/pairs/v21-H1-2500.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from compatibility.posterior import clean_log_probs
from compatibility.dependence import js_dep, pointwise_terms
from evaluation.eval_diag_fp32 import fp32_forward, load_model, MANIFEST, MANIFEST_SHA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 50258
DIAG_PROTOCOL_VERSION = "v1"
PRECISION_MODE = "true_fp32_mirrored_forward"


def run(model, step, param_dtypes, model_path, ckpt, weights, chunk, tag, out):
    records = [json.loads(line) for line in open(MANIFEST)]
    per = dict(delta=[], delta_abs=[], d_ab=[], pmi_ab=[], pmi_ba=[],
               js_dep=[], js_ab=[], js_ba=[], local_ce=[], token_acc=[],
               entropy=[], confidence=[], manifest_index=[])
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for start in range(0, len(records), chunk):
            ch = records[start:start + chunk]
            c = len(ch)
            x_t = torch.stack([torch.tensor(r["initial_state"]) for r in ch]).to("cuda")
            x0 = torch.stack([torch.tensor(r["x0"]) for r in ch]).to("cuda")
            i = torch.tensor([r["i"] for r in ch], device="cuda")
            j = torch.tensor([r["j"] for r in ch], device="cuda")
            a = torch.tensor([r["a"] for r in ch], device="cuda")
            b = torch.tensor([r["b"] for r in ch], device="cuda")
            sigma = torch.tensor([r["sigma"] for r in ch], device="cuda")
            idx = torch.arange(c)
            s_C = fp32_forward(model, x_t, sigma)
            x_Ca = x_t.clone(); x_Ca[idx, i] = a
            x_Cb = x_t.clone(); x_Cb[idx, j] = b
            s_Ca = fp32_forward(model, x_Ca, sigma)
            s_Cb = fp32_forward(model, x_Cb, sigma)
            logp_C = clean_log_probs(s_C, D)
            logp_Ca = clean_log_probs(s_Ca, D)
            logp_Cb = clean_log_probs(s_Cb, D)
            mask = x_t == D - 1
            for k in range(c):
                ii, jj = int(i[k]), int(j[k])
                pt = pointwise_terms(logp_C[k], logp_Ca[k], logp_Cb[k],
                                     ii, jj, int(a[k]), int(b[k]))
                js = js_dep(logp_C[k], logp_Ca[k], logp_Cb[k], ii, jj)
                per["delta"].append(float(pt["delta"].item()))
                per["delta_abs"].append(float(abs(pt["delta"].item())))
                per["d_ab"].append(float(pt["d_ab"].item()))
                per["pmi_ab"].append(float(pt["pmi_ab"].item()))
                per["pmi_ba"].append(float(pt["pmi_ba"].item()))
                per["js_dep"].append(float(js["js_dep"].item()))
                per["js_ab"].append(float(js["js_ab"].item()))
                per["js_ba"].append(float(js["js_ba"].item()))
                pos = mask[k].nonzero().squeeze(1)
                gold = x0[k, pos]
                lp = logp_C[k, pos].double()
                per["local_ce"].append((-lp[torch.arange(pos.numel()), gold]).mean().item())
                per["token_acc"].append((lp.argmax(-1) == gold).float().mean().item())
                p = lp.exp()
                per["entropy"].append((-p * lp).sum(-1).mean().item())
                per["confidence"].append(lp.max(-1).values.exp().mean().item())
                per["manifest_index"].append(start + k)
            del s_C, s_Ca, s_Cb, logp_C, logp_Ca, logp_Cb, x_t, x0
    peak = torch.cuda.max_memory_allocated() / 1e9
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
        compute_dtype="fp32（镜像 forward）", extraction_dtype="fp32", aggregation_dtype="fp64",
        manifest_file=os.path.relpath(MANIFEST, ROOT), manifest_sha256=MANIFEST_SHA,
        n_samples=len(records), chunk=chunk,
        per_sample=per, peak_vram_gb=peak)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"FP32 pairs（{tag}）: n={len(records)} |δ|mean={sum(per['delta_abs'])/len(per['delta_abs']):.4f} "
          f"JS_dep mean={sum(per['js_dep'])/len(per['js_dep']):.4f} "
          f"D_ab mean={sum(per['d_ab'])/len(per['d_ab']):+.4f} VRAM {peak:.2f}GB")
    print(f"结果: {out}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--weights", default="ema", choices=["ema", "raw"])
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--tag", default="diag-fp32-pairs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    digest_man = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    assert digest_man == MANIFEST_SHA, "manifest sha 变化，硬停"
    model, step, param_dtypes = load_model(args.model_path, args.ckpt, args.weights)
    run(model, step, param_dtypes, args.model_path, args.ckpt, args.weights,
        args.chunk, args.tag, args.out)


if __name__ == "__main__":
    main()
