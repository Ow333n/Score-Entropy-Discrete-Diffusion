"""Phase 1 FP32 diagnostic evaluator（DIAG_PROTOCOL_VERSION=v1；precision-ladder 裁决
Level C = true-FP32 mirrored scoring forward，2026-10-07）。

定位：
- 新 versioned diagnostic evaluator，**绝不替换** frozen v1.2/v2.1 evaluator
- 唯一目的：移除 frozen forward 中 transformer blocks + output layer 内部的 bf16
  autocast（model/transformer.py:282），使 scoring forward 真正 FP32
- **不修改任何 frozen 文件**（model/transformer.py / losses.py / compatibility/* /
  evaluation/eval_cpi*.py 零改动）；镜像 forward 以相同 frozen 子模块按相同调用顺序
  复刻 transformer.py L275–305，仅去掉 bf16 autocast

精度政策（precision_mode = "true_fp32_mirrored_forward"）：
- model.eval() + torch.no_grad()；dropout 评估时禁用（eval 模式）
- 参数权重不转换、不修改、不重新保存（模型参数原 dtype 记录进输出）
- scoring forward：fp32（镜像去 autocast）
- log-prob / score extraction：fp32（score 已 fp32，log_softmax fp32）
- aggregation（均值/分位数/SD）：fp64
- 明确剩余限制：本 evaluator 消除 frozen 路径的 bf16 模型级量化与提取量化；
  不声称"FP32 = ground truth"（模型参数与训练过程本身的近似不变）

输出指标（frozen manifest 口径，delta-swap 3 forward/pair）：
- CPI_abs / CPI_RMS / signed δ mean / δ_SD / |δ| quantiles（P50/P75/P90/P95）/
  signed δ quantiles（P10/P25/P50/P75/P90/P95）/ local CE（masked-position NLL）/
  token acc / NLL lattice stats（off-bf16-grid fraction）
- per-sample 全落盘（delta / delta_abs / local_ce / token_acc，fp64）
- provenance：DIAG_PROTOCOL_VERSION / precision_mode / source hash（git + 本文件
  sha256）/ manifest sha / checkpoint 信息 / VRAM / runtime / chunk

用法:
  .venv/bin/python evaluation/eval_diag_fp32.py --model_path DIR \
      --ckpt checkpoint_2500.pth --weights ema --tag v21-H1-2500 \
      --out results/phase1_diag/H1_2500.json
  （--indices-file 指定子集索引 JSON；默认全 manifest 500）
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from hydra import initialize, compose

from model import SEDD
from model.ema import ExponentialMovingAverage
from compatibility.posterior import clean_log_probs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 50258
DIAG_PROTOCOL_VERSION = "v1"
PRECISION_MODE = "true_fp32_mirrored_forward"
MANIFEST = os.path.join(ROOT, "manifests/regime_a_eval_v1.jsonl")
MANIFEST_SHA = "1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3"


# --------------------------------------------------------------------------
# fp32 镜像 forward
# --------------------------------------------------------------------------

def fp32_forward(model, indices, sigma):
    """镜像 model/transformer.py forward（L275–305）的调用顺序，唯一改动：
    去掉 blocks 循环 + output_layer 外层的 `autocast(dtype=bfloat16)`（transformer.py:282）。

    忠实性（mirror-fidelity）由 GPU 单测保证：若人为恢复 bf16 块精度，本镜像与
    frozen forward 同输入逐位一致——证明与 frozen 的差别只来自 precision policy。
    """
    x = model.vocab_embed(indices)
    c = F.silu(model.sigma_map(sigma))
    rotary_cos_sin = model.rotary_emb(x)
    for i in range(len(model.blocks)):
        x = model.blocks[i](x, rotary_cos_sin, c, seqlens=None)
    x = model.output_layer(x, c)
    if model.scale_by_sigma:
        esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma),
                                 sigma.exp() - 1).log().to(x.dtype)[:, None, None]
        x = x - esigm1_log - np.log(x.shape[-1] - 1)
    x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
    return x


# --------------------------------------------------------------------------
# 模型加载（与 frozen evaluator 同款容器语义，权重零修改）
# --------------------------------------------------------------------------

def load_model(model_path, ckpt, weights="ema"):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="vanilla_256")
    model = SEDD(cfg).to("cuda").eval()
    loaded = torch.load(os.path.join(model_path, ckpt), map_location="cuda", weights_only=False)
    if weights == "ema":
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
        ema.load_state_dict(loaded["ema"])
        ema.copy_to(model.parameters())
    elif weights == "raw":
        model.load_state_dict(loaded["model"], strict=False)
    else:
        raise ValueError(f"weights 必须是 ema|raw，得到 {weights}")
    step = loaded["step"]
    param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    return model, step, param_dtypes


# --------------------------------------------------------------------------
# 评估核心（fp32 提取 + fp64 聚合）
# --------------------------------------------------------------------------

def run_eval(model, samples, chunk=8):
    """delta-swap 3 forward/pair（fp32 镜像 forward）+ local CE + token acc。

    返回 dict（全部 fp64/float）：
      delta, delta_abs, local_ce, token_acc（per-sample）、
      nll_values（全部 masked-position NLL，lattice 检测用）、
      vram_gb, runtime_s
    """
    device = "cuda"
    per_sample = dict(delta=[], delta_abs=[], local_ce=[], token_acc=[])
    nll_values = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(samples), chunk):
            ch = samples[start:start + chunk]
            c = len(ch)
            x_t = torch.stack([s["x_t"] for s in ch]).to(device)
            x0 = torch.stack([s["x0"] for s in ch]).to(device)
            i = torch.tensor([s["i"] for s in ch], device=device)
            j = torch.tensor([s["j"] for s in ch], device=device)
            a = torch.tensor([s["a"] for s in ch], device=device)
            b = torch.tensor([s["b"] for s in ch], device=device)
            sigma = torch.tensor([s["sigma"] for s in ch], device=device)
            idx = torch.arange(c)

            s_C = fp32_forward(model, x_t, sigma)
            x_Ca = x_t.clone(); x_Ca[idx, i] = a
            x_Cb = x_t.clone(); x_Cb[idx, j] = b
            s_Ca = fp32_forward(model, x_Ca, sigma)
            s_Cb = fp32_forward(model, x_Cb, sigma)

            logp_C = clean_log_probs(s_C, D)          # fp32 log_softmax（score 已 fp32）
            logp_Ca = clean_log_probs(s_Ca, D)
            logp_Cb = clean_log_probs(s_Cb, D)

            delta = (logp_C[idx, i, a].double() + logp_Ca[idx, j, b].double()
                     - logp_C[idx, j, b].double() - logp_Cb[idx, i, a].double())
            per_sample["delta"].extend(delta.tolist())
            per_sample["delta_abs"].extend(delta.abs().tolist())

            mask = x_t == D - 1
            for k in range(c):
                pos = mask[k].nonzero().squeeze(1)
                gold = x0[k, pos]
                nll = (-logp_C[k, pos, gold]).double()
                nll_values.extend(nll.tolist())
                per_sample["local_ce"].append(nll.mean().item())
                per_sample["token_acc"].append(
                    (logp_C[k, pos].argmax(-1) == gold).float().mean().item())
            del s_C, s_Ca, s_Cb, logp_C, logp_Ca, logp_Cb, x_t, x0
    runtime = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    return dict(per_sample=per_sample, nll_values=np.array(nll_values),
                vram_gb=vram, runtime_s=runtime)


def summarize(per_sample, nll_values):
    delta = torch.tensor(per_sample["delta"], dtype=torch.float64)
    ad = delta.abs()
    ce = torch.tensor([v for v in per_sample["local_ce"]], dtype=torch.float64)
    acc = torch.tensor(per_sample["token_acc"], dtype=torch.float64)

    def sem(v):
        return v.std(unbiased=True).item() / v.numel() ** 0.5

    def q(v, p):
        return torch.quantile(v, p).item()

    nll_t = torch.tensor(nll_values, dtype=torch.float64)
    off_bf16 = float((nll_t.to(torch.bfloat16).double() != nll_t).float().mean())
    off_fp16 = float((nll_t.half().double() != nll_t).float().mean())

    return dict(
        n_samples=delta.numel(),
        cpi_abs=ad.mean().item(),
        cpi_abs_sem=sem(ad),
        cpi_rms=(delta ** 2).mean().sqrt().item(),
        delta_mean=delta.mean().item(),
        delta_sem=sem(delta),
        delta_sd=delta.std(unbiased=True).item(),
        delta_abs_quantiles={f"p{p}": q(ad, p / 100) for p in (50, 75, 90, 95)},
        delta_quantiles={f"p{p}": q(delta, p / 100) for p in (10, 25, 50, 75, 90, 95)},
        local_ce=ce.mean().item(),
        local_ce_sem=sem(ce),
        token_acc=acc.mean().item(),
        token_acc_sem=sem(acc),
        nll_lattice=dict(n_positions=len(nll_values),
                         off_bf16_grid_fraction=off_bf16,
                         off_fp16_grid_fraction=off_fp16),
    )


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------

def load_samples(indices=None):
    records = [json.loads(line) for line in open(MANIFEST)]
    if indices is None:
        indices = list(range(len(records)))
    samples = []
    for k in indices:
        r = records[k]
        samples.append(dict(
            x_t=torch.tensor(r["initial_state"], device="cuda"),
            i=r["i"], j=r["j"], a=r["a"], b=r["b"],
            x0=torch.tensor(r["x0"], device="cuda"), sigma=r["sigma"],
            manifest_index=k))
    return samples, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint 文件名；缺省为 checkpoints-meta/checkpoint.pth")
    parser.add_argument("--weights", default="ema", choices=["ema", "raw"])
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--indices-file", default=None,
                        help="子集索引 JSON（ladder subset 等）；缺省全 manifest 500")
    parser.add_argument("--tag", default="diag-fp32")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    digest_man = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    assert digest_man == MANIFEST_SHA, "manifest sha 变化，硬停"

    indices = None
    if args.indices_file:
        sub = json.load(open(args.indices_file))
        idx_digest = hashlib.sha256(open(args.indices_file, "rb").read()).hexdigest()
        expected = open(args.indices_file + ".sha256").read().strip()
        assert idx_digest == expected, "subset 索引文件被改动"
        indices = sub["indices"]

    samples, indices = load_samples(indices)
    model, step, param_dtypes = load_model(args.model_path, args.ckpt, args.weights)

    res = run_eval(model, samples, chunk=args.chunk)
    summ = summarize(res["per_sample"], res["nll_values"])

    evaluator_path = os.path.abspath(__file__)
    payload = dict(
        tag=args.tag,
        diag_protocol_version=DIAG_PROTOCOL_VERSION,
        precision_mode=PRECISION_MODE,
        evaluator_file=os.path.relpath(evaluator_path, ROOT),
        evaluator_sha256=hashlib.sha256(open(evaluator_path, "rb").read()).hexdigest(),
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        model_path=args.model_path,
        ckpt=args.ckpt,
        weights=args.weights,
        step=step,
        model_param_dtypes=param_dtypes,
        compute_dtype="fp32（镜像 forward）",
        extraction_dtype="fp32（log_softmax/δ/CE）",
        aggregation_dtype="fp64",
        manifest_file=os.path.relpath(MANIFEST, ROOT),
        manifest_sha256=MANIFEST_SHA,
        n_samples=len(samples),
        indices_file=os.path.relpath(args.indices_file, ROOT) if args.indices_file else None,
        manifest_indices=indices,
        chunk=args.chunk,
        summary=summ,
        per_sample=dict(
            manifest_index=indices,
            delta=res["per_sample"]["delta"],
            delta_abs=res["per_sample"]["delta_abs"],
            local_ce=res["per_sample"]["local_ce"],
            token_acc=res["per_sample"]["token_acc"],
        ),
        vram_gb=res["vram_gb"],
        runtime_s=res["runtime_s"],
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    s = summ
    print("=" * 60)
    print(f"Phase 1 FP32 diagnostic evaluator（{PRECISION_MODE}，{DIAG_PROTOCOL_VERSION}）")
    print(f"  CPI_abs = {s['cpi_abs']:.4f} ± {s['cpi_abs_sem']:.4f} | RMS = {s['cpi_rms']:.4f} "
          f"| δ̄ = {s['delta_mean']:+.4f} | δ_SD = {s['delta_sd']:.4f}")
    print(f"  |δ| quantiles: { {k: round(v, 4) for k, v in s['delta_abs_quantiles'].items()} }")
    print(f"  local CE = {s['local_ce']:.4f} | token acc = {s['token_acc']:.4f}")
    print(f"  NLL lattice: off-bf16 {s['nll_lattice']['off_bf16_grid_fraction']:.1%} / "
          f"off-fp16 {s['nll_lattice']['off_fp16_grid_fraction']:.1%} "
          f"（n={s['nll_lattice']['n_positions']}）")
    print(f"  VRAM {res['vram_gb']:.2f}GB | {res['runtime_s']:.1f}s | chunk={args.chunk}")
    print(f"结果: {args.out}")


if __name__ == "__main__":
    main()
