"""Phase 1 FP32 diagnostic evaluator 的 GPU 单测（DIAG_PROTOCOL_VERSION=v1）。

覆盖（用户验收清单）：
  0. mirror fidelity：镜像在人为 bf16 块精度下与 frozen forward 逐位一致
     （证明 Level C 与 frozen 的差别只来自 precision policy，不是镜像写错）
  A. deterministic rerun：同 checkpoint/samples 两次运行逐位一致
  B. no bf16 lattice：NLL off-bf16-grid fraction 显著非零（不得重现 frozen lattice）
  C. chunk invariance：chunk 1/2/4/8 下 CPI/CE 在 fp32 容差内一致
  D. no RNG contamination：运行前后 global CPU/CUDA RNG state 不变
  E. batch-order invariance：样本重排恢复后 per-sample 输出一致

用法: .venv/bin/python scripts/test_diag_fp32.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from evaluation.eval_diag_fp32 import (fp32_forward, load_model, load_samples,
                                       run_eval, summarize)
from model import utils as mutils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "exp_local/regime_a/v21pilot-H1-230242")
CKPT = "checkpoint_2500.pth"
LADDER_SUBSET = os.path.join(ROOT, "results/phase1_preflight/ladder_subset_indices.json")
N_PASS = 0


def ok(name, cond, detail=""):
    global N_PASS
    assert cond, f"{name} FAIL: {detail}"
    N_PASS += 1
    print(f"PASS {name}" + (f"（{detail}）" if detail else ""))


def main():
    print("== 0. mirror fidelity（bf16 块精度下镜像 == frozen forward 逐位一致）==")
    model, step, dtypes = load_model(MODEL_DIR, CKPT, "ema")
    x = torch.randint(0, 50257, (4, 256), device="cuda")
    sigma = torch.tensor([0.5, 0.8, 1.2, 2.0], device="cuda")
    with torch.no_grad():
        out_frozen = model(x, sigma)

        def mirror_bf16(m, indices, sigma_):
            import torch.nn.functional as F
            xm = m.vocab_embed(indices)
            c = F.silu(m.sigma_map(sigma_))
            rotary_cos_sin = m.rotary_emb(xm)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for i in range(len(m.blocks)):
                    xm = m.blocks[i](xm, rotary_cos_sin, c, seqlens=None)
                xm = m.output_layer(xm, c)
            if m.scale_by_sigma:
                esigm1_log = torch.where(sigma_ < 0.5, torch.expm1(sigma_),
                                         sigma_.exp() - 1).log().to(xm.dtype)[:, None, None]
                xm = xm - esigm1_log - np.log(xm.shape[-1] - 1)
            xm = torch.scatter(xm, -1, indices[..., None], torch.zeros_like(xm[..., :1]))
            return xm

        out_mirror = mirror_bf16(model, x, sigma)
    ok("mirror_fidelity_bitwise", torch.equal(out_mirror, out_frozen),
       "bf16 块精度下镜像与 frozen forward 逐位一致")

    print("== 数据加载（32-sample ladder subset）==")
    samples, indices = load_samples(json.load(open(LADDER_SUBSET))["indices"])
    assert len(samples) == 32

    print("== A. deterministic rerun ==")
    r1 = run_eval(model, samples, chunk=8)
    r2 = run_eval(model, samples, chunk=8)
    s1, s2 = summarize(r1["per_sample"], r1["nll_values"]), summarize(r2["per_sample"], r2["nll_values"])
    same = all(np.array_equal(np.array(r1["per_sample"][k]), np.array(r2["per_sample"][k]))
               for k in ("delta", "delta_abs", "local_ce", "token_acc"))
    ok("deterministic_rerun", same and s1["cpi_abs"] == s2["cpi_abs"],
       f"CPI {s1['cpi_abs']:.6f} vs {s2['cpi_abs']:.6f}")

    print("== B. no bf16 lattice ==")
    frac = s1["nll_lattice"]["off_bf16_grid_fraction"]
    ok("no_bf16_lattice", frac > 0.9, f"off-bf16 {frac:.1%}（frozen 为 0%）")

    print("== C. chunk invariance（1/2/4/8）==")
    ref_cpi, ref_ce = None, None
    for ck in (1, 2, 4, 8):
        r = run_eval(model, samples, chunk=ck)
        s = summarize(r["per_sample"], r["nll_values"])
        if ref_cpi is None:
            ref_cpi, ref_ce = s["cpi_abs"], s["local_ce"]
        else:
            assert abs(s["cpi_abs"] - ref_cpi) < 1e-4, f"chunk={ck} CPI 漂移"
            assert abs(s["local_ce"] - ref_ce) < 1e-4, f"chunk={ck} CE 漂移"
    ok("chunk_invariance", True, f"CPI {ref_cpi:.6f} / CE {ref_ce:.6f} 全 chunk 一致(<1e-4)")

    print("== D. no RNG contamination ==")
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    run_eval(model, samples, chunk=8)
    ok("rng_isolation", torch.equal(cpu_state, torch.get_rng_state())
       and torch.equal(cuda_state, torch.cuda.get_rng_state()),
       "evaluator 不推进全局 CPU/CUDA RNG")

    print("== E. batch-order invariance ==")
    order = list(range(len(samples)))
    shuffled = [samples[i] for i in order[::-1]]
    r_orig = run_eval(model, samples, chunk=8)
    r_shuf = run_eval(model, shuffled, chunk=8)
    max_diff = max(abs(np.array(r_orig["per_sample"][k]) - np.array(r_shuf["per_sample"][k])[::-1]).max()
                   for k in ("delta", "delta_abs", "local_ce", "token_acc"))
    ok("batch_order_invariance", max_diff < 1e-6, f"max per-sample diff={max_diff:.2e}")

    print(f"\n全部 {N_PASS} 项 GPU 单测通过 ✅")


if __name__ == "__main__":
    main()
