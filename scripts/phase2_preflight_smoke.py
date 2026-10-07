"""Phase 2 preflight smoke（GPU，frozen Phase 2 协议口径验证，不启动 bulk）。

- 模型：v21pilot-H1-2500 EMA（Level C fp32 镜像 forward）
- 样本：frozen ladder 32 子集的前 16 个（确定性；provenance 记录）
- decoder：block {1,2,4,8} × confidence-first greedy × 16 样本；sampled（block=2、
  seed=0、T=1.0）× 4 样本
- 验证：NFE accounting（total == rounds、select == 0）、block=1 为 sequential
  参考（每轮恰 1 位置）、确定性（重跑逐位一致）、endpoints 有效范围、无 NaN
- pairs 评估器（δ/D_ab/JS_dep/CE）同 16 样本：有限性 + JS_dep ∈ [0, ln2]
- 输出：results/phase2/preflight_smoke.json

用法: .venv/bin/python scripts/phase2_preflight_smoke.py
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from evaluation.eval_diag_fp32 import MANIFEST, MANIFEST_SHA, load_model
from evaluation.phase2_block_decoder import block_decode
import evaluation.eval_diag_fp32_pairs as pairs_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "phase2")
MODEL_DIR = os.path.join(ROOT, "exp_local/regime_a/v21pilot-H1-230242")
CKPT = "checkpoint_2500.pth"
LADDER = os.path.join(ROOT, "results/phase1_preflight/ladder_subset_indices.json")
N_SAMPLES = 16


def main():
    digest_man = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    assert digest_man == MANIFEST_SHA, "manifest sha 变化，硬停"
    ladder = json.load(open(LADDER))
    indices = ladder["indices"][:N_SAMPLES]
    records = [json.loads(line) for line in open(MANIFEST)]
    records = [records[k] for k in indices]

    model, step, param_dtypes = load_model(MODEL_DIR, CKPT, "ema")
    model.eval()

    report = dict(model="v21pilot-H1-2500", step=step,
                  sample_indices=indices, n_samples=N_SAMPLES,
                  ladder_subset_sha256=hashlib.sha256(open(LADDER, "rb").read()).hexdigest(),
                  decoder={}, pairs={}, checks={})

    print("=== block decoding smoke（confidence-first greedy）===")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    dec = {}
    for bs in (1, 2, 4, 8):
        rows = []
        for k in indices:
            r = records[indices.index(k)]
            res = block_decode(
                model,
                torch.tensor(r["initial_state"], device="cuda"),
                torch.tensor(r["x0"], device="cuda"),
                r["sigma"], (r["span_start"], r["span_end"]),
                block_size=bs, select="confidence", mode="greedy", seed=0)
            rows.append(dict(sample=k, recovery=res["token_recovery"],
                             exact=res["exact_span"], pseudo_nll=res["pseudo_nll"],
                             n_rounds=res["n_rounds"], nfe=res["nfe"]))
        rec = float(np.mean([x["recovery"] for x in rows]))
        nfe_ok = all(x["nfe"]["total"] == x["n_rounds"] and x["nfe"]["select"] == 0
                     for x in rows)
        dec[f"block{bs}"] = dict(mean_recovery=rec, nfe_accounting_ok=nfe_ok, rows=rows)
        print(f"  block={bs}: mean recovery={rec:.4f} NFE-ok={nfe_ok}")

    # 确定性检查（block=4 重跑两次逐位一致）
    r0 = records[0]
    a = block_decode(model, torch.tensor(r0["initial_state"], device="cuda"),
                     torch.tensor(r0["x0"], device="cuda"), r0["sigma"],
                     (r0["span_start"], r0["span_end"]), 4, "confidence", "greedy")
    b = block_decode(model, torch.tensor(r0["initial_state"], device="cuda"),
                     torch.tensor(r0["x0"], device="cuda"), r0["sigma"],
                     (r0["span_start"], r0["span_end"]), 4, "confidence", "greedy")
    det_ok = torch.equal(a["final_state"], b["final_state"]) and \
        a["pseudo_nll"] == b["pseudo_nll"]
    print(f"  determinism（block=4 重跑）: {det_ok}")

    # block=1 参考性质：每轮恰 1 位置
    b1 = block_decode(model, torch.tensor(r0["initial_state"], device="cuda"),
                      torch.tensor(r0["x0"], device="cuda"), r0["sigma"],
                      (r0["span_start"], r0["span_end"]), 1, "confidence", "greedy")
    b1_ok = all(len(rnd["positions"]) == 1 for rnd in b1["rounds"])
    print(f"  block=1 每轮 1 位置: {b1_ok}")

    # sampled 模式（block=2, seed=0, T=1.0）× 4 样本
    samp = []
    for k in indices[:4]:
        r = records[indices.index(k)]
        res = block_decode(model, torch.tensor(r["initial_state"], device="cuda"),
                           torch.tensor(r["x0"], device="cuda"), r["sigma"],
                           (r["span_start"], r["span_end"]), 2, "confidence",
                           "sampled", seed=0)
        samp.append(dict(sample=k, recovery=res["token_recovery"],
                         nfe=res["nfe"]))
    print(f"  sampled 模式（block=2 × 4 样本）: recovery="
          f"{np.mean([s['recovery'] for s in samp]):.4f}")

    report["decoder"] = dict(by_block=dec, determinism_ok=det_ok,
                             block1_one_per_round_ok=b1_ok, sampled=samp)
    report["checks"]["decoder"] = dict(
        nfe_total_eq_rounds=all(x["nfe_accounting_ok"] for x in dec.values()),
        nfe_select_zero=all(all(row["nfe"]["select"] == 0 for row in x["rows"])
                            for x in dec.values()),
        deterministic=det_ok, block1_reference=b1_ok,
        recovery_in_range=all(0 <= x["mean_recovery"] <= 1 for x in dec.values()))

    print("=== pairs 评估器 smoke（δ/D_ab/JS_dep/CE，16 样本）===")
    payload = pairs_mod.run(model, step, param_dtypes, MODEL_DIR, CKPT, "ema",
                            8, "phase2-preflight-pairs",
                            os.path.join(OUT_DIR, "preflight_pairs.json"))
    ps = payload["per_sample"]
    js = np.array(ps["js_dep"])
    checks = dict(
        all_finite=all(np.isfinite(ps[k]).all() for k in
                       ("delta", "js_dep", "d_ab", "local_ce")),
        js_in_bounds=bool((js >= 0).all() and (js <= np.log(2) + 1e-9).all()),
        delta_algebra=bool(np.allclose(np.array(ps["delta"]),
                                       np.array(ps["pmi_ab"]) - np.array(ps["pmi_ba"]),
                                       atol=1e-9)),
    )
    report["pairs"] = dict(js_dep_mean=float(js.mean()), checks=checks)
    print(f"  JS_dep mean={js.mean():.4f} checks={checks}")

    report["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    report["runtime_s"] = time.time() - t0
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "preflight_smoke.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nVRAM {report['vram_gb']:.2f}GB | {report['runtime_s']:.1f}s")
    print(f"saved: {os.path.join(OUT_DIR, 'preflight_smoke.json')}")


if __name__ == "__main__":
    main()
