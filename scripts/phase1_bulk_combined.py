"""Phase 1 bulk 合并执行器（bulk_manifest FROZEN；单 checkpoint 单次模型加载）。

每个 checkpoint 在同一进程、同一模型实例下依次执行：
  1. core（evaluation/eval_diag_fp32.py 的 run_eval + summarize——frozen 模块函数，
     直接 import，零改动；payload 逐字段复刻其 main()）
  2. finite-path OrderGap（evaluation/eval_diag_fp32_ordergap.py 的 run()）
  3. aux（evaluation/eval_diag_fp32_aux.py 的 run()）
然后卸载模型。

目的：模型加载/卸载从 165 次（3 类型 × 55 checkpoint 独立进程）降到 ≤55 次
（每 checkpoint 一次），减少 WSL/WDDM create_allocation churn。
- 不改变任何 evaluator 数学定义 / sample manifest / FP32 precision policy / 输出 schema
- skip-if-exists 逐文件守卫（与既有 pretrained-step0 core 输出无缝衔接）
- checkpoint 之间执行 gc.collect() + torch.cuda.empty_cache()——常规资源回收
  （进程内多 checkpoint 顺序执行的 hygiene），**不描述为 WDDM 异常根治措施**
- allocator 配置（expandable_segments）由 run_phase1_bulk.sh 注入

用法: .venv/bin/python scripts/phase1_bulk_combined.py
输出: results/phase1_diag/{core,ordergap,aux}/<id>.json（与 standalone 逐字段一致）
"""
import gc
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from evaluation.eval_diag_fp32 import (DIAG_PROTOCOL_VERSION, MANIFEST, MANIFEST_SHA,
                                       PRECISION_MODE, load_model, load_samples,
                                       run_eval, summarize)
import evaluation.eval_diag_fp32_ordergap as og_module
import evaluation.eval_diag_fp32_aux as aux_module

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(ROOT, "results", "phase1_diag")
BM_SHA = "3b970159ecc6818beeefd2250d141ae54c7ea309a0dbb85e84415abe9c7bcc6c"
PROTOCOL_SHA = "ec61491736dac5e54a3d857896585e2c0a49361405a705dd5b09a76dd8d46ee7"
CORE_SHA = "ef1ef03ea08da87fb75eda6049311e2dd4ae91ab1a9f29dfa9d57cdcc6064030"
CHUNK = 8
OG_CHUNK = 16


def write_core(model, step, param_dtypes, c, out):
    """core 评估：frozen eval_diag_fp32.py 的 run_eval + summarize；payload 逐字段
    复刻其 main()（evaluator_file/sha 指向 frozen 文件，与 standalone 输出一致）。"""
    samples, indices = load_samples(None)
    res = run_eval(model, samples, chunk=CHUNK)
    summ = summarize(res["per_sample"], res["nll_values"])
    evaluator_path = os.path.join(ROOT, "evaluation", "eval_diag_fp32.py")
    payload = dict(
        tag=f"phase1-core-{c['id']}",
        diag_protocol_version=DIAG_PROTOCOL_VERSION,
        precision_mode=PRECISION_MODE,
        evaluator_file=os.path.relpath(evaluator_path, ROOT),
        evaluator_sha256=hashlib.sha256(open(evaluator_path, "rb").read()).hexdigest(),
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        model_path=c["path"],
        ckpt=c["ckpt"],
        weights="ema",
        step=step,
        model_param_dtypes=param_dtypes,
        compute_dtype="fp32（镜像 forward）",
        extraction_dtype="fp32（log_softmax/δ/CE）",
        aggregation_dtype="fp64",
        manifest_file=os.path.relpath(MANIFEST, ROOT),
        manifest_sha256=MANIFEST_SHA,
        n_samples=len(samples),
        indices_file=None,
        manifest_indices=indices,
        chunk=CHUNK,
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
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"core {c['id']}: CPI={summ['cpi_abs']:.4f} CE={summ['local_ce']:.4f}", flush=True)
    return payload


def main():
    # ---- 守卫 ----
    digest_man = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    assert digest_man == MANIFEST_SHA, "eval manifest sha 变化，硬停"
    digest_proto = hashlib.sha256(open(os.path.join(
        ROOT, "protocol/phase1_compatibility_estimation_dynamics_DRAFT.md"), "rb").read()).hexdigest()
    assert digest_proto == PROTOCOL_SHA, "Phase 1 协议 sha 变化，硬停"
    core_digest = hashlib.sha256(open(os.path.join(
        ROOT, "evaluation/eval_diag_fp32.py"), "rb").read()).hexdigest()
    assert core_digest == CORE_SHA, "core FP32 evaluator 被改动，硬停"
    digest_bm = hashlib.sha256(open(os.path.join(DIAG, "bulk_manifest.json"), "rb").read()).hexdigest()
    assert digest_bm == BM_SHA, "bulk manifest 被改动，硬停"
    print("OK: 守卫通过（manifest/protocol/core-evaluator/bulk-manifest sha）", flush=True)

    man = json.load(open(os.path.join(DIAG, "bulk_manifest.json")))
    n_loaded = 0
    for c in man["checkpoints"]:
        cid = c["id"]
        out_core = os.path.join(DIAG, "core", f"{cid}.json")
        out_og = os.path.join(DIAG, "ordergap", f"{cid}.json")
        out_aux = os.path.join(DIAG, "aux", f"{cid}.json")
        if os.path.exists(out_core) and os.path.exists(out_og) and os.path.exists(out_aux):
            print(f"SKIP {cid}（三类型已存在）", flush=True)
            continue
        print(f"=== {cid}（加载模型） ===", flush=True)
        model, step, param_dtypes = load_model(c["path"], c["ckpt"], "ema")
        n_loaded += 1
        try:
            if not os.path.exists(out_core):
                write_core(model, step, param_dtypes, c, out_core)
            else:
                print(f"SKIP core {cid}（已存在）", flush=True)
            if not os.path.exists(out_og):
                og_module.run(model, step, param_dtypes, c["path"], c["ckpt"],
                              "ema", OG_CHUNK, f"phase1-ordergap-{cid}", out_og)
            else:
                print(f"SKIP ordergap {cid}（已存在）", flush=True)
            if not os.path.exists(out_aux):
                aux_module.run(model, step, param_dtypes, c["path"], c["ckpt"],
                               "ema", CHUNK, f"phase1-aux-{cid}", out_aux)
            else:
                print(f"SKIP aux {cid}（已存在）", flush=True)
        finally:
            # 常规资源回收（hygiene）：进程内多 checkpoint 顺序执行的显存/对象释放。
            # 不描述为 WDDM 异常（dxg EOVERFLOW）的根治措施——根治在 Windows 侧重启。
            del model
            gc.collect()
            torch.cuda.empty_cache()
    print(f"OK: Phase 1 bulk 合并执行完成（模型加载 {n_loaded} 次）", flush=True)


if __name__ == "__main__":
    main()
