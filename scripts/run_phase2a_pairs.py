"""Phase 2A pairs diagnostics 批量执行器（复用 frozen bulk_manifest 的 55 checkpoint）。

每 checkpoint 单次模型加载 → eval_diag_fp32_pairs.run（500 样本 per-sample：
δ/|δ|/D_ab/PMI_forward/PMI_reverse 可由 D_ab±δ/2 派生/JS_dep/CE/acc/entropy/conf）。
输出 results/phase2/pairs/<id>.json；逐文件 skip-if-exists；gc/cache cleanup 为
常规 hygiene（非 WDDM 根治）。allocator 配置由调用方 shell 注入。

用法: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python scripts/run_phase2a_pairs.py
"""
import gc
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from evaluation.eval_diag_fp32 import MANIFEST, MANIFEST_SHA, load_model
import evaluation.eval_diag_fp32_pairs as pairs_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(ROOT, "results", "phase1_diag")
OUT = os.path.join(ROOT, "results", "phase2", "pairs")
BM_SHA = "3b970159ecc6818beeefd2250d141ae54c7ea309a0dbb85e84415abe9c7bcc6c"
CORE_SHA = "ef1ef03ea08da87fb75eda6049311e2dd4ae91ab1a9f29dfa9d57cdcc6064030"


def main():
    digest_man = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    assert digest_man == MANIFEST_SHA, "eval manifest sha 变化，硬停"
    digest_bm = hashlib.sha256(open(os.path.join(DIAG, "bulk_manifest.json"), "rb").read()).hexdigest()
    assert digest_bm == BM_SHA, "bulk manifest 被改动，硬停"
    core_digest = hashlib.sha256(open(os.path.join(
        ROOT, "evaluation/eval_diag_fp32.py"), "rb").read()).hexdigest()
    assert core_digest == CORE_SHA, "core FP32 evaluator 被改动，硬停"
    print("OK: 守卫通过", flush=True)

    man = json.load(open(os.path.join(DIAG, "bulk_manifest.json")))
    n_loaded = 0
    for c in man["checkpoints"]:
        cid = c["id"]
        out = os.path.join(OUT, f"{cid}.json")
        if os.path.exists(out):
            print(f"SKIP {cid}（已存在）", flush=True)
            continue
        print(f"=== pairs {cid} ===", flush=True)
        model, step, param_dtypes = load_model(c["path"], c["ckpt"], "ema")
        n_loaded += 1
        try:
            pairs_mod.run(model, step, param_dtypes, c["path"], c["ckpt"], "ema",
                          8, f"phase2a-pairs-{cid}", out)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    print(f"OK: Phase 2A pairs 完成（模型加载 {n_loaded} 次）", flush=True)


if __name__ == "__main__":
    main()
