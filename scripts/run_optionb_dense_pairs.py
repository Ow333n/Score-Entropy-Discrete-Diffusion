"""Option B dense checkpoint 的 pairs 评估（FP32，frozen manifest 500 pairs）。

26 = 2 seeds × 13 dense steps。输出 results/phase2/dense_pairs/{s1|s2}-{step}.json；
逐文件 skip-if-exists；模型每 checkpoint 加载一次。

用法: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python scripts/run_optionb_dense_pairs.py
"""
import gc
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from evaluation.eval_diag_fp32 import MANIFEST, MANIFEST_SHA, load_model
import evaluation.eval_diag_fp32_pairs as pairs_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "phase2", "dense_pairs")
STEPS = [50, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1020, 1500, 2500]


def main():
    digest_man = hashlib.sha256(open(MANIFEST, "rb").read()).hexdigest()
    assert digest_man == MANIFEST_SHA, "manifest sha 变化，硬停"
    dirs = {}
    for seed in (1, 2):
        ds = sorted(glob.glob(os.path.join(ROOT, f"exp_local/regime_a/optionb-dense-s{seed}-*")))
        assert ds, f"missing optionb-dense-s{seed}"
        dirs[seed] = ds[-1]
    n = 0
    for seed, d in dirs.items():
        for st in STEPS:
            cid = f"s{seed}-{st}"
            out = os.path.join(OUT, f"{cid}.json")
            if os.path.exists(out):
                print(f"SKIP {cid}", flush=True)
                continue
            print(f"=== pairs {cid} ===", flush=True)
            model, step, param_dtypes = load_model(d, f"checkpoint_{st}.pth", "ema")
            n += 1
            try:
                pairs_mod.run(model, step, param_dtypes, d, f"checkpoint_{st}.pth",
                              "ema", 8, f"optionb-pairs-{cid}", out)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
    print(f"OK: optionb dense pairs 完成（模型加载 {n} 次）", flush=True)


if __name__ == "__main__":
    main()
