#!/usr/bin/env bash
# Phase 1 bulk FP32 dynamics evaluation driver（bulk_manifest FROZEN）
#   55 checkpoint × 3 评估类型（core / finite-path OrderGap / aux），全部 Level C FP32：
#     core:    evaluation/eval_diag_fp32.py（500 样本：CE/CPI_abs/RMS/signed δ/δ_SD/quantiles/acc）
#     ordergap: evaluation/eval_diag_fp32_ordergap.py（frozen §9.5 64 子集 + frozen 6 路径）
#     aux:     evaluation/eval_diag_fp32_aux.py（entropy/confidence/reliability）
#   输出 results/phase1_diag/{core,ordergap,aux}/<id>.json；逐文件 [ -f ] 守卫，可重跑。
# 预计 55 × ~2.8min ≈ 2.5–3h（请在 tmux 跑）。
set -euo pipefail
cd "$(dirname "$0")/.."

# allocator 工程配置（RL 阶段同款分类：memory/allocator engineering config，
# 非算法改动）——减少 create_allocation 调用次数，缓解 WDDM 侧分配记账异常
# （2026-10-07 bulk 首跑 dxg EOVERFLOW 后启用）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python

# --- 守卫 ---
EXPECTED_MANIFEST=1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3
AM=$(sha256sum manifests/regime_a_eval_v1.jsonl | cut -d' ' -f1)
[ "$AM" == "$EXPECTED_MANIFEST" ] || { echo "FATAL: eval manifest hash 变化"; exit 1; }
EXPECTED_PROTOCOL=ec61491736dac5e54a3d857896585e2c0a49361405a705dd5b09a76dd8d46ee7
AP=$(sha256sum protocol/phase1_compatibility_estimation_dynamics_DRAFT.md | cut -d' ' -f1)
[ "$AP" == "$EXPECTED_PROTOCOL" ] || { echo "FATAL: Phase 1 协议 sha 变化"; exit 1; }
EXPECTED_CORE=ef1ef03ea08da87fb75eda6049311e2dd4ae91ab1a9f29dfa9d57cdcc6064030
AC=$(sha256sum evaluation/eval_diag_fp32.py | cut -d' ' -f1)
[ "$AC" == "$EXPECTED_CORE" ] || { echo "FATAL: core FP32 evaluator 被改动"; exit 1; }
EXPECTED_BM=3b970159ecc6818beeefd2250d141ae54c7ea309a0dbb85e84415abe9c7bcc6c
AB=$(sha256sum results/phase1_diag/bulk_manifest.json | cut -d' ' -f1)
[ "$AB" == "$EXPECTED_BM" ] || { echo "FATAL: bulk manifest 被改动"; exit 1; }
echo "OK: 守卫通过（manifest/protocol/core-evaluator/bulk-manifest sha）"

$PY - <<'EOF'
import json, os, subprocess, sys
man = json.load(open("results/phase1_diag/bulk_manifest.json"))
for c in man["checkpoints"]:
    cid = c["id"]
    tasks = [
        ("core", "evaluation/eval_diag_fp32.py", f"results/phase1_diag/core/{cid}.json", []),
        ("ordergap", "evaluation/eval_diag_fp32_ordergap.py", f"results/phase1_diag/ordergap/{cid}.json", []),
        ("aux", "evaluation/eval_diag_fp32_aux.py", f"results/phase1_diag/aux/{cid}.json", []),
    ]
    for kind, script, out, extra in tasks:
        if os.path.exists(out):
            print(f"SKIP {kind} {cid}（已存在）")
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(f"=== {kind} {cid} ===", flush=True)
        # weights 为 manifest 顶层全局字段（全部 checkpoint 均 ema）
        r = subprocess.run([".venv/bin/python", script,
                            "--model_path", c["path"], "--ckpt", c["ckpt"],
                            "--weights", c.get("weights", "ema"),
                            "--tag", f"phase1-{kind}-{cid}",
                            "--out", out] + extra)
        if r.returncode != 0:
            print(f"FATAL: {kind} {cid} 失败 exit={r.returncode}")
            sys.exit(1)
print("OK: Phase 1 bulk 三类型评估全部完成")
EOF
echo "bulk 完成。下一步: scripts/phase1_dynamics_analysis.py"
