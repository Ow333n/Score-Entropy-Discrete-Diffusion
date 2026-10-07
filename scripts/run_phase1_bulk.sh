#!/usr/bin/env bash
# Phase 1 bulk FP32 dynamics evaluation driver（bulk_manifest FROZEN；合并执行）
#   55 checkpoint，每 checkpoint **单次模型加载**，同进程依次 core → finite-path
#   OrderGap → aux（见 scripts/phase1_bulk_combined.py）。模型加载 165 → ≤55 次。
#   输出 results/phase1_diag/{core,ordergap,aux}/<id>.json；逐文件 skip-if-exists，可重跑。
# 预计 ~2–2.5h（请在 tmux 跑）。
set -euo pipefail
cd "$(dirname "$0")/.."

# allocator 工程配置（RL 阶段同款分类：memory/allocator engineering config，
# 非算法改动）——减少 create_allocation 调用次数，缓解 WDDM 侧分配记账异常
# （2026-10-07 bulk 首跑 dxg EOVERFLOW 后启用）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python

# --- 守卫（合并执行器内部同样校验；此处快速失败）---
EXPECTED_MANIFEST=1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3
AM=$(sha256sum manifests/regime_a_eval_v1.jsonl | cut -d' ' -f1)
[ "$AM" == "$EXPECTED_MANIFEST" ] || { echo "FATAL: eval manifest hash 变化"; exit 1; }
echo "OK: eval manifest sha 未变"

$PY scripts/phase1_bulk_combined.py
echo "bulk 完成。下一步: scripts/phase1_dynamics_analysis.py"
