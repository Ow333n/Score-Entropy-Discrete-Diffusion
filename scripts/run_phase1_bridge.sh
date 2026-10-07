#!/usr/bin/env bash
# Phase 1.2 bridge driver（protocol phase1 §4 + bridge_manifest FROZEN）
#   每 checkpoint 跑两套 evaluator（同 500 样本 manifest）：
#     old: evaluation/eval_cpi.py（frozen BF16，chunk 8，EMA）
#     new: evaluation/eval_diag_fp32.py（Level C FP32，chunk 8，EMA）
#   输出 results/phase1_bridge/old_<id>.json + new_<id>.json；逐文件 [ -f ] 守卫，可重跑。
# 预计 15 × (old ~2.5min + new ~1min) ≈ 55–70min。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=results/phase1_bridge
mkdir -p "$OUT"

# --- 守卫 ---
EXPECTED_MANIFEST=1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3
AM=$(sha256sum manifests/regime_a_eval_v1.jsonl | cut -d' ' -f1)
[ "$AM" == "$EXPECTED_MANIFEST" ] || { echo "FATAL: eval manifest hash 变化"; exit 1; }
EXPECTED_PROTOCOL=ec61491736dac5e54a3d857896585e2c0a49361405a705dd5b09a76dd8d46ee7
AP=$(sha256sum protocol/phase1_compatibility_estimation_dynamics_DRAFT.md | cut -d' ' -f1)
[ "$AP" == "$EXPECTED_PROTOCOL" ] || { echo "FATAL: Phase 1 协议 sha 变化"; exit 1; }
EXPECTED_OLD=e8e250ba3a13243380de14212278c27775a759d9406ee348525472e33bcee07c
AO=$(sha256sum evaluation/eval_cpi.py | cut -d' ' -f1)
[ "$AO" == "$EXPECTED_OLD" ] || { echo "FATAL: old evaluator 被改动"; exit 1; }
EXPECTED_NEW=ef1ef03ea08da87fb75eda6049311e2dd4ae91ab1a9f29dfa9d57cdcc6064030
AN=$(sha256sum evaluation/eval_diag_fp32.py | cut -d' ' -f1)
[ "$AN" == "$EXPECTED_NEW" ] || { echo "FATAL: new evaluator 被改动"; exit 1; }
EXPECTED_BM=986dcf1d0c79e7ff6fd21f9d00cf782ce81707556d62f3ed2b0425ea98d6587b
AB=$(sha256sum results/phase1_bridge/bridge_manifest.json | cut -d' ' -f1)
[ "$AB" == "$EXPECTED_BM" ] || { echo "FATAL: bridge manifest 被改动"; exit 1; }
echo "OK: 全部守卫通过（manifest/protocol/old/new/bridge-manifest sha）"

# --- 逐 checkpoint 双 evaluator ---
$PY - <<'EOF'
import json, subprocess, os, sys
man = json.load(open("results/phase1_bridge/bridge_manifest.json"))
for c in man["checkpoints"]:
    cid = c["id"]
    for tag, script in (("old", "evaluation/eval_cpi.py"), ("new", "evaluation/eval_diag_fp32.py")):
        out = f"results/phase1_bridge/{tag}_{cid}.json"
        if os.path.exists(out):
            print(f"SKIP {tag} {cid}（已存在）")
            continue
        print(f"=== {tag} {cid} ===", flush=True)
        args = [".venv/bin/python", script,
                "--model_path", c["path"], "--ckpt", c["ckpt"],
                "--weights", c["weights"],
                "--tag", f"bridge-{tag}-{cid}", "--out", out]
        if tag == "old":
            args += ["--chunk", "8"]
        else:
            args += ["--chunk", "8"]
        r = subprocess.run(args)
        if r.returncode != 0:
            print(f"FATAL: {tag} {cid} 失败 exit={r.returncode}")
            sys.exit(1)
print("OK: bridge 双 evaluator 全部完成")
EOF
echo "bridge 完成。下一步: scripts/phase1_bridge_analysis.py"
