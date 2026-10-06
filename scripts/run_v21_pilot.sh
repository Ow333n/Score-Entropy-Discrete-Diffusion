#!/usr/bin/env bash
# v2.1 mechanism pilot 正式训练 driver（protocol v2.1 §10/§14 步骤 7，preflight P6）
#   1. 守卫：manifest hash / 协议 sha / frozen 文件 / maps sha（任一变化 → exit 1）
#   2. U1→H1→L1→U2→H2→L2 顺序训练（每 run 2500 步，~20 min，总 ~2h，请 tmux 跑）
# 输出: exp_local/regime_a/v21pilot-<run>-<HHMMSS>/（checkpoint 500/1020/2500 EMA +
#   raw 2500 + run_metadata.json）
# 中断后重跑本脚本即可（逐 run 完成守卫：日志含"完成: 2500"且"与 dryrun 一致"）。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python

EXPECTED_MANIFEST=1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3
ACTUAL=$(sha256sum manifests/regime_a_eval_v1.jsonl | cut -d' ' -f1)
[ "$ACTUAL" == "$EXPECTED_MANIFEST" ] || { echo "FATAL: manifest hash 变化: $ACTUAL"; exit 1; }
echo "OK: manifest hash 未变"

EXPECTED_PROTOCOL=af312d864d41d8f67da343d55a8dee8a29a91c37dea622e6b2c4b22ef441fc61
AP=$(sha256sum protocol/mechanism_complementary_exposure_v2_1.md | cut -d' ' -f1)
[ "$AP" == "$EXPECTED_PROTOCOL" ] || { echo "FATAL: 协议 sha256 变化: $AP"; exit 1; }
echo "OK: protocol sha256 未变"

if ! git diff --quiet HEAD -- losses.py graph_lib.py noise_lib.py data.py \
     training/vanilla.py task_data/corruption.py task_data/policy_corruption.py model/ \
     task_data/v21_policy.py evaluation/eval_cpi_pairs.py evaluation/eval_order_gap_subset.py; then
  echo "FATAL: frozen 文件被改动 — 停止"
  exit 1
fi
echo "OK: frozen 文件零改动"

$PY - <<'EOF'
import json, hashlib, os
mdir = "exp_local/regime_a/mechpilot_v21_maps"
man = json.load(open(os.path.join(mdir, "maps_manifest.json")))
for fn, expected in man["file_sha256"].items():
    got = hashlib.sha256(open(os.path.join(mdir, fn), "rb").read()).hexdigest()
    assert got == expected, f"maps 被改动: {fn}"
print("OK: maps sha256 全部一致")
EOF

declare -A RUNS=( [U1]="U 1" [H1]="H 1" [L1]="L 1" [U2]="U 2" [H2]="H 2" [L2]="L 2" )
for run in U1 H1 L1 U2 H2 L2; do
  set -- ${RUNS[$run]}
  pol=$1; rep=$2
  meta=$(ls -1t exp_local/regime_a/v21pilot-${run}-*/run_metadata.json 2>/dev/null | head -1 || true)
  if [ -n "$meta" ]; then
    log=${meta%/*}/train.log
    if grep -q "完成: 2500 optimizer steps" "$log" && grep -q "与 dryrun 一致" "$log"; then
      echo "SKIP 训练 $run（已完成且与 dryrun 一致）"
      continue
    fi
  fi
  echo "=== 训练 $run (policy=$pol replicate=$rep) ==="
  $PY training/pilot_v21.py mechanism_v21.policy=$pol mechanism_v21.replicate=$rep \
      training.n_iters=2500 training.name=v21pilot-$run
  log=$(ls -1t exp_local/regime_a/v21pilot-${run}-*/train.log | head -1)
  tail -3 "$log"
  grep -q "完成: 2500 optimizer steps" "$log" || { echo "FATAL: $run 未完成"; exit 1; }
  grep -q "与 dryrun 一致" "$log" || { echo "FATAL: $run 与 dryrun 不一致"; exit 1; }
done
echo "OK: 6 个训练 run 全部完成且与 dryrun 一致"
echo "下一步: scripts/run_v21_evals.sh（评估 ~3-3.5h，tmux 跑）"
