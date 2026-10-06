#!/usr/bin/env bash
# Mechanism pilot Stage P2 driver（protocol v1.2 §9 staged）：C1/D1/C2/D2 训练 + 评估。
# 与 P1 driver 相同守卫与幂等语义；step0 共享 baseline 复用（已有则跳过）。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
mkdir -p results/mechanism_pilot

EXPECTED=1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3
ACTUAL=$(sha256sum manifests/regime_a_eval_v1.jsonl | cut -d' ' -f1)
[ "$ACTUAL" = "$EXPECTED" ] || { echo "FATAL: manifest hash 变化"; exit 1; }
echo "OK: manifest hash 未变"
git diff --quiet HEAD -- losses.py graph_lib.py noise_lib.py data.py \
  training/vanilla.py task_data/corruption.py model/ \
  || { echo "FATAL: frozen 文件被改动"; exit 1; }
echo "OK: frozen 文件零改动"

declare -A TRAIN=( [C1]="mechanism.policy=C mechanism.replicate=1" \
                   [D1]="mechanism.policy=D mechanism.replicate=1" \
                   [C2]="mechanism.policy=C mechanism.replicate=2" \
                   [D2]="mechanism.policy=D mechanism.replicate=2" )
for run in C1 D1 C2 D2; do
  meta=$(ls -1t exp_local/regime_a/mechpilot-${run}-*/run_metadata.json 2>/dev/null | head -1 || true)
  if [ -n "$meta" ]; then
    log=${meta%/*}/train.log
    grep -q "完成: 2500 optimizer steps" "$log" \
      && grep -q "与 preflight 一致" "$log" \
      && { echo "SKIP 训练 $run（已完成且 hash 一致）"; continue; }
  fi
  echo "=== 训练 $run: ${TRAIN[$run]} ==="
  $PY training/pilot_mechanism.py ${TRAIN[$run]} training.n_iters=2500 training.name=mechpilot-$run
  log=$(ls -1t exp_local/regime_a/mechpilot-${run}-*/train.log | head -1)
  tail -3 "$log"
  grep -q "完成: 2500 optimizer steps" "$log" || { echo "FATAL: $run 未完成"; exit 1; }
  grep -q "与 preflight 一致" "$log" || { echo "FATAL: $run schedule hash 不一致"; exit 1; }
done
echo "OK: P2 四个训练 run 全部完成且 schedule hash 一致"

S0=exp_local/regime_a/mechpilot_shared
[ -f "$S0/checkpoint_0000.pth" ] || { echo "FATAL: step0 共享 artifact 不存在"; exit 1; }
echo "OK: step0 共享 baseline 复用（cpi_step0/task_step0 已存在）"

shopt -s nullglob
for run in C1 D1 C2 D2; do
  dirs=(exp_local/regime_a/mechpilot-${run}-*)
  [ ${#dirs[@]} -eq 1 ] || { echo "FATAL: mechpilot-${run}-* 目录数 ${#dirs[@]}"; exit 1; }
  dir=${dirs[0]}
  for step in 500 1020 2500; do
    if [ ! -f results/mechanism_pilot/cpi_${run}_${step}.json ]; then
      echo "=== CPI ${run}@${step} ==="
      $PY evaluation/eval_cpi.py --model_path "$dir" --ckpt checkpoint_${step}.pth \
          --weights ema --tag mechpilot-${run}-${step} \
          --out results/mechanism_pilot/cpi_${run}_${step}.json
    fi
    if [ ! -f results/mechanism_pilot/task_${run}_${step}.json ]; then
      echo "=== task ${run}@${step} ==="
      $PY evaluation/eval_task.py --model_path "$dir" --ckpt checkpoint_${step}.pth \
          --weights ema --tag mechpilot-${run}-${step} \
          --out results/mechanism_pilot/task_${run}_${step}.json
    fi
  done
done
echo "OK: P2 全部 12 checkpoint 评估完成"
