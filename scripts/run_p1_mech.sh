#!/usr/bin/env bash
# Mechanism pilot Stage P1 driver（protocol v1.2 §9 staged，2026-10-06 批准）
#   1. A1/B1/A2/B2 训练（顺序，每个 ~20 min）
#   2. step0 共享 baseline 评估（一次）
#   3. 每 run × checkpoint {500,1020,2500}：CPI（含 per-example delta/delta_abs）+ task NLL/acc
# 输出: results/mechanism_pilot/（独立目录，绝不覆盖任何 frozen 结果）
# 中断后重跑本脚本即可（逐任务 [ -f ] 守卫，训练完成守卫）。
# 硬停: manifest hash 变化 / frozen 文件被改动 / 训练异常（脚本内检查）→ exit 1
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
mkdir -p results/mechanism_pilot

# --- 预检 1: frozen manifest 未变（v4.1 §5A 不可变） ---
EXPECTED=1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3
ACTUAL=$(sha256sum manifests/regime_a_eval_v1.jsonl | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "FATAL: manifest hash 变化: $ACTUAL (期望 $EXPECTED) — 停止"
  exit 1
fi
echo "OK: manifest hash 未变"

# --- 预检 2: frozen 文件未被改动 ---
if ! git diff --quiet HEAD -- losses.py graph_lib.py noise_lib.py data.py \
     training/vanilla.py task_data/corruption.py model/; then
  echo "FATAL: frozen 文件被改动 — 停止"
  exit 1
fi
echo "OK: frozen 文件零改动"

# --- Stage P1 训练（顺序） ---
declare -A TRAIN=( [A1]="mechanism.policy=A mechanism.replicate=1" \
                   [B1]="mechanism.policy=B mechanism.replicate=1" \
                   [A2]="mechanism.policy=A mechanism.replicate=2" \
                   [B2]="mechanism.policy=B mechanism.replicate=2" )
for run in A1 B1 A2 B2; do
  echo "=== 训练 $run: ${TRAIN[$run]} ==="
  $PY training/pilot_mechanism.py ${TRAIN[$run]} training.n_iters=2500 training.name=mechpilot-$run
  log=$(ls -1t exp_local/regime_a/mechpilot-${run}-*/train.log | head -1)
  tail -3 "$log"
  grep -q "完成: 2500 optimizer steps" "$log" || { echo "FATAL: $run 未完成"; exit 1; }
  grep -q "与 preflight 一致" "$log" || { echo "FATAL: $run schedule hash 不一致"; exit 1; }
done
echo "OK: 四个训练 run 全部完成且 schedule hash 一致"

# --- step0 共享 baseline（一次） ---
S0=exp_local/regime_a/mechpilot_shared
[ -f "$S0/checkpoint_0000.pth" ] || { echo "FATAL: step0 共享 artifact 不存在"; exit 1; }
if [ ! -f results/mechanism_pilot/cpi_step0.json ]; then
  echo "=== step0 CPI ==="
  $PY evaluation/eval_cpi.py --model_path "$S0" --ckpt checkpoint_0000.pth \
      --weights ema --tag mechpilot-step0 --out results/mechanism_pilot/cpi_step0.json
fi
if [ ! -f results/mechanism_pilot/task_step0.json ]; then
  echo "=== step0 task ==="
  $PY evaluation/eval_task.py --model_path "$S0" --ckpt checkpoint_0000.pth \
      --weights ema --tag mechpilot-step0 --out results/mechanism_pilot/task_step0.json
fi
echo "OK: step0 评估完成"

# --- 每 run × checkpoint 评估 ---
shopt -s nullglob
for run in A1 B1 A2 B2; do
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
echo "OK: P1 全部 12 checkpoint 评估完成"
echo "driver 完成: 训练 4 + step0 2 + checkpoint 24 评估"
