#!/usr/bin/env bash
# v2.1 mechanism pilot 评估 driver（protocol v2.1 preflight P7，训练完成后执行）
#   1. global：6 runs × 3 steps × (CPI + task) = 36 文件（frozen evaluator 原样复用）
#   2. treated：H/L × 2 reps × 3 steps = 12（replicate map；step0 × 两 map 已在
#      训练前冻结于 results/mechanism_pilot_v21/cpi_treated_step0_r{1,2}.json）
#   3. heldout：U/H/L × 2 reps × 3 steps = 18（step0 已冻结）
#   4. §9.5 OrderGap subset + path-score variance：6 runs × step2500
#      （step0 已冻结于 order_gap_subset_step0.json）
# 输出: results/mechanism_pilot_v21/；逐文件 [ -f ] 守卫，中断重跑安全。
# 预计 ~3-3.5h（pair evals 31×~3.5min + global 36×~2min + subset 6×~1.5min），请 tmux 跑。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=results/mechanism_pilot_v21
mkdir -p "$OUT"

EXPECTED_MANIFEST=1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3
ACTUAL=$(sha256sum manifests/regime_a_eval_v1.jsonl | cut -d' ' -f1)
[ "$ACTUAL" == "$EXPECTED_MANIFEST" ] || { echo "FATAL: manifest hash 变化"; exit 1; }
echo "OK: manifest hash 未变"

EXPECTED_PROTOCOL=af312d864d41d8f67da343d55a8dee8a29a91c37dea622e6b2c4b22ef441fc61
AP=$(sha256sum protocol/mechanism_complementary_exposure_v2_1.md | cut -d' ' -f1)
[ "$AP" == "$EXPECTED_PROTOCOL" ] || { echo "FATAL: 协议 sha256 变化"; exit 1; }
EXPECTED_ERRATA=d099aaaaa7732d98467a3cc43d5d6b936f32322252b7f291c298a0731af951b0
AE=$(sha256sum protocol/mechanism_v2_1_errata.md | cut -d' ' -f1)
[ "$AE" == "$EXPECTED_ERRATA" ] || { echo "FATAL: errata sha256 变化"; exit 1; }
EXPECTED_EXECMAN=5cb816f32e6c53d696625a3a233bd11e54f3be229aff38933c69593821eb3b25
AM=$(sha256sum protocol/mechanism_v2_1_execution_manifest.json | cut -d' ' -f1)
[ "$AM" == "$EXPECTED_EXECMAN" ] || { echo "FATAL: execution manifest sha256 变化"; exit 1; }
echo "OK: 协议/errata/execution manifest sha256 未变"

declare -A RUNS=( [U1]="U 1" [H1]="H 1" [L1]="L 1" [U2]="U 2" [H2]="H 2" [L2]="L 2" )
shopt -s nullglob

for run in U1 H1 L1 U2 H2 L2; do
  set -- ${RUNS[$run]}
  pol=$1; rep=$2
  dirs=(exp_local/regime_a/v21pilot-${run}-*)
  [ ${#dirs[@]} -eq 1 ] || { echo "FATAL: v21pilot-${run}-* 目录数 ${#dirs[@]}"; exit 1; }
  dir=${dirs[0]}

  # --- 1. global（frozen evaluator，不动）---
  for step in 500 1020 2500; do
    if [ ! -f "$OUT/cpi_global_${run}_${step}.json" ]; then
      echo "=== global CPI ${run}@${step} ==="
      $PY evaluation/eval_cpi.py --model_path "$dir" --ckpt checkpoint_${step}.pth \
          --weights ema --tag v21-${run}-${step} --out "$OUT/cpi_global_${run}_${step}.json"
    fi
    if [ ! -f "$OUT/task_${run}_${step}.json" ]; then
      echo "=== task ${run}@${step} ==="
      $PY evaluation/eval_task.py --model_path "$dir" --ckpt checkpoint_${step}.pth \
          --weights ema --tag v21-${run}-${step} --out "$OUT/task_${run}_${step}.json"
    fi
  done

  # --- 2. treated（仅 H/L，各自 replicate 的 map）---
  if [ "$pol" != "U" ]; then
    for step in 500 1020 2500; do
      if [ ! -f "$OUT/cpi_treated_${run}_${step}.json" ]; then
        echo "=== treated CPI ${run}@${step} ==="
        $PY evaluation/eval_cpi_pairs.py --map-type replicate --replicate "$rep" \
            --model_path "$dir" --ckpt checkpoint_${step}.pth --weights ema \
            --tag v21-trt-${run}-${step} --out "$OUT/cpi_treated_${run}_${step}.json"
      fi
    done
  fi

  # --- 3. heldout（全部 6 runs）---
  for step in 500 1020 2500; do
    if [ ! -f "$OUT/cpi_heldout_${run}_${step}.json" ]; then
      echo "=== heldout CPI ${run}@${step} ==="
      $PY evaluation/eval_cpi_pairs.py --map-type heldout \
          --model_path "$dir" --ckpt checkpoint_${step}.pth --weights ema \
          --tag v21-held-${run}-${step} --out "$OUT/cpi_heldout_${run}_${step}.json"
    fi
  done

  # --- 4. §9.5 OrderGap subset（仅 step2500）---
  if [ ! -f "$OUT/order_gap_subset_${run}_2500.json" ]; then
    echo "=== OrderGap subset ${run}@2500 ==="
    $PY evaluation/eval_order_gap_subset.py --model_path "$dir" \
        --ckpt checkpoint_2500.pth --weights ema --tag v21-og-${run}-2500 \
        --out "$OUT/order_gap_subset_${run}_2500.json"
  fi
done
echo "OK: v2.1 评估全部完成（global 36 + treated 12 + heldout 18 + subset 6 + step0 4 冻结）"
