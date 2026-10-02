#!/usr/bin/env bash
# 正式 G1/G2 评估: 2 seeds × 3 analysis checkpoints (1020/5100/10200)
# 每 checkpoint: eval_cpi + eval_order_gap(固定+adaptive 路径) + eval_task (G1)
# 全部使用 frozen manifest (paired 口径) 与 EMA 权重 (预注册 primary)。
# 预计总时长 ~3.5-4h。中断后重跑本脚本即可 (各 step 独立)。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
S1=exp_local/regime_a/formal-vanilla-s1-191414
S2=exp_local/regime_a/formal-vanilla-s2-204215
mkdir -p results/vanilla

for seed_dir in "$S1" "$S2"; do
  seed=$(basename "$seed_dir" | sed 's/formal-vanilla-//; s/-.*//')
  for step in 1020 5100 10200; do
    echo "=== $seed @ $step ==="
    [ -f "results/vanilla/cpi_${seed}_${step}.json" ] || \
      "$PY" evaluation/eval_cpi.py \
        --model_path "$seed_dir" --ckpt "checkpoint_${step}.pth" \
        --tag "${seed}-${step}" --out "results/vanilla/cpi_${seed}_${step}.json"
    [ -f "results/vanilla/og_${seed}_${step}.json" ] || \
      "$PY" evaluation/eval_order_gap.py \
        --model_path "$seed_dir" --ckpt "checkpoint_${step}.pth" --adaptive \
        --tag "${seed}-${step}" --out "results/vanilla/og_${seed}_${step}.json"
    [ -f "results/vanilla/g1_${seed}_${step}.json" ] || \
      "$PY" evaluation/eval_task.py \
        --model_path "$seed_dir" --ckpt "checkpoint_${step}.pth" \
        --tag "${seed}-${step}" --out "results/vanilla/g1_${seed}_${step}.json"
  done
done

echo "ALL FORMAL EVALS DONE"
