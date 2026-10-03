#!/bin/bash
# Post-run CPI / OrderGap evaluation (protocol v1.0 §36): rlpilot snapshots 50/100/250/500 × raw/ema
# 与 formal compatibility evaluation 完全同一 pipeline (frozen manifest / 同一 eval CLI / 同 chunk=4 用法同 in-run step0)
set -e
RUN=exp_local/regime_a/rlpilot-185545
OUT=results/rl_pilot
mkdir -p "$OUT"
for step in 50 100 250 500; do
  for w in raw ema; do
    echo "== cpi step$step $w =="
    .venv/bin/python evaluation/eval_cpi.py --model_path "$RUN" --ckpt "eval_snapshot_step${step}.pth" \
      --weights "$w" --chunk 4 --out "$OUT/cpi_step${step}_${w}.json"
    echo "== og step$step $w =="
    .venv/bin/python evaluation/eval_order_gap.py --model_path "$RUN" --ckpt "eval_snapshot_step${step}.pth" \
      --weights "$w" --chunk 4 --out "$OUT/og_step${step}_${w}.json"
  done
done
echo "ALL_POSTRUN_EVALS_DONE"
