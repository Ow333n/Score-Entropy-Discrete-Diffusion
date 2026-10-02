#!/usr/bin/env bash
# P1 early-dynamics 评估 (protocol v4.2 §3.2, regime_a_protocol_v4.2.yaml)
#   Tier 1: EMA × 全 dense 点 × 2 seeds —— CPI 全指标 + buckets         (~4.5h)
#   Tier 2: EMA × 全 dense 点 × 2 seeds —— TF OrderGap(固定6路径) + G1   (~8h)
#   Tier 3: raw × 全 dense 点 × 2 seeds —— CPI + TF OrderGap + G1       (~12h)
# dense 点 = [50, 100, 250, 500, 750, 1020, 2500]; step 0 = 冻结的 v4.1
# pretrained 基线 (cpi/og/g1_pretrained.json)，不重评。
# 输出: results/p1_dense/ —— 独立目录, 绝不覆盖 v4.1 的 results/vanilla。
# 中断后重跑本脚本即可 (逐任务 [ -f ] 守卫, 各任务独立)。
# 预检 (只读): ① frozen manifest hash 不变 ② P1 checkpoint_1020 与 v4.1 formal
#              同 seed 逐字节一致 (训练复现性门, 不一致必须停下排查)。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
DENSE="50 100 250 500 750 1020 2500"
mkdir -p results/p1_dense

# --- 训练目录发现 (必须恰好一个 p1-s1 / p1-s2) ---
shopt -s nullglob
S1=(exp_local/regime_a/p1-s1-*)
S2=(exp_local/regime_a/p1-s2-*)
[ ${#S1[@]} -eq 1 ] || { echo "FATAL: 需要恰好一个 exp_local/regime_a/p1-s1-* 目录, 现有 ${#S1[@]} 个"; exit 1; }
[ ${#S2[@]} -eq 1 ] || { echo "FATAL: 需要恰好一个 exp_local/regime_a/p1-s2-* 目录, 现有 ${#S2[@]} 个"; exit 1; }
S1=${S1[0]}; S2=${S2[0]}
echo "P1 run dirs: $S1 / $S2"

# --- 预检 1: frozen manifest 未变 (v4.1 §5A 不可变) ---
EXPECTED=1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3
ACTUAL=$(sha256sum manifests/regime_a_eval_v1.jsonl | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "FATAL: manifest hash 变化: $ACTUAL (期望 $EXPECTED) — 停止, 不允许重新生成 manifest"
  exit 1
fi
echo "OK: manifest hash 未变"

# --- 预检 2: P1 checkpoint_1020 vs v4.1 formal 同 seed (两级复现检查, reports/p1_preflight_audit.md §2) ---
for pair in "$S1:formal-vanilla-s1-191414" "$S2:formal-vanilla-s2-204215"; do
  p1_dir=${pair%%:*}; v41_dir=${pair##*:}
  if cmp -s "$p1_dir/checkpoint_1020.pth" "exp_local/regime_a/$v41_dir/checkpoint_1020.pth"; then
    echo "OK: $p1_dir/checkpoint_1020.pth 与 v4.1 $v41_dir 逐字节一致 (gate 2 PASS)"
  else
    echo "WARN: 逐字节不一致 → 第二级 tensor-level diff (model/EMA/optimizer/scaler, rel 阈值 1e-5):"
    if "$PY" scripts/compare_checkpoints.py \
        "$p1_dir/checkpoint_1020.pth" "exp_local/regime_a/$v41_dir/checkpoint_1020.pth"; then
      echo "OK: 差异在 kernel 非确定性噪声量级 (gate 2 PASS-CAVEAT, 证据见上方输出)"
    else
      echo "FATAL: 训练轨迹实质分歧 → 硬停排查, 不要继续评估"
      exit 1
    fi
  fi
done

run_cpi() { # $1 seed_dir $2 seed $3 step $4 weights
  [ -f "results/p1_dense/cpi_${2}_${3}_${4}.json" ] || \
    "$PY" evaluation/eval_cpi.py --model_path "$1" --ckpt "checkpoint_${3}.pth" \
      --weights "$4" --tag "p1-${2}-${3}-${4}" \
      --out "results/p1_dense/cpi_${2}_${3}_${4}.json"
}
run_og() { # TF OrderGap 固定 6 路径口径 (v4.2 §3.2; 不加 --adaptive)
  [ -f "results/p1_dense/og_${2}_${3}_${4}.json" ] || \
    "$PY" evaluation/eval_order_gap.py --model_path "$1" --ckpt "checkpoint_${3}.pth" \
      --weights "$4" --tag "p1-${2}-${3}-${4}" \
      --out "results/p1_dense/og_${2}_${3}_${4}.json"
}
run_g1() {
  [ -f "results/p1_dense/g1_${2}_${3}_${4}.json" ] || \
    "$PY" evaluation/eval_task.py --model_path "$1" --ckpt "checkpoint_${3}.pth" \
      --weights "$4" --tag "p1-${2}-${3}-${4}" \
      --out "results/p1_dense/g1_${2}_${3}_${4}.json"
}

echo "=== Tier 1: EMA × CPI ==="
for sd in "$S1 s1" "$S2 s2"; do set -- $sd; for st in $DENSE; do run_cpi "$1" "$2" "$st" ema; done; done

echo "=== Tier 2: EMA × OrderGap + G1 ==="
for sd in "$S1 s1" "$S2 s2"; do set -- $sd; for st in $DENSE; do
  run_og "$1" "$2" "$st" ema
  run_g1 "$1" "$2" "$st" ema
done; done

echo "=== Tier 3: raw × CPI + OrderGap + G1 ==="
for sd in "$S1 s1" "$S2 s2"; do set -- $sd; for st in $DENSE; do
  run_cpi "$1" "$2" "$st" raw
  run_og "$1" "$2" "$st" raw
  run_g1 "$1" "$2" "$st" raw
done; done

echo "ALL P1 EVALS DONE (results/p1_dense/)"
