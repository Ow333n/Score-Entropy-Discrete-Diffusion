# RL K-Ablation Diagnostic Report（K=1 vs K=4, 100 steps）

> 日期：2026-10-04 夜（unattended）。协议：`protocol/rl_k_ablation_protocol.md`（含
> amendment #1 RAM gate 修订）。诊断目的：K=1 是否可能是 RL-1 weak task signal 的
> estimator-density 瓶颈。**诊断性结果，非 formal result，不覆盖/不修改 formal RL-1。**

## 1. 执行摘要

- Preflight（修订后）：PASS。K=4 5-step memory smoke：**5/5 PASS**。
- 两个 matched run 均 **100/100 完成**，零 OOM / NaN / hard failure / 新 dxg anomaly。
- **结论：Scenario B（K=4 ≈ K=1）** —— 单纯提高 K 不足以解决 weak task signal；
  下一瓶颈更可能在 reward / group-relative objective / credit-assignment mechanism 本身。
- K=4 的额外 wall-time 成本 ~6%（sequential accumulation 生效，peak VRAM 与 K=1 相同）。

## 2. 核心对比表（formal64，RAW primary / EMA secondary）

| metric | step0 | K1-50 | K1-100 | K4-50 | K4-100 |
|---|---|---|---|---|---|
| NLL raw | 7.0490 | 7.0535 | 7.0529 | 7.0494 | **7.0398** |
| NLL ema | 7.0490 | 7.0614 | 7.0476 | 7.0472 | 7.0447 |
| sampled64 raw | 0.2562 | 0.2564 | **0.2589** | 0.2570 | 0.2570 |
| sampled64 ema | 0.2562 | 0.2559 | 0.2605 | 0.2589 | 0.2588 |
| greedy raw | 0.3365 | 0.3346 | 0.3378 | 0.3352 | **0.3403** |
| greedy ema | 0.3365 | 0.3353 | 0.3333 | 0.3384 | 0.3368 |

- 100 步内两组 task 指标均为噪声级移动（与 formal RL-1 前 100 步同量级）。
- K4 的 NLL（−0.009）与 greedy（+0.004）比 K1（+0.004 / +0.001）移动略大，
  但全部远低于判读阈值；sampled 方向相反（K1 +0.0027 vs K4 +0.0008）。

## 3. Training diagnostics

| metric | K1-100 | K4-100 |
|---|---|---|
| drift ‖Δθ‖/‖θ0‖ | 1.61e-4 | 1.62e-4 |
| reward first / last / mean | 0.1459 / 0.2691 / 0.2033 | 0.1459 / 0.2732 / 0.2045 |
| zvg | 26/400 (6.5%) | 26/400 (6.5%) |
| grad norm max / mean | 1.02 / 0.49 | 0.79 / 0.47 |
| nan_rollouts | 0 | 0 |
| soft_neg elements/steps | 8/8 | 8/8 |
| J support / trunc | 0.926 / 0.074 | 0.926 / 0.074 |

- 两组 reward 序列几乎重合（step1 逐位相同 = j_rng 隔离生效；后续微小分叉来自参数更新差异）。
- drift、zvg、软门统计几乎相同 → K 不改变优化轨迹的宏观性质。

## 4. Compatibility secondary check（RAW，current-code）

| run | CPI_abs | CPI_RMS | signed δ mean | OrderGap_raw | OrderGap/token |
|---|---|---|---|---|---|
| step0（SFT init） | 0.2871 | 0.5469 | −0.0077 | 8.7475 | 0.5338 |
| K1-100 | 0.2891 | 0.5547 | −0.0025 | 8.7488 | 0.5334 |
| K4-100 | 0.2852 | 0.5469 | −0.0040 | 8.7312 | 0.5319 |

- 100 步 short horizon 下两组均无结构性变化（±0.002 / ±0.02 为 harness 噪声量级）。
- K 不改变 compatibility 轨迹方向。

## 5. Compute cost（estimator efficiency）

| metric | K1 | K4 |
|---|---|---|
| sec/optimizer step（稳态） | 28.6 | 30.3（**+5.7% wall time**） |
| peak allocated / reserved | 6.36 / 7.40 GB | 6.36 / 7.41 GB（**相同**） |
| RAM final（avail / swap） | 3.2 GiB / 0.87 GiB | 2.9 GiB / 0.58 GiB |

- sequential backward 生效：K=4 的显存峰值与 K=1 完全相同；recompute 只占 step 时间的小头
  （rollout 主导），因此 K=4 的额外成本 ~6%，远低于 4×。
- **task gain per GPU-hour：K=4 无实质 gain（Scenario B）→ 效率上 K=1 仍占优。**

## 6. Scenario 判定

**Scenario B：K=4 ≈ K=1。**

"单纯提高 K 不足以解决 weak task signal，下一瓶颈可能在 reward /
group-relative objective / credit assignment mechanism 本身。"

（K4 的 NLL/greedy 微幅优势与 sampled 的微幅劣势方向不一致、幅度均为噪声级，
不足以支持 Scenario A/D。）

## 7. STOP / 异常记录

- 无 OOM / NaN / Inf / hard failure / model-state corruption / checkpoint 写损坏。
- dxg：全程 27（= baseline），无新增。
- RAM：最低 2.9 GiB available（> 2.5 阈值）、swap 峰值 0.87 GiB（< 1.5 阈值）→ 未触发
  runtime RAM STOP。
- Disk：D: 22GB 起始，运行后 ~19GB（新增 ~3GB：2×1.4GB 快照 + smoke/日志），> 10GB 阈值。

## 8. 新增磁盘占用

- `exp_local/regime_a/rl-k-ablation/`：k1-100（1.4GB 快照）、k4-100（1.4GB 快照）、
  smoke-k4（<10MB）≈ **2.8GB**；rolling checkpoints 已按协议删除。
- `results/rl_k_ablation/`：4 个兼容评估 JSON + 3 个对比 JSON（<1MB）。

## 9. 说明与措辞纪律

- 每 condition 一个 training seed → 所有差异为 sample-level / 单 seed 观察，
  **不声称 seed-level significance**。
- 本诊断的 K=1 与 formal RL-1 的实现差异（J 采样改用独立 j_rng，用于 K=1/K=4 匹配）
  已在协议 §4 记录；科学设置全同。
- Original preflight stopped because the absolute 8-GiB available-memory threshold
  exceeded the WSL instance's total 7.3-GiB memory. The diagnostic was resumed only
  after replacing it with a machine-capacity-aware safety gate.
