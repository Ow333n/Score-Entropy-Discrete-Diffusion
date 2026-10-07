# Option B Case Adjudication（人工最终裁定）

- 日期：2026-10-08（用户裁定）
- 状态：**FINAL — Case A, with a replicated but non-independent D_ab correlate**
- 机械脚本输出：`B_candidate_unadjudicated`（2026-10-08 已修正，见
  `scripts/optionb_dense_analysis.py`：independence 判据未实现时脚本禁止输出正式 Case B）
- 依据数据：`results/phase2/dense_pairs/`（26 files，2 seeds × 13 dense steps，
  FP32，frozen manifest 500 pairs）+ shared pretrained step0

## 1. Dense trajectory（per-checkpoint 均值）

| step | s1 \|δ\| | s1 CE | s1 JS_dep | s1 D_ab | s2 \|δ\| | s2 CE | s2 JS_dep | s2 D_ab |
|------|---------|-------|-----------|---------|---------|-------|-----------|---------|
| 0    | .3242   | 4.276 | .0461     | .160    | .3242   | 4.276 | .0461     | .160    |
| 50   | .3238   | 4.274 | .0460     | .160    | .3239   | 4.274 | .0460     | .160    |
| 100  | .3228   | 4.266 | .0457     | .161    | .3227   | 4.268 | .0458     | .160    |
| 150  | .3234   | 4.255 | .0455     | .162    | .3223   | 4.258 | .0455     | .161    |
| 200  | .3237   | 4.241 | .0451     | .163    | .3219   | 4.241 | .0451     | .162    |
| 250  | .3217   | 4.224 | .0447     | .164    | .3213   | 4.223 | .0448     | .164    |
| 300  | .3175   | 4.204 | .0442     | .166    | .3172   | 4.200 | .0444     | .167    |
| 350  | .3134   | 4.182 | .0439     | .169    | .3132   | 4.177 | .0440     | .170    |
| 400  | .3089   | 4.161 | .0435     | .172    | .3098   | 4.154 | .0437     | .172    |
| 500  | .2998   | 4.120 | .0432     | .176    | .3052   | 4.117 | .0436     | .176    |
| 750  | .2908   | 4.059 | .0432     | .189    | .2878   | 4.051 | .0430     | .191    |
| 1020 | .2833   | 4.020 | .0426     | .201    | .2769   | 4.020 | .0430     | .201    |
| 1500 | .2749   | 3.987 | .0432     | .206    | .2717   | 3.986 | .0431     | .207    |
| 2500 | .2819   | 3.938 | .0432     | .212    | .2839   | 3.946 | .0433     | .218    |

（step0 为全体 run 共享的 pretrained init；数值来自 `optionb_dense_analysis.json`
trajectory 段，4 位小数展示）

## 2. CPI active window

- 0–250：变化很小（|δ| 降 ~0.003）
- **250–750：主要下降窗口**（s1 −0.0267 / s2 −0.0294，两 seed 一致）
- 750–1020：继续下降（−0.0075 / −0.0109）
- 1020–1500：接近最低点（s1 0.2749 / s2 0.2717）
- 2500：轻微 rebound（+0.007 / +0.012）

与 frozen P1 的"CPI 衰减定位 250–750"一致；dense 分辨率进一步确认
底部在 1020–1500、终点微反弹。

## 3. 为什么 mechanical Case B 不成立

1. **脚本缺判据**：committed 的 `optionb_dense_analysis.py` 只实现了 §7 的方向
   一致 + 幅度（>0.001）判据；"不能被 CE/NLL 单独解释"（CE-adjusted
   independence）从未实现。机械输出因此不满足 Case B 的完整预注册标准。
2. **JS_dep 不构成 independent transition**：全程 0→2500 下降
   −0.002890（−6.27%，s1）/−0.002786（−6.04%，s2）——modest but
   reproducible；但 temporal profile 与 CPI active window 不对齐：0–250
   （active 之前）已下降 −0.001383（−3.00%，s1）/−0.001255（−2.72%，s2），
   占全程下降的 ~48%（s1）/~45%（s2）；active window（250–750）为
   −0.001516（−3.39%，s1）/−0.001860（−4.15%，s2），与 active 前同量级、
   远小于同期 CPI 的 −9.61%/−10.42%，无独立 transition；后期（1020–2500）
   基本 plateau。关键论据是 **timing mismatch / lack of independent
   transition**，而非"几乎没变"。
3. **D_ab 与 CE 完全共单调**：Within each seed, CE and D_ab are perfectly
   rank-monotonic in opposite directions（per-seed Spearman ρ = −1；各 14 个
   checkpoint 均值，CE 严格单调降、D_ab 严格单调升、无 ties、无任何脱钩；
   pooled 28 点 ρ = −0.996）——这是 generic sharpening 的 correlate 特征，
   不是独立 mechanism 应有的行为。
4. **1020 后 D_ab 不 plateau**：CPI 在 1020–1500 已近底，D_ab 仍继续上升
   （1020–2500：+0.006/+0.011），与"1020 后同步 plateau"的 Case B 判据不符。

## 4. 为什么最终裁定 Case A

1. CPI / |δ| 下降严格复现，active window 清楚（§2）。
2. CE/NLL 与 CPI 在同一训练阶段持续改善（250–750：CE −0.145/−0.149），支持
   generic conditional-estimation improvement 的解释。
3. JS_dep 不满足 independent structural transition（§3.2）。
4. D_ab 虽两 seed 高度复现且变化明显，但目前不能认定为独立 mechanism（§3.3/§3.4）。

→ **Case A：generic conditional-estimation improvement 为最简解释。**

## 5. D_ab 的正确定位

D_ab 是：

> a robust correlate of training progress / compatibility improvement whose
> independence from conditional-estimation improvement is not established.

- 两 seed 复现性好、方向一致、变化集中在 active window——是**强 correlate**；
- 但**不称 mechanism**：其与 CE 的完全共单调 + 后期继续上升，使它更接近
  generic training progress / conditional sharpening 的伴随量。

## 6. JS_dep 的正确定位

JS_dep changes modestly and reproducibly, but its temporal profile does not
align with the CPI active window: part of the decrease already occurs before
the main CPI transition, and no distinct structural transition appears during
250–750.

- 全程 0→2500：−0.002890（−6.27%，s1）/−0.002786（−6.04%，s2）；
- 其中 0–250（active 前）已占全程下降的 ~48%/~45%；
- 250–750：−3.39%/−4.15%（vs 同期 CPI −9.61%/−10.42%），无独立 transition；
- 后期（1020–2500）基本 plateau。
→ **不构成 independent structural transition**；关键论据是 timing mismatch，
而非变化幅度为零。

## 7. Reproduction evidence

- **s2 deterministic reproduction gate（strict PASS，worst_max_rel = 0.0）**：
  10 个共享 checkpoint（50/100/150/200/250/300/350/400/500/750）EMA
  shadow_params 130/130 全部 `torch.equal`；step 500 anchor 的 model 131/131
  全部 `torch.equal`；step / decay / num_updates 全一致；43 条重叠
  train/eval/lr 日志逐位一致；曲线 500/1000 行逐位一致。
  → 见 `results/phase2/optionb_s2_repro_gate.json`。
- **official Option B reproduction gate（PASS）**：s1/s2 @2500 EMA+model 与
  frozen p1-s1/p1-s2 逐位一致；s1/s2 @1020 EMA 与 formal-vanilla 幸存
  checkpoint 逐位一致；4 组 frozen 指标 diff（cpi/rms/ce/acc）全为 0.0。
  → 见 `results/phase2/optionb_repro_gate.json`。
- **loss 曲线**：新 s2 的 5 个 eval 点（500/1000/1500/2000/2500）与 frozen
  p1-s2 学习曲线 fp32 逐位一致。
- 文件级 SHA256 不同系 torch.save 序列化层差异（file-object vs path 写法），
  已实证 tensor 内容全等——SHA256 仅作辅助证据（用户 2026-10-07 修正案 #1）。

## 8. allocator OOM 不影响 trajectory 的 tensor-exact 证据

- 首次启动（23:04）在 step 2 因 CUDA OOM 碎片化死亡：该目录
  `optionb-dense-s2-230404` 只含 train.log（1 步）与空 learning_curve.csv，
  **未写出任何 checkpoint**，无半截权重。
- 重启（23:06）加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：
  纯分配器配置，不改变任何数值路径 / 训练数学。
- 决定性证据：新 run（带该 flag）与旧 s2 215621（不带该 flag）在 10 个
  共享 checkpoint 上 tensor 级全等（§7）+ loss 日志逐位一致 → 分配器差异对
  trajectory 零影响。
- 终点验证：2500 的 train/eval 与 frozen p1-s2 逐位一致
  （6.504166126251221 / 6.909730724059045）。

## 9. 后续禁止（用户裁定）

不启动：新 seed / 新 hand-crafted metric / Phase 2C / 新 intervention /
full block decoding bulk / PAPL / Swap / PAPL+Swap / 2×2 matrix。

## 10. 关联文件

- `results/phase2/optionb_dense_analysis.json`（case_label=B_candidate_unadjudicated，
  case_b_full_criteria_met=false，case_b_missing_criterion=independence_from_ce）
- `results/phase2/dense_pairs/`（26 files）
- `results/phase2/optionb_s2_repro_gate.json`、`optionb_repro_gate.json`
- run 目录：`exp_local/regime_a/optionb-dense-s2-230610`
