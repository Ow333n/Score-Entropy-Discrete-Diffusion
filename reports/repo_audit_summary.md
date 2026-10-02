# Repo Audit Summary — pre-RL（2026-10-02）

> 快照：`pre-rl-compatibility-v1` @ `de69bb9`（origin main 已同步）。
> 本文回答：进入 RL 阶段前，repo 里哪些组件可直接复用、哪些需要新增。

## 1. 可直接复用的组件（RL 不动它们）

| 组件 | 文件 | 复用方式 |
|---|---|---|
| 模型主体 | `model/transformer.py` (SEDD, 12-block DDiT, gradient checkpointing ON) | 直接复用；RL 前向/反向走同一 forward |
| score 装配 | `model/utils.py::get_score_fn` | rollout 用 `train=False, sampling=True`（返回 exp score）；训练重算用 `train=True`（返回 log score） |
| 图 / rate 矩阵 | `graph_lib.py::Absorbing` | `reverse_rate` / `transp_transition` / `staggered_score` 是 policy 构造的数学基础（见 reverse-transition 分析） |
| noise schedule | `noise_lib.py::LogLinearNoise` | σ(t) = −log(1−(1−ε)t)，rollout 与 SFT 腐蚀同 schedule |
| 采样器 | `sampling.py::get_pc_sampler` | RL rollout 直接复用；predictor 选择是 policy 定义的一部分（协议 §14） |
| 采样原语 | `catsample.py::sample_categorical`（gumbel argmax） | 复用于 rollout 采样 |
| SFT 训练循环骨架 | `training/vanilla.py`（hydra、work_dir、logging、checkpoint save/restore、metadata） | RL 训练循环按同款式新写 `training/rl.py`，沿用 save_ckpt/restore_ckpt 格式 |
| optimizer / warmup / clip | `losses.py::get_optimizer / optimization_manager` | 复用（fresh optimizer，LR pilot 见协议 §28） |
| EMA | `model/ema.py` | 复用（RL 期间 EMA 照常更新，评估口径同 v4.2 双权重） |
| 腐蚀 / span 构造 | `task_data/corruption.py::corrupt_span_batch` | RL rollout 的初始状态构造与 SFT 完全相同（同一 schedule、同一 span 规则） |
| 评估三件套 | `evaluation/eval_task.py / eval_cpi.py / eval_order_gap.py` | RL checkpoint 评估直接复用（EMA/raw 双权重） |
| frozen manifest | `manifests/regime_a_eval_v1.jsonl`（SHA `1897bd14…`） | RL 的 CPI/OrderGap 评估继续用同一 manifest（协议 §31） |
| 概率提取口径 | `compatibility/posterior.py::clean_log_probs` | log p̂(v|C) = log_softmax(score[..., :D−1]) —— 与 CPI 度量内部口径一致，是 RL policy log-prob 的推荐定义（协议 §14） |
| Swap 重计算显存配方 | `training/swap.py`（Pass A no-grad + Pass B 逐状态 backward） | RL 训练时逐 transition forward/backward 的显存模式参考（峰值 = 单状态） |

## 2. RL 需要新增的组件

| 组件 | 说明 |
|---|---|
| `training/rl.py` | rollout / reward / PG 更新主循环（v0.1 DRAFT 协议 §32-33） |
| `rl/rollout.py`（或并入 rl.py） | 用 pc_sampler 生成 span 重建 trajectory；no_grad；保存 (x_t, σ, action, old_logπ) detached |
| `rl/reward.py` | 位置级 token 重建准确率（协议 §18），纯函数、no_grad、不接触模型 |
| `rl/transition.py` | policy 分布构造 + log π 计算（协议 §14-16 的公式实现，独立小模块便于单测） |
| `rl/advantage.py` | group-relative advantage（协议 §22） |
| `tests/test_rl_*.py` | 协议 §34 的 14 项 correctness 单测 |
| `evaluation/eval_rollout.py` | rollout-level 任务指标（mean reward、exact-match、贪心重建） |

## 3. 必须小心的既有实现细节

1. **训练/采样 score 的域不同**：`get_score_fn(train=True)` 返回 **log score**（losses 内 exp）；`sampling=True` 返回 **exp score**。RL 代码任何地方取 score 都要明确域，建议统一在 `rl/transition.py` 内做域转换。
2. **scatter 置零**：forward 尾部把 x_t 自身 token 的 logit 置 0（`transformer.py` 末尾 `torch.scatter`）。MASK 位置 score[MASK] = 0 不是有效 logit —— 概率提取必须像 `clean_log_probs` 一样只在干净词表上 softmax。
3. **采样器是 τ-leaping / staggered 近似**（Euler 用 rate·dt·dσ 一阶；analytic 用 staggered_score 近似 p 比值）。rollout 采样分布与"干净的正规化 policy"之间存在近似差——RL 的 old/new logπ 必须按**同一公式**重算（协议 §14 预注册口径）。
4. **EMA 加载**必须走 `ema.load_state_dict(loaded["ema"])` + `ema.copy_to(...)`，直接 `model.load_state_dict(ema_dict)` 会静默空载（Day 4 记录）。
5. **显存基线**：SFT train 峰值 ~6.4GB（batch 32×256 + gradient checkpointing）。RL 的 rollout 是 eval 模式 no_grad（~1.3-2.4GB），训练步重算单状态 forward+backward（≈ SFT 量级）。8GB 可行，但 batch 需要回到 ~8-16 级（协议 §27）。
6. **checkpoint 格式**：`{model, ema, optimizer, scaler, step}` 五键，~2.7GB/个（fp32 raw+EMA+AdamW×2）。RL checkpoint 计划见协议 §31。
7. **随机源**：`torch.manual_seed`（模型/优化器）+ per-batch `corr_g` generator + `sample_categorical` 走全局 CPU RNG（`torch.rand_like`）。RL 要新增 rollout RNG 控制并在 metadata 记录（协议 §41）。

## 4. 无需改动 / 保持不变

- protocol v4.1（FROZEN_FINAL）、v4.2、regime_a yaml：不修改。
- Stage-4 判决 B_DIAGNOSTIC_ONLY：永久固定。
- P1 全部产物（exp_local/regime_a/p1-*、results/p1_dense）：只读。
- frozen manifest 与 SHA 侧车：只读。
- v4.1 formal / pilot / pretrained 结果与 checkpoint：只读。
