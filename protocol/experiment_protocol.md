# Experiment Protocol Freeze — Post-Training Reveal-Order Compatibility (v4.1)

> 本文件用于在正式 confirmatory experiments 开始前冻结实验协议。
> 协议版本：`v4.1`。
> 一旦冻结，不根据正式实验结果临时修改。
> 任何实现/协议修订必须递增版本号，并使所有受影响 run 失效重跑。

## 1. Training Budget

Vanilla pilot 固定：

```text
pilot_seed = 0
```

Pilot 仅用于 calibration，不用于 seed variance estimation。

### \(N_{\text{pilot,max}}\)

\[
N_{\text{pilot,max}}
\]

首先定义为 optimization horizon，而不是 wall-clock 能跑多少 step。

选择依据：

1. 既有 SEDD-small 训练经验；
2. 预计任务学习所需优化 horizon；
3. Stage 2 smoke 只检查本机是否可执行。

若因硬件缩减 horizon：

```text
hardware_driven_design_choice = true
```

并记录原因。

Pilot 开始前冻结：

```text
N_pilot_max
eval_interval M
plateau_window K
relative_improvement_threshold epsilon
safety_factor alpha
pilot_seed = 0
```

若首次 plateau step 为 \(s_p\)：

\[
N=
\min(
N_{\text{pilot,max}},
\lceil \alpha s_p\rceil
)
\]

若无 plateau：

\[
N=N_{\text{pilot,max}}
\]

正式 Vanilla / PAPL / Swap / PAPL+Swap 使用相同 optimizer-step budget \(N\)。

正式比较不使用 method-specific early stopping。

## 2. Checkpoints

正式 analysis checkpoints：

\[
s_{\text{early}}=\operatorname{round}(0.1N)
\]

\[
s_{\text{mid}}=\operatorname{round}(0.5N)
\]

\[
s_{\text{late}}=N
\]

训练代码必须在这三个显式 step 保存 analysis checkpoints。

所有条件使用完全相同的实际整数 step。

### Preemption snapshots

preemption snapshot 与 analysis checkpoint 分离。

例如：

```text
snapshot_freq_for_preemption = 50–100
```

只用于恢复，不决定正式分析点。

## 2A. Optimizer / LR / EMA Freeze

Vanilla pilot 后、正式实验前冻结：

```text
optimizer_type
base_learning_rate
lr_schedule
warmup_steps_or_ratio
weight_decay
adam_beta1
adam_beta2
adam_eps
gradient_clipping_rule
gradient_clip_value
ema_enabled
ema_decay
mixed_precision_mode
```

正式 Vanilla / PAPL / Swap / PAPL+Swap 使用相同 base optimizer recipe。

若修改 shared optimizer / LR / scheduler / corruption recipe：

> 所有共享该因素的正式条件全部失效重跑。

## 2B. Equal-Budget / Data-Matching Rule

正式主 2×2：

> **data-matched / optimizer-budget matched**

所有 conditions 必须使用：

\[
B_{\text{eff}}
=
B_{\text{micro}}
\times
\text{grad\_accum}
\times
n_{\text{gpu}}
\]

相同的：

- effective batch；
- optimizer-step budget \(N\)；
- unique training sequence exposure；
- unique source-token exposure。

若 Swap micro-batch 更小：

> 增大 gradient accumulation 恢复相同 effective batch。

只有整数约束导致无法完全一致时允许：

\[
\frac{|T_c-T_{\text{ref}}|}{T_{\text{ref}}}\le1\%
\]

超过 1%：

> formal factorial 不允许开始，必须重新设计 accumulation。

每个 formal run 记录：

```text
unique_training_sequences_seen
unique_source_tokens_seen
model_forward_tokens
optimizer_steps
```

注意：

> Swap 的额外 forward 不计作额外 unique seen data。

主 factorial 不要求 compute-matched。

G6 matched-control 才要求在 data-matched 基础上进一步尽量：

> compute-matched。

## 3. Swap Pair

- \(i\neq j\)
- \(i,j\) 都必须来自 target span 当前 masked positions。
- masked target positions < 2 时，不产生 pair。
- 默认：

\[
K_{\text{pair}}=1
\]

每 sample 每 step 最多一个 pair。

## 4. Swap Reduction

\[
L_{\text{swap}}
=
\lambda
\frac{1}{N_{\text{pair}}}
\sum_i
\delta_i^2
\]

如果 \(N_{\text{pair}}=0\)，该 microbatch 只计算 task loss并记录 valid pair count。

## 4A. Valid-Pair Fraction

在 calibration 阶段估计：

\[
p_{\ge2}
=
P(
\text{target 中至少有 2 个 masked positions}
)
\]

正式 run 记录：

```text
valid_pair_fraction
expected_valid_pair_fraction
```

正式 run 前冻结 tolerance：

\[
\tau_{\text{pair}}
\]

若：

\[
|p_{\text{observed}}-p_{\ge2}|>\tau_{\text{pair}}
\]

持续出现，则视为 corruption / pair-construction protocol violation。

不得使用任意固定 50% 阈值。

## 5. Stop-Gradient Semantics

真实 objective：

\[
\lambda\delta^2
\]

recomputation 使用：

\[
2\lambda\operatorname{sg}(\delta)\nabla\delta
\]

作为 exact first-order gradient implementation。

## 6. Teacher-Forced Path

初始 masked target set：

\[
\mathcal M_0
\]

\[
m=|\mathcal M_0|
\]

只对 initially masked target positions 计算：

\[
Q_\pi(x)
=
\sum_{k=1}^{m}
\log p_\theta(x_{i_k}|C,x_{i_{<k}})
\]

## 7. Confidence Tie-Break

confidence 相同时：

> 选择 position index 更小的位置。

记录 tie count / fraction。

## 8. Sampling

主结果：

- greedy
- \(T=1.0\)

Sensitivity：

- \(T=0.5\)
- \(T=1.5\)

## 9. Evaluation Manifest

所有正式 checkpoint 使用完全相同 frozen manifest。

固定：

- samples
- pairs
- sigma
- random paths
- fixed confidence path
- sampling seeds

## 10. Seeds

分别记录：

```text
model_seed
data_order_seed
corruption_seed
eval_manifest_seed
sampling_seed
```

训练 seed 可变化，evaluation manifest 不变化。

## 10A. DataLoader Worker Seed

如果：

```text
num_workers > 0
```

必须设置 deterministic `worker_init_fn`。

每个 worker 的 seed 从：

```text
data_order_seed
```

确定性派生。

正式 run config 记录：

```text
dataloader_worker_seed_rule
```

## 11. Lambda Calibration

第一阶段：

\[
B_{\text{calib}}=10
\]

分别计算：

\[
r_b=
\frac{\|g_{\text{swap},b}^{(\lambda=1)}\|}
{\|g_{\text{SFT},b}\|}
\]

报告 mean / median / IQR / CV。

以 median 为主要 calibration scale。

若 10 batch 后：

\[
CV>0.5
\]

则增加到：

\[
B_{\text{calib}}=30
\]

如果 30 batch 后仍：

\[
CV>0.5
\]

则：

- 使用 median / IQR；
- 标记：

```text
heavy_tail_lambda_calibration = true
```

- 不再继续增加 batch。

目标辅助梯度比例：

\[
0.05,\quad0.1,\quad0.2
\]

formal run 前冻结最终 λ。

## 12. Instability

默认：

```text
gradient_monitor_interval = 50 optimizer steps
instability_patience = 3 intervals
```

如果连续 3 个 intervals：

\[
\frac{\|g_{\text{swap}}\|}{\|g_{\text{SFT}}\|}>2
\]

或者出现 non-finite loss / gradient：

> calibration 中判 unstable。

允许 λ 减半，最多两次。

正式 run 中不动态修改 λ / LR。

若最终 \(N\) 极小使 150-step patience 明显不合理，必须在 formal run 前修改并形成 protocol revision。

## 13. Failure Handling

正式 run 出现：

- NaN；
- Inf；
- unrecoverable OOM；
- crash；

标记 `failed`。

不得修改 recipe 后继续视作同一 formal run。

### Systematic failure

如果某 condition 在：

\[
\ge2
\]

个正式 seed 中 failed：

> 暂停该 condition，返回 calibration。

重新冻结 recipe 后：

- 修改 shared base LR / optimizer / scheduler / corruption：
  - 所有共享该因素的 conditions 全部失效重跑；
- 仅修改 Swap-specific λ / implementation：
  - Swap 与 PAPL+Swap 失效重跑；
  - Vanilla / PAPL 可保留。

不得混用旧 recipe 与新 recipe 的正式结果。

## 14. Extreme Delta

记录：

```text
delta_min
delta_max
delta_mean
delta_std
delta_abs_p95
delta_abs_p99
```

finite extreme values 不静默跳过。

## 15. Regime B G1

在 pilot 前定义。

如果 metric 适合 ceiling normalization：

\[
G=
\frac{M_{\text{SFT}}-M_{\text{pre}}}
{M_{\text{ceiling}}-M_{\text{pre}}}
\ge0.2
\]

且 bootstrap CI lower bound > 0。

如果 metric 不适合 ceiling normalization：

> 必须在 pilot 前写出 task-specific metric、threshold、practical-effect criterion 和 uncertainty criterion。

不得看完 pilot 后改 threshold 让任务通过。

## 15A. Regime B Protocol Artifact

Regime B task-specific G1 和数据生成规则写入：

```text
protocol/regime_b_protocol.yaml
```

必须在 Regime B pilot 前冻结。

记录：

```text
regime_b_protocol_version
regime_b_protocol_sha256
```

formal Regime B run config 必须包含该 hash。

## 16. OrderGap

主报告：

\[
OrderGap_{\text{raw}}
\]

和：

\[
OrderGap_{\text{per-token}}
=
\frac{OrderGap_{\text{raw}}}{m}
\]

单位 nats / revealed token。

## 16A. G5 Mechanism Gate

不使用：

> “L_swap 至少下降 50%”

之类 arbitrary hard threshold。

### Optimization check

要求：

\[
L_{\text{swap,late}}<L_{\text{swap,early}}
\]

并报告 relative change。

### Structural check

Swap vs Vanilla：

\[
\Delta CPI<0
\]

\[
\Delta OG<0
\]

主判断要求：

- 两个 training seeds effect direction 一致；
- effect size 非平凡；
- 每个 seed 内的 paired sample analysis 支持相同方向。

如果两个 training seeds 方向相反：

```text
G5_status = unstable
```

sample-level CI 不解释为 training-seed-level significance。

## 17. Statistics

正式 comparison：

- paired mean difference；
- SEM；
- paired bootstrap CI；
- paired standardized effect size；
- per-seed result；
- seed mean / spread。

### 2-seed rule

第一轮只有 2 个 training seeds 时：

- 主文展示两个 seed 各自结果和 seed mean；
- sample-level paired bootstrap CI 只解释为 evaluation-sample uncertainty；
- 不把它当作 training-seed-level significance；
- 两个 seed effect direction 相反时，标记 `unstable`；
- 不通过不断增加 seed 来追求显著性。

## 17A. Manifest Versioning

正式 manifest 命名：

```text
regime_a_eval_v1.jsonl
regime_a_eval_v2.jsonl
regime_b_eval_v1.jsonl
```

同一版本号内容不可变化。

每个 manifest 保存 SHA-256。

正式 run config 记录：

```text
manifest_filename
manifest_version
manifest_sha256
```

如果发现 manifest bug：

1. 原版本保留；
2. 新版本号；
3. protocol version 递增；
4. 受影响 formal runs 标记 invalid；
5. 重跑受影响 conditions。

## 18. Environment

每个 run 保存：

```text
python_version
torch_version
cuda_runtime_version
gpu_driver_version
numpy_version
transformers_version
git_commit
protocol_version
recipe_version
```

## 18A. Protocol / Implementation Bug Rule

如果正式实验中发现实现或协议 bug：

1. 记录 bug 描述；
2. 递增 protocol version；
3. 记录修复前后 git commit；
4. 所有受影响 run 标记 `invalid`；
5. 重跑所有受影响 conditions；
6. 不允许继续使用受影响旧 run 数据。

## 18B. Protocol Version vs Recipe Version

### protocol_version

规则层版本，例如：

```text
v4.1
```

控制：

- budget；
- checkpoint；
- evaluation；
- failure；
- statistics；
- manifest rules。

### recipe_version

具体 condition 配置版本，例如：

```text
vanilla_v1
papl_v1
swap_v2
papl_swap_v2
```

Protocol revision 不必然使所有 recipes 失效；按修改影响范围判定。

每个 formal run 同时记录：

```text
protocol_version
recipe_version
```

## 19. Formal Freeze Rule

一旦进入正式 confirmatory experiment：

> 不因为结果“不好看”而改变 checkpoint、seed、manifest、temperature、pair selection、training budget、λ、learning rate、G1 threshold 或 failure handling。

## 20. K-Bucket Freeze

K buckets 必须在 formal CPI comparison 前冻结。

允许查看：

> frozen corruption-process support / pilot distribution

不得查看：

> formal method comparison results

后再改 bucket。

例如仅作为候选：

```text
K = 0 / 1 / 2 / 3 / 4+
```

实际边界写入冻结协议。

## 21. Manifest Eligibility

正式 CPI / OrderGap manifest samples 必须满足：

\[
m=|\mathcal M_0|\ge2
\]

不满足者在 manifest construction 阶段跳过并记录 skip count。

## 22. Final Freeze Statement

正式 confirmatory experiments 开始后，不得根据结果质量修改：

```text
N
analysis checkpoint steps
base LR
LR schedule
optimizer
EMA
gradient clipping
dropout
pair selection
K_pair
effective_batch
unique_data_exposure
lambda
temperature
manifest
G1 threshold
failure handling
statistics protocol
```

合法 bug fix / protocol revision 必须：

> 新 protocol version + 所有受影响 formal runs invalidation / rerun。

## 23. Canonical Protocol File

唯一维护的协议文件：

```text
protocol/experiment_protocol.md
```

不并行维护内容重复的版本副本。

协议版本通过文件内部：

```text
protocol_version
```

管理。
