# 本地执行计划 — Post-Training Reveal-Order Compatibility（v4.1 FROZEN）

> 配套研究提案：`order_alignment_vs_compatibility_research_proposal_v2.md`
>
> 本文是当前研究方案的**最终本地执行版**，已经吸收前几轮所有关键审查意见，并加入正式的 `experiment protocol freeze`。
>
> 核心原则：
>
> 1. **先验证测量，再验证现象，再验证机制，最后实现干预。**
> 2. **所有 confirmatory experiment 在看到结果前冻结协议。**
> 3. **正式 run 中不动态修改超参数、停止规则、评估规则或样本选择规则。**
> 4. **每次比较尽量只改变一个科学变量。**
>
> 协议版本：`v4.1`
>
> 正式 confirmatory experiment 开始后，任何协议/实现修订必须递增版本号，并使所有受影响 run 失效重跑。
>
> 标记：
>
> - ✅ = 已实测
> - 📐 = 估算，必须通过 smoke test 校准
> - 🛑 = hard stop / falsification gate
> - 🔒 = 正式实验前必须冻结

---

# 0. 当前研究定位

## 0.1 核心问题

本项目不再把主要贡献表述为：

> “发现 diffusion language model 存在 reveal-order incompatibility。”

已有工作已经对 order dependence、local circulation、pseudo-joint deviation 等现象进行了诊断。

当前核心研究问题收紧为：

> **后训练是否系统性改变 masked / absorbing diffusion language model 的 reveal-order compatibility？这种变化是否与 post-training 后的 order sensitivity 相联系？针对 reveal-order compatibility 的显式正则，是否能提供区别于 planner/order alignment 的额外作用？**

核心研究链条：

\[
\text{Pretrained}
\rightarrow
\text{Post-training}
\rightarrow
\Delta \text{Compatibility}
\rightarrow
\Delta \text{Order Sensitivity}
\rightarrow
\text{Compatibility Intervention}
\]

定义：

\[
\Delta CPI
=
CPI_{\text{post}}
-
CPI_{\text{pretrained}}
\]

不预设符号：

- \(\Delta CPI > 0\)：post-training **amplifies** incompatibility；
- \(\Delta CPI < 0\)：post-training **attenuates** incompatibility；
- \(\Delta CPI \approx 0\)：post-training approximately **preserves** incompatibility。

因此核心假设写为：

> **H1：Post-training changes / amplifies / attenuates reveal-order compatibility.**

不再写：

> SFT creates incompatibility.

---

# 1. 与现有工作的边界

## 1.1 Reveal-order diagnostics

已有工作已经表明：

- order-agnostic / masked diffusion LM 的不同 reveal order 可能给同一完整序列不同的 pseudo-joint score；
- local adjacent swaps 可以刻画局部 circulation / incompatibility；
- global order gap 可以和这些 local circulations 联系起来。

因此本文不再声称：

> “首次发现不同 reveal order 会得到不同 likelihood。”

也不把 local swap residual 本身作为主要数学 novelty。

---

## 1.2 Order alignment / planner-aware training

PAPL、TRIMS、Scheduling Thoughts 等工作主要处理：

> **应该优先走哪条 reveal path。**

本文研究的是：

> **不同合法 reveal path 之间的 conditional probability 是否互相兼容。**

这两个问题相关，但不相同。

---

## 1.3 Diffusion-time path consistency

CDLM / MPDC 研究：

\[
x_t \rightarrow x_s \rightarrow x_0
\]

不同 diffusion noise levels / posterior bridges 之间的 consistency。

本文研究：

\[
C\rightarrow C+a\rightarrow C+a+b
\]

与：

\[
C\rightarrow C+b\rightarrow C+a+b
\]

之间的 **token reveal-order compatibility**。

因此当前 novelty 空间定义为：

1. **post-training dynamics**；
2. **reveal-order-specific intervention**；
3. **alignment vs compatibility interaction**；
4. **task-dependent boundary conditions**。

---

# 2. 硬件与 Repo 约束

## 2.1 已确认硬件

| 项目 | 当前状态 |
|---|---|
| GPU | RTX 5060 Ti 8GB，Windows WDDM |
| SEDD-small | ✅ 约 170M，可训练 |
| SEDD-medium | ✅ 完整 AdamW + EMA 本机训练不可行，仅 inference-only |
| 1024-seq baseline | ✅ 已跑通 |
| 训练吞吐 | ✅ 约 2.2 training-loop steps/s；actual batch semantics 在 G0 重新确认 |
| 采样 | ✅ 当前 `sampling.py` 无 KV cache |
| gradient checkpointing | ✅ 当前 transformer 未实现 |
| WDDM | 必须留显存余量，不再以 95% VRAM 为长期目标 |

---

## 2.2 Batch semantics 必须重新核实

原记录中的：

```text
batch 4 / accum 2 = eff 8
```

不能继续作为未经验证的事实。

G0 中打印：

```text
config.training.batch_size
actual DataLoader batch.shape[0]
training.accum
ngpus
optimizer_step_count
sequences_per_optimizer_step
```

统一采用：

\[
B_{\text{effective}}
=
B_{\text{actual micro}}
\times
\text{accum}
\times
n_{\text{gpu}}
\]

后续每个 run 都记录：

```text
config_batch
actual_micro_batch
grad_accum
effective_batch
seq_len
tokens_per_optimizer_step
```

---

## 2.3 短序列原则

Regime A / B 任务优先控制：

\[
L\le256
\]

必要时：

\[
L\le128
\]

注意：

- attention 总复杂度 1024 → 256 理论上约下降 16×；
- per-token attention cost 约下降 4×；
- vocab projection 等部分仍是 \(O(L)\)；
- 增大 batch 会重新提高总 token 数。

因此：

> **不再预写 20–60 steps/s。**

所有新 pipeline 先运行：

> **300-step smoke calibration**

记录：

```text
actual_micro_batch
grad_accum
effective_batch
optimizer_steps_per_second
tokens_per_second
peak_vram_allocated
peak_vram_reserved
wall_clock_per_optimizer_step
```

---

# 3. 🔒 Experiment Protocol Freeze

本节是正式 confirmatory experiments 的冻结协议。

任何 pilot / development 阶段允许调试，但一旦进入正式实验，以下规则不得根据结果临时改变。

---

## 3.1 总训练预算与 Vanilla pilot

先运行一个：

> **Vanilla pilot**

它不进入最终方法比较。

Pilot 固定：

```text
pilot_seed = 0
```

pilot 的目的仅是 calibration，不用于估计 training-seed variance。

### Pilot 的作用

pilot 用于冻结：

- stable learning rate；
- optimizer；
- LR scheduler；
- weight decay；
- EMA 配置；
- gradient clipping；
- 总训练预算 \(N\)；
- eval frequency；
- analysis checkpoint steps；
- preemption snapshot frequency；
- span difficulty；
- 256-seq 实际 throughput。

### \(N_{\text{pilot,max}}\) 的原则

\[
N_{\text{pilot,max}}
\]

首先定义为：

> **optimization horizon**

而不是“某张 GPU 在固定小时数内能跑多少 step”。

其选择依据优先级：

1. 现有 SEDD-small / 1024-seq 训练经验；
2. 预计 task learning 所需优化 horizon；
3. Stage 2 的 256-seq smoke test 用于检查该 horizon 在本机是否现实。

如果硬件预算要求缩减：

> 必须记录为 `hardware_driven_design_choice`。

不得因为换了更快 GPU 就自动改变科学定义下的 \(N_{\text{pilot,max}}\)。

### Pilot 如何确定 \(N\)

在 pilot 开始前冻结：

```text
N_pilot_max
eval_interval M
plateau_window K
relative_improvement_threshold epsilon
safety_factor alpha
pilot_seed = 0
```

pilot 至少运行到：

- learning curve 出现预注册 plateau；
- 或达到 \(N_{\text{pilot,max}}\)。

plateau 定义：

> 连续 \(K\) 次 validation evaluation 的 relative improvement 均小于 \(\epsilon\)。

若首次在 step \(s_p\) 满足 plateau，则：

\[
N
=
\min(
N_{\text{pilot,max}},
\lceil \alpha s_p\rceil
)
\]

其中：

\[
\alpha>1
\]

必须在 pilot 开始前冻结。

如果直到 \(N_{\text{pilot,max}}\) 都未出现 plateau：

\[
N=N_{\text{pilot,max}}
\]

pilot 必须保存：

- 完整 learning curve；
- plateau 检测结果；
- \(N\) 的选择依据。

正式 factorial experiment 中：

> **Vanilla / PAPL / Swap / PAPL+Swap 使用相同 optimizer-step budget \(N\)。**

正式比较不使用 method-specific early stopping。

## 3.2 Checkpoint protocol

正式训练开始前冻结总训练预算：

\[
N
\]

主分析 checkpoint 定义为：

\[
s_{\text{early}}
=
\operatorname{round}(0.1N)
\]

\[
s_{\text{mid}}
=
\operatorname{round}(0.5N)
\]

\[
s_{\text{late}}
=
N
\]

其中：

> **analysis checkpoint 与 preemption snapshot 分离管理。**

### Analysis checkpoints

训练代码必须在以下显式 step 保存正式分析 checkpoint：

```text
early_step
mid_step
late_step
```

所有条件使用完全相同的实际整数 step。

不得依赖普通 snapshot frequency 去“碰巧命中”这些位置。

### Preemption snapshots

仅用于崩溃恢复，例如：

```text
snapshot_freq_for_preemption = 50–100
```

它不决定正式分析 checkpoint。

正式 protocol 中预先记录：

```text
early_step
mid_step
late_step
```

不得：

> 看完 validation curve 后再挑“最好看的” early / mid / late。

如需 best-validation checkpoint，只能作为附加 exploratory analysis，不替代主协议 checkpoint。

## 3.3 正式停止标准

主停止标准：

\[
\text{optimizer steps}=N
\]

安全停止条件：

- non-finite loss；
- non-finite gradients；
- unrecoverable OOM；
- hard crash。

正式 run 不因 validation plateau 提前停止。

如果某个 method 经常失败：

> 返回 calibration 阶段重新确定 recipe，然后重新开始正式比较。

不能：

> 在已经开始的 confirmatory factorial 中临时改 recipe 后继续把结果放在同一个主比较表里。

---

## 3.3A Optimizer / LR / EMA protocol

以下配置必须在 Vanilla pilot 后、正式实验前冻结：

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

正式 Vanilla / PAPL / Swap / PAPL+Swap：

> 使用相同 base optimizer recipe。

若某个方法因其特有 auxiliary objective 需要修改 optimizer / LR recipe：

> 必须视为 protocol revision，并使所有共享被修改因素的正式条件失效重跑。

## 3.3B Equal-Budget / Data-Matching Protocol

主 factorial 的公平性定义为：

> **data-matched + optimizer-budget matched**

而不是 compute-matched。

正式 Vanilla / PAPL / Swap / PAPL+Swap 必须满足：

1. 相同 optimizer-step budget：

\[
N
\]

2. 相同 effective training batch：

\[
B_{\text{eff}}
=
B_{\text{micro}}
\times
\text{grad\_accum}
\times
n_{\text{gpu}}
\]

3. 相同 unique training sequence exposure；
4. 相同 unique source-token exposure。

如果 Swap 因显存限制必须减小 micro-batch：

> 必须优先通过增大 gradient accumulation 恢复相同的 \(B_{\text{eff}}\)。

例如：

```text
Vanilla: micro=16, accum=1 -> effective batch=16
Swap:    micro=4,  accum=4 -> effective batch=16
```

### Tolerance

理想情况：

\[
T_c=T_{\text{ref}}
\]

其中 \(T_c\) 是 condition \(c\) 的 unique source-token exposure。

只有在 repo / dataloader 的整数约束导致无法完全匹配时，允许：

\[
\frac{
|T_c-T_{\text{ref}}|
}{
T_{\text{ref}}
}
\le 1\%
\]

该 1% 仅作为工程离散误差 tolerance，不作为常规预算放宽。

如果超过 1%：

🛑 **不得进入正式主 factorial。**

必须重新设计 micro-batch / gradient accumulation / step accounting。

### Unique data exposure 与 compute exposure 分开记录

Swap 中的：

\[
C,\quad C+a,\quad C+b
\]

是同一批 source examples 的不同 model states。

因此正式记录必须区分：

```text
unique_training_sequences_seen
unique_source_tokens_seen
model_forward_tokens
optimizer_steps
```

其中：

- 数据公平性看：
  - `unique_training_sequences_seen`
  - `unique_source_tokens_seen`

- 计算成本看：
  - `model_forward_tokens`
  - forward/backward count
  - wall-clock
  - estimated FLOPs（如可得）

Swap 的额外 forward **不能**计入额外 unique seen data。

### 主 factorial 与 G6 control 的公平性不同

主 2×2：

> **data-matched / optimizer-budget matched**

G6 matched control：

> 在 data-matched 基础上进一步尽量 **compute-matched**。

主 factorial 不要求 Vanilla 人为增加无意义 compute 去匹配 Swap。

## 3.4 Swap pair protocol

对每个训练样本：

1. 只从 target span 内当前 masked positions 选择 pair；
2. \(i\neq j\)；
3. 如果 masked target positions < 2，则该 sample 不产生 Swap pair；
4. 每个 sample 每次训练固定最多：

\[
K_{\text{pair}}=1
\]

作为默认配置；
5. 如果未来做 \(K_{\text{pair}}>1\) ablation，必须作为独立实验。

这样避免：

> 长 span / high-mask 样本天然贡献更多 Swap loss。

---

## 3.5 Swap reduction

当前 batch / microbatch 中有效 pair 数：

\[
N_{\text{pair}}
\]

Swap objective：

\[
L_{\text{swap}}
=
\lambda
\frac{1}{N_{\text{pair}}}
\sum_{n=1}^{N_{\text{pair}}}
\delta_n^2
\]

如果：

\[
N_{\text{pair}}=0
\]

该 microbatch 仅计算 task loss，并记录：

```text
swap_valid_pair_count = 0
```

同时报告：

```text
valid_pair_fraction
```

不得静默忽略。

---

## 3.5A Valid-pair fraction protocol

根据 frozen corruption process，在 pilot / calibration 阶段估计：

\[
p_{\ge2}
=
P(
\text{target 中至少存在 2 个 masked positions}
)
\]

正式 run 记录：

```text
valid_pair_fraction
expected_valid_pair_fraction
```

不得使用任意固定阈值（例如 50%）判断 protocol violation。

正式 run 的 valid-pair fraction 必须落在预先冻结的 tolerance band 内：

\[
|p_{\text{observed}}-p_{\ge2}|
\le\tau_{\text{pair}}
\]

其中 \(\tau_{\text{pair}}\) 在 formal run 前由 pilot / binomial variation 预先设定。

若持续超出 tolerance band：

> 标记为 corruption / pair-construction protocol violation，停止该 run 并检查实现。

## 3.6 Stop-gradient 语义

真实 objective：

\[
L_{\text{swap}}
=
\lambda\delta^2
\]

其中 \(\delta\) 本身依赖模型参数。

8GB recomputation 实现中使用：

\[
\operatorname{sg}(\delta)
\]

作为外部 coefficient。

因此计算的梯度为：

\[
2\lambda
\operatorname{sg}(\delta)
\nabla\delta
\]

这与：

\[
\nabla(\lambda\delta^2)
\]

数学上相同。

文档中不能写成：

> “\(\delta\) 本身是 detached objective。”

正确表述：

> **The coefficient is detached during recomputation, while the target gradient remains the exact first-order gradient of \(\lambda\delta^2\).**

---

## 3.7 Teacher-forced path length

定义初始 masked target set：

\[
\mathcal M_0
\]

其中只包含：

> evaluation 开始时 target span 内 initially masked positions。

Manifest builder 必须保证用于 CPI / OrderGap 的正式样本满足：

\[
m=|\mathcal M_0|\ge2
\]

若候选样本不满足，则在 manifest 构建阶段跳过并记录 skip count，而不是在正式 evaluation 时临时处理。


定义：

\[
m=|\mathcal M_0|
\]

Teacher-forced path likelihood：

\[
Q_\pi(x)
=
\sum_{k=1}^{m}
\log
p_\theta(
x_{i_k}
|
C,x_{i_1},...,x_{i_{k-1}}
)
\]

只对：

\[
i_k\in\mathcal M_0
\]

求和。

初始 visible target positions 不参与 \(Q_\pi\)。

---

## 3.8 Confidence-first tie-breaking

每步对所有当前 masked target positions 计算 confidence。

如果两个或多个位置 confidence 完全相同：

> 选择 sequence index 最小的位置。

记录：

```text
confidence_tie_count
confidence_tie_fraction
```

如果 tie 比例异常高，需要作为模型 confidence resolution 的 diagnostic。

---

## 3.9 Sampling protocol

主 sampled results 固定：

### Greedy

等价于：

> argmax token selection

### Standard sampling

\[
T=1.0
\]

固定 sampling seeds。

Sensitivity-only：

\[
T=0.5
\]

与：

\[
T=1.5
\]

不进入主 hypothesis test，除非预先升级为主实验。

---

## 3.10 Evaluation paths

所有固定路径写入 evaluation manifest。

至少保存：

- L2R；
- R2L；
- fixed random paths；
- pretrained-derived confidence-first path。

正式 checkpoint comparison 不重新生成 fixed paths。

---

## 3.11 Training seed protocol

不同训练 seed：

> 使用不同 training randomness。

包括：

```text
model_seed
data_order_seed
corruption_seed
```

但 evaluation：

> 完全使用同一个 frozen manifest。

单独记录：

```text
eval_manifest_seed
sampling_seed
```

---

## 3.11A DataLoader worker seed

如果：

```text
num_workers > 0
```

则必须：

- 设置 deterministic `worker_init_fn`；
- 每个 worker 的 seed 从：

```text
data_order_seed
```

确定性派生；
- 正式 run config 中记录：

```text
dataloader_worker_seed_rule
```

这样可以确保：

> 相同的 training seed / data-order configuration 能对应可复现的 DataLoader worker 随机流。

如果：

```text
num_workers = 0
```

则该规则记为：

```text
dataloader_worker_seed_rule = N/A
```

## 3.12 Instability protocol

instability 不使用：

> “loss > initial loss × 10”

这种任意阈值。

主要监控：

- non-finite loss；
- non-finite gradient；
- auxiliary gradient norm；
- SFT gradient norm；
- total gradient norm；
- optimizer update norm；
- delta statistics。

在 calibration 阶段，如果：

\[
\frac{
\|g_{\text{swap}}\|
}{
\|g_{\text{SFT}}\|
}
>2
\]

连续多个 monitoring interval，

或者出现 non-finite gradients，

则当前 λ 判为 unstable。

calibration 阶段允许：

\[
\lambda\leftarrow \lambda/2
\]

最多重试两次。

正式 confirmatory run：

> 不动态调整 λ 或 learning rate。

---

## 3.13 NaN / crash / systematic failure protocol

### Calibration 阶段

允许：

- 调 learning rate；
- 调 λ；
- 调 batch；
- 调 optimizer numerical settings；
- 修实现 bug。

### Formal run

如果出现：

- NaN；
- Inf；
- unrecoverable OOM；
- crash；

则：

> 标记该 run 为 `failed`。

不能：

> 临时改 learning rate / λ / optimizer 后继续把 continuation 视作同一个正式 run。

### Systematic failure

如果某个 condition 在：

\[
\ge2
\]

个正式 seed 中 failed，则暂停该 condition 的正式比较，返回 calibration。

重新冻结 recipe 后：

> **所有共享了被修改因素的 condition 都必须失效并重跑。**

例如：

- 如果修改 shared base LR / optimizer / scheduler / corruption pipeline：
  - Vanilla
  - PAPL
  - Swap
  - PAPL+Swap  
  全部正式 run 失效重跑。

- 如果只修改 Swap-specific λ / Swap numerical implementation：
  - Swap
  - PAPL+Swap  
  失效重跑；
  - Vanilla / PAPL 可继续保留。

不得：

> 只重跑某个失败 seed，而保留同一受影响 condition 的旧 recipe 结果混在主分析中。

最终报告：

```text
failed_run_count
failure_rate
failure_reason
recipe_version
```

## 3.14 Extreme delta handling

每步记录：

```text
delta_min
delta_max
delta_mean
delta_std
delta_abs_p95
delta_abs_p99
```

实现使用稳定的：

```python
log_softmax
```

不得因为 \(|\delta|\) 很大而静默跳过 sample / batch。

只有：

> non-finite \(\delta\)

可以触发 numerical failure handling。

如果极端 finite delta 大量出现：

> 先诊断概率下溢、错误 posterior extraction 或真实 heavy-tail behavior。

如未来使用 clipping：

> 必须作为显式、预注册的独立 ablation。

---

## 3.15 Regime B G1 protocol

G1 threshold 必须在看到 Regime B pilot result 前定义。

若 task metric 适合 ceiling normalization，默认使用：

\[
G
=
\frac{
M_{\text{SFT}}
-
M_{\text{pre}}
}{
M_{\text{ceiling}}
-
M_{\text{pre}}
}
\]

默认 gate：

\[
G\ge0.2
\]

同时：

> paired / bootstrap CI lower bound > 0

并要求：

- 超过 trivial baseline；
- 未发生 ceiling saturation。

### 如果 metric 不适合 ceiling normalization

必须在 Regime B pilot 开始前，写入：

```text
task_specific_metric
task_specific_G1_threshold
practical_effect_threshold
uncertainty_criterion
```

并解释其与 task difficulty / baseline 的关系。

不得：

> 看完 pilot 结果以后修改 G1 threshold 让该任务通过。

pilot 不通过：

> reject / redesign task。

## 3.16 OrderGap normalization

主报告：

\[
OrderGap_{\text{raw}}
=
\max_\pi Q_\pi
-
\min_\pi Q_\pi
\]

同时报告：

\[
OrderGap_{\text{per-token}}
=
\frac{
OrderGap_{\text{raw}}
}{
m
}
\]

单位：

> nats / revealed token

不使用：

> OrderGap / mean(|Q|) < 1%

这种无理论依据 hard threshold。

---

## 3.17 Statistics protocol

所有 checkpoint / method comparison 优先：

> paired analysis

因为 evaluation manifest 完全相同。

报告：

- mean；
- SEM；
- paired bootstrap CI；
- raw paired difference；
- standardized paired effect size。

例如：

\[
d_{\text{paired}}
=
\frac{
E[\Delta CPI]
}{
SD(\Delta CPI)
}
\]

不依赖：

> “看起来差不多。”

---

## 3.17A Protocol / implementation bug rule

正式实验开始后，如果发现：

- protocol 文本本身有错误；
- manifest builder 有 bug；
- pair selection 有 off-by-one；
- metric implementation 有错误；
- checkpoint selection implementation 与冻结协议不一致；

允许修复，但必须：

1. 写 bug 描述；
2. 递增 protocol version；
3. 记录修复前后 git commit；
4. 标记所有受影响 run 为 `invalid`；
5. 重新运行所有受影响条件；
6. 不允许修复后继续沿用旧 run 的数据。

Protocol revision 示例：

```text
v4.0 -> v4.1
```

任何正式分析表必须记录：

```text
protocol_version
```

## 3.18 Environment protocol

每个正式 run 保存：

```text
python_version
torch_version
cuda_runtime_version
gpu_driver_version
numpy_version
transformers_version
git_commit
hostname_or_machine_label
```

必要时增加：

```text
cudnn_version
```

---

## 3.19 Formal Freeze Rule

一旦进入正式 confirmatory experiment：

> **不得因为结果“不好看”而修改已经冻结的实验规则。**

以下项目在 formal run 开始前必须冻结：

- 总训练预算：

\[
N
\]

- analysis checkpoint steps；
- effective batch：

\[
B_{\text{eff}}
\]

- unique training sequence exposure；
- unique source-token exposure；
- base learning rate；
- LR schedule；
- optimizer；
- EMA；
- gradient clipping；
- dropout；
- pair selection rule；
- \(K_{\text{pair}}\)；
- \(\lambda\)；
- sampling temperature；
- evaluation manifest；
- Regime B G1 threshold；
- failure handling；
- statistics protocol。

正式结果出现后，不允许为了改善结果临时修改：

```text
checkpoint
seed
manifest
temperature
pair selection
training budget
effective batch
lambda
learning rate
G1 threshold
failure handling
statistics rule
```

### 合法 bug fix / protocol revision

如果正式实验过程中发现：

- 实现 bug；
- protocol 文本错误；
- metric bug；
- manifest bug；
- off-by-one；
- checkpoint saving 与冻结协议不一致；

允许修复，但必须：

1. 递增 `protocol_version`；
2. 记录 bug 描述；
3. 记录修复前后 git commit；
4. 标记所有受影响 formal runs 为：

```text
invalid
```

5. 重新运行所有受影响 conditions；
6. 不允许把旧 recipe / 旧协议下的结果与修复后的正式结果混用。

完整、唯一权威规则以：

```text
protocol/experiment_protocol.md
```

为准。

# 4. 阶段 0 — G0 正确性门

预计：

- 编码：0.5–1 天
- GPU：<1h

🛑 **G0 不通过，不进入研究结论。**

---

## 4.1 Batch semantics

打印：

```text
config.training.batch_size
actual DataLoader batch.shape[0]
training.accum
ngpus
optimizer_step_count
sequences_per_optimizer_step
```

冻结实际语义。

---

## 4.2 Normalized conditional posterior

目标：

\[
\hat p_\theta(v|C,t)
\]

要求：

- clean vocabulary only；
- `[MASK]` excluded；
- masked position 上：

\[
\sum_v \hat p_\theta(v|C,t)=1
\]

验证：

1. normalization；
2. mask exclusion；
3. time scalar cancellation；
4. repeated deterministic evaluation consistency。

---

## 4.3 Fixed-sigma CPI

一个 quartet：

\[
C,\quad C+a,\quad C+b
\]

必须共享完全相同的：

\[
t/\sigma
\]

---

## 4.4 Swap defect

\[
\delta_{\text{swap}}
=
\log p(a|C)
+
\log p(b|C,a)
-
\log p(b|C)
-
\log p(a|C,b)
\]

主指标：

\[
CPI_{\text{abs}}
=
E[|\delta|]
\]

辅助：

\[
CPI_{\text{RMS}}
=
\sqrt{E[\delta^2]}
\]

同时记录：

- signed mean；
- median；
- p90；
- p99。

---

## 4.5 Compatible toy test

显式构造 joint：

\[
p(a,b|C)
\]

推导 conditionals。

要求：

\[
|\delta|<\epsilon
\]

---

## 4.6 Incompatible toy test

人工指定 incompatible conditionals。

要求：

\[
\delta_{\text{code}}
\approx
\delta_{\text{manual}}
\]

---

## 4.7 Residual time-dependence test

同一 context 上取：

\[
t_1,t_2,t_3
\]

计算 posterior JSD。

正式结果按：

```text
sigma_low
sigma_mid
sigma_high
overall
```

分桶。

---

## 4.8 Strict reveal decoder test

Teacher-forced 和 sampled 两种模式都验证：

```text
mask_count[t+1] = mask_count[t] - 1
only selected position changes
selected position was MASK
```

---

## 4.9 Full-autograd vs recomputation gradient test

tiny model：

```text
vocab = 10
hidden = 8
seq_len = 4
batch = 2
dropout = 0
```

完整：

\[
L=\lambda\delta^2
\]

得到：

\[
g_{\text{full}}
\]

recompute 得到：

\[
g_{\text{recompute}}
\]

比较：

\[
\epsilon_g
=
\frac{
\|g_{\text{full}}-g_{\text{recompute}}\|_2
}{
\|g_{\text{full}}\|_2+\epsilon
}
\]

目标：

\[
\epsilon_g\lesssim10^{-5}
\]

或至少处于 fp32 浮点误差量级。

🛑 不通过则禁止进入 Swap-SFT。

---

# 5. 阶段 1 — Pretrained Compatibility Baseline

预计：

- 编码：0.5 天
- GPU：<1h

---

## 5.1 Data CPI

语料：

> WikiText-103

每个 evaluation example：

- visible context \(C\)；
- 两个 target masked positions \(i,j\)；
- \(i\neq j\)；
- gold tokens \(a,b\)；
- fixed sigma。

三次 forward：

\[
C
\]

读取：

\[
\log p(a|C),\quad
\log p(b|C)
\]

然后：

\[
C+a
\]

读取：

\[
\log p(b|C,a)
\]

最后：

\[
C+b
\]

读取：

\[
\log p(a|C,b)
\]

---

## 5.2 Evaluation pair selection

evaluation manifest 中：

- \(i\neq j\)；
- \(i,j\) 均位于 target span；
- 二者在 evaluation 初始状态均 masked；
- 如果可用 masked target positions < 2，则跳过该 candidate sample；
- 默认随机选择一对；
- random seed 固定并保存；
- pair distance：

\[
|i-j|
\]

写入 manifest。

---

## 5.3 Frozen evaluation manifest

保存：

```text
sample_id
source_offset
context_tokens
target_span
initial_masked_positions
i
j
a
b
sigma
seq_len
span_len
mask_ratio
revealed_target_count
pair_distance
random_path_seeds
```

推荐：

```text
manifests/regime_a_eval_v1.jsonl
```

后续所有 checkpoint 使用完全相同 manifest。

---

## 5.4 Pretrained metrics

### Compatibility

- CPI_abs；
- CPI_RMS；
- median；
- p90；
- p99；
- signed mean。

### Prediction

- local CE/NLL；
- token accuracy。

### Buckets

- sigma；
- seq length；
- mask ratio；
- revealed target count；
- span length；
- pair distance。

### Effect-size baseline

不拿真实 CPI 与 toy numerical noise 直接比较。

重点记录真实分布：

\[
SD(CPI_{\text{pre}})
\]

用于后续标准化：

\[
\frac{
\Delta CPI
}{
SD(CPI_{\text{pre}})
}
\]

以及 paired standardized effect。

---

# 5A. Manifest Versioning Protocol

Evaluation manifest 文件采用不可变版本号：

```text
regime_a_eval_v1.jsonl
regime_a_eval_v2.jsonl
regime_b_eval_v1.jsonl
```

一旦某个版本进入正式 run：

> **同一文件名 / 版本号下内容不得变化。**

每个 manifest 同时保存 SHA-256：

```text
manifest_sha256
```

每个正式 run config 必须记录：

```text
manifest_filename
manifest_version
manifest_sha256
```

如果发现 manifest bug：

1. 原版本保留；
2. 新建下一个版本；
3. protocol version 递增；
4. 所有使用受影响旧 manifest 的 formal runs 标记 invalid；
5. 重跑受影响条件。

# 6. 阶段 2 — Order Dependence Evaluator + 256 Smoke

预计：

- 编码：0.5–1 天
- GPU：约 1h
- 实际以 smoke 为准

---

## 6.1 Teacher-forced path likelihood

给定：

\[
\pi=(i_1,\dots,i_m)
\]

其中：

\[
m=|\mathcal M_0|
\]

定义：

\[
Q_\pi(x)
=
\sum_{k=1}^{m}
\log
p_\theta(
x_{i_k}
|
C,x_{i_1},\dots,x_{i_{k-1}}
)
\]

每一步：

> policy 决定位置，实际 reveal gold token。

---

## 6.2 OrderGap

\[
OrderGap_{\text{raw}}
=
\max_\pi Q_\pi
-
\min_\pi Q_\pi
\]

\[
OrderGap_{\text{per-token}}
=
\frac{
OrderGap_{\text{raw}}
}{
m
}
\]

以及：

\[
Var_\pi(Q_\pi)
\]

---

## 6.3 Policies

至少：

1. L2R；
2. R2L；
3. fixed random；
4. confidence-first。

random：

> 每个 example 固定 3–5 个 random paths。

---

## 6.4 Fixed vs adaptive confidence path

### Fixed-policy

用 pretrained 生成：

\[
\pi_{\text{conf-pre}}
\]

所有 checkpoint 共享。

这是主 cross-checkpoint comparison。

### Adaptive-policy

每个 checkpoint 自己生成：

\[
\pi_{\text{conf-ckpt}}
\]

这是 behavior / self-preferred path analysis。

两者不混。

---

## 6.5 Tie-breaking

confidence 完全相同时：

> position index 较小者优先。

记录 tie rate。

---

## 6.6 Sampled rollout

主结果：

- greedy；
- \(T=1.0\)。

固定 seeds。

sensitivity-only：

- \(T=0.5\)；
- \(T=1.5\)。

---

## 6.7 256 training smoke

300-step Vanilla 256-seq smoke。

记录：

```text
actual_micro_batch
grad_accum
effective_batch
optimizer_steps_per_second
tokens_per_second
peak_vram_allocated
peak_vram_reserved
```

Stage 3 所有 wall-clock 预算由此更新。

---

# 7. 阶段 3 — Regime A Conditional SFT

## 7.1 Pipeline

正式 task training：

> 统一使用 256-token pipeline。

1024 pipeline 仅作为：

- pretrained reproduction；
- sanity check。

---

## 7.2 WikiText span infilling

target span 初始候选：

\[
10\sim50 \text{ tokens}
\]

最终范围由 Vanilla pilot 在正式 experiment 前冻结。

span 外：

\[
x_t^{context}=x_0^{context}
\]

始终 visible。

span 内：

\[
x_t^{target}
\sim
q_t(x_t^{target}|x_0^{target})
\]

即 partial reveal / partial absorbing corruption。

---

## 7.3 Loss support

task loss 只计算：

> target span 中当前 masked positions。

训练自然覆盖：

```text
[M] [M] C [M]
A   [M] C [M]
[M] B   C [M]
A   B   [M] D
```

而不是只覆盖：

```text
[M] [M] [M] [M]
```

---

# 8. G0.5 — State-Support Gate

目标：

> 排除 CPI 变化只是 partial-state exposure mismatch。

---

## 8.1 不使用 exact pattern low-density hard cut

高维 mask pattern 本身概率低不等于 OOD。

---

## 8.2 State statistics overlap

比较训练和 evaluation 的：

- sigma；
- target mask ratio；
- revealed target count \(K\)；
- span length；
- pair distance；
- partial reveal depth。

先根据 frozen corruption process 得到 \(K\) 的支持范围和经验分布，再在正式 CPI comparison 前冻结 K buckets。

例如仅作为初始候选：

```text
K = 0 / 1 / 2 / 3 / 4+
```

实际 bucket 边界必须在 formal comparison 前冻结。

估计：

\[
P(K=k|t)
\]

检查：

\[
C,\ C+a,\ C+b
\]

对应 state 是否落在训练 distribution 的正常覆盖区域。

---

## 8.3 Sensitivity analysis

报告：

\[
CPI_K
=
E[|\delta|\mid K]
\]

以及：

```text
CPI by mask-ratio bucket
CPI by sigma bucket
CPI by span-length bucket
CPI by pair-distance bucket
```

低覆盖区域作为 sensitivity / limitation，不改变主 estimand。

---

# 9. Regime A Training Protocol

## 9.1 Dropout

所有条件统一：

```text
dropout = 0
```

包括：

- Vanilla；
- PAPL；
- Swap；
- PAPL+Swap。

---

## 9.2 Vanilla pilot

不进入最终方法比较。

目的：

1. 确定总训练预算 \(N\)；
2. 确定稳定 LR；
3. 确定 eval/checkpoint frequency；
4. 确定 span difficulty；
5. 确定 256 pipeline throughput；
6. 确认 G1 可学。

pilot 完成后冻结 formal protocol。

---

## 9.3 Formal checkpoints

正式所有条件：

\[
0.1N,\quad0.5N,\quad N
\]

分别作为：

- early；
- mid；
- late。

---

## 9.4 G1 — Task Learnability

比较：

> pretrained vs Vanilla SFT

至少：

- masked-token NLL；
- token accuracy；
- span exact match；
- sampled infilling quality。

要求：

- non-trivial improvement；
- 超过 trivial baseline；
- 非 near-zero；
- 非 ceiling saturated。

G1 不通过：

🛑 重设计 task，不进入 compatibility interpretation。

---

# 10. G2 — Post-Training Compatibility Dynamics

每个 checkpoint 测：

### Task

- CE/NLL；
- token accuracy；
- exact match。

### Compatibility

- CPI_abs；
- CPI_RMS；
- quantiles；
- sigma buckets。

### Path dependence

- fixed-policy TF OrderGap；
- adaptive-policy TF OrderGap；
- raw；
- per-token。

### Behavior

- sampled rollout。

核心问题：

> task learning 过程中 compatibility 与 path dependence 如何变化？

不预设方向。

---

# 11. 阶段 4 — 第一科学 Hard Stop

## Q1

\[
\Delta CPI
=
CPI_{\text{SFT}}
-
CPI_{\text{pretrained}}
\]

是否稳定偏离 0？

---

## Q2

\[
\Delta OrderGap_{\text{TF}}
\]

是否稳定变化？

---

## Q3

\[
CPI
\leftrightarrow
OrderGap_{\text{TF}}
\]

是否在 checkpoint trajectory 中共同移动？

---

## 11.1 Outcome A — Hard Stop

如果：

\[
|\Delta CPI|\approx0
\]

且：

\[
CPI\not\leftrightarrow OrderGap
\]

在主要 buckets 中也没有稳定结构，

🛑 停止 Swap 主路线。

---

## 11.2 Outcome B — Diagnostic Only

如果：

\[
\Delta CPI\neq0
\]

但：

\[
CPI\not\leftrightarrow OrderGap
\]

则：

> compatibility dynamics 是现象，但暂不支持 mechanism claim。

---

## 11.3 Outcome C — Proceed

如果：

\[
\Delta CPI\neq0
\]

且：

\[
CPI\leftrightarrow OrderGap
\]

稳定联动，

进入：

- PAPL；
- Swap；
- factorial。

---

# 12. 阶段 5 — Simplified PAPL-Style Baseline

## 12.1 Weight

不使用：

\[
p_{\max}
\]

使用：

\[
w_i
\propto
\hat p_\theta(x_{0,i}|x_t)
\]

并 detach。

---

## 12.2 Normalization

\[
\tilde w_i
=
\frac{
w_i
}{
\operatorname{mean}_{masked}(w)+\epsilon
}
\]

任何 clipping 必须在 pilot 阶段冻结。

正式称为：

> **simplified PAPL-style planner-aware weighting**

不声称 strict P-ELBO reproduction。

---

# 13. 阶段 6 — Swap-SFT

## 13.1 Objective

\[
\delta
=
\log p(a|C)
+
\log p(b|C,a)
-
\log p(b|C)
-
\log p(a|C,b)
\]

\[
L_{\text{swap}}
=
\lambda
\frac{1}{N_{\text{pair}}}
\sum_n
\delta_n^2
\]

---

## 13.2 Recomputation

### Pass A

no-grad：

```text
C
C+a
C+b
```

得到：

\[
\operatorname{sg}(\delta)
\]

---

### Pass B

State \(C\)：

\[
2\lambda
\operatorname{sg}(\delta)
[
\log p(a|C)-\log p(b|C)
]
\]

State \(C+a\)：

\[
2\lambda
\operatorname{sg}(\delta)
\log p(b|C,a)
\]

State \(C+b\)：

\[
-2\lambda
\operatorname{sg}(\delta)
\log p(a|C,b)
\]

梯度累积后统一 optimizer step。

---

## 13.3 Compute

普通：

\[
1F+1B
\]

Swap：

\[
6F+3B
\]

初始调度按：

\[
3.5\sim4\times Vanilla
\]

真实结果全部用 smoke 实测覆盖。

---

# 14. λ Calibration

## 14.1 Gradient-norm calibration

不使用单个 batch 决定 λ。

第一阶段：

\[
B_{\text{calib}}=10
\]

在 10 个 representative training batches 上分别计算：

\[
r_b
=
\frac{
\|g_{\text{swap},b}^{(\lambda=1)}\|
}{
\|g_{\text{SFT},b}\|
}
\]

记录：

- median；
- mean；
- IQR；
- CV。

使用：

\[
r_{\text{median}}
=
\operatorname{median}_b(r_b)
\]

作为主要 calibration scale。

### CV fallback

如果 10 batch 后：

\[
CV>0.5
\]

则将：

\[
B_{\text{calib}}=30
\]

重新估计完整 ratio distribution。

如果 30 batch 后仍：

\[
CV>0.5
\]

则：

- 不再继续无限增加 batch；
- 使用 median / IQR 作为主要 calibration statistic；
- 标记：

```text
heavy_tail_lambda_calibration = true
```

并在实验记录中保存完整 \(r_b\) distribution。

目标辅助梯度比例：

\[
r_{\text{target}}
\in
\{0.05,0.1,0.2\}
\]

基于 median ratio：

\[
\lambda_0
=
\frac{
r_{\text{target}}
}{
r_{\text{median}}
}
\]

development sweep：

\[
\{0.3\lambda_0,\lambda_0,3\lambda_0\}
\]

formal run 前冻结最终 λ。

## 14.2 Stability criterion

正式 protocol 默认：

```text
gradient_monitor_interval = 50 optimizer steps
instability_patience = 3 intervals
```

即：

> 每 50 optimizer steps 检查一次 gradient ratio。

如果连续 3 个 monitoring intervals：

\[
\frac{
\|g_{\text{swap}}\|
}{
\|g_{\text{SFT}}\|
}>2
\]

或者任一 interval 出现：

- non-finite loss；
- non-finite gradient；

则该 λ 在 calibration 阶段判为 unstable。

calibration 阶段允许：

\[
\lambda\leftarrow\lambda/2
\]

最多两次。

上述 interval / patience 在 Vanilla pilot 后冻结；若最终 \(N\) 极小导致 150 steps 明显不合适，必须在 formal run 前调整并记录 protocol revision。

正式 confirmatory run：

> 不动态修改 λ、LR 或 optimizer recipe。


# 15. G5 — Mechanism Gate

G5 不使用人为固定百分比阈值，例如：

> “\(L_{\text{swap}}\) 必须下降至少 50%”。

而分成两层。

## 15.1 Mechanism optimization check

检查：

\[
L_{\text{swap,late}}
<
L_{\text{swap,early}}
\]

并报告相对变化：

\[
R_L
=
\frac{
L_{\text{late}}-L_{\text{early}}
}{
|L_{\text{early}}|+\epsilon
}
\]

不预设 50% 等 arbitrary hard threshold。

---

## 15.2 Structural effect check

比较：

\[
\Delta CPI
=
CPI_{\text{Swap}}
-
CPI_{\text{Vanilla}}
\]

以及：

\[
\Delta OG
=
OG_{\text{Swap}}
-
OG_{\text{Vanilla}}
\]

要求：

1. 两个正式 training seeds effect direction 一致；
2. paired effect size 具有非平凡大小；
3. 每个 seed 内的 frozen evaluation samples 上，paired bootstrap evidence 支持相同方向；
4. CPI 与 OrderGap 同时朝预期方向变化。

注意：

> sample-level paired CI 只表示 evaluation-sample uncertainty，不能当作 training-seed-level significance。

如果两个 training seeds 方向相反：

> 标记 `unstable`，G5 不通过。

如果 \(L_{\text{swap}}\) 下降，但 CPI / OrderGap 没有一致 structural effect：

> mechanism claim 弱化，只能保留 optimization / diagnostic observation。

# 16. G6 — Matched Control

## 16.1 Permuted-Coefficient Control

每个真实 sample 正常算：

\[
\delta_i
\]

真实 states 不变：

\[
C_i,\ C_i+a_i,\ C_i+b_i
\]

但 batch 内打乱 coefficient：

\[
\tilde\delta_i
=
\delta_{\pi(i)}
\]

使用：

\[
2\lambda
\tilde\delta_i
\nabla\delta_i
\]

保留：

- states；
- forward count；
- backward count；
- coefficient magnitude distribution；
- wall-clock；

只破坏：

> defect ↔ correct cycle

对应关系。

---

## 16.2 Optional Random-Sign Control

\[
\tilde\delta_i
=
s_i|\delta_i|
\]

其中：

\[
s_i\in\{-1,+1\}
\]

作为二级 matched control。

---

# 17. 阶段 7 — Regime A 2×2

| Condition | Planner alignment | Compatibility reg |
|---|---:|---:|
| Vanilla | ✗ | ✗ |
| PAPL | ✓ | ✗ |
| Swap | ✗ | ✓ |
| PAPL + Swap | ✓ | ✓ |

---

## 17.1 Seeds

第一轮：

\[
4\times2=8\text{ runs}
\]

若：

- effect 大；
- 两 seed 同方向；
- variance 可接受；

再补关键条件第 3 seed。

若两个 seed 方向相反：

> 首先视为 unstable effect，不自动扩 seed 寻找显著性。

---

## 17.2 Equal-budget rule

所有正式条件严格遵守：

> **§3.3B Equal-Budget / Data-Matching Protocol**

正式 Regime A 2×2 中，Vanilla / PAPL / Swap / PAPL+Swap 必须使用：

- 相同 optimizer-step budget：

\[
N
\]

- 相同 effective training batch：

\[
B_{\text{eff}}
\]

- 相同：

```text
unique_training_sequences_seen
```

- 相同：

```text
unique_source_tokens_seen
```

- 相同 analysis checkpoint steps；
- 相同 frozen evaluation manifest。

注意：

> `unique_source_tokens_seen` 与 `model_forward_tokens` 是两个不同概念。

Swap 因为对同一 source example 计算：

\[
C,\quad C+a,\quad C+b
\]

会产生更高的：

```text
model_forward_tokens
```

但不能把这些额外 forward 计为额外：

```text
unique_source_tokens_seen
```

如果由于 repo / DataLoader 的整数约束无法做到完全一致，只允许 §3.3B 已冻结的 engineering tolerance：

\[
\frac{
|T_c-T_{\text{ref}}|
}{
T_{\text{ref}}
}
\le 1\%
\]

超过该 tolerance：

🛑 **不得进入正式主 factorial。**

必须重新设计：

- micro-batch；
- gradient accumulation；
- optimizer-step accounting。

本节不重复完整预算协议；如有任何解释冲突：

> **以 §3.3B 为唯一权威定义。**

# 18. 阶段 8 — Regime B

不预注册：

> Swap 必然有害。

使用 competing hypotheses。

---

## 18.1 Hypothesis B1

如果任务具有真实顺序非对称性：

> 强迫 reveal-order compatibility 可能限制 reasoning / task learning。

---

## 18.2 Hypothesis B2

如果 Swap 主要修复概率模型内部不自洽：

> 即使 decoder 不使用 arbitrary order，Swap 也可能中性或有益。

---

## 18.3 Candidate tasks

- template arithmetic；
- controlled multi-step arithmetic；
- last-letter concatenation；
- CLUTRR-like synthetic reasoning。

优先：

\[
L\le128
\]

---

# 19. Regime B Dataset Freeze

每个候选任务必须记录：

```text
dataset_generator_version
train_size
val_size
test_size
train_seed
val_seed
test_seed
difficulty_parameters
deduplication_rule
metric
trivial_baseline
```

必须确认：

- train / val / test 不共享相同实例；
- 不存在明显 template leakage；
- 不存在答案直接复制等 trivial shortcut。

正式数据生成后保存 manifest。

不得：

> 每次 run 重新随机生成测试集。

---

# 20. Regime B G1

在 pilot 前定义：

\[
G
=
\frac{
M_{\text{SFT}}-M_{\text{pre}}
}{
M_{\text{ceiling}}-M_{\text{pre}}
}
\]

默认：

\[
G\ge0.2
\]

且：

> bootstrap CI lower bound > 0

并要求：

- 超过 trivial baseline；
- 未 ceiling saturated。

不通过：

🛑 reject / redesign task。

不得：

> 根据 pilot 实际数字临时修改 G1 threshold。

---

## 20.1 Regime B protocol artifact

Regime B task-specific G1 与数据生成配置必须写入：

```text
protocol/regime_b_protocol.yaml
```

并在 Regime B pilot 开始前冻结。

该文件必须保存：

```text
regime_b_protocol_version
regime_b_protocol_sha256
```

正式 Regime B run config 同时记录该 hash。

# 21. GSM8K

仅作为 capability probe。

流程：

1. Vanilla SFT；
2. G1；
3. near-zero 则停止。

不能把：

> 170M capacity failure

解释为 compatibility result。

---

# 22. SEDD-Medium

仅做 pretrained diagnostic：

- CPI；
- TF OrderGap；
- CPI ↔ OrderGap relationship。

不能报告：

\[
\Delta CPI_{\text{medium}}
\]

因为没有 medium post-trained checkpoint。

定位：

> cross-scale diagnostic validation

建议放 robustness / appendix。

---

# 23. Final Statistics

主报告至少包含：

### Compatibility

\[
CPI_{\text{abs}}
\]

\[
CPI_{\text{RMS}}
\]

以及 quantiles。

### Path dependence

\[
OrderGap_{\text{raw}}
\]

\[
OrderGap_{\text{per-token}}
\]

### Task

task-specific metrics。

### Uncertainty

- SEM；
- paired bootstrap CI；
- paired standardized effect size。

### Training variability

- per-seed result；
- seed mean；
- seed spread。

### 2-seed reporting rule

第一轮只有 2 个 training seeds 时：

> **不把 pooled sample-level significance 当作 training-seed-level significance。**

主文必须展示：

```text
seed_1
seed_2
seed_mean
```

可以对同一 seed 内的 frozen evaluation samples 做 paired bootstrap，但其 CI 只反映：

> evaluation-sample uncertainty

不代表：

> across-training-seed uncertainty。

如果两个 seed effect direction 相反：

> 直接标记为 `unstable`。

不得：

> 因为方向相反就不断增加 seed 直到得到显著结果。

只有当两个 seed 方向一致、效应量值得继续时，才补第 3 seed 作为 confirmation。

# 24. Scientific Outcome Tree

## Outcome A — strongest

Vanilla：

\[
CPI\uparrow,\quad OrderGap\uparrow
\]

PAPL：

\[
Task\uparrow
\]

但 CPI 改善有限。

Swap：

\[
CPI\downarrow,\quad OrderGap\downarrow
\]

PAPL+Swap：

\[
>PAPL
\]

支持：

> compatibility intervention provides value beyond planner alignment.

---

## Outcome B

Vanilla：

\[
Task\uparrow,\quad CPI\downarrow
\]

说明：

> ordinary post-training attenuates incompatibility。

研究问题变成：

> 什么训练因素决定 compatibility 的方向？

---

## Outcome C

PAPL：

\[
CPI\downarrow,\quad OrderGap\downarrow
\]

且：

\[
PAPL+Swap\approx PAPL
\]

说明：

> planner-aware training may implicitly regularize compatibility。

Swap utility 弱，但机制结论仍有价值。

---

## Outcome D

Swap：

\[
CPI\downarrow,\quad OrderGap\downarrow
\]

但：

\[
Task\approx unchanged
\]

说明：

> compatibility 是可控 structural property，但不是当前 task bottleneck。

---

## Outcome E — Hard Stop

如果：

\[
|\Delta CPI|\approx0
\]

且：

\[
CPI\not\leftrightarrow OrderGap
\]

并且：

- fixed sigma；
- state support；
- sample size；
- buckets；
- measurement correctness；

均已控制，

🛑 停止 Swap 方法路线。

---

# 25. 时间预算

## 25.1 第一目标：3–5 天

完成：

```text
G0
↓
Pretrained CPI
↓
Teacher-forced OrderGap
↓
256 training smoke
↓
Vanilla pilot
↓
Protocol freeze
↓
Formal Vanilla SFT
↓
G1/G2
↓
Stage-4 go/no-go
```

---

## 25.2 第二目标：10–14 天

如果 signal 成立：

- PAPL calibration；
- Swap λ calibration；
- Swap；
- matched control；
- Regime A 2×2。

---

## 25.3 第三目标：2–3 周

如果值得继续：

- Regime B；
- 第 3 seed；
- medium diagnostic；
- NFE；
- figures；
- final statistics。

---

# 26. 风险表

| 风险 | 类型 | 处理 |
|---|---|---|
| batch semantics 错误 | 工程 | G0 runtime printing |
| 256 speed 低于预期 | 性能 | 300-step smoke |
| CPI 混入 timestep | 测量 | fixed sigma |
| residual time dependence | 模型 | JSD + sigma buckets |
| exposure mismatch | 科学 | partial-reveal + G0.5 |
| sampled OrderGap 噪声 | 测量 | teacher-forced primary |
| checkpoint path 变化 | 测量 | fixed/adaptive split |
| PAPL baseline 不忠实 | baseline | GT-token posterior |
| Swap gradient 错 | 工程 | exact gradient toy test |
| dropout mismatch | 优化 | dropout=0 for all |
| λ 不合理 | 优化 | gradient-norm calibration |
| generic aux effect | 因果 | permuted coefficient control |
| extreme finite delta | 数值 | monitor, no silent skip |
| NaN | 稳定性 | formal run fail, no dynamic LR |
| G1 选择偏差 | 科学 | threshold frozen before pilot |
| Regime B leakage | 数据 | manifests + dedup |
| seed 不稳定 | 统计 | report instability |
| medium 不能训练 | 硬件 | pretrained diagnostic only |

---

# 26A. Protocol Version vs Recipe Version

定义：

### protocol_version

表示：

> 实验规则层版本。

例如：

```text
v4.1
v4.2
```

它控制：

- budget rules；
- checkpoint rules；
- failure handling；
- statistics；
- manifest rules；
- evaluation definitions。

### recipe_version

表示：

> 某个 condition 的具体训练配置版本。

例如：

```text
vanilla_v1
papl_v1
swap_v2
papl_swap_v2
```

一次 protocol revision：

> 不一定使所有 recipe 失效。

必须根据修改影响范围判断哪些 conditions 受影响。

每个 formal run 必须同时记录：

```text
protocol_version
recipe_version
```

# 27. Reproducibility Checklist

每个正式 run 保存：

```text
git_commit
protocol_version
recipe_version
full_config
seed
model_seed
data_order_seed
corruption_seed
eval_manifest_seed
sampling_seed
dataloader_worker_seed_rule
actual_micro_batch
grad_accum
effective_batch
seq_len
tokens_per_optimizer_step
unique_training_sequences_seen
unique_source_tokens_seen
model_forward_tokens
optimizer_steps_per_second
tokens_per_second
peak_vram_allocated
peak_vram_reserved
wall_clock
checkpoint_steps
eval_manifest_version
eval_manifest_sha256
python_version
torch_version
cuda_runtime_version
gpu_driver_version
numpy_version
transformers_version
```

---

# 28. 推荐目录结构

```text
project/
├── configs/
│   ├── vanilla_256.yaml
│   ├── papl_256.yaml
│   ├── swap_256.yaml
│   └── papl_swap_256.yaml
│
├── protocol/
│   ├── experiment_protocol.md       # 冻结协议
│   ├── regime_a_protocol.yaml        # Regime A 具体配置
│   └── regime_b_protocol.yaml        # Regime B 具体配置
│
├── compatibility/
│   ├── posterior.py
│   ├── cpi.py
│   ├── order_gap.py
│   ├── paths.py
│   └── tests/
│       ├── test_posterior.py
│       ├── test_swap_delta.py
│       ├── test_strict_reveal.py
│       └── test_swap_gradient.py
│
├── task_data/
│   ├── infilling.py
│   ├── corruption.py
│   └── regime_b.py
│
├── evaluation/
│   ├── build_manifest.py
│   ├── eval_cpi.py
│   ├── eval_order_gap.py
│   ├── eval_task.py
│   └── bootstrap.py
│
├── training/
│   ├── vanilla.py
│   ├── papl.py
│   ├── swap.py
│   └── controls.py
│
├── manifests/
│   ├── regime_a_eval_v1.jsonl
│   └── regime_b_eval_v1.jsonl
│
├── results/
│   ├── pretrained/
│   ├── vanilla/
│   ├── papl/
│   ├── swap/
│   └── controls/
│
└── scripts/
    ├── smoke_256.sh
    ├── run_regime_a.sh
    └── run_matrix.sh
```

---

# 28A. Canonical File Maintenance Rule

最终只维护两个主文件：

```text
post_training_reveal_order_compatibility_local_plan_v4.1_FROZEN.md
protocol/experiment_protocol.md
```

其中：

- 主计划：研究结构、阶段、科学决策树；
- `experiment_protocol.md`：正式执行规则的唯一权威来源。

不再并行维护内容重复的：

```text
experiment_protocol_v4_FROZEN.md
experiment_protocol_v4.1_FROZEN.md
```

协议版本通过文件内部：

```text
protocol_version
```

管理，而不是靠复制多个同内容文件维护。

# 29. Final Execution Order

```text
G0
├── batch semantics
├── posterior normalization
├── fixed-sigma CPI
├── toy compatible/incompatible
├── residual-time JSD
├── strict reveal decoder
└── gradient equivalence
        ↓
Pretrained CPI
        ↓
Frozen eval manifest
        ↓
Teacher-forced OrderGap
        ↓
256 training smoke
        ↓
Vanilla pilot
        ↓
🔒 Formal protocol freeze
        ↓
Partial-reveal formal Vanilla SFT
        ↓
G0.5 state-support
        ↓
G1
        ↓
G2
        ↓
Stage-4 scientific gate
      /        \
 HARD STOP    SIGNAL
                ↓
          PAPL calibration
                ↓
          λ calibration
                ↓
             Swap-SFT
                ↓
         Matched controls
                ↓
          Regime A 2×2
                ↓
          Regime B pilot
                ↓
          Regime B G1
                ↓
          Regime B 2×2
                ↓
      seeds / NFE / medium
```

---

# 30. Day 0 / Day 1 Immediate Tasks

- [ ] 打印实际 batch semantics；
- [ ] 实现 `normalized_clean_posterior()`；
- [ ] 写 compatible / incompatible \(\delta\) toy tests；
- [ ] 实现 fixed-sigma CPI；
- [ ] 写 residual-time JSD sanity test；
- [ ] 实现 teacher-forced strict reveal；
- [ ] 写 full-autograd vs recompute gradient test；
- [ ] 冻结 Regime A eval manifest；
- [ ] 跑 pretrained CPI；
- [ ] 跑 pretrained TF OrderGap；
- [ ] 跑 300-step 256 Vanilla smoke；
- [ ] 跑 Vanilla pilot；
- [ ] 冻结 \(N\)、checkpoint fractions、LR、span range、G1；
- [ ] 保存 `experiment_protocol.md`；
- [ ] 开始 formal Vanilla SFT。

在这些完成之前：

> **不实现正式 PAPL，不实现正式 Swap，不跑 2×2。**

---

# 31. 最终原则

只有当实验已经支持：

\[
\boxed{
\text{Post-training changes reveal-order compatibility}
}
\]

并且最好进一步看到：

\[
\boxed{
\text{Compatibility tracks reveal-order path dependence}
}
\]

才进入 Swap 方法路线。

整个项目的控制原则压缩为：

\[
\boxed{
\text{Same states}
+
\text{same sigma}
+
\text{same paths}
+
\text{same compute}
+
\text{same training budget}
}
\]

每一次正式比较尽量只改变一个变量。

---

# 32. 一句话版本

> **先证明 post-training 真的改变 reveal-order compatibility，而且这种变化真的与 path dependence 有关；只有这两个事实成立，才值得做 Swap-SFT。**
