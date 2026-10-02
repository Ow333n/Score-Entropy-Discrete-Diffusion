# 文献核查 — vanilla downstream SFT 前后 reveal-order compatibility 变化（v4.2 持续任务）

> 核查问题：是否已有工作**直接研究 vanilla downstream SFT 前后的 reveal-order
> compatibility / CPI / OrderGap 变化**（pretrained → SFT 的 paired 对比）？
> 日期：2026-10-02（首轮；后续持续更新）。

## 结论（首轮）

**未发现直接测量 "vanilla SFT 前后 ΔCPI/ΔOrderGap" 的工作。** 与我们的测量对象最接近的
工作分三类：① 同一数学量（local curl = 我们的 δ_swap）的推理时诊断；② 在已微调模型上的
单点快照测量；③ MLM 条件不兼容的存在性理论。三者的交集（吸收扩散 LM + swap/curl 指标 +
post-training 前后 paired 对比）目前为空。我们的发现暂以
**"potentially novel empirical observation"** 表述。

## 逐条记录

### 1. Path-Dependent Denoising: A Non-Conservative Field Perspective on Order Collapse in Diffusion LMs
- Kim, arXiv:2605.09303（2026-05）
- **与我们的关系（最密切）**：定义 local denoising circulation (curl) =
  同一对未解析位置两种揭示顺序的 pseudo-joint 的 log 比——**这正是我们的 δ_swap**。
  Theorem 1（curl = 交换顺序的 log-density 比）、Proposition 1（order consistency ⇔
  每 elementary square curl-free）、Theorem 3（**Bayes 最优 uniform masking 下 curl=0，
  非零 curl 来源于 finite capacity / incomplete masking coverage / imperfect
  optimization / calibration mismatch，而非语言结构本身**）、Theorem 4（零 curl 下
  parallel 更新仍有 TC 代价）。
- 定位：**inference-only diagnostics**；作者自述 limitation 包括"缺乏跨 DLM 家族的
  广泛实证验证"。
- **Gap**：无任何 post-training 前后的 Δcurl/ΔCPI 测量。我们的 v4.1 发现（SFT 降低
  CPI）可按其 Theorem 3 解释为 SFT 改善 masked-prediction 的优化/校准 → curl 下降——
  这是 v4.2 机制叙事的候选来源（消融阶段再验证）。

### 2. Decoding in Order-Agnostic Language Models: Chain-Rule Deviation and Uniform Spreading
- arXiv:2606.00997（2026-06）
- 在 **LLaDA-2.1-mini**（已指令微调）上测量不同揭示顺序的 pathwise log-likelihood
  变化 ~0.5 nats/token——**单点快照**，无 pretrained→SFT 的 before/after 对比。
- Gap：不回答"post-training 是否改变 compatibility"。

### 3. Mixing Times of Glauber Dynamics on Masked Language Models
- Sana, Wolf, Mehta, Shah, Shaikh, Goodman, Levine, arXiv:2605.16378（2026-05）
- MLM conditionals **内在**不对应一致 joint（rectangle test 证明 + 现代 MLM 实证）；
  Glauber 动力学的 mixing time / 低温亚稳态理论。
- Gap：存在性理论，无 post-training 动力学。

### 4. Inconsistencies in Masked Language Models
- Young, Chen, You, arXiv:2301.00068
- 不同 masking pattern 的分布不能来自同一 joint（BERT→UL2-20B 普遍存在）；提出推理时
  "Ensemble of Conditionals"。
- Gap：同 3，无 post-training 前后对比。

### 5. On the Consistent Recovery of Joint Distributions from Conditionals
- Majid et al., AISTATS 2025（proceedings.mlr.press/v258/majid25a.html）
- path/swap consistency 的定义与理论；BERT 上的快照测量（E_C ≈ 0.09-0.48）。
- Gap：定义来源（我们引用），无 DLM、无 fine-tuning 对比。

### 6. 顺序感知 post-training 方法（非直接相关但需区分）
- **TRIMS**（arXiv:2604.00666）：trajectory-aware masking SFT——关注"教什么路径"，
  不测量 vanilla SFT 对 compatibility 的副作用。
- **Scheduling Thoughts / SAS**（ICML 2026）：学习揭示顺序策略——顺序策略侧，不测
  conditional family 的变化。
- **DTM**（ICML 2026 slides 65069）：reward 下 local unmasking posterior matching——
  RL 后训练方法，非 vanilla SFT 动力学测量。
- 自洽性理论（arXiv:2605.00161）：diffusion consistency loss 的唯一最小化解理论——
  理论分析，无 SFT 实证对比。

## 对我们定位的含义

1. **数学对象非我们首创**：δ_swap 就是 Path-Dependent Denoising 的 local curl
   （及 Majid 的 swap consistency）。论文必须如此引用，不能声称发明指标。
2. **测量对象（post-training 前后的 Δ）目前空白**：上述工作要么是推理时诊断、
   要么是快照、要么是存在性理论。v4.1 的 paired before/after 测量 + 两 seed 稳健性
   是这个 gap 的第一次（据本核查）实证填补。
3. **机制候选已有理论基础**：Theorem 3（imperfect optimization / calibration → curl）
   为"为什么 vanilla SFT 降低 CPI"提供了可检验的假设，v4.2 机制消融可据此设计。
4. **风险**：这些工作发表时间均晚于 2026 年中，与我们的实验并行；论文提交前必须再查
   一轮（v4.2 §6 持续任务），尤其关注 Path-Dependent Denoising 的后续实证扩展。
