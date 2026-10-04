# 最终汇报大纲（8 页，中文，总时长 5–8 分钟）

> 汇报标题：《离散扩散语言模型的监督微调、强化学习与 Reveal-Order Compatibility 研究》
> 每个指标第一次出现时给中文解释 + 英文缩写；公式保留；不出现大段英文。

---

## 第 1 页 · 研究背景与问题定义（约 40 秒）

**核心 bullet**

- 离散扩散语言模型（Diffusion Language Model, dLLM）以迭代去噪方式生成文本，token
  揭示顺序（Reveal Order）不再固定为从左到右
- 揭示顺序的灵活性带来一个新问题：模型对不同揭示顺序的条件预测是否一致？
- 研究问题：**后训练过程会如何改变 dLLM 对 token reveal order 的敏感性？**
- 项目位置：SEDD-small（~170M）、WikiText103 partial-reveal span-infilling 任务

**推荐图/表**：Research Story 总流程图（demo_assets/figures/research_story_figure.svg）

**一句话结论**：后训练不只是改变任务表现，还会系统性改变模型对生成顺序的依赖结构。

---

## 第 2 页 · SEDD 与离散扩散语言模型原理（约 50 秒）

**核心 bullet**

- 前向 corruption：按噪声 schedule σ 把 span 内 token 逐位置随机替换为 [MASK]
  （absorbing 离散扩散）
- 反向生成（Reverse Diffusion）：从部分 Mask 状态出发，128 步逐位置决定
  "保持 MASK 或跳到干净 token"——与自回归模型（AR）严格从左到右生成有本质区别
  （模型一次 forward 可同时为多个 MASK 位置给出预测，但 sampler 在多个 reverse
  steps 中随机决定哪些位置被 reveal：parallel prediction + iterative revealing）
- 模型输出离散 score（Discrete Score）：对每个位置、每个词表 token 打分的
  [B, L, 50258] log-score；采样路径用 staggered score（跨时间步修正后的离散 score）
  乘以 transition kernel 得到转移权重 w
- 显存约束下的关键选择：128 reverse steps（vs 1024）、chunk 化 rollout、
  gradient checkpointing 沿用

**推荐图/表**：Demo Tab 2 的 trajectory 关键帧（MASK → 逐步揭示）

**一句话结论**：dLLM 的生成是"从噪声中逐步恢复"，顺序灵活性是它的结构特性而不是缺陷。

---

## 第 3 页 · 项目整体技术路线（约 40 秒）

**核心 bullet**

- 路线：预训练基线 → Partial-Reveal SFT → Reveal-Order Compatibility 分析
  → RL-1 强化学习探索 → K=1 vs K=4 诊断实验 → 交互式 Demo
- 评估体系：frozen manifest（500 样本，SHA 冻结）+ CPI / OrderGap / G1 三件套 +
  formal64 任务指标（NLL / sampled / greedy，RAW 主口径、EMA 辅助口径）
- 方法论：全链路预注册 gate（RL-G0 初始化验证、25/25 单测、128-vs-1024 gate、
  LR probe、protocol v1.0 freeze）

**推荐图/表**：技术路线时间线（本页）

**一句话结论**：每个阶段的数值结论都有可复现的评估协议与可追溯的 provenance。

---

## 第 4 页 · Partial-Reveal SFT：方法与结果（约 50 秒）

**核心 bullet**

- 任务设计：partial-reveal span-infilling——span 内只有部分 token 初始可见，其余
  为 [MASK]，模型重建全部 MASK 位置
- 训练：SEDD-small + WikiText103，正式实验两 seed **每个训练 10200 steps**，在
  1020 / 5100 / 10200 等 checkpoint 上跟踪任务性能与 compatibility（2500 只是
  P1 early-dynamics 的观测点，不是正式训练长度）；checkpoint_10200 为 RL 初始化点
- 结果：SFT 后 CPI_abs 0.3301 → 0.2871（−13%）、OrderGap 10.2246 → 8.7475（−14%），
  同时任务指标（masked NLL、token acc）明显改善
- P1 早期动力学：CPI 衰减集中在 warmup 后期（250–750 窗口）的连续过程，非瞬时阶跃

**推荐图/表**：三阶段 CPI / OrderGap 柱线图 + P1 dense 曲线

**一句话结论**：Vanilla SFT 在提升任务能力的同时，系统性降低了模型的顺序敏感性。

---

## 第 5 页 · Reveal-Order Compatibility：CPI 与 OrderGap（约 60 秒）

**核心 bullet**

- 局部指标 CPI（局部两-token reveal-order 不一致程度）：
  δ = log p(a|C) + log p(b|C,a) − log p(b|C) − log p(a|C,b)，CPI = E|δ|
- 全局指标 OrderGap（完整 reveal path 的全局顺序敏感性）：
  OrderGap = max_π Q_π(x) − min_π Q_π(x)
- 两个度量互补：CPI 看单对 token 的条件不兼容性，OrderGap 看整条路径的概率差异
- 为什么重要：如果模型对"先揭示谁"很敏感，采样路径的选择会实质影响生成结果

**推荐图/表**：Demo Tab 3 的 pair 四项分解 + 三阶段对比卡片

**一句话结论**：CPI 与 OrderGap 共同刻画了"生成顺序依赖"这个 AR 模型不具备的维度。

---

## 第 6 页 · RL 探索与 K-ablation（约 60 秒）

**核心 bullet**

- RL-1 设计：pure on-policy REINFORCE、128 步 analytic rollout、G=4 组内相对优势
  （Group-relative Advantage）、M0-only exact-token reward、safe-region/truncated PG
  （σ≥0.05 才采样 timestep，91.9% 支撑、8.1% 截断）
- LR probe（1e-6 / 3e-6 / 1e-5 × 150 步）→ formal LR=3e-6；500 步 formal run：
  任务指标近似平稳，CPI 0.2871 → 0.2852、OrderGap 8.7475 → 8.7442（CI 含 0）
- K=1 vs K=4 匹配诊断（j_rng 隔离，100 步）：两组 reward / drift / zvg 几乎相同 →
  在当前设置、单 seed、100-step matched diagnostic 下，提高 K 没有带来一致的
  task-learning improvement
- 诚实结论：目前**没有证据表明**单纯提高 timestep sampling density 是主要瓶颈；
  继续简单增加 K 不是最有价值的下一步，后续更值得检查 reward 与 objective-level
  credit assignment

**推荐图/表**：RL task 三指标曲线（step 0–500）+ K-ablation 对比表

**一句话结论**：短程 RL 没有带来额外结构变化，且这一"无变化"是受控实验测出来的；
不把 null result 外推成"K 完全不重要"或"RL 天然不会改变 compatibility"。

---

## 第 7 页 · 工程问题、调试与经验总结（约 50 秒）

**核心 bullet**

- WSL2/WDDM dxgkrnl 显存记账损坏：EOVERFLOW 小分配 OOM，只有 Windows 完整重启能
  重置；expandable_segments 仅作 allocator 工程配置
- 8GB 显存下的 RL：f32 one-hot、chunk=2、phase 间 empty_cache、K=4 sequential backward
  （峰值与 K=1 完全相同）
- 评估 provenance 教训：早期评估代码未进 git → 不可 bit-reproduce；Demo 全链改为
  current-code harmonized 复算
- 短 horizon sampled 指标分辨率不足 → greedy 作为 deterministic 补充指标

**推荐图/表**：deviation record + acceptance 检查清单摘要

**一句话结论**：研究可信度的一半来自工程纪律（gate、checksum、provenance、对拍）。

---

## 第 8 页 · Demo、结论与后续工作（约 50 秒）

**核心 bullet**

- 交互式 Demo（无 GPU、全预计算）：生成对比 / 反向扩散轨迹 / Reveal Order 一致性
  分析 / 训练与研究结果四个 Tab
- 统一结论：
  "实验表明，Vanilla SFT 会明显降低 masked diffusion language model 的
  reveal-order sensitivity，表现为 CPI 和 OrderGap 同时下降。随后进行的
  short-horizon RL-1 在任务学习信号较弱的情况下基本保持了 SFT 后形成的
  compatibility 结构。进一步的 K=1 与 K=4 匹配诊断实验没有发现单纯提高 timestep
  sampling density 能带来一致的任务学习改善，因此后续更值得关注 objective-level
  credit assignment，而不是继续简单增加 K。"
- Future Work：① 更强的 RL credit assignment（group-relative / PPO-style
  per-position objective）② 多 seed + cross-task 验证 ③ 更大模型与更多算力

**推荐图/表**：Demo 首页截图 + 三阶段结论卡片

**一句话结论**：这是一个"结构发现明确、RL 边界诚实、方法链路完整"的可复现研究。

---

## 时间分配与节奏

| 页 | 时间 | 重点 |
|---|---|---|
| 1 | 40s | 问题动机 |
| 2 | 50s | 原理直觉（非公式堆砌） |
| 3 | 40s | 路线可信度 |
| 4 | 50s | SFT 结果 |
| 5 | 60s | CPI / OrderGap 定义 |
| 6 | 60s | RL 诚实结论 + K-ablation |
| 7 | 50s | 工程亮点 |
| 8 | 50s | 收束 + Demo 引导 |

合计约 6.7 分钟，留 1–2 分钟余量，总 5–8 分钟。
