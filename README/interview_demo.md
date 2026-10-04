# 面试演示文档（中文版，Interview Demo Scripts）

> 面向中国面试官。启动 Demo：`cd Score-Entropy-Discrete-Diffusion && .venv/bin/python demo/app.py`
> → http://127.0.0.1:7860（无 GPU 要求，全预计算模式）。

## 一、30 秒项目介绍（口语版）

"这个项目基于 SEDD 做离散扩散语言模型的后训练。我先实现了 partial-reveal SFT，然后
重点研究 dLLM 一个特有的问题：不同 token reveal order 会不会导致不一致的概率判断。
我用 CPI 和 OrderGap 分别衡量局部和全局的顺序敏感性，发现 Vanilla SFT 会明显降低
这两个指标。之后我实现了 reverse-diffusion RL；当前 RL 的 task signal 较弱，
compatibility 基本保持不变，我又通过 K=1 和 K=4 的控制实验进一步分析了它的瓶颈。
最后整个过程做成了交互式 Demo。"

## 二、2 分钟 Demo 讲稿（顺序：Tab 2 → Tab 3 → Tab 4）

**Tab 2（反向扩散轨迹）：**

"我先展示一下 dLLM 实际是怎么生成文本的。这里是真实的 SEDD reverse diffusion
trajectory。模型不是像 GPT 一样严格从左往右生成，而是从包含 MASK 的状态开始，在多个
reverse step 中逐渐恢复 token。"

（拖动关键帧 slider：指出绿色高亮的"新揭示 token"与灰色 [MASK]；提到模型一次 forward
可以同时给多个 MASK 位置预测，但采样器是多步逐位置揭示，即 parallel prediction +
iterative revealing。）

**Tab 3（Reveal Order 一致性分析）：**

"因为 token 的揭示顺序不是固定的，所以我研究了一个问题：如果先揭示 A 再揭示 B，和先 B
再 A，模型自己的概率判断是否一致。这里展示的是同一对 token 在两种顺序下的四项 log
概率：log p(a|C)、log p(b|C,a)、log p(b|C)、log p(a|C,b)。δ 是两种顺序的联合概率差，
|δ| 越接近 0，说明交换揭示顺序后模型的条件概率越一致。下面两张卡片是聚合指标：
CPI 衡量局部两-token 的不一致程度，OrderGap 衡量同一序列在不同完整 reveal path 下的
全局概率差异。"

**Tab 4（训练与研究结果）：**

"我把这个现象沿着 Pretrained、SFT、RL 三个阶段进行了跟踪。SFT 之后 CPI 从 0.3301 降到
0.2871，OrderGap 从 10.2246 降到 8.7475，说明顺序敏感性明显下降。后续 500-step RL 的
task improvement 比较弱，同时 CPI 和 OrderGap 也基本不变，所以目前更准确的结论是：
在当前实验设置和训练 horizon 下，我们观察到的主要 compatibility restructuring 发生在
SFT 阶段，而当前 weak-learning RL 基本保持了 SFT 后的结构。"

## 三、5 分钟技术讲稿（6 模块，口语化）

**模块 1：任务背景——dLLM 与 SEDD（约 50 秒）**

"离散扩散语言模型和自回归模型最本质的区别在生成方式：GPT 严格从左往右、一个 token
一个 token 地生成；dLLM 是从部分 Mask 的状态出发，通过多步反向扩散逐步恢复 token。
在 absorbing discrete diffusion 里，前向过程会按噪声 schedule 逐步把 token 替换成
MASK，反向过程就是学这个逆。SEDD 模型输出的是离散 score 的 log 表示——它不是普通
AR logits，更接近两个离散状态之间的概率比 s_θ(x,y,t) ≈ p_t(y)/p_t(x)，用来比较
MASK 位置上不同候选 token 的相对概率。模型一次 forward 可以同时为多个 MASK 位置给出
预测，但实际 sampler 会在多个 reverse steps 中随机决定哪些位置在当前 step 被 reveal，
所以是 parallel prediction + iterative revealing，不是一次性把所有 token 定死。"

**模块 2：Partial-Reveal SFT（约 40 秒）**

"SFT 采用 partial-reveal span-infilling：目标 span 内部分 token 可见、部分被 Mask，
loss 只计算当前被 Mask 的目标位置——注意，把 corruption 限制在 span 内是本项目的
任务设计，不是 SEDD 的一般原理。正式实验跑了两个 seed，每个训练 10200 steps，并在
1020 / 5100 / 10200 等多个 checkpoint 上跟踪任务性能和 compatibility。两个 seed 都
复现了 CPI 与 OrderGap 的下降。"

**模块 3：Reveal Order 问题与两个指标（约 70 秒）**

"dLLM 的揭示顺序不固定，这引出一个问题：先揭示 A 再揭示 B，和先 B 再 A，模型自评的
条件概率是否一致。局部指标 CPI 比较两种顺序：δ = log p(a|C) + log p(b|C,a) −
log p(b|C) − log p(a|C,b)，CPI = E|δ|。全局指标 OrderGap 比较同一序列在不同完整
reveal path 下的最大与最小路径似然差。两个指标互补：局部一致不保证全局一致。"

**模块 4：SFT 核心结果（约 60 秒）**

"这是项目最干净的结论：Pretrained 上 CPI_abs = 0.3301、OrderGap = 10.2246；SFT 之后
降到 0.2871 和 8.7475，下降约 13% 和 14%，两 seed 稳定复现。P1 early-dynamics 还观察
到一个有意思的细节：全局 OrderGap 在 step 50 就已经小幅下降，而局部 CPI 的明显下降
出现在 warmup 后期——局部和全局结构变化并不完全同步。这是观察加解释方向，不是证明。"

**模块 5：RL-1 与 K-ablation（约 90 秒）**

"之后我实现了基于 SEDD reverse process 的 RL pipeline：128 步 analytic rollout，
G=4 组内相对优势，M0-only exact-token reward。M0 就是 rollout 开始时真正被 Mask 的
位置集合，只在 M0 上算 exact-match——初始已经可见的 token 相当于答案直接给了模型，
算进去会产生'免费正确率'，稀释真正的重建信号。训练用 safe-region truncated policy
gradient：σ≥0.05 的 step 才进入 timestep 采样，σ<0.05 的尾段保留完整 rollout 但不
进 PG 目标。500 步 formal run 的结果：task 指标近似平稳，compatibility 无可检测变化。
针对'是不是 K=1 的 timestep estimator 太稀疏'，我做了 K=1 vs K=4 的匹配诊断——用
独立 RNG 保证两组 rollout 随机流逐位相同。结论是：在当前设置、单 seed、100 步下，
提高 K 没有带来一致的 task-learning improvement，所以目前没有证据表明 timestep
sampling density 是主要瓶颈，继续简单加 K 不是最有价值的下一步。"

**模块 6：Limitations、Future Work 与 Demo（约 50 秒）**

"诚实的边界：单训练 seed，所有统计只表述 sample-level 不确定性；exact-token reward
对语义等价输出会判 0，这个局限我在 Demo 里用真实案例标注了。后续三件事：RL-3 的
per-position PPO-style objective 直接针对 credit assignment；多 seed + cross-task
验证 SFT attenuation 的普遍性；更大模型看现象是否随规模保持。最后，整个研究——
生成对比、反向扩散轨迹、Reveal Order 一致性、三阶段结果——我做成了一个无 GPU 依赖的
交互式 Demo，所有数据都是正式 checkpoint 的真实 inference 导出，可以随时演示。"

## 四、指标解释（Demo 内图注口径）

| 指标 | 中文解释 |
|---|---|
| CPI | 局部两-token reveal-order 不一致程度：E\|δ\|，δ = log p(a\|C)+log p(b\|C,a)−log p(b\|C)−log p(a\|C,b) |
| OrderGap | 完整 reveal path 的全局顺序敏感性：max_π Q_π − min_π Q_π |
| M0 精确重建率 | rollout 初始 Mask 位置上生成 token 与标准答案的 exact-match 比例（M0-only exact-token reward） |
| Formal64 NLL | 64 样本正式评估集的 masked NLL（冻结 corruption realization） |
| 局部 Mask Token CE | compatibility 样本口径的逐样本 masked CE（与 Formal64 NLL 的评估样本集不同） |
| sampled64 / greedy | 64 样本正式评估集上的采样 / 贪心（argmax）重建 reward |

## 五、Limitations 与 Provenance（中文统一口径）

- 单训练 seed：所有统计只表述 sample-level uncertainty，不声称 seed-level significance。
- 历史 early-eval 结果由当时未提交的评估代码生成，无法 bit-level 复现；Demo 主链统一
  使用 current-code harmonized 复算值，历史记录保留在
  protocol/errata_v4.2_eval_provenance.md，聚合差异很小、不改变定性结论。
- exact-token reward 对语义等价输出（article→story）判 0，已用真实案例在 Tab 1 标注。
- Demo 覆盖 15 个固定样本（9 代表性 + 6 案例研究）；轨迹展示 ~10 个关键帧而非全 128 帧。
