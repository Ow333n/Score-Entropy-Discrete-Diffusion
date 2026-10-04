# 面试演示文档（中文版，Interview Demo Scripts）

> 面向中国面试官。启动 Demo：`cd Score-Entropy-Discrete-Diffusion && .venv/bin/python demo/app.py`
> → http://127.0.0.1:7860（无 GPU 要求，全预计算模式）。

## 一、30 秒项目介绍（口语版）

"这个项目是基于 SEDD 的离散扩散语言模型后训练研究。我先完成了 partial-reveal SFT，
然后重点研究一个问题：dLLM 的 token 可以按不同顺序被揭示，那不同 reveal order 下模型
的条件概率是否一致。我用 CPI 和 OrderGap 分别衡量局部和全局的顺序敏感性，发现 Vanilla
SFT 会明显降低这两个指标。之后我又实现了基于 reverse diffusion rollout 的 RL-1，但当前
短程 RL 的任务提升比较弱，compatibility 也基本保持不变。最后我进一步做了 K=1 和 K=4 的
诊断实验，并把整个生成轨迹和研究结果做成了一个可交互 Demo。"

## 二、2 分钟 Demo 讲稿（顺序：Tab 2 → Tab 3 → Tab 4）

**Tab 2（反向扩散轨迹）：**

"我先展示一下 dLLM 实际是怎么生成文本的。这里是真实的 SEDD reverse diffusion
trajectory。模型不是像 GPT 一样严格从左往右生成，而是从包含 MASK 的状态开始，在多个
reverse step 中逐渐恢复 token。"

（拖动关键帧 slider，指出绿色高亮的"新揭示 token"与灰色 [MASK]，提到 σ 随 step 下降。）

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
task improvement 比较弱，同时 CPI 和 OrderGap 也基本不变，所以目前更准确的结论是：主要
结构变化发生在 SFT 阶段，而当前 weak-learning RL 基本保持了 SFT 后的结构。"

## 三、5 分钟技术讲稿（中文口语，公式保留）

**1. 什么是 dLLM。**
离散扩散语言模型把文本生成建模成一个离散去噪过程：前向过程按噪声 schedule σ 把 token
逐位置替换成 [MASK]（absorbing 离散扩散），反向过程从 Mask 状态出发，一步步把 token
"揭示"出来。

**2. 和 AR 的区别。**
自回归模型严格从左到右、一个 token 一个 token 地生成；dLLM 每次反向 step 同时更新多个
位置，token 揭示顺序不固定——这也是我这个项目研究的起点。

**3–4. Forward corruption 与 Reverse Diffusion。**
前向 corruption 只发生在目标 span 内：每个位置以概率 1−e^(−σ) 被替换为 MASK。反向
扩散就是学习这个过程的逆：给定当前状态 x_t 和 σ，输出每个位置跳到每个干净 token 的
转移权重，采 128 步从 σ₀ 回到 σ≈0，最后一步用 denoiser 收尾。

**5. 离散 score 是什么。**
模型输出的是 [B, L, 50258] 的 log-score——对每个位置、每个词表 token 的打分。采样时
exp 成真正的 score，再和 transition kernel 相乘得到转移权重 w；w 归一化后就是该位置的
categorical 策略分布。

**6. 为什么会出现 Reveal Order。**
因为反向过程每一步多个位置可以同时揭示，同一个目标序列可以由很多条不同的揭示顺序
（reveal path）生成出来。如果模型是完美的一致性条件分布，先揭示谁不应该影响联合概率。

**7–8. CPI 与 OrderGap。**
CPI 是局部指标：取两个 MASK 位置 a、b，比较"先 a 后 b"和"先 b 后 a"两种顺序下模型
自评的条件概率，δ = log p(a|C) + log p(b|C,a) − log p(b|C) − log p(a|C,b)，
CPI = E|δ|。OrderGap 是全局指标：同一序列在不同完整 reveal path 下的最大-最小似然差。
两个指标一起刻画模型的顺序敏感性。

**9. Partial-Reveal SFT。**
SFT 任务设计成 partial-reveal span-infilling：span 内部分 token 初始可见，模型要重建
全部 MASK 位置。训练两 seed、各 2500 步。结果是任务指标明显改善，同时 CPI 从 0.3301
降到 0.2871、OrderGap 从 10.2246 降到 8.7475——顺序敏感性被系统性降低了。

**10–11. RL-1 与 M0 reward。**
RL-1 是 pure on-policy REINFORCE：128 步 analytic rollout（no_grad），每条轨迹从
safe-region（σ≥0.05）采样 K 个 timestep 做可微重算，advantage 用 G=4 组内相对优势。
reward 用 M0 口径：只统计初始就被 Mask 的位置上的 exact-token 重建率——因为模型从来没
被要求重建初始就可见的 token，把它们计入会稀释信号。

**12–13. K=1 与 K=4 诊断。**
K=1 每条轨迹只采样一个 timestep 做策略梯度，可能带来高方差。我用独立 j_rng 做了 K=1
vs K=4 的匹配诊断：两组 rollout 随机流逐位相同，唯一变量是 K。结果是 100 步内两组的
reward、参数漂移、zvg、compatibility 几乎相同——单纯提高 timestep sampling density
不能改善 task learning。

**14. 主要结果。**
一句话：Vanilla SFT 明显降低 reveal-order sensitivity；short-horizon RL-1 在任务学习
信号较弱的情况下基本保持了 SFT 后的 compatibility 结构；K-ablation 说明瓶颈不在 K。

**15. Limitations。**
单训练 seed；语义等价输出被 exact-token reward 判 0（真实案例已标注）；8GB WDDM 环境
限制了规模；RL-1 没有产生明确 task-learning signal。

**16. Future Work。**
① 更强的 RL credit assignment（group-relative / PPO-style per-position objective）；
② 多 seed + cross-task 验证 SFT attenuation 的普遍性；③ 更大模型与更多算力。

## 四、指标解释（Demo 内图注口径）

| 指标 | 中文解释 |
|---|---|
| CPI | 局部两-token reveal-order 不一致程度：E\|δ\|，δ = log p(a\|C)+log p(b\|C,a)−log p(b\|C)−log p(a\|C,b) |
| OrderGap | 完整 reveal path 的全局顺序敏感性：max_π Q_π − min_π Q_π |
| M0 精确重建率 | 初始 Mask 位置上生成 token 与标准答案的 exact-match 比例（M0-only exact-token reward） |
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
