# 面试 Q&A 准备（中文口语版）

> 每题 2–5 句、技术准确、不过度包装、面试时可直接说出口。

**Q：为什么不用普通自回归模型？**

这个项目研究的问题本身就是 dLLM 特有的：AR 模型的 token 揭示顺序被训练目标固定成从左
到右，而 dLLM 每一步可以同时揭示多个 token，顺序敏感性才成为一个可以研究、可以测量的
结构属性。用 AR 就没有这个研究维度了。

**Q：SEDD 的 score 到底是什么？**

score 是模型对每个位置、每个词表 token 打的 log 分数，形状是 [B, L, 50258]。它的含义
接近"这个位置在给定噪声状态下应该变成哪个 token"的对数倾向。采样时把 log-score exp 成
真正的 score，再乘以 transition kernel，就得到该位置的离散转移权重。

**Q：raw score 和 staggered score 有什么区别？**

raw score 就是模型在单个时间点输出的 score。staggered score 是跨时间步修正后的离散
score：因为每个 reverse step 的 σ 变化量 dσ 有限，一步内真正发生的转移概率需要对 score
做 e^{dσ} 缩放并补偿 MASK 列的权重。简单说：raw score 是模型打分，staggered score 是
把它换算成"这一步实际会怎么跳"的概率质量。

**Q：为什么还需要乘 transition kernel？**

因为离散扩散的一步转移不是纯由 score 决定的：即使模型强烈倾向某个 token，一步之内
能转移的质量也受 dσ 限制，大部分概率还会停留在当前 token 或 MASK 上。transition kernel
编码了这个"一步内最多能转移多少"的物理约束，乘上 score 才是完整的一步转移权重 w。

**Q：为什么 SFT 会降低 CPI？**

这是观察到的实验结果，机理上没有做因果实验，我不强行归因。一个合理的解释方向是：
SFT 让模型在同一个 span 内反复见过"部分揭示、部分 Mask"的状态，模型学会了在给定
上下文下做更一致的条件预测，因此先揭示谁对联合概率的影响变小了。这是后续值得做的
解释性实验，不是已证明的结论。

**Q：你的 RL 最后没什么提升，是不是失败了？**

不是失败，是一个受控测出来的边界结果。500 步、lr=3e-6 的 RL-1 任务指标近似平稳，
compatibility 也无检测变化，这在预注册的解释矩阵里就是 Scenario B：主要结构变化发生
在 SFT 阶段。而且我做了 K=1 vs K=4 诊断，证明瓶颈不在 timestep sampling density。
科学上，"测出一个受控的无变化"本身就是有效信息。

**Q：为什么 K=1？**

K=1 是协议预注册的起点：每条轨迹只采样一个 timestep 做可微重算，显存和算力最省，
也能保持 rollout 与 logπ 完全同源。只有当诊断表明 variance 或 credit assignment 是
瓶颈时才增加 K。后来我做了 K=4 诊断，发现增加 K 没有带来一致改善，说明当初的保守
选择没有掩盖掉什么。

**Q：K=4 为什么没提升？**

K=4 是把每条轨迹的 timestep MC 估计从 1 个样本变成 4 个样本的平均，理论上只降低
estimator variance。诊断结果两组几乎重合，说明当前 weak task signal 的瓶颈不在
timestep estimator 的密度，而在更高层——比如组内相对 advantage 的 credit assignment
机制或 reward 本身的信息量。所以下一步应该动 objective 设计，而不是继续加 K。

**Q：为什么不用 GRPO？**

GRPO 的核心是 group-relative advantage 加 ratio clipping。我的 RL-1 已经用了
group-relative advantage，但保持 pure on-policy REINFORCE、没有 ratio/clip，这是
协议预注册的最小干预设计——先把单一变量的行为测清楚，再决定加什么。RL-3 阶段
（per-position PPO-style clip）就是朝这个方向的下一步，但按纪律要先过 review。

**Q：为什么不用 DeepSpeed？**

单卡 8GB、模型 ~170M，显存瓶颈主要在 rollout 的 one-hot 张量和重算图，而不是参数或
优化器状态的分片。DeepSpeed 解决的是大模型多卡问题，在这个规模下是额外复杂度。我用
的是更直接的工程手段：f32 one-hot、chunk 化、phase 间显存清理、expandable_segments
allocator，实测峰值稳定在 ~7GB 以内。

**Q：只有一个 RL seed，结果可靠吗？**

结论表述上和只有一个 seed 是匹配的：我只说"within this formal run / sample-level
paired evaluation"，不声称跨 seed 的显著性。SFT 的 compatibility attenuation 是两
seed 验证过的；RL 部分单 seed 是成本与范围的取舍，Future Work 里明确列了多 seed
验证。

**Q：历史 evaluator 复现不了会不会影响结论？**

不会改变定性结论。历史早期 formal 结果是用当时未提交的评估代码跑的，无法 bit-level
复现，但聚合差异很小（CPI ~2e-3、OrderGap ~8e-3），SFT attenuation 的方向和量级在
current-code 全链复算下完全保持。我已经把这个 provenance limitation 写进了 append-only
errata，Demo 主链全部改用 current-code harmonized 值，保证三阶段同口径。

**Q：你的创新点是什么？**

三点。第一，把 reveal-order compatibility 作为一个可测量的结构属性，用 CPI（局部）和
OrderGap（全局）两个互补指标跟踪后训练全程，发现 Vanilla SFT 系统性降低顺序敏感性。
第二，在离散扩散模型上实现了带 safe-region 截断的 on-policy 策略梯度训练（RL-1），
并给出了受控的"无进一步结构变化"边界结论。第三，方法论上做了一套可复现的评估与
诊断体系：预注册 gate、逐位对拍、K-ablation 的 RNG 隔离，以及一个全预计算的交互 Demo。

**Q：如果再给你两周你会做什么？**

三件事按优先级：先做 RL-3 的 per-position PPO-style objective，直接针对诊断指出的
credit assignment 瓶颈；然后加第二个 RL seed 并做 cross-task 验证，检验 SFT
attenuation 的普遍性；最后如果有算力，在更大模型上跑同样的协议，看这个现象是否随
规模保持。
