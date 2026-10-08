# 面试 Q&A 准备（中文口语版，41 题）

> 每题 2–5 句、技术准确、不过度结论、明确区分"实验事实"与"合理解释"。
> 板块：A 基础原理 / B 生成与采样 / C RL 设计 / D 评估指标 / E 工程与复现 / F 研究定位与后续。

## A. 基础原理

**A1. 为什么不用普通自回归模型？**

这个项目研究的问题本身就是 dLLM 特有的：AR 模型的 token 揭示顺序被训练目标固定成从左
到右，而 dLLM 的采样允许更灵活的揭示顺序，顺序敏感性才成为一个可以研究、可以测量的
结构属性。用 AR 就没有这个研究维度了。

**A2. SEDD 的 score 到底是什么？**

SEDD 输出的是离散 score 的 log 表示。连续扩散中的 score 是 ∇x log p(x)，但离散 token
没有连续梯度方向，所以离散 score 更接近两个离散状态之间的概率比：s_θ(x,y,t) ≈
p_t(y)/p_t(x)。代码里模型输出 [B,L,V] 的 log-score，对 MASK 位置它可以比较不同候选
token 的相对概率，但它不是普通 autoregressive logits，也不能把 raw output 直接 softmax
就当作 finite-step reverse transition。

**A3. raw score 和 staggered score 有什么区别？**

raw score 描述的是同一个噪声时刻下，候选离散状态与当前状态之间的相对概率关系；但
reverse sampler 要从当前时间 t 走到更干净的 t−Δt，需要把这个 same-time ratio 转换成
适用于一个有限 reverse interval 的跨时间概率比。staggered score 就是做这个跨时间步
修正：raw score = 当前时刻的相对概率关系；staggered score = 为当前这一个有限 reverse
step 修正后的概率比因子。它仍然不是最终 transition probability。

**A4. 为什么还要乘 transition kernel？**

从 Bayes 角度，reverse transition 可以写成：
p(x_{t−Δt}=y | x_t=x) ∝ p(x_t=x | x_{t−Δt}=y) × p_{t−Δt}(y) / p_t(x)。
staggered score 对应后面的跨时间概率比，transition kernel 对应 forward corruption 的
转移概率。口语版：staggered score 只回答"这个更干净的候选状态有多合理"，transition
kernel 回答"这个候选状态经过这一小段 forward corruption 后，能不能合理地产生当前状态"。
记忆法：**score 管合理性，kernel 管可达性。** 两个都乘上，才是完整的一步 reverse
transition 权重。

**A5. 为什么不能直接把 raw score 做 softmax？**

因为模型输出的是 score parameterization，真正的 finite-step reverse transition 还需要
staggered correction 和 forward transition kernel。最终使用的是构造出的非负 transition
weights，再做线性归一化，而不是直接对 raw score 做普通 softmax。直接 softmax 得到的是
"同时间刻下的相对打分"，不是"这一步实际转移去哪"的分布。

**A6. SEDD 的 score entropy 在优化什么？**

score entropy 是离散扩散的训练损失：对每个被 Mask 的位置，让模型给 ground-truth token
的 log-score 尽可能高，并按当前 σ 的权重（本项目里是 dσ 加权）聚合。直觉上，它在
"当前噪声状态"和"干净答案"之间建立概率比的回归目标；在本项目的 partial-reveal SFT
里，loss 只计算当前被 Mask 的目标位置。

**A7. 为什么离散 token 不能直接用连续 score？**

连续 score 是 ∇x log p(x)，它需要输入空间有连续梯度方向；离散 token 之间没有"插值"
或"方向"可言。所以离散扩散把 score 重新参数化为状态间的概率比——这是 SEDD 论文里
score parameterization 的核心设计，而不是工程妥协。

**A8. absorbing diffusion 是什么？为什么 MASK 是 absorbing state？**

absorbing（吸收型）离散扩散的前向过程只做一件事：按一定速率把干净 token 替换成
[MASK]，而 MASK 一旦出现就保持为 MASK——所以 MASK 是吸收态。反向过程就是从吸收态
出发，把 MASK 逐步恢复成干净 token。这个设计的好处是反向生成天然对应"填空"，
而且与 MLM 预训练的结构高度兼容。

## B. 生成与采样

**B1. 为什么模型一次 forward 能预测多个位置，但仍然需要 128-step sampler？**

模型一次 forward 可以同时为多个 MASK 位置给出预测，但实际 reverse sampler 会在多个
reverse steps 中随机决定哪些位置在当前 step 被 reveal，所以是 parallel prediction +
iterative revealing，而不是一次性把所有 token 定死。每步能转移多少概率还受 dσ 限制
（finite-step transition），128 步是这个离散化 schedule 的长度。

**B2. 为什么选择 128 steps 而不是 1024？**

做了专门的 gate 实验：同一 SFT checkpoint 上跑 128 与 1024 步的完整 rollout 对比，
1024 步的 M0 重建 reward 均值 0.2192 只比 128 步的 0.2035 高约 0.016，残存 MASK 率
两者都是 0，但 1024 步的 runtime 是 279.4s vs 35.8s，约 7.8 倍。质量接近、成本差距
明显，所以裁决 128 步（results/rl_gates/128_vs_1024.json）。

**B3. safe-region σ≥0.05 是怎么来的？**

是经验数值有效性边界，不是理论常数。证据：在 SEDD-small + analytic predictor +
128-step schedule 下，σ≥0.05 区全程零硬失效（三条 run、数千次 rollout step）；
σ<0.05 尾段才出现低概率的负权重元素（chunk-step 级约 0.15%、element 级约 1e-7~1e-8）。
协议 v1.0 里明确写：换模型 / 采样器 / schedule 必须重新验证这个边界。

**B4. truncated policy gradient 会不会有 bias？**

会，这是设计里明确承认的。J timestep 只从 σ≥0.05 的 safe region 采样（约 91.9% 的
支撑），σ<0.05 的尾段（约 8.1%）被排除在 PG 目标之外，所以 objective 是截断的条件
期望，不是全 128 步的 uniform 期望。协议因此把方法命名为 safe-region / truncated
policy gradient，明确不声称 full reverse-trajectory unbiased。截断是为了数值稳健性，
代价就是这 8.1% 的支撑偏倚。

**B5. 为什么 rollout 用 no_grad，J timestep 再 recompute？**

rollout 阶段是行为采样（on-policy 数据收集）：128 步里只需要缓存被抽中的 J 步的
(x_J, action, σ_J, dσ_J, mask)，如果保留整条 128 步的计算图，8GB 显存扛不住。
训练时用当前参数对 J 步重新 forward 出 logπ 再求导——梯度等于当前策略的梯度，
保持了 on-policy 语义，显存也只付单步的账。

## C. RL 设计

**C1. 为什么 REINFORCE 的 loss 是 -A logπ？**

policy gradient 定理给出 ∇E[R] = E[A · ∇logπ]，所以构造 loss = −A·logπ，梯度就正好
等于这个期望方向。A 用组内相对优势代替原始 reward，是为了减方差：同一 prompt 的
G=4 条轨迹共享初始腐蚀，组内 mean-center 后，绝对 reward 水平被消掉，剩下的是
"这条轨迹相对同组好多少"的信号。

**C2. group-relative advantage 为什么不用 critic？**

第一版刻意最小干预：不引入 critic 网络，就没有额外参数的训练噪声和 bootstrap 误差；
组内共享初始腐蚀的 4 条轨迹天然构成同 prompt 基线。代价是方差可能比有 critic 高，
这是设计选择，不是优越性证明；要不要 critic 留给后续 ablation。

**C3. P=4、G=4 分别是什么意思？**

P=4 是每个 optimizer step 采样 4 个不同 prompt；G=4 是每个 prompt 展开 4 条 rollout
（共享同一初始腐蚀，只有采样随机性不同），组内做 relative advantage。所以每个
optimizer step 实际有 16 条轨迹。

**C4. zero-variance group 是什么？**

同一 prompt 的 4 条 rollout reward 完全相同 → 组内 advantage 全 0 → 这一组对梯度
贡献为 0。formal run 里占比约 6–7%，均匀出现。这是合理行为：组内没有差异就没有
相对信号，协议不强制重采样，只记录计数。

**C5. 为什么当前 RL 不直接叫 GRPO？**

GRPO 一般包含 ratio / clipping（PPO 风格）甚至 reference KL；我的 RL-1 是 pure
on-policy REINFORCE + group-relative advantage：没有 ratio、没有 clip、没有 replay。
这个命名是协议 v0.2 定案的：先用最小干预的版本把单一变量的行为测清楚，ratio+clip
版本（RL-3）是预注册的下一步。

**C6. PPO clipping 是解决什么问题？**

限制更新后的策略与 rollout 时的旧策略偏离太远：如果没有 clip，某个大 advantage 的
轨迹可能把对应动作概率推到极端，一步就破坏策略稳定性。RL-3 预注册的是 per-position
clip（ρ_i = 新旧 logπ 之差 exp 后在每个位置独立 clip），避免 joint ratio 的乘积爆炸。

**C7. 为什么 K=1？**

K=1 是预注册起点：每条轨迹只采样一个 timestep 做可微重算，显存和算力最省，rollout
与 logπ 也完全同源。诊断实验表明提高 K 没有一致改善（见 C9），说明当初的保守选择
没有改变结论方向。

**C8. 你的 RL 最后没什么提升，是不是失败了？**

当前 formal RL 的 task improvement 确实比较弱，所以我不会包装成 RL 提升了模型。但
pipeline 本身稳定更新了参数（drift 3e-4），compatibility 也保持稳定。更重要的是，
我没有把这个 null result 直接解释成"RL 天然不会改变 compatibility"，而是继续做
K=1 vs K=4 的 matched diagnostic，发现单纯提高 timestep sampling density 仍然没有
一致改善，因此下一步更值得检查 reward 和 objective-level credit assignment。

**C9. K=4 为什么没提升？**

实验事实：在当前 SEDD-small、当前 reward、当前 REINFORCE objective、单 seed、
100-step matched diagnostic 下，把 K 从 1 提高到 4 没有带来一致的 task-learning
improvement。合理解释（不是结论）：timestep estimator 的方差可能不是当前 weak signal
的主要来源。所以目前没有证据表明单纯提高 timestep sampling density 是主要瓶颈，
也说明继续简单增加 K 不是最有价值的下一步。

## D. 评估指标

**D1. CPI 和 OrderGap 的区别是什么？**

CPI 是局部指标：取两个 MASK 位置 a、b，比较"先 a 后 b"和"先 b 后 a"两种揭示顺序下
模型自评的条件概率差 δ，CPI = E|δ|。OrderGap 是全局指标：同一个目标序列在不同完整
reveal path（如 l2r / r2l / random）下的最大-最小路径似然差。

**D2. 为什么两个指标都需要？**

因为局部一致不保证全局一致：单个 pair 的 δ 都很小，多步路径累积起来仍然可能产生
可观的全局差异。CPI 定位"哪里不一致"，OrderGap 量化"整条路径差多少"，两个一起才能
同时看到局部结构和全局结构的变化。

**D3. CPI 是你原创的吗？**

不是。这个数学量（swap δ / local curl）在已有工作中出现：Path-Dependent Denoising
论文里叫 local curl，Majid et al. 里有 swap consistency。指标本身是 SEDD 论文和
后续工作定义好的，我没有发明指标。我的测量对象才是新的：**vanilla SFT 前后、
Pretrained→SFT→RL 轨迹上的 paired ΔCPI/ΔOrderGap**，据文献核查这是目前没有直接
覆盖的空白。

**D4. 为什么 SFT 会降低 CPI？**

这是实验观察（两 seed 稳定复现），机理上没有做因果实验，我不强行归因。一个合理的
解释方向：SFT 让模型在同一个 span 内反复见过"部分揭示、部分 Mask"的状态，masked
prediction 的优化和校准改善会让条件预测更一致，先揭示谁对联合概率的影响变小——这也
和 Path-Dependent Denoising 的理论方向（imperfect optimization → 非零 curl）吻合。
注意：这是解释候选，不是已证明的机制。

**D5. 为什么 SFT attenuation 值得研究？**

因为 reveal order 影响采样器的实际行为：如果模型对不同揭示顺序的条件概率差异很大，
不同采样路径会生成不同质量的结果，采样器设计（path 选择）就是一个实际自由度。我们的
发现说明：后训练（而不只是任务数据本身）会系统改变这个结构属性——这对理解 dLLM 的
post-training 动力学有直接含义。

**D6. 为什么 OrderGap 早于 CPI 出现变化？**

这是 P1 early dynamics 的观察：OrderGap 在 SFT step 50 已经有小幅但可检测的下降，
而 CPI 当时还在平台，CPI 的明显下降出现在 warmup 后期（250–750 窗口）。合理解释
（非结论）：全局路径敏感性是多步累积量，对参数的早期微小变化更敏感；局部两-token
条件分布需要更强的参数移动才显著改变。

## E. 工程与复现

**E1. 只有一个 RL seed，结果可靠吗？**

结论表述上和只有一个 seed 是匹配的：所有统计只表述 sample-level uncertainty 和
"within this formal run"，不声称跨 seed 显著性。SFT 的 compatibility attenuation
是两 seed 验证的；RL 单 seed 是成本与范围的取舍，Future Work 明确列了多 seed。
如果 reviewer 要求，边界就在那里，我不反驳，只说明哪些结论已经跨 seed、哪些没有。

**E2. 历史 evaluator 复现不了会不会影响结论？**

不会改变定性结论。历史早期 formal 结果是用当时未提交的评估代码跑的，无法 bit-level
复现，但聚合差异很小（CPI ~2e-3、OrderGap ~8e-3），SFT attenuation 的方向和量级在
current-code 全链复算下完全保持。这个 provenance limitation 已经写进 append-only
errata，Demo 主链全部改用 current-code harmonized 值。

**E3. 为什么不用 DeepSpeed？**

单卡 8GB、模型 ~170M，显存瓶颈在 rollout 的 one-hot 张量和重算图，不在参数/优化器
状态的分片。DeepSpeed 解决的是大模型多卡问题，在这个规模是额外复杂度。我用的工程
手段更直接：f32 one-hot、chunk 化、phase 间显存清理、expandable_segments allocator。

**E4. K=4 为什么 peak VRAM 没增加？**

因为实现是 sequential backward：每个 J 独立建一张计算图，backward 完立即释放，四个
J 的梯度在参数上累加。显存峰值由单张重算图决定，与 K 无关，所以 K=4 和 K=1 的
peak VRAM 完全相同（6.36GB）。

**E5. 为什么 K=4 wall time 只增加约 6%？**

每个 optimizer step 的时间由 rollout 主导：128 步 × 8 个 chunk 的 forward 占大头；
K 只改变 recompute 的次数，而 recompute 在总时间里的占比小。所以 4 倍的重算量只
带来约 6% 的总 step 时间增长。

**E6. WSL2 的 dxgkrnl OOM 是怎么回事？**

是这个项目最难的工程问题：CUDA 在 148MB 级别的小分配上报 OOM，空闲显存显示回绕到
2^64 量级，dmesg 出现 EOVERFLOW 的 create_allocation 失败。定位到 Windows 侧 WDDM
driver 的分配记账状态损坏，`wsl --shutdown` 无效，完整 Windows Restart 才重置。之后
用 expandable_segments 减少分配次数作为工程保险，重启前先做默认 allocator 的因果
验证，确认是重启（而不是配置）解决了问题。

**E7. 为什么 Demo 不用 live inference？**

演示可靠性优先：live inference 意味着现场依赖 GPU、依赖驱动状态、还可能撞上这个
项目实际遇到过的 WDDM 异常。precomputed 模式启动快、无 GPU 要求、现场不会翻车。
live mode 可以后置做，但第一版明确不作为 blocker。

**E8. 为什么 precomputed outputs 的 Demo 仍然有效？**

因为所有预计算内容都来自正式 checkpoint 的真实 inference 导出：轨迹是确定性 seed
的 rollout（跨进程逐位验证过），pair 四项与正式 per-sample δ 逐位一致，聚合指标与
正式结果文件对拍过。Demo 只是把"计算"和"展示"解耦，展示的是真实模型行为，不是
演示性伪造数据。

## F. 研究定位与后续

**F1. 你的研究贡献是什么？（不问"创新点"）**

我不把 dLLM 存在 reveal-order dependence 本身当作我的贡献，因为已有工作已经研究过
order sensitivity 和 conditional incompatibility。我的重点是研究这种结构在
post-training 过程中怎么变化。第一，沿着 Pretrained→SFT→RL 的训练轨迹同时跟踪局部
CPI 和全局 OrderGap，发现 Vanilla SFT 会稳定降低两者（两 seed 复现）；第二，P1
early-dynamics 分析观察到全局 OrderGap 很早就开始变化、局部 CPI 的明显下降稍晚，
说明局部和全局结构变化不完全同步；第三，实现了基于 SEDD reverse process 的 RL
pipeline，用与 compatibility 无关的 task reward 检查后续 RL 是否会继续改变这种结构，
并用 K-ablation 诊断 weak learning 的来源。工程上还建立了冻结 manifest、统一
evaluator、pair-level 对拍、protocol/errata 记录和交互式 Demo。

**F2. 和 PAPL / TRIMS 等 order-aware 工作的关系是什么？**

它们是"方法侧"的工作：TRIMS 用 trajectory-aware masking 教模型走什么路径，PAPL 把
多种 path 用于训练，方向是设计 order-aware 训练法。我的项目是"评估侧"：不改变训练
方法，而是测量 vanilla SFT / RL 对 order sensitivity 的副作用。文献核查的结论是：
"post-training 前后的 paired ΔCPI/ΔOrderGap"目前没有直接覆盖的工作，这是本项目
最接近新贡献的位置——并且我明确引用 Path-Dependent Denoising 的 curl 定义和
Majid 的 swap consistency，不声称指标原创。

**F3. 如果给你 8×H100，第一批扩展实验怎么设计？**

三件事按顺序：① RL credit assignment——直接做预注册的 RL-3（per-position PPO-style
objective），加上第二个 RL seed；② cross-task / cross-model 验证——换任务和模型族，
检验 SFT attenuation 是否普遍；③ 更大模型（SEDD-medium 级别）跑同一协议，看现象
是否随规模保持。都是验证性扩展，不是新指标或新方法。

**F4. 如果 reviewer 说 compatibility 下降只是 task performance 提升的副作用，你怎么回应？**

诚实回答：当前实验无法排除这个解释，我没有做控制 task performance 的对照。可行的
验证方案是构造任务难度梯度（同模型同数据、不同 span 难度）观察 CPI/OrderGap 是否
随 task 表现单调变化；或者反过来，在 RL 里用与任务无关的 reward 看 compatibility
是否独立响应。这是 Future Work 的候选，不是已经排除了的解释。

**F5. 如果再给你两周你会做什么？**

先做 RL-3 的 per-position PPO-style objective，直接针对诊断指出的 credit assignment
方向；然后加第二个 RL seed 并做 cross-task 验证，检验 SFT attenuation 的普遍性；
有余力就在更大模型上跑同一协议。三件事都写在 Future Work 里，不临时起意。
