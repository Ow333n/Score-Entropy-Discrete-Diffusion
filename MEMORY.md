# MEMORY.md — SEDD 大作业每日任务记录

> 研究提案 v2（2026-10-01）：`order_alignment_vs_compatibility_research_proposal_v2.md`（Order Alignment vs. Order Compatibility）
> 本地执行计划（8GB 单卡适配版）：`local_execution_plan.md`（阶段 0-8 + Swap-SFT 显存配方 + 止损点）

> 大作业三大目标：
> 1. 读懂 SEDD 论文（arXiv:2310.16834）——Score Entropy 损失、连续→离散 score matching 的推广
> 2. 实现 SEDD 的监督微调（SFT），在测试集上评估
> 3. 探索离散扩散模型的 RL 路线（DDPO/GRPO 等）
> 4. （加分）跟踪前沿研究、自主探索新方案

## 环境信息

- WSL2, Linux
- GPU: RTX 5060 Ti (8GB, Blackwell sm_120), driver CUDA 13.1
- 包管理器: uv 0.12.15（待创建虚拟环境 .venv）
- 官方 environment.yml 是 conda + CUDA 11.8 + torch 2.0.1，**本机 Blackwell 显卡必须换用 torch ≥2.7 (cu128)**，flash-attn 2.2.2 也需替换或打补丁

---

## 2026-10-01（Day 4）研究项目开工：阶段 0 + 阶段 1 + 阶段 2 完成 ✅（v4.1 协议版）

> ⚠️ 本日重要更正：**Day 3 记录的 "batch 4/accum 2 = eff 8" 是错的**。data.py 的 DataLoader 把 `config.batch_size // (ngpus×accum)` 当 micro-batch → config 里的 batch_size **本身就是 effective batch**。运行时核实（G0 §4.1，scripts/check_batch_semantics.py）：config batch=4/accum=2 → 实际 micro=2、optimizer step 吃 2×2=4 条序列、**B_eff=4**（Day 3 smoke 实际是 eff 4）。这就是 v4.1 协议 §2.2 坚持运行时核实的原因。

### G0 正确性门：20/20 单元测试 + 3 项 GPU 检查全部通过 ✅
- [x] §4.1 batch semantics 运行时核实：config 4 → micro 2 → eff **4**（见上，记录已更正）
- [x] §4.2/4.4/4.5/4.6 玩具测试（compatibility/tests/：test_posterior 归一化/MASK排除/时间标量消掉/确定性、test_swap_delta 兼容 δ=0 不兼容 ln(3.2)、test_strict_reveal TF+sampled 记账/4 顺序/tie-breaking/fixed path/温度/TF 路径概率手算值）
- [x] §4.7 residual time-dependence：同一状态 JSD(σ=0.5,1.5)=0.004、JSD(1.5,3.0)=0.012、JSD(0.5,3.0)=0.018（轻微时间依赖，量级小）；确定性检查逐位一致 ✓
- [x] §4.8 严格揭示：不变式（mask−1/仅选中位置变/选中前是 MASK）作为运行时断言，违反抛错（含违反测试）；TF 与 sampled 两模式
- [x] §4.9 全 autograd vs 重计算梯度等价：**fp32 ε_g < 1e-5、fp64 ε_g < 1e-9 通过** → Swap-SFT 进入门已开（training/swap.py 的 Pass A/B + sg(δ) 配方，逐状态 backward 保持显存峰值=单状态）
- [x] 目录重构（§28）：compatibility/{posterior,cpi,order_gap}.py + tests/、training/swap.py、evaluation/{build_manifest,eval_cpi,eval_order_gap}.py、manifests/、protocol/、scripts/、results/{pretrained,pilot}/；旧 cpi_eval.py/test_cpi.py/run_cpi_baseline.py 已删除（迁移），旧结果移入 results/pilot/（标记为 exploratory，非正式基线）

### 冻结资产（protocol v4.1）
- [x] **manifest 已冻结**：`manifests/regime_a_eval_v1.jsonl`（500 样本，SHA-256=1897bd14…，§5A 不可变）：span-structured（wikitext103 test 256-chunk、span 10~50、σ 网格 {0.5,1.5,3.0} round-robin、span 外恒可见、i≠j 在 span 内、pair_distance 记录）+ 6 条路径（l2r/r2l/3 seeded random/pretrained 置信路径，tie_fraction=0.0072）
- [x] **正式 pretrained 基线（N=500，results/pretrained/）**：
  - **CPI_abs = 0.328 ± 0.023；CPI_RMS = 0.605；δ 均值 +0.003 ± 0.027（无偏）；median 0.0；|δ| p90=0.94 p99=2.29；δ_SD=0.605（效应量基线）**
  - local CE = 4.281 ± 0.081；token acc = 0.322 ± 0.010；峰值显存 1.33GB
  - 分桶结构（重要观察）：CPI 随 **σ** 升（0.23→0.48）、随 **mask_ratio** 升（0.19→0.47）、随 **span_len** 降（0.44→0.21）、随 **pair_distance** 强烈降（**距离 1-4 → 0.68；距离 15-41 → 0.11**）——相邻 mask 对的 swap 不兼容远大于远距离对，符合 n-gram 耦合直觉
  - 注：span 结构 CPI (0.33) ≫ 昨日全序列随机腐蚀 pilot (0.039) 的原因正是 pair_distance 分布不同——协议改 span 结构是对的
- [x] **正式 pretrained TF OrderGap（results/pretrained/order_gap.json）**：
  - Q：l2r −61.99 / r2l −61.32 / random×3 ≈ −62.1~−62.2 / **confidence −65.78（最差）**
  - **OrderGap_raw = 10.24 ± 0.35 nats；per-token = 0.605 ± 0.031 nats/revealed-token；Var_π(Q) = 24.3**；m̄=21.7；峰值显存 2.36GB
  - 早期观察：pretrained 按自己置信度优先揭示的路径 TF 似然最差——与 Flexibility Trap 叙事方向一致（order choice matters），可作报告观察点，但不作声明
- [ ] 下一步（协议顺序）：Vanilla pilot（已启动，见下）→ 🔒 协议冻结 → partial-reveal 正式 Vanilla SFT → G0.5 状态支持 → G1/G2 → 阶段 4 止损

---

## 2026-10-01（Day 4 晚）256-seq 管线 + Vanilla smoke + pilot 启动

### 显存调试大工程（重要，勿重踩）
- [x] **现象**：batch 32×256 训练 forward 峰值 10.32GB 超 8GB → WDDM 换页 → 0.2 steps/s 灾难降速（首跑 smoke 卡 125 步/11 分钟，已 kill）
- [x] **排查过程**（scripts/profile_mem.py 逐段打点）：SDPA 在 sm_120 上 flash kernel 正常（profiler 实证 `pytorch_flash::flash_fwd_kernel`，非连续视图也走 flash，**注意力矩阵不是元凶**）；真正原因是 **train 模式每 block 保留 ~540MB 激活**（fp32 LayerNorm 输出 + MLP 中间量 + autograd 图）→ 12 blocks ≈ 6.5GB
- [x] **修复**：model/transformer.py 12 个 block 加 `torch.utils.checkpoint`（train 模式专用；§9.1 dropout=0 使 backward 重算逐位一致，无 RNG 差异；eval/forward 输出不变 → **冻结的 pretrained CPI/OrderGap 基线不受影响**）
- [x] **修复后 smoke 通过**：300 步 145s（**2.1 steps/s、17k tokens/s**）、峰值 **5.73GB**（reserved 6.49GB，桌面余量充足）、loss 9.86→5.56 下降
- [x] 训练管线：`task_data/corruption.py`（§7.2 span 腐蚀：span 外恒可见、span 内 1-e^{-σ} partial absorbing）、`training/vanilla.py`（§27 协议字段全记录 + plateau 检测 §3.1 + analysis checkpoint 机制 §3.2 + resume）、`configs/vanilla_256.yaml`（batch 32 eff、dropout 0.0、pilot 参数）、loss reduction 冻结为 mean_over_masked_span（已写入 regime_a_protocol.yaml）
- [x] **pilot 参数在跑之前已冻结**（§3.1）：N_pilot_max=30000、M=500、K=6、ε=0.5%、α=1.2、pilot_seed=0（protocol/regime_a_protocol.yaml `vanilla_pilot:` 节）
- [x] **Vanilla pilot 已启动**（后台，30k steps ≈ 4.5h，2.1 steps/s）：exp_local/regime_a/pilot-*，plateau 检测自动算 N = min(30000, ceil(1.2·s_p))
- [x] `evaluation/eval_task.py` 已写（G1 §9.4：masked NLL/token acc 复用 manifest 协议 + span l2r 贪心解码命中率/精确匹配；支持 HF 或本地 checkpoint 的 EMA 权重）——pilot 完成后先跑 pretrained 基线再跑 pilot checkpoint
- [ ] pilot 完成后：eval_task（pretrained vs pilot）→ 冻结 N/LR/recipe → 正式 Vanilla SFT（analysis checkpoints 0.1N/0.5N/N）→ G0.5 → G1/G2 → 阶段 4 止损

### Vanilla pilot 首跑发现（calibration 信号，已按 §3.13 调整）
- [x] pilot 跑到 5000/30000 步时 kill：**eval loss 在 warmup 结束（LR 达 3e-4）后系统性上升**（7.27@500 → 7.17@1500 最低 → 8.01@2500 → 8.20@5000）——过拟合签名：170M 模型 + weight_decay=0 + LR 3e-4 对该任务过激进（§9.2.2 "确定稳定 LR" 正是 pilot 的职责）
- [x] 已启动 LR 校准探针（后台串行）：3e-5 与 1e-4 各 3500 步（3e-4 探针 = 已终止 pilot 的数据，不必重跑）；选定后重跑 pilot
- [x] 附带发现：plateau 检测用相对改善 ε=0.5% 对 eval 噪声（±0.5-1%）过严，eval 波动会不断重置 K 窗口 → 30k 步内可能永不触发（pilot 数据验证：1500→2000 变差后又出现 +0.6-0.7% 的噪声改善）。**处理：ε 等 pilot 参数已冻结不动**，但若重跑 pilot 仍不触发则按 §3.1 取 N=N_max=30000（协议已有此分支，不需修订）
- [x] **LR 校准完成：选定 3e-5**（探针数据：3e-4 发散 ✗；1e-4 warmup 后摇摆 7.30→7.31→7.26；3e-5 warmup 后单调 7.27→7.24→**7.19@3500** ✓）
- [x] **工作流变更（用户要求）**：>2h 的任务给出命令由用户在 tmux 自己跑，跑完告诉我 run 目录再继续分析
- [ ] pilot（lr 3e-5, 30k 步）已交用户 → 完成后读曲线/plateau → 冻结 N/LR/recipe → 正式 Vanilla SFT

---

## 2026-10-01（Day 4 深夜）pilot 完成 → 协议冻结 → G1 pilot 检查通过 ✅

- [x] **pilot-150849 完成**（30k 步，2.2 steps/s，6.41GB）：eval 7.55@500 → **7.05@6500 最优** → plateau@8500 → **N = min(30000, ⌈1.2×8500⌉) = 10200**；20000 步后晚段过拟合（+0.3 nats），N=10200 恰好避开
- [x] **协议已冻结**（protocol/regime_a_protocol.yaml `training_frozen:`）：N=10200、analysis checkpoints [1020, 5100, 10200]、LR=3e-5、warmup 2500、AdamW(0.9,0.999,1e-8)、wd=0、EMA 0.9999、grad_clip 1.0、eff batch 32、dropout 0、span [10,50]、formal seeds [(1,1,1),(2,2,2)]
- [x] **G1 pilot 检查通过**（eval_task.py，EMA 权重）：NLL 4.281→3.828（−10.6%）、token acc 0.322→0.362、贪心命中 0.443→0.472
- [x] **踩坑**：EMA state_dict 格式是 `{decay, num_updates, shadow_params: [tensor列表]}` 非按参数名索引——`model.load_state_dict(ema_dict)` 全 key 不匹配静默空载 → 输出层零初始化 → NLL=ln(50257)=10.81 零方差（这个签名=均匀分布）。正确姿势：`ema.load_state_dict(...)` + `ema.copy_to(model.parameters())`（eval_task.py 已修；training/vanilla.py 的 restore 本来就对）
- [x] 正式 Vanilla SFT ×2 seeds 已交用户在 tmux 跑（~1.5h/seed）
- [ ] 正式 run 完成后：eval_task/eval_cpi/eval_order_gap 于 3 analysis checkpoints × 2 seeds → G2 兼容性动力学 → G0.5 状态支持 → 阶段 4 止损门

### 正式实验前实现核查（用户要求，已完成 ✅）
- [x] **核查 ① scheduler**（scripts/check_scheduler.py）：`g['lr'] = lr * min(step/warmup, 1.0)`，warmup 后**位级恒定 3e-5**；n_iters=10200 与 30000 两条 trajectory 共享 step 逐点位级一致，30000 轨迹在 10200 后仍恒定 → **trajectory 与 total_steps 无关**（n_iters 只进训练循环终止条件）
- [x] **核查 ② EMA**（scripts/check_ema.py）：torch 同款镜像逐位验证 update rule；**存在 decay warmup**（pytorch_ema 标准）：effective_decay = min(0.9999, (1+n)/(10+n))，第 89990 次 update 才达 0.9999 → **本实验全程（10200 步）处于 warmup 区间，effective decay 0.9990→0.99912**；**无 bias correction**；eval/checkpoint 用 shadow（EMA），raw weights 同时保存
- [x] **G1/G2 评估口径 + Stage-4 gate 判据已预注册**（protocol/regime_a_protocol.yaml `g2_evaluation_pre_registered:` + `stage4_gate_pre_registered:`，看结果前冻结）：primary=EMA 权重 + late checkpoint；paired bootstrap 10k CI；stable=两 seed 同符号且 CI 不含 0；联动=seed 内 sign(CPI_late−CPI_early)==sign(OG_late−OG_early) 且 CI 不含 0；gate 映射 A_HARD_STOP/B_DIAGNOSTIC_ONLY/C_PROCEED；seed 相反→unstable 不加 seed；pilot 30k G1 仅 sanity；24k spike/晚段退化仅审计
- [x] eval_cpi.py/eval_order_gap.py 已支持本地 checkpoint 的 EMA 加载（与 eval_task 同款；pilot 目录冒烟通过）。**注意 pilot CPI 冒烟显示 SFT 后 CPI 可能下降（4-8 桶 0.357→0.275 等）——预览而已，不得预判 Stage-4，以预注册判据为准**
- [x] 正式 Vanilla ×2 seeds 命令已交用户（tmux）

---

## 2026-10-02（Day 5 凌晨）正式 Vanilla 2 seeds 完成 → G0.5 PASS → Stage-4 判决 **B_DIAGNOSTIC_ONLY**

- [x] **formal-vanilla-s1-191414 / s2-204215** 完成：各 10201 步、零 NaN、3 analysis checkpoints（raw+EMA 双权重）
- [x] 训练期 eval 监控器噪声已诊断（交叉评估证明），正式评估用 frozen manifest 固定 σ 网格不受影响
- [x] **G0.5 PASS**（6/6）：σ 网格在训练支持内（P(σ≤3)=95.1%）、K=0/1 状态可达（P=3.8%/3.9%）、span/pair 规则一致
- [x] **正式 G1/G2 全指标**：18 项评估（2 seeds × 3 checkpoints × CPI/OG/G1），全部 paired per-sample 落盘
- [x] **Stage-4 判决（预注册判据，results/vanilla/stage4_gate.json）**：
  - ΔCPI stable ✅：s1 −0.040 [−0.072,−0.007]，s2 −0.046 [−0.078,−0.014]（SFT 降 CPI ~12-14%，两 seed 同负）
  - ΔOrderGap stable ✅：s1 −1.49，s2 −1.43（OG 10.24→~8.8，−14%）
  - **联动 ❌**：early→late 轨迹平坦（CPI_el CI 含 0）→ 判 B_DIAGNOSTIC_ONLY
  - **关键科学发现**：SFT 的全部效应在头 ~1020 步（warmup 期）内完成，之后平台——"瞬时阶跃"模式使预注册的轨迹联动判据天然不适用；s1/s2 在 OG early→late 符号分歧（噪声内）
  - 方向与原始 H1 假设（SFT 制造不兼容）**相反**：Vanilla SFT 无正则即减弱不兼容 + 顺序敏感度 → proposal §24 Outcome B 叙事（"什么因素决定 compatibility 方向"）
- [x] **按预注册规则：不进入 PAPL/Swap 方法路线**（gate 要求 C_PROCEED 才放行）
- [ ] 下一步选项（待用户定）：① 诊断论文路线（写 Stage-4 报告 + 探索"什么训练因素决定方向"：lr/wd/data 消融）② 若用户与外部 AI 讨论后认为联动判据应修订（如"阶跃+平台"模式的判据），需 protocol revision v4.2 + 说明理由，不可因结果不好看而改

---

## 2026-10-02（Day 5 早）用户裁定 + 三项审计 + protocol v4.2 建立

### 用户裁定（固定，不得偏离）
- [x] **v4.1 不修改不重新解释；Stage-4 判决 B_DIAGNOSTIC_ONLY 永久固定**
- [x] 核心现象（固定表述）："Vanilla SFT 在当前 SEDD-small + WikiText103 span-infilling 设置下使 CPI 从 0.328 降到约 0.283–0.289、OrderGap 从 10.24 降到约 8.8，主要变化在第一个 1020-step checkpoint 前已经出现"
- [x] 叙事口径："potentially novel empirical observation"，不声称普适规律
- [x] 方法路线（PAPL/Swap 2×2）**降级**：Vanilla 已自行降 CPI，原始动机弱化

### 三项审计（全部确认）
- [x] **审计 ①**：formal run 实际 optimizer steps = **10201**（循环 `step < n_iters+1` 继承 repo 惯例）；checkpoint 精确在 1020/5100/10200 ✓；偏离 0.01% 记录在案
- [x] **审计 ②**：EMA effective decay 更正——n=1 时 **0.1818**（此前报告"0.9990→0.99912"错误）；n=50→0.850、n=1020→**0.9913**、n=5100→0.9982、n=10200→0.9991。early ckpt 的 EMA 半衰期 ~70 步（v4.2 密集研究重要）；stage4 报告已修正
- [x] **审计 ③**：G0.5 分布级检查重做（首版 K 定义混乱 + 只有可达性）——OC: σ=0.81、mask_ratio=0.62、span_len=0.95、pair_distance=0.92、K_revealed=0.75，**PASS**（results/vanilla/g05_state_support_distributional.json）

### protocol v4.2（独立，不动 v4.1）
- [x] `protocol/experiment_protocol_v4.2.md` + `regime_a_protocol_v4.2.yaml` 已建
- [x] P1 早期动力学：dense checkpoints [50,100,250,500,750,1020,2500]、seeds (1,1,1)/(2,2,2)、raw+EMA 双权重、同一 frozen manifest、全指标 + buckets；评估分 3 个预注册 Tier（全部执行，顺序只管先后）；变化定位 = 第一个相邻点对 paired Δ CI 不含 0
- [x] P2 复现：P2a openwebtext（必做，新冻结 manifest）；P2b 任务族（尽力）；P2c 模型规模 stretch（medium 8GB 不可训，替代=自定义 tiny 从零预训练）；判据：ΔCPI CI 不含 0 = 重塑；跨设置符号分歧 = 方向依赖
- [x] 机制消融/PAPL-Swap 延迟到跨设置复现之后
- [x] 代码就绪：vanilla.py 循环改为精确 N 步（v4.2 语义）；eval 三脚本加 `--weights raw|ema`

### 文献核查首轮（reports/literature_check.md）
- [x] **未发现直接测量 vanilla SFT 前后 ΔCPI/ΔOrderGap 的工作**。最密切：① Path-Dependent Denoising（arXiv:2605.09303）的 local curl **就是我们的 δ_swap**（inference-only diagnostics，自述缺实证，Theorem 3: curl 源自 finite capacity/imperfect optimization/calibration 而非数据结构——是我们发现的现成机制假设）② Decoding in Order-Agnostic LMs（2606.00997）在已微调 LLaDA 上的单点快照 ③ Mixing Times（2605.16378）存在性理论 ④ Inconsistencies in MLMs（2301.00068）⑤ Majid AISTATS 2025 定义来源。TRIMS/SAS/DTM 均不测 vanilla SFT 的兼容性副作用

### formal-s1 完成 + 重要诊断（eval 监控器噪声，勿误判）
- [x] **formal-vanilla-s1-191414 完成**：10201 步/4474s/2.3 steps/s/6.41GB/零 NaN/3 个 analysis checkpoints（各 2.7GB，含 raw+EMA 双权重）✓
- [x] **异常排查**：s1 训练期 eval 曲线 ~8.6-8.7 vs pilot ~7.1-7.2（差 1.6 nats）。交叉诊断（scripts/cross_eval_check.py，同一 eval 种子下互换模型）证明**是 eval 腐蚀种子噪声而非模型差异**：同一模型换 eval 种子 7.05↔8.67；同一 eval 种子下 **s1-10200 (7.049) 好于 pilot-30000 (7.420)**。根因：训练期 eval 64 块×每块 1 个 t，loglinear 的 dsigma 权重跨 t 变化 3 个数量级（t≈0.5 权重 2 vs t≈0.97 权重 33），(chunk,t) 配对差异主导均值。**正式评估用 frozen manifest 的固定 σ 网格，不受影响**；formal run 里的 plateau 检测仅信息性（N 已冻结）。勿据此改任何东西

---

## 2026-10-02（Day 5 重启后）完整性审计 ✅ + 10201 审计定案 + v4.2 P1 启动准备

- [x] **重启后完整性审计全部通过**：Stage-4 快照 SHA 24/24 + pilot 快照 11/11；23 个 canonical 结果文件与登记 hash 逐一 MATCH；manifest 500 行、hash `1897bd14...` 与冻结记录一致（未重新生成）；18/18 评估 JSON 全部 parse；8/8 formal checkpoint torch.load 成功（内部 step 精确 1020/5100/10200）；关键数值全一致；verdict 仍为 **B_DIAGNOSTIC_ONLY（永久固定，不允许用后续分析回头改）**。无缺失、无截断、无重启损坏
- [x] **10201 审计定案**（`reports/audit_optimizer_step_count_10201.md`）：v4.1 循环条件 `step < n_iters+1`（继承 run_train.py:145 惯例）→ **真实发生了 10201 次 optimizer.step()，不是 logging 假象**；但计数器 1-based，`checkpoint_{k}` = 恰好 k 次 update 后的状态 → 1020/5100/10200 全部精确，第 10201 次 update 从未落盘；仅 metadata 计数多算 1 步（0.01%）。**不重跑、不修改 v4.1**
- [x] v4.2 P1 代码就绪：`training/vanilla.py` 精确 N 步循环（`step < n_iters`）+ metadata `loop_semantics: exact_N` + PROTOCOL_VERSION=v4.2；eval 三脚本 protocol_version=v4.2（eval 配置仍读 v4.1 yaml 的 frozen manifest 指向，manifest 本身不变）；`scripts/run_p1_evals.sh`（Tier1 EMA-CPI → Tier2 EMA-OG/G1 → Tier3 raw 全指标；输出 `results/p1_dense/` 独立目录不碰 v4.1；两条只读预检：manifest hash 不变 + **P1 checkpoint_1020 与 v4.1 formal 同 seed 逐字节一致 = 训练复现性门**）
- [x] **P1 preflight audit**（`reports/p1_preflight_audit.md`）：scheduler 一致性通过——`losses.py:67` closed-form `g['lr']=lr*min(step/warmup,1)` 无 n_iters 项；2500 vs 10200 逐 step 位级 0 差异；v4.1 formal 日志 101+101 点与 closed-form 对拍全一致；check_scheduler 回归过。**措辞更正：warmup 内 LR 线性爬坡（step 1 = 1.2e-8），step≥2500 才恒定 3e-5**
- [x] gate 2 升级为两级（run_p1_evals.sh + `scripts/compare_checkpoints.py`）：先 `cmp` 逐字节；不一致则 tensor-level diff（model/EMA/optimizer/scaler；RNG 未序列化 → 权重一致即隐含 corruption 流一致），max rel < 1e-5 = kernel 噪声 PASS-CAVEAT，≥1e-5 或结构不一致 = 硬停
- [ ] P1 训练 2 seeds（n_iters=2500 精确步、save_at=[50,100,250,500,750,1020,2500]、seeds (1,1,1)/(2,2,2)、lr=3e-5，~18min/seed）已交用户 tmux → 完成后跑 run_p1_evals.sh（总 ~23h GPU）

---

## 2026-10-01（Day 4 早）研究项目开工：阶段 0 + 阶段 1 完成 ✅（旧协议版，已被上面 v4.1 版取代）

### G0 正确性门（`test_cpi.py`，9/9 CPU 单元测试过）
- [x] 条件概率提取公式（RADD 桥）：absorbing 下 mask 位置 `score[v] = log r(t) + log p̂(v|C)`（从 Absorbing.score_entropy 最优解 exp(score)=ratio·p 推出）→ `p̂ = softmax(score[i,:D-1])`，时间标量 r(t)=1/(e^σ−1) 是加性常数、softmax 中消掉；MASK 条目必须排除（transformer.py:288 scatter 会把 score[i,MASK] 置 0，不是有效 logit）
- [x] 玩具测试：兼容联合分布 → δ_swap=0 精确；故意不兼容 → ln(3.2) 手算值；batch 化=逐样本；对抗性 MASK logit=1e9 保证排除逻辑真被测试
- [x] 严格逐 token 解码记账（CPU）：每步 mask 数 -1、可见位置不动、路径 logp 正确、4 顺序位置选择正确、categorical 采样落在干净词表

### 阶段 1：Data CPI 评估器 + pretrained 基线（已冻结）
- [x] `cpi_eval.py`（提取 + delta_swap 3-forward + 严格解码器 + 批量评估 + 分桶汇总）、`run_cpi_baseline.py`（协议：t~U[eps,1]→σ→腐蚀→无放回采样 (i,j) 对；样本冻结到 `results/cpi_samples_pretrained.pt`，后续 checkpoint 一律 `--load_samples` 复用做 paired 对比）
- [x] **冻结数字（N=998，seed=0）：CPI = 0.0393 ± 0.0038；δ 均值 +0.0044 ± 0.0040（≈0，模型基本无偏）；local CE = 3.469 ± 0.052**
- [x] 早期观察：CPI 随 σ 单调升（0.019→0.067，4 桶）也随 mask 数单调升（0.019→0.066）——mask 越多上下文越少、条件越噪声，方向符合直觉（报告可用，但样本少不作声明）
- [x] 踩坑记录：① `graph.sample_transition(x0, sigma[None,None])` sigma 变 3D 会把 x_t 广播成 [1,1,L]（正确是 `sigma[:,None]`，eval_ppl.py 同款）② 3 个状态的 logits+logp 同活 → 峰值 8.13GB 超上限（WDDM 危险区）→ 改为用完即 del，降到 **3.20GB** ③ `torch.randperm` 不接受 cuda generator 且走全局 CPU RNG → main 里必须 `torch.manual_seed` ④ strict decode 里 `torch.zeros(B)` 默认 fp32 把 fp64 logp 降精度（dtype 跟随 logp 修复）
- [ ] 阶段 2：GPU smoke 严格解码（真实模型 + infilling 提示）→ 4 顺序 × ~100 样本测 OrderGap + Rollout CPI

---

## 2026-09-30（Day 3 晚）

### 训练 smoke test 首跑：CUDA OOM + Windows 侧 explorer.exe 崩溃（已诊断，待重跑）
- [x] 跑两次 `run_train.py`（batch 8 / n_iters 300 / wikitext103），均在 **step 1 崩**：`torch.OutOfMemoryError: Tried to allocate 786.00 MiB`，位于 `model/transformer.py:292`（vocab logits = batch 8 × 1024 × 50257 × fp16 ≈ 786MB）
- [x] 崩溃模式确认：step 0 train/eval loss 正常（~1.11e4 nats）→ backward 后显存碎片化 → step 1 的 786MB 连续分配失败（0 bytes free，PyTorch 报已分配 10.12GB > 7.96GB 卡容量，且 "17179869184 GiB in use" 是 WDDM 驱动上报的垃圾值）
- [x] 静态开销账：模型 678MB + 梯度 678MB + AdamW 1.36GB + EMA 678MB ≈ 3.4GB，batch 8 动态部分超预算
- [x] **explorer.exe fail-fast 的定性**：5060 Ti 同时是 Windows 显示 GPU，WSL2 CUDA 走 WDDM 驱动；CUDA OOM 破坏驱动状态 → 桌面图形栈不稳 → explorer（GPU 合成的 shell）崩溃。属 Windows/驱动侧症状，**不是训练代码 bug**。缓解：更新 NVIDIA 驱动、别把显存吃满（留 1-2GB 给桌面）、完全重启 Windows（`wsl --shutdown` 不够）
- [x] 修复方案确定（最终版）：`training.batch_size=4 training.accum=2`（有效 batch 8）+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `eval.batch_size=2`。**关键约束：data.py:203-206 断言 batch_size 和 eval.batch_size 必须被 ngpus×accum 整除**——所以 batch=2/accum=4（2%4≠0）和另一 LLM 的 batch=1/accum=8（1%8≠0）都非法，batch=4/accum=2 是保持 effective 8 的合规选择（峰值预计 ~6GB）。另一 LLM 建议 batch=1/accum=8 已否掉：batch 8 的 step 0 已完整成功（峰值≈7.9GB）→ batch 4 余量足够
- [x] 顺手确认：① accum 是标准梯度累积（losses.py:92 `loss/accum`、每 micro backward、accum_iter==accum 才 optimizer.step+EMA+step+=1）→ effective batch = 2×4 = 8；② repo 有 final checkpoint 逻辑（run_train.py:178-183，step==num_train_steps 时存 checkpoint_0.pth）
- [x] smoke test 通过标准冻结（5 条）：① 300 steps 跑完 ② loss 全程 finite 无 NaN/Inf ③ step 100/200 eval 正常完成 ④ 显存不逐 step 单调增长 ⑤ loss 稳定/缓慢下降（warmup 2500 → 300 步 lr 仅 ~3.6e-5，只期望降 1-3%，不要求大幅下降）
- [x] **重要发现：训练入口是 `train.py` 不是 `run_train.py`！** run_train.py 是纯库模块（无 `__main__`，直接跑 = import 完 exit 0 静默退出，我之前两次"启动失败无输出"就是这个原因）。证据：21:57 崩溃运行的 `.hydra/hydra.yaml` 里 `job.name: train`。`train.py` 在 git 索引里（HEAD/upstream 都有，1555 字节）但工作区被删了（git status 却没报 D——原因待查，可能与你 shell 历史的 `git update` 别名或误删有关）。已从 upstream 恢复并验证（py_compile ✓、`--cfg hydra` 组合 ✓）。注意 train.py 把训练包在 try/except 里（失败也 exit 0），所以**退出码不能当验收信号，必须看日志内容**
- [x] 附带确认：`python run_train.py` 时唯一输出 FutureWarning 来自 `model/fused_add_dropout_scale.py` 三个 `@torch.jit.script`（torch 2.14 弃用，无害，以后可清）
- [x] **smoke test：300 steps 全部通过 ✅ Day 3 完成**（batch=4/accum=2/eval=2，22:39:03→22:41:00，~2.2 steps/s）。5 条验收全过：① 300 steps ✓ ② loss 全程 finite 零 NaN（grep 命中 4 条是 huggingface.co/.../info 的 "inf" 子串）✓ ③ eval@0/100/200/300 ✓（1.117e4→1.071e4→9.70e3→7.92e3）④ 显存 7.75-7.82GB 平台、无单调增长 ✓（峰值 7821/8151 MiB = 95%，紧但稳）⑤ loss 下降 ✓（train 1.091e4→8.53e3 = -22%；eval -29%）
- [x] **resume 验证通过 ✅**（load_dir=223903 + 把 checkpoint_0.pth 拷入 checkpoints-meta）："Starting training loop at step 301"（全新训练是 step 0）+ 10 个 backward/optimizer 步 + **loss 连续**（8.53e3@300 → 7.64e3@310，趋势无缝衔接，证明 model/optimizer/EMA 三者恢复正确）
- [x] resume 路径实测踩出 3 个 repo 坑并已修复：① train.py load_dir 分支引用 `cfg.work_dir`，但 configs/config.yaml 从不定义 work_dir（运行时注入），.hydra/config.yaml 里没有 → 修复 `work_dir = os.path.abspath(load_dir)`（先捕获 load_dir 再替换 cfg）② utils.py `torch.load` 缺 `weights_only=False`，torch 2.14 默认 True 拒载 EMA 自定义类 → 已加（Day 2 记忆标记的疑点，实测实锤）③（设计事实）resume 只读 checkpoints-meta/checkpoint.pth；checkpoints/checkpoint_*.pth 需手动拷入 meta；scaler 不入 checkpoint（丢失只重头缩放）；load_state_dict strict=False
- [x] 清理完成（用户授权）：215716/215953（5.4GB）+ 223642 + exp_local/openwebtext 下 5 个失败尝试 junk 目录 + tmux 调试日志；磁盘 26G→20G；223903 已还原为原始状态（checkpoint_0.pth=step-300，meta=step-1，config 已还原 n_iters=300）
- [x] **Day 4 关键建议**：`training.snapshot_freq_for_preemption` 从 100000 降到 50~100——否则 meta 只在 step 0 写过一次，SFT 中途崩了只能从 step 0 恢复；续训传参用 `+load_dir=`（config 无此键，裸 `load_dir=` 被 hydra 拒）；默认 `data.train=openwebtext`，不带 override 跑会把 junk 目录建到 exp_local/openwebtext/
- [ ] Day 4：SFT（从 louaaron/sedd-small 预训练权重继续训练 + eval_ppl.py 1000-t 评估）



### 已完成（调试大工程 🐛）
- [x] `model/transformer.py`: flash-attn import 改 try/except + SDPA fallback
- [x] `model/rotary.py`: torchscript rotary 在 torch 2.14 崩溃 → 改纯 torch 实现（数学与 flash-attn 2.2.2 triton kernel 逐项核对一致）
- [x] `configs/config.yaml`: submitit_slurm → basic launcher；ngpus 8→1；batch_size 512→16；删掉 SLURM 专属键
- [x] `data.py`: `load_dataset("wikitext"...)` → `"Salesforce/wikitext"`（datasets 5.x + hf-hub 1.33 兼容）
- [x] 验证链: checkpoint 权重 131/131 与 HF 一致 → score_entropy 玩具用例单元测试通过 → wikitext103 PPL 从 4657（坏）修到 ~40（论文 v3 Table 1: SEDD-small Absorb **≤ 40.62**, GPT-2 small 41.60）→ 实现正确性确认 ✅
- [x] 无条件生成 ✅ 输出连贯英文（电影评论风格）
- [x] 条件生成（infilling）✅ 前后缀钉住、中间自由生长

### 关键教训（debug 复盘）
1. **SDPA 形状坑**: `scaled_dot_product_attention` 要 `[B,H,S,D]`，不是 `[B,S,H,D]`——形状对得上、数值量级正常，但语义全错（1024 个"头"）。损失 4-8 nats 看起来"正常"，对照论文 PPL 上界（≤43.14，PPL 4657 明显越界）才暴露。教训：验证要查一手来源的基准数字。
2. **torch 2.14 的 TorchScript** 不兼容 2024 年的索引写法（`vector::_M_range_check` 报错）→ rotary 纯 torch 化。
3. **损失符号**: Score Entropy 理论最优值是**正数 ≈3.3 nats/位置**（= ln PPL），不是负的。
4. **uv venv 默认不装 pip**，检查依赖要用 `.venv/bin/python -c "import x"`。
5. **datasets 5.x 与 hf-hub 1.33 的 URI 解析不兼容**（裸 repo id 报错）→ 用带命名空间 id。

### 待办（接下来几天）
- [x] **Day 3 第一件事：严格核对评估协议（论文 v3 Appendix C.6）——已完成 ✅**
  - [x] 1000 个随机 timestep 的 MC 估计（论文原话 "We randomly sample with 1000 timesteps to Monte Carlo estimate our likelihoods"）→ 已写 `eval_ppl.py` 严格实现
  - [x] invertible tokenizer：byte-level GPT-2 ✓（data.py 用 GPT2TokenizerFast）
  - [x] WikiText103 test set ✓（仅取了前 16 块 = **pilot estimate，非完整复现**）
  - [x] unconditional、无 sliding window ✓（非重叠 1024 分块）
  - [x] schedule ε = 1e-3 ✓
  - [x] prior KL 项（Theorem 3.6/Eq.9）：p_base = MASK + 泄漏（C.1），KL ≈ ε·log(d−1) ≈ 0.011 nats → 已加回
  - [x] **16-block paper-protocol pilot: PPL = 40.19**（DWDSE 3.683 + KL 0.011；16 块 × 1000 t）
- [x] **冻结状态：Baseline implementation FROZEN ✅ / 16-block reproduction FROZEN ✅ / Full-test reproduction = 可选最终验证（264 块 × 1000 t，以后有算力再跑）**
  - 结论表述（报告用）："Our 16-block estimate of 40.19 is very close to the paper's reported WikiText103 bound of ≤40.62 (SEDD-small Absorb, v3 Table 1; GPT-2 small = 41.60). Given the limited 16-block subset and Monte Carlo estimation, we regard this as strong evidence that our implementation reproduces the reported likelihood behavior, rather than claiming an improvement over the original model."
  - 统计口径：±0.283 是 block-to-block SD（文本难度差异），不是 estimator 的 MC uncertainty；16 块均值的 SEM = SD/√16 ≈ 0.071。此前 1-t 版 38.5 有更大的 timestep sampling variance，与 40.19 数值接近但不作显著性对比；**后续一律用 1000-t estimator**
- [ ] Day 3: 小规模训练 smoke test（wikitext103、ngpus=1、batch 8~16）
- [ ] Day 4: 实现 SFT（从预训练 SEDD 继续训练 + 测试集评估，用 eval_ppl.py 的 1000-t estimator）
- [ ] Day 5+: RL 路线调研与实现

### 已完成
- [x] 通读仓库全部源码（graph_lib / noise_lib / losses / sampling / model / data / train）
- [x] 第一课：SEDD 核心概念讲解（前向 CTMC、rate matrix、具体 score、Score Entropy、采样器）
- [x] 逐文件功能清单
- [x] 环境配置方案确定（uv + Python 3.12 + torch cu128）

---

## 2026-09-30（Day 2 上午）

### 已完成
- [x] uv 创建 .venv（Python 3.12.3）
- [x] 安装 torch 2.14.1+cu130，CUDA 可用（RTX 5060 Ti 验证通过）
- [x] 第二课：用大白话重讲 8 个概念板块（类比版）

### 发现的问题（下一步处理）
- [x] `.venv` 里只装了 torch，**其余依赖（einops/hydra/datasets/transformers 等）和 flash-attn 都还没装** → 用户已装齐
- [x] `model/transformer.py:8` 仍直接 `import flash_attn` → 已打补丁

### 待办（接下来几天）
- [x] Day 2 下午: 安装其余依赖 + flash-attn 兼容补丁（fallback 到 SDPA）
- [x] Day 2 下午: 下载 `louaaron/sedd-small` 权重，跑通 `run_sample.py` 生成文本
- [ ] Day 3: 理解 loss 数值路径：手动构造 batch，单步 forward + score_entropy 计算，与公式对齐
- [ ] Day 4: 小规模训练 smoke test（wikitext103、ngpus=1、batch 8~16、8GB 显存约束）
- [ ] Day 5: 实现 SFT（从预训练 SEDD 继续在目标数据集上训练，或加前缀条件）
- [ ] Day 6: SFT 测试集评估（perplexity + 生成样本质量）
- [ ] Day 7+: RL 路线调研（DDPO / Discrete Diffusion Reward Guidance / SEDD-RLHF），设计并实现一个最小 RL 训练循环
- [ ] 前沿论文调研：D3PM、MDLM、SEDD-RLHF 等（见笔记）

---

## 2026-10-02（Day 5 深夜）P1 启动 → C: 爆满崩溃 → 迁移 D: → 恢复审计 ✅

- [x] P1 s1/s2 在 tmux 启动：s1 (1,1,1) **完成**（2500 精确步、7 dense ckpt + meta 全落盘、零错误）；s2 (2,2,2) 在 **step 100** 处 C: 盘满硬崩溃——checkpoint_100.pth 截断 324MB/2.71GB（CORRUPT），无 checkpoints-meta → 无法 resume
- [x] 恢复审计（只读，全通过）：s1 8/8 ckpt torch.load + 内部 step 与文件名一致 + 全张量 finite PASS；s2 checkpoint_50 PASS；**Gate 2 s1 逐字节 PASS**（P1 s1 checkpoint_1020 == formal-s1 checkpoint_1020，位级复现 → recipe 完全确定）；manifest SHA 1897bd14… 不变；v4.1 注册哈希 24/24 + pilot 11/11 全 MATCH；崩溃后项目内零文件变动
- [x] WSL 已迁 D:：`D:\WSL\Ubuntu-24.04\ext4.vhdx`（87.2GB）；C: 87.3G free / D: 34.8G free；vhdx 真实余量 = D: 余量；HF/datasets cache 已随迁移在 D: 上，无需重定向；无 Docker
- [x] 保存逻辑确认：torch.save 直接写最终路径（无 .tmp/无 atomic）→ 无瞬时双倍空间；每 seed 8 个完整 ckpt = 21.7GB
- [x] s2 已按用户指令隔离+重跑完成（`p1-s2-110355`，2500 精确步/1138s，8/8 ckpt 完整性 PASS，step 1 loss 与崩溃前逐位一致）；**Gate 2 s2 逐字节 PASS**（checkpoint_1020 == formal-s2）→ P1 训练阶段两个 seed 全部完成，位级复现证据 2/2
- [x] smoke 垃圾已清（用户授权）：vanilla256-* 4 目录 + exp_local/wikitext103（Day-3 smoke），回收 ~13GB（WSL 用量 82G→69G）
- [x] **P1 评估完成（84/84，~4h，远快于预注册的 23h 保守估计）**：Gate 1 manifest hash PASS + Gate 2 两 seed checkpoint_1020 逐字节 PASS；84 JSON 全部 parse、per-sample 全 N=500
- [x] **P1 判读完成**（reports/p1_early_dynamics_analysis.md，§3.3 预注册口径）：CPI 变化定位 s1=(250,500) / s2=(500,750)（EMA，同窗口方向一致）→ **连续衰减非瞬时阶跃**，0→250 平台、250→1020 下降、1020→2500 平台；OrderGap 更早启动（0→50 已显著）且 s2 持续到 2500；signed δ 全程 ≈0（对称收缩非方向偏置）；pair-distance 下降由 d[1,4] 主导、σ=3 桶主导
- [x] **pre-RL Git 固化完成**：`pre-rl-compatibility-v1` tag @ de69bb9（171 文件：协议/报告/结果 JSON/manifest/figs 全入库；.gitignore 补 *.pt/.venv、manifest jsonl 例外）；main 已 push origin（当前 HEAD 6b9b110 含审计产出）
- [x] **RL 阶段规划（未训练）**：reports/repo_audit_summary.md（复用/新增组件清单）+ reports/rl_reverse_transition_analysis.md（log-score→reverse rate→τ-leaping/staggered 两采样器的完整推导；policy=per-position categorical；rollout 与 logπ 公式必须同源）+ post_training_rl_plan_v0.1_DRAFT.md（47 节，DRAFT 未冻结）
- [x] 关键结论：model 输出 log-score（采样时 exp）；CPI 口径 clean_log_probs=log_softmax(score[...,:D-1])；Euler=τ-leaping 一阶近似（可能负概率）、analytic=staggered-score 一步转移；RL 起点=formal-s1 EMA-10200 + fresh optimizer；reward=span 位置级 token acc；G=4 group-relative；LR probe 3e-6 优先；pilot save_at 0/50/100/250/500
- [ ] 存储 blocker 依旧：D: ~19GB free，RL pilot（5 全量 ckpt ≈13.5GB + trajectory 缓存）前必须先清理/compact/迁盘
- [x] **RL plan v0.2 DRAFT 完成**（external review 11 条修正全纳入）：§14 policy=w/Σw 同源口径 + §14.1 validity gate（Euler 负权重→取消候选资格）；§18 reward=M0 初始 mask 口径 + G 共享初始腐蚀；§22 mean-centered advantage；§25 uniform timestep MC target（unbiased 声明限定）；§26 PPO min 形式 + per-position ratio（RL-1 joint on-policy 无 clip）；§13 EMA num_updates reset + raw primary；§31/§43 轨迹禁止全量落盘 + checkpoint 精简（pilot <10GB、前置 free disk ≥30GB）；§34 单测 22 项
- [x] ⚠️ external review 的 P1 OrderGap "0→50 initial increase" 表述与 frozen 数据方向相反（数据为 −0.03 nats 下降），v0.2 §2 按数据写并 flag（§46.10 待与 review 确认）
- [x] **review 第二轮定案已入 v0.2**：analytic=primary policy（Euler 降级 debug baseline）、K=1、|M0|=0 才重采样、constant LR（无 warmup，避免 confound）、执行顺序 A-J 写入 §45
- [x] **rl/transition.py + rl/reward.py + 22 项单测已实现**（2ba3210）：policy=w/Σw 同源、validity gate、mean-centered advantage、PPO min 形式、M0 reward；单测已过 7 项（含 K=1 MC toy 解析对拍——期间修了我 toy 的 inverse-CDF 反号 bug 与 0.5 链式因子遗漏）
- [x] **负权重疑云已澄清（非 blocker）**：采样路径 score[MASK] = exp(0) = **1**（scatter 置 0 后过 exp，非 0）→ stag_MASK = e^{dσ} + (1−e^{dσ})·Σscore；完整 analytic rollout 实测（真实 σ,dσ 匹配网格、SFT s1 EMA-10200、128 步）：**全程零负权重、validity 全 PASS**，stay 频率 ~0.99→末步 0.17。理论负值条件 Σscore > e^{dσ}/(e^{dσ}−1) 在真实轨迹不满足（Σscore ≈ 1/(e^σ−1)）。此前"恒为负"结论错误已修正；gate 保留防御语义（测试 test_validity_gate_rejects_negative_stag）
- [x] **24 个 RL 单测全过**（含 sampler factorization、frequency 对拍、K=1 MC 解析对拍、PPO clip 手算、M0 reward、G 共享腐蚀）
- [ ] 待用户 GPU 空闲后：跑探针 → 定 policy 口径 → 修单测 → 128vs1024 gate → RL-G0 → RL-1 smoke → LR probe

- [ ] P2a 前必须先解决长期存储（D: 现剩 ~19GB，openwebtext 数据集 15-40GB 放不下；可选 Optimize-VHD 回收 ~10G slack / 清 lrprobe+pilot-132632 ~7.8G / vhdx 迁 E:/F:）

## 2026-10-03（Day 6）RL-1 smoke 重启：model-load test FAIL——OOM 异常状态未随 WSL 重启消除

- [x] 状态恢复确认：HEAD 3f88959（memory hardening 已 commit/push）、tag pre-rl-compatibility-v1 在、frozen recipe 确认（config vanilla_256.yaml rl 默认 = smoke 配方：lr 3e-6 constant / G=4 / P=4 / pool=64 / 128 steps / 150 步 / chunk 2 / init=formal-vanilla-s1-191414 checkpoint_10200 EMA）
- [x] 诊断脚本 `scripts/rl1_model_load_test.py` 新增（复刻 training/rl.py 初始化序列，未 commit）
- [x] **model-load test FAIL**：模型参数 648.6MB 上卡后，EMA shadow 第一次 clone（148.00 MiB）OOM，allocator 报 6.15GB free；torch OOM 消息 non-PyTorch memory = **17179869184.00 GiB（2^64 回绕）**——与重启前异常同签名（上轮 rl1-smoke-231852 日志仅 2 行即死，同一点）
- [x] **新证据（dmesg）**：`dxgvmb_send_create_allocation failed ffffffb5` + `dxgkio_create_allocation: Ioctl failed: -75(EOVERFLOW)`，4 条时间戳精确对应本次失败尝试 → 失效在 **dxgkrnl↔Windows WDDM 边界**，Windows 侧 driver 记账状态损坏（`wsl --shutdown` 只重置 Linux 侧）
- [x] **阈值探针 PASS**：裸进程 648MB(2MB×324) + 148MB 全部成功 → 非容量/VA 阈值，model-load 大量异构分配 pattern 才触发（推测：Windows 侧 per-process allocation 记账溢出，仅推测）
- [x] 未启动 150-step smoke（按 OOM 规则停止）。候选出路已裁定：**① Windows 重启**（确定重置，已执行见下节）② 单次试 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（减少 create_allocation 次数，重启后 Step 3 情况 A 再做）③ 驱动 591.86 Blackwell+WDDM bug 排查（仅推测，重启后仍 FAIL 才进入）

## 2026-10-03（Day 6 深夜）裁定：方案 A——完整 Windows Restart

- [x] 重启前检查：nvidia-smi 正常（5060 Ti、driver 591.86、632MiB/8GB、无 GPU 进程）；无 python/train 进程；无 tmux 会话 → 无需要保存的运行中状态
- [x] **方案 A 已执行**：Windows Restart（`shutdown /r`，非 `wsl --shutdown`）。目标：重置 dxgkrnl↔WDDM 边界的 Windows 侧 allocation 记账状态
- [x] **重启后执行顺序（严格）**：
  1. [x] Step 1 基础 CUDA 检查：nvidia-smi 正常（5060 Ti、driver 591.86、671MiB/8151MiB、无 GPU 进程）；torch 2.14.1+cu130 CUDA=True、allocated 0.0GB、reserved 0.0GB、free 6.83GB/total 7.96GB
  2. [x] Step 2 默认 allocator model-load：**PASS**（`MODEL_LOAD_TEST_PASS`）；EMA shadow 完整构建（130 shadow、0 mismatch）、148MB clone 成功、step=10200；post-load allocated=680.4MB reserved=734.0MB free=6581.9MB peak_alloc=1360.2MB peak_reserved=1442.8MB；512MB 测试 PASS；forward smoke finite=True；dmesg 前后均无 dxg EOVERFLOW（基线仅 boot 时 dxgkrnl 注册行）
  3. [x] Step 3 分支：**情况 A**。expandable_segments:True 全新进程 model-load 也 **PASS**（无 unsupported/ignored warning）：post-load allocated=678.8MB reserved=696.3MB free=6619.7MB peak_alloc=1357.0MB peak_reserved=1367.3MB → torch 2.14 支持，reserved 略低于默认 allocator
  4. [x] Step 4 正式 RL-1 smoke 条件 ①—⑤ 全部满足 → 等用户 go-ahead 后从 **step 0** 重跑 150-step RL-1 smoke（不 resume 之前中断 run）；expandable_segments 若启用只记为 allocator 工程设置（入 metadata，非 algorithm change）
- [ ] **scientific recipe 不变**：SFT seed1 EMA checkpoint_10200 init / analytic predictor / 128 reverse steps / G=4 / K=1 / pure on-policy REINFORCE / group mean-centered advantage / M0-only reconstruction reward / constant LR / 当前 sigma-safe PG rule / 150 steps；内存工程优化保留（f32 one_hot、chunk=2、cleanup fixes）；expandable_segments 只算 allocator 工程设置，必须记入 metadata，不描述为 algorithm change
- [ ] **新 clean run 若再 OOM**：第一次 OOM 立即停止，不自动重启；保存 optimizer step / rollout|recompute|backward|eval 阶段 / requested / allocated / reserved / max allocated / max reserved / mem_get_info / nvidia-smi / dmesg dxg lines / |M0| / rollout index / timestep J / sigma 后回报

## 2026-10-03（凌晨，重启后）RL-1 smoke clean run：rl1-smoke-022319（进行中）

- [x] 用户 go-ahead 后启动：tmux `rl1-smoke` / PID 1875 / work_dir `exp_local/regime_a/rl1-smoke-022319` / HEAD 3f88959 / `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；allocator 设置已写入 work_dir 的 `allocator_config.json`（分类：memory/allocator engineering configuration，明确 not algorithmic change / not scientific intervention）；从 step 0 全新跑、无 resume；recipe 全部来自 config 默认（lr 3e-6/G=4/P=4/pool=64/128 步/chunk 2/init=formal-vanilla-s1-191414 checkpoint_10200 EMA/analytic/K=1/σ-gate 0.05）
- [x] **step 1 全链路 PASS**（rollout→recompute→backward→optimizer）：reward=0.1290±0.1061、zvg=0/4、grad_norm=0.53、loss=0.0631、0.041 steps/s（≈24.4 s/step → 150 步预计 ~65-75min + 3 次 eval）、script vram(max_alloc)=4.29GB、nvidia-smi 5345MiB used/2547MiB free、dmesg 0 条 dxg 错误
- [x] **150 步全部跑完，零 OOM，dxg_errors=0**（3456s ≈ 57.6min，0.043 steps/s ≈ 23 s/step）：
  - reward：无趋势、在噪声内震荡（logged mean 0.211，首 6 点 0.222 vs 末 5 点 0.226，final 0.183±0.264；per-step std 0.06–0.26）
  - fixed eval：eval8_reward 0.1883→0.1837→0.1837 平；eval_loss(EMA) 7.0186→7.0130→7.0171（无 NLL 退化）
  - grad_norm：0.06–1.87（clip 1.0），均值 ~0.66，无发散；3/16 点触 clip
  - zvg（zero-variance group fraction）：cumulative 43/600 = **7.2%**，均匀增长
  - sigma-boundary：soft_neg_elements=20 / soft_steps=19（全部在 σ<0.05 软门区）；nan_rollouts=0、无硬门 RuntimeError → 真实轨迹无硬负权重，验证之前"负权重非 blocker"结论在 RL 训练全程成立
  - VRAM：max_allocated 4.40GB 平台 → step 50 首次 eval 后升至 5.84GB 平台并保持（无单调增长）；driver 层峰值 5345MiB；结束后 729MiB
  - 产物已验证：run_metadata.json + learning_curve.csv + eval_snapshot.pth（1.36GB，step=150、130 shadow、ema_num_updates=150）
  - advantage statistics 未逐 group 打点（仅有 reward std / zvg 代理）——若 review 需要可下轮加 logging
- [x] **Review 裁定 SMOKE=PASS**；不跑 2500-step probe / 不跑 500-step pilot；补 3 个 audit + 2 条短 LR probe（1e-6 / 1e-5，各 150 步；3e-6 复用 smoke 数据）
- [x] **Audit 完成**（scripts/rl1_post_smoke_audit.py，eval-only 未 commit）：
  - step0 fixed eval（smoke 同 seed 同代码）：reward=0.1979（smoke step50 0.1883 → 100 0.1837 → 150 0.1837，小幅下降但噪声内）；step0 NLL=7.0490（→7.0186→7.0130→7.0171，微升）
  - step0 greedy（audit-only 新代码，argmax 变体同 J）：0.2375 vs sampled 0.1979；greedy 不在 smoke 记录中，50/100/150 的 greedy 用 snapshot 离线补
  - sigma 口径修正：153,600 = 150 更新 × 8 chunk × 128 步的 **validity_check 调用数**（= 307,200 trajectory-steps = 3.95e12 elements）；soft_steps=19 = 软区失败 chunk-steps、soft_neg_elements=20 = 负权重元素总数（均只在 σ<0.05 尾段，即 step ≥112 区域）
  - J support：σ≥0.05 步 mean 117.6/128 = 91.9%；被截断 objective support mean 10.4/128 = **8.1%**（per-prompt 7–16 步，全部在 σ→0 尾段）；instrumented 3072 prompt-steps 样本 0 负权重（与 smoke 全局率一致）
  - drift：‖θ150_raw−θ0‖/‖θ0‖ = **1.54e-4**；‖θ150_ema−θ0‖/‖θ0‖ = **1.44e-4**
- [x] training/rl.py 纯 logging 增强（用户 C 段允许清单内，objective/sampling/update 零改动，未 commit）：A mean/std/mean_abs、torch.cuda 全内存指标、fixed eval RAW(PRIMARY)+EMA(SECONDARY)（EMA NLL 路径逐位保持 smoke 口径：corruption_seed+1000 首次使用仍为 EMA pass）、metadata 加 allocator_env+git_head
- [x] **lrprobe-1e6 首轮 aborted + 已重启**：首轮（122325）step50 eval 发现 raw NLL 读数异常（RAW=25.78 vs EMA=7.04，eval8 RAW=0.1979=step0 → 权重没坏）；离线复现定位：run_eval raw pass 用的 seed+2000 corruption 流抽到 σ=0.002 极端低-sigma 样本（单块 loss=1009.6 → 64 块均值 22.84；seed+1000 流为 7.05）→ **定性为 high-variance / pathological realization（非错误数据、非模型退化）**；修复：raw/EMA 共用同一冻结 corruption realization（seed+1000，独立 generator 对象，配对比较，EMA 逐位保持 smoke 口径）；证据 /tmp/lrprobe-1e6-aborted-artifact.log；aborted 至 ~step60 后 kill 重启（lrprobe-1e6-125140），step1 逐位复现 0.1290±0.1061
- [x] 注：eval NLL 对 σ≈0 的 corruption 抽取高方差敏感（seed+1000 流最差块 σ=0.040/loss=73.3 也偏大）——**跨 run 比较必须用同一冻结 corruption realization**；修复属 logging，非 evaluation protocol / recipe 变化
- [x] 用户定案（最终 review 要求）：① raw/EMA NLL 用完全相同冻结 corruption realization ② seed+2000 记为 high-variance/pathological realization ③ **J support=91.9% / truncation=8.1% 必须进入 protocol revision** ④ 禁 CPI/OrderGap 选 LR ⑤ 三条 probe 完成后停止，统一回报后再定 LR
- [x] **lrprobe-1e6-125140 完成**（3493s，零 OOM）：raw NLL 修复验证成功（RAW 7.0510→7.0431→7.0424 vs EMA 7.0407→7.0335→7.0448，配对跟踪；step50 RAW≈step0 基线 7.0490）；eval8 RAW 0.1979→0.1979→0.1933（step0 0.1979 → 平）；zvg 38/600=6.3%；peak_alloc 5.84GB / peak_resv 7.08GB 平台稳定；step 1 逐位复现
- [x] **lrprobe-1e5-135019 已启动**（13:50，recipe 唯一差异 lr=1e-5）；step 1 逐位一致（0.1290±0.1061）→ 三条 run seed 流同源第三次实证
- [x] **lrprobe-1e5-135019 完成**（3489s，零 OOM）：eval8 RAW 0.1883→0.1798→0.1709（step150 raw<EMA 0.1798，1e-5 下 raw/EMA 首次可见分歧）；NLL RAW 7.0128→7.0272→7.0228（≤step0 7.0490）；zvg 45/600=7.5%；soft_neg 15/15；nan_rollouts=0
- [x] **LR review 离线收尾完成**（scripts/rl1_lr_review_offline.py，reports/lr_probe_summary.json；7/7 交叉验证与 logged 逐位吻合）：drift 线性标度（1e-6: 5.3e-5 / 3e-6: 1.5e-4 / 1e-5: 5.3e-4）；NLL 全 ≤ step0（无退化）；**greedy eval8 全 LR 均高于 step0 0.2375（3e-6: 0.2803 最大 / 1e-6: 0.2625 / 1e-5: 0.2622）→ 无采样噪声口径下可见真实 task learning 信号**；sampled eval8 受 gumbel-stable 分辨率限制（多数 per-prompt 与 step0 相同）
- [x] **统一 LR review 汇报已交付**：三条 150-step calibration 全部完成；按用户 E 段**停止**，等 LR review（不自动选 LR / 不跑 2500 / 不开 500-step pilot）；protocol revision 议题已列：J support 91.9% / truncation 8.1%、冻结 corruption realization、greedy 作为补充 eval 口径
- [x] **LR review 裁定：formal LR = 3e-6**（逐字表述入 v1.0 §0.1；不声称统计显著；LR calibration 终止，不再搜中间值）
- [x] **protocol v1.0 FROZEN 完成**（post_training_rl_plan_v1.0_FROZEN.md，SHA 13e97be7…，commit 4f855b5，8 文件 1149 行新增）：8 项修订全部入协议（LR=3e-6 / safe-region truncated PG 命名 / σ=0.05 经验边界语义 / NLL 冻结 corruption realization / RAW primary+EMA secondary+greedy supplementary / formal fixed eval = manifest idx 64–127 的 64 样本 @ 0/50/100/250/500 / allocator=memory engineering config / recipe 八项保持）；一致性检查：config rl 默认与 frozen recipe 逐项一致（lr 3e-6/G4/P4/pool64/128/chunk2/seed0/init s1-10200）、manifest SHA OK（1897bd14…）、working tree clean、D: 35GB ≥30GB 前置满足（pilot 启动前复核）
- [x] **Freeze review PASS + implementation review PASS**（25/25 单测实跑核对 → errata #1；formal eval acceptance 17/17：64 samples / 零 overlap / IDs 落盘 / 冻结 corruption / greedy deterministic / 权重零变化 / RNG 正确 / 无泄漏 / 无 dxg；commit 2c59a6f）
- [x] **500-step formal RL pilot 已启动**：`rlpilot-185545`（tmux `rlpilot`，18:55，HEAD 2c59a6f，expandable_segments:True，命令 rl.n_steps=500 rl.lr=3e-6 rl.name=rlpilot）；GPU 干净基线（1017MiB/1%）；D: 35GB
- [x] step0 严格顺序执行完成：RL init 验证（step10200）→ **step0 formal task eval PASS**（nll 7.0490 / sampled 0.2562 / greedy 0.3365，与 acceptance 基线逐位一致）→ **step0 CPI PASS**（cpi_abs=0.2871 ∈ v4.1 frozen 区间 0.283–0.289，manifest SHA OK）→ OrderGap 跑完后进 optimizer step 1
- [x] **500-step formal pilot 完成**（rlpilot-185545，14131s ≈ 3.9h，零 OOM/NaN，nan_rollouts=0，soft_neg 61/60，zvg 117/2000=5.85%，peak 7.03GB 无泄漏）：
  - formal eval 轨迹（64-sample 子集）：nll_raw 7.0490→7.0100→7.0069→7.0477→**7.0342**；sampled_raw 0.2562→0.2585→0.2573→0.2579→**0.2619**；greedy_raw 0.3365→0.3373→0.3293→0.3277→**0.3380**（非单调、幅度噪声级）
  - drift @500：raw 2.99e-4 / ema 2.75e-4（vs 150 步 1.54e-4，亚线性积累符合梯度抵消）
  - 产物全部验证：checkpoint_step500（model+ema+opt+scaler）、eval snapshots 50/100/250/500、5 个 eval JSON、metadata（protocol v1.0-rl1/allocator/git 全记录）
  - **⚠️ dxg 异常披露**：kernel 71402–71514s（≈22:10–22:12，step ~410–450 窗口）出现 **27 条 dxgkio_create_allocation EOVERFLOW(-75)**，112 秒窗口后自行消失；运行未受影响（无 crash/NaN/OOM，全部产物在窗口之后正常写出并验证通过）；疑似与并发 GPU 负载有关（用户游戏？待确认）——按 launch 规则属"dxg anomaly 但未造成 failure"，运行已自然完成
- [x] **post-run CPI/OrderGap 完成**（16 evals，commit 5be2fdf，dxg 无新增）：RAW CPI_abs 0.2871→0.2852、OG_raw 8.7475→8.7442 全程平（±0.003/±0.037 噪声级）；paired bootstrap（500 samples，B=10000）ΔCPI_abs −0.0016 CI[−0.0072,+0.0037]、ΔOG_raw −0.0033 CI[−0.032,+0.026] —— **CI 均含 0，无 sample-level 可检测变化**；sigma/pair_distance buckets 无系统性变化；§40 矩阵 → 最接近 **Scenario B**（Task≈stable + CPI≈stable + OG≈stable；候选解读：SFT 是主要 compatibility restructuring 阶段，short-horizon RL 基本保留 SFT 后结构）
- [x] deviation record：protocol/deviation_formal_rlpilot_dxg.md（27×EOVERFLOW @step410–450 112s 窗口，无 observed model-state corruption，root cause 未确认）
- [x] 科学汇报已交付（15 项）；**已停止，等 scientific review**：RL-2/GRPO、第二 seed、P2、semantic 均不动；RL-1 task signal 弱 → 优先按 estimator/credit-assignment 效率解释（纯 on-policy REINFORCE + K=1），review 后再定
