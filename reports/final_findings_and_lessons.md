# 最终研究结论与经验总结（Final Findings / Challenges / Lessons，中文版）

> 面向中国面试官 / 中国团队的汇报文档。所有指标为 current-code harmonized 口径
> （历史 early-eval 数值的 provenance 说明见 protocol/errata_v4.2_eval_provenance.md）。

## 一、核心 Findings（研究发现）

### F1. Vanilla SFT 明显降低 reveal-order sensitivity

- 预训练 → SFT：CPI_abs 0.3301 → 0.2871（约 −13%），OrderGap 10.2246 → 8.7475（约 −14%）
- 方向与量级和项目冻结的 v4.1 发现一致：监督训练信号本身会系统性地重塑模型对 token
  揭示顺序的依赖。
- 这是本项目最干净、最可复现的结论。

### F2. Short-horizon RL-1 基本保持 SFT 后的 compatibility 结构

- RL-500（lr=3e-6、safe-region/truncated PG、K=1、500 步）：
  CPI_abs 0.2871 → 0.2852，OrderGap 8.7475 → 8.7442；
  step500 vs step0 paired bootstrap CI 全部含 0 → 无可检测的进一步结构变化。
- 任务指标：NLL 7.0490 → 7.0342，sampled64 0.2562 → 0.2619，greedy 0.3365 → 0.3380
  —— approximately stable / weak improvement signal（近似平稳 / 弱提升信号），
  不表述为显著提升。
- 对应协议 §40 矩阵的 Scenario B：主要结构变化发生在 SFT 阶段；当前 weak-learning RL
  基本保留 SFT 后形成的结构。

### F3. K=1 → K=4 不能带来一致的 task-learning 改善（诊断实验）

- 匹配诊断（100 步、唯一变量 K、j_rng 隔离保证 rollout 随机流逐位相同）：
  K1 与 K4 的 reward 序列、参数漂移（1.61e-4 vs 1.62e-4）、zvg（同为 26/400）、
  compatibility 全部近似相同；K4 的 NLL/greedy 微幅优势与 sampled 的微幅劣势方向不一致。
- 判定 Scenario B：单纯提高 timestep sampling density 不是当前 weak task signal 的瓶颈；
  下一瓶颈候选在 objective-level credit assignment（reward / group-relative 机制）。

### F4. 工程可靠性结论（对本项目同样重要）

- 8GB WDDM 环境下完成了 500 步 RL 训练 + 全链路评估：expandable_segments 显存管理、
  f32 one-hot、chunk=2、phase 间显存清理的组合是稳定可行的；
- 25/25 correctness 单测、RL-G0 逐字段初始化验证、128-vs-1024 gate、LR probe、
  K-ablation 全部完成——研究链路的每个数值都有可追溯的验证。

## 二、Challenges（挑战与解决）

### C1. WSL2 / WDDM 的 dxgkrnl allocation 异常（最难的工程问题）

- 现象：CUDA 在 148MB 级别的小分配上 OOM，torch 报错中的空闲显存回绕成 2^64 量级；
  dmesg 出现 `dxgkio_create_allocation: Ioctl failed: -75 (EOVERFLOW)`。
- 排查路径：模型加载探针 → 阈值分配实验 → dmesg 时间戳对齐 → 定位到
  dxgkrnl↔Windows WDDM 边界的状态损坏；`wsl --shutdown` 无效，**完整 Windows Restart**
  才能重置。
- 结论与经验：Windows 侧 driver 记账状态不在 Linux 侧可控范围内；重启后先跑默认
  allocator 的因果验证（确认是重启还是 allocator 配置起作用），再决定是否加
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（只记作 memory/allocator
  engineering configuration，不当作算法改动）。
- 后续遗留：formal run 中出现过一次 27 条 EOVERFLOW 的 112 秒 transient 窗口，
  未造成任何可观测损坏，root cause 未确认（protocol/deviation_formal_rlpilot_dxg.md）。

### C2. 评估口径的一致性（provenance 教训）

- 历史 pretrained / Vanilla SFT formal 结果由当时**未提交的评估代码**产生，无法
  bit-level 复现（聚合差异 CPI ~2e-3、OrderGap ~8e-3，定性结论不变）。
- 教训：**评估代码与结果必须同时进 git**；跨阶段比较必须同一 harness。Demo 主链改为
  current-code harmonized 全链复算（单进程原子生成保证 pair 对拍逐位一致）。
- 附带发现：bf16 GEMM 存在跨进程低概率数值漂移（6 进程 1 次 ~3e-3）——对拍类验证
  要优先"同一进程内原子生成"而非跨进程复算。

### C3. 短 horizon 下 sampled 指标分辨率不足

- 小参数漂移（5e-5~5e-4）下 gumbel 采样轨迹高度重合（多数 per-prompt 输出与 step0
  完全相同），sampled reward 几乎测不出变化；引入 greedy（argmax 变体、同 seed 同 J）
  作为无采样噪声的补充任务指标，才看到 step0→150 的干净信号。
- 教训：离散轨迹的采样噪声比连续 RL 场景更隐蔽——评估体系需要 deterministic
  supplementary metric。

### C4. 8GB 显存下的 RL 工程化

- 128 步 rollout + K 个 timestep 的重算必须 sequential backward（每个 J 独立计算图、
  立即释放）；K=4 与 K=1 的 peak VRAM 完全相同（6.36/7.40GB），额外 wall-time 仅 ~6%。

## 三、Lessons（方法论沉淀）

1. **预注册 + gate 机制**：每一步（RL-G0、128 gate、LR probe、freeze review）都有
   明确的 PASS 条件和失败停止规则；实验没有因中间结果好看/难看而改过 recipe。
2. **科学表述纪律**：单 seed 结果只表述 sample-level uncertainty；"approximately
   stable / weak improvement signal"不等于"显著提升"；"no detectable change"不等于
   "没有变化"。
3. **数值可追溯**：manifest SHA、protocol SHA、allocator 配置、git HEAD、evaluation
   code path 全部落盘进 run metadata；Demo 的每个展示数字都能回溯到正式结果文件。
4. **诊断实验要匹配变量**：K-ablation 用独立 j_rng 保证 K=1 与 K=4 的 rollout 随机流
   逐位相同，才敢说差异来自 K 本身。
5. **面向面试的表达**：先讲"问题是什么、为什么值得研究"，再讲"我做了什么、发现了
   什么"，最后诚实说"哪里还不行、下一步怎么做"。

## 四、Future Work（统一口径，只保留三项）

1. **更强的 RL credit assignment**：group-relative / PPO-style per-position objective
   （当前 K-ablation 证据表明瓶颈不在 timestep sampling density）；
2. **多 seed + cross-task 验证**：验证 SFT compatibility attenuation 是否具有普遍性；
3. **更大模型与更多算力**：验证该现象是否随模型规模保持。
