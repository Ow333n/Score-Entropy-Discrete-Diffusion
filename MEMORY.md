# MEMORY.md — SEDD 大作业每日任务记录

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
