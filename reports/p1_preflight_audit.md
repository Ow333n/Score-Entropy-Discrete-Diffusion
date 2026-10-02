# P1 Preflight Audit — scheduler 一致性 + 硬性 gate 定义（protocol v4.2）

- 日期：2026-10-02（P1 训练启动前）
- 状态：**scheduler 核查通过，P1 放行**（训练命令不变，等待用户执行）。
- 关联：`reports/audit_optimizer_step_count_10201.md`（10201 审计）、`protocol/experiment_protocol_v4.2.md`。

## 1. Scheduler 一致性核查（用户要求的 P1 启动前最后一项）

### 1.1 问题（含我此前的措辞错误，已更正）

我之前说"warmup 2500 全覆盖，LR 恒 3e-5"**不准确**：scheduler 是 **linear warmup**，
step < 2500 时 LR = 3e-5 × step/2500 逐 step 线性爬坡（step 1 = 1.2e-8），只有
step ≥ 2500 才恒定 3e-5。本核查验证的正确命题是：

> **P1 的 step 1–2500 LR trajectory 与 v4.1 formal Vanilla 的前 2500 step 完全一致，
> 且 scheduler 不依赖 n_iters（10200 vs 2500）。**

### 1.2 代码路径

- `losses.py:67`（closed-form）：`g['lr'] = lr * np.minimum(step / warmup, 1.0)`。
  `step` 由训练循环传入 `state["step"]`（`training/vanilla.py:209`
  `optimize_fn(optimizer, scaler, score_model.parameters(), step=state["step"])`，
  该调用行 v4.2 修改中未动）。**scheduler 无任何 n_iters 项**；n_iters 只出现在
  循环终止条件。
- 文件时间戳：`losses.py`（09-30 16:57）与 `configs/vanilla_256.yaml`
  （10-01 13:04）均早于 formal s1 启动（19:14），且 losses.py 为 git 跟踪文件、
  formal run 的 git_commit=84dc4b4 与其一致 → **formal 与 P1 使用同一 scheduler 代码**。
- P1 命令行 override（n_iters=2500 / save_at_steps / lr=3e-5 / seeds / name）
  **不含 warmup**；`optim.warmup: 2500` 来自未修改的 config；lr=3e-5 与 v4.1 formal
  override 同值。

### 1.3 三项验证结果

| 验证 | 方法 | 结果 |
|---|---|---|
| ① n_iters 独立性（2500 vs 10200） | 同一 closed-form 代码路径，逐 step 1..2500 驱动 optimize_fn 并位级比较 | **0 步不一致**（2500/2500 位级相等） |
| ② v4.1 实际轨迹对拍 | 解析 s1/s2 train.log 全部 step≤2500 的 LR（各 101 点，log_freq=25 + step 1），与 closed-form 打印精度比对 | **s1 101/101、s2 101/101 完全一致** |
| ③ 回归（10200 vs 30000） | `scripts/check_scheduler.py`（Day 4 预注册核查 ①） | ✅ 通过：warmup 后位级恒定 3e-5、trajectory 与 n_iters 无关 |

附：v4.1 日志本身已含证据——step 1 lr=1.20e-08 = 3e-5/2500 精确、step 25 lr=3.00e-07
= 3e-5×25/2500 精确，与 closed-form 位级吻合。

### 1.4 结论

P1 的 step 1..2500 LR trajectory 与 v4.1 formal 前 2500 步**逐点位级一致**（同代码、
同 closed-form、同参数），scheduler 不依赖 n_iters。**放行 P1，训练命令不变。**

## 2. 训练结束后的两个硬性 preflight gate（最终定义）

### Gate 1：frozen manifest 未变（只读）

`manifests/regime_a_eval_v1.jsonl` SHA-256 必须仍为
`1897bd14bf92e61b2664a4950a3b60f14e09ac0d454d58d22e88815207f263d3`。
不一致 → 硬停，不允许重新生成 manifest。

### Gate 2：P1 checkpoint_1020 与 v4.1 formal 同 seed 复现检查（只读，两级）

1. **第一级：逐字节比较**（`cmp`）。一致 → **PASS**，直接进入评估。
2. **第二级（仅当逐字节不一致）**：`scripts/compare_checkpoints.py` 做 tensor-level
   diff——model parameters（131 keys）、EMA（decay/num_updates/shadow_params[130]）、
   optimizer state（exp_avg/exp_avg_sq/step）、scaler state、step。报告每个 section 的
   **max absolute / max relative difference**。RNG state 未序列化于 checkpoint（save_ckpt
   只存 model/ema/optimizer/scaler/step）→ N/A；corruption 流的分歧会经 loss/梯度进入
   权重，故权重一致即隐含 corruption 流一致（num_workers=0 + 固定 generator）。
3. **判读**：结构一致且全局 max rel < **1e-5** → 判定为 GPU kernel 非 bitwise
   确定性噪声（fp32 累积误差典型 ≤1e-6 rel；真实轨迹分歧在 1020 步内放大到 ≥1e-3 rel，
   阈值留足边际），**PASS-CAVEAT**，差异证据入档后继续。≥ 1e-5 或结构不一致 →
   **硬停排查**，不要继续烧 23 小时评估。

## 3. P1 执行纪律（预注册，结果前冻结，不因 Tier 1 结果修改）

- checkpoint grid / manifest / evaluation plan / training recipe **不再修改**：
  dense 点 [50, 100, 250, 500, 750, 1020, 2500]，n_iters=2500 精确步，
  seeds (1,1,1)/(2,2,2)，raw+EMA 双权重，同一 frozen manifest。
- 评估顺序：**Tier 1（EMA×CPI）→ Tier 2（EMA×OG 固定6路径+G1）→ Tier 3（raw 全指标）**，
  三个 Tier 全部预注册、全部执行；Tier 1 只回答一个问题——CPI attenuation 发生在
  step 0→50→100→250→500→750→1020→2500 的哪一段。不用 Tier 1 结果裁剪 Tier 2/3，
  不改变 primary/secondary 口径。
- step 0 继续复用冻结的 v4.1 pretrained 基线（cpi/og/g1_pretrained.json），不重新评估。
- **措辞固定**：当前只能说"attenuation 已经在第一个正式测量 checkpoint（1020）出现
  并随后平台"；不能提前说"变化在 1020 前瞬时完成"。P1 的目的就是补出时间分辨率。
  50 步已完成大部分下降 / 50→500 平滑下降 / EMA 滞后而 raw 更早变化——都是合法结果，
  如实报告。
- 本阶段不实现、不运行 PAPL / Swap / PAPL+Swap。
