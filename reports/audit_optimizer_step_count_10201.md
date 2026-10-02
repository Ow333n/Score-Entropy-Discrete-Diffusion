# 审计记录 — v4.1 formal Vanilla "实际 optimizer steps = 10201" 定性核查

- 日期：2026-10-02（机器重启后完整性审计通过当日）
- 状态：**定案**。v4.1 所有 checkpoint / 评估结果**不需要修改、不需要重跑**。
- 关联：Stage-4 报告审计注记 2（`reports/stage4_report/report.md`）；protocol v4.2 §3.1。

## 1. 问题

`run_metadata.json` 记录 `optimizer_steps=10201`；协议 N=10200；analysis checkpoint 内部
`step` 字段=10200。问："10201" 究竟是 0-based/inclusive logging 的循环迭代假象，
还是真实发生了 10201 次 optimizer update？

## 2. 结论（定案）

**真实发生了 10201 次 optimizer.step() 调用——不是 logging 假象。**
但 **checkpoint_10200 恰好是"10200 次 optimizer update 之后"的状态，与协议完全一致**；
多出的第 10201 次 update 只作用于从未落盘的最终内存状态。**无实质错误，不重跑。**

## 3. 证据

### 3.1 循环代码（v4.1 语义）

formal run 使用 v4.1 版 `training/vanilla.py`，循环条件为
`while state["step"] < n_iters + 1`（继承 repo 惯例 `run_train.py:145`
`while state['step'] < num_train_steps + 1`）。v4.2 起已改为
`while state["step"] < n_iters`（见 `training/vanilla.py` 内注释与 protocol v4.2 §3.1）。

循环体每次迭代（accum=1 → 每迭代恰好一次）依次执行：

1. `state["step"] += 1` —— 计数器先自增，**1-based**（= 已完成 optimizer steps 数）；
2. `optimize_fn(...)` → **optimizer.step()** —— 每迭代恰好一次；
3. `ema.update(...)`；日志 / eval / checkpoint 均以自增后的计数器值为准；
4. analysis checkpoint 在 `state["step"] in save_at_steps` 时保存（update 之后）。

n_iters=10200、条件 `step < 10201` → 迭代 1..10201 共 **10201 次**，即
**10201 次 optimizer.step()**。计数器终值 10201（10201 < 10201 为假 → 退出）。

### 3.2 日志证据（formal-vanilla-s1-191414 / s2-204215）

- `^step` 行共 429 = 409 条 train（step 1 + 25 的倍数至 10200）+ 20 条 eval（500 的倍数）✓
- 最后一条 train 日志为 `step 10200`（10201 % 25 ≠ 0 不打印——与"第 10201 次更新
  静默发生"一致）
- 收尾行：`完成: 10201 optimizer steps in ...` ✓（两 run 均如此）

### 3.3 checkpoint 内部证据（torch.load 实测）

- `checkpoint_1020/5100/10200.pth` 内部 `step` 字段 = 1020 / 5100 / 10200（精确）；
- `checkpoints-meta/checkpoint.pth` 内部 `step` = 10200
  （snapshot_freq=100 整除 10200 → meta 在第 10200 次 update 后落盘；
  10201 % 100 ≠ 0 → 第 10201 次 update 之后的状态从未落盘）。

### 3.4 计数语义

计数器为 1-based "已完成 optimizer steps 数"，`checkpoint_{k}` = 恰好 k 次
optimizer update 之后的状态。因此：

- checkpoint_1020  = 1020 次 update ✓（协议 ✓）
- checkpoint_5100  = 5100 次 update ✓（协议 ✓）
- checkpoint_10200 = 10200 次 update ✓（协议 ✓）

## 4. 影响评估

- **对 checkpoint / 评估结果：零影响。** 三个 analysis checkpoint 全部精确，无实质错误。
- 第 10201 次 update 的副作用（一次 EMA 更新 n=10201、decay≈0.99912；一次梯度清零；
  一次额外 corruption batch 消费）只存在于退出循环前一刻的内存状态，从未保存。
- 仅 metadata 计数类字段多算 1 步（偏离 1/10201 ≈ 0.01%）：
  `optimizer_steps=10201`（checkpoint 语义为 10200）、
  `unique_training_sequences_seen=326432`（=10201×32；10200 步应为 326400）、
  `unique_source_tokens_seen` / `model_forward_tokens` 同比例。
  这些字段已在 Stage-4 报告审计注记 2 记录在案，不影响任何分析结论。

## 5. 处置（按用户指令执行）

- 不重跑、不修改任何 v4.1 checkpoint / 结果 / 协议文件。
- v4.2 起循环语义修正为精确 N 步（`vanilla.py`: `while state["step"] < n_iters`）；
  run_metadata 新增 `loop_semantics: "exact_N"` 字段（protocol v4.2 §3.1 要求）。
- **复现性验证钩子（只读）**：P1 early-dynamics 使用相同 seeds (1,1,1)/(2,2,2)、
  相同 recipe、相同代码路径重训至 1020 步，其 checkpoint_1020 应与 v4.1 formal
  checkpoint_1020 **逐字节一致**（`scripts/run_p1_evals.sh` 预检，不一致则停止评估先排查）。
