# K=1 vs K=4 Diagnostic Ablation —— Preflight STOP Record

> 日期：2026-10-04 夜（unattended 任务）
> 状态：**PREFLIGHT FAIL → 按协议 STOP，未启动任何训练/smoke，未占用 GPU。**

## Preflight 结果

| Gate | 要求 | 实测 | 判定 |
|---|---|---|---|
| A. GPU 空闲 | 无 heavy GPU process | 8151MiB total / 1056MiB used / 6836MiB free / 0% util / 无 compute 进程 | ✅ PASS |
| B. System RAM | available ≥ 8 GB | **MemTotal 7699948 kB (7.3 GiB)、MemAvailable 6682940 kB (6.4 GiB)** | ❌ **FAIL** |
| C. Disk | ≥ 20 GB free | / 852GB free；D:（vhdx 宿主）22GB free | ✅ PASS |
| D. dxg baseline | 记录 | 27 条 EOVERFLOW（全部为 formal pilot 期间的既有记录，无新增） | ✅ PASS（记录） |

Swap：2.0Gi total / 482Mi used / 1.5Gi free（系统空闲，无 swap 压力）。

## 判定与执行

- 用户协议明确："如果 available RAM < 8 GB：STOP。不要为了跑实验大量 swap。"
- **available = 6.4 GiB < 8 GB → 触发 STOP**，未启动 smoke、未启动任何训练进程、未写任何 run 目录。
- 额外事实（供明天裁定）：本机 **MemTotal 仅 7.3 GiB** —— 8GB available 的门控在本机
  **物理上不可满足**（available 恒 < total）。当前 6.4 GiB available 为系统空闲状态下的
  健康水平（占 total 87%），swap 仅轻度使用。RL 训练的实测 CPU RAM 需求约 2–3 GB
  （模型/优化器状态在 GPU；CPU 侧为 dataset 缓存与 snapshot 搬运）。
- 未删除/未修改任何既有结果与 checkpoint；工作区无变动。

## 明天待裁定选项

1. 将 RAM 门控调整为机器可满足的阈值（如 available ≥ 5 GB 且 swap used < 1 GiB），
   批准后重跑 preflight → smoke → matched runs；
2. 维持原门控（则本机无法执行该诊断，需换机或扩 WSL 内存）；
3. 用户指定其他处理。

## 未做事项确认（协议 §15 合规）

未启动 RL-2/GRPO/PPO、未做第二 seed、未改 reward、未做 LR/K sweep、
未改 frozen protocol、未覆盖结果、未改 Demo 结论。GPU 已处于空闲状态。
