# Protocol v1.0 Errata / Audit Record（append-only）

> 本文件为 `post_training_rl_plan_v1.0_FROZEN.md`（SHA 13e97be7…）的 append-only 勘误记录。
> 已 hash 的 v1.0 正文不被修改；所有澄清/修正以条目形式追加于此，按时间顺序编号。

## Errata #1（2026-10-03）：correctness test 计数 24 → 25

- **v1.0 原文**：§0.3 与 §34 写"24 项全过"。
- **勘误**：实际为 **25 个 test 函数，全部通过**（在 git HEAD bc479ac 上以
  `scripts/run_rl_tests.py` 实跑验证：`全部 25 个 RL 单测通过 ✅`，25 PASS / 0 FAIL）。
- **来源解释**：
  - v0.2 §34 定义 **22 个编号 correctness items**（协议条目粒度）；
  - 实现把这些 items 拆成 **25 个 test 函数**：`tests/test_rl_transition.py` 21 个 +
    `tests/test_rl_reward.py` 4 个（部分 item 覆盖多个 test 函数）；
  - 此前任务日志写"24 个 RL 单测全过"为陈旧计数；v1.0 freeze summary 沿用该数字，
    与 runner 实际口径差 1。
- **结论**：纯文档计数问题，**没有测试被增删**，无科学影响。协议含义（"§34 全部通过"）不变。
- 验证证据：本文件写入时在 bc479ac 上两次实跑 `scripts/run_rl_tests.py`，均输出
  "全部 25 个 RL 单测通过 ✅"，逐项列表 25 条 PASS 全部核对。
