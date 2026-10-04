# Demo Asset Export Report（Phase B/C/E acceptance）

> 日期：2026-10-04。范围：Demo 资产导出（Phase B/C）与 Gradio Demo（Phase D）验收。

## Phase B：Example selection

- 15 个样本（9 representative + 6 curated case studies），规则与 reason 全部落盘于
  `demo_assets/examples.json`。
- Representative 规则：分位最近邻（SFT local_ce p10/50/90、pretrained delta_abs p10/50/90、
  SFT token_acc p80/50/20），tie-break manifest index 升序，去重保序。
- Case studies 明确标记 "curated"，非统计代表性。
- Semantic mismatch（cs6 s069）：**确认真实案例**（"article"→"story" approximately
  equivalent；"fall"→"spring" incorrect），人工离线标注，无 LLM judge。

## Phase C：Asset export acceptance（用户 10 项）

| # | 检查 | 结果 |
|---|---|---|
| 1 | sampled fixed-seed repeatability | ✅ 跨进程重导 3 样本 × 4 stages × 2 modes = **24/24 逐位一致**（scripts/verify_demo_trajectories.py） |
| 2 | greedy determinism | ✅ 同上（argmax 确定性 + 跨进程一致） |
| 3 | final trajectory state == exported final output | ✅ 15 文件 × 8 (stage,mode) 组合全部 final == 最后关键帧 |
| 4 | pair delta 与 formal delta 对拍 | ✅ **18/18 逐位一致**（单进程原子生成：pair 四项与 harmonized per_sample δ 来自同一批 logp 张量 → 构造保证；非复算对拍） |
| 5 | OG Q_by_path 来自正式结果 | ✅ 纯 JSON 提取自 harmonized OG 资产，零重算 |
| 6 | zero NaN | ✅ token 合法域 / sigma 有限全查 |
| 7 | zero invalid hard-region transitions | ✅ 硬区失败会 raise（导出零 raise）；gate 字段只有软区计数 |
| 8 | no new dxg anomaly | ✅ dmesg 计数 = 27（基线，无新增） |
| 9 | no backward | ✅ 全部 no_grad；脚本无 optimizer/backward 调用 |
| 10 | no model weight mutation | ✅ 权重 checksum 守卫（导出前后一致） |

## 对拍失败排查记录（重要）

- 初始 pair 对拍失败（s375 pretrained：−0.6094 vs 存储 −0.5625）触发暂停。
- 排查结论：历史 pretrained/vanilla formal CPI/OG 由 **84dc4b4 时代未提交的评估代码**生成，
  不可 bit-reproduce（详见 `protocol/errata_v4.2_eval_provenance.md`）。
- 用户裁定方案 1：Demo 主链统一用 current-code harmonized 复算值。
- 附加发现：bf16 GEMM 跨进程存在低概率数值漂移（6 进程 1 次偏离 ~3e-3），
  跨进程 bit-exact 无保证 → pair 四项改为**单进程原子生成**，对拍由构造保证。
- Harmonized 链值：pretrained CPI 0.3301 / OG 10.2246；SFT 0.2871 / 8.7475；
  RL-500 0.2852 / 8.7442。科学结论不变（SFT clear attenuation；RL-1 no detectable
  additional change）。

## Phase D/E：Demo acceptance（用户 10 项）

| # | 检查 | 结果 |
|---|---|---|
| 1 | 无 GPU 可完整启动 | ✅ HTTP 200；**torch 未 import**（gradio_api info 正常） |
| 2 | examples 全部可切换 | ✅ 15/15 render（Tab1/Tab2/Tab3 全部通过） |
| 3 | trajectories 全部可播放 | ✅ 15 文件 × 4 stages × 2 modes 关键帧结构完整（9–10 帧）且可渲染 |
| 4 | sampled repeatability 已离线验证 | ✅ 见上（24/24 逐位） |
| 5 | CPI pair decomposition 与 formal 对拍 | ✅ 18/18 逐位（构造保证） |
| 6 | Dashboard 数字与正式报告完全一致 | ✅ 程序化对拍通过（三阶段链 + RL task step0/500，容差 1e-3 显示精度） |
| 7 | Representative / Case Study 标签明确 | ✅ 下拉框与 examples.json 双处标注 |
| 8 | 数据 provenance 可查看 | ✅ Tab3/Tab4 页脚（中英双语）+ 资产内字段 + README/demo.md |
| 9 | scientific wording 不过度声称 | ✅ "approximately stable / weak improvement signal" / "no detectable additional change"，无 "significant" |
| 10 | README/demo.md 完整 | ✅ 9 节（purpose/panels/provenance/precomputed/launch/GPU/limitations/reward limitation/summary） |

## 输出文件

- 资产：`demo_assets/{examples.json, trajectories/*.json ×15, compatibility_examples/*.json ×47,
  metrics/harmonized/*.json ×7, metrics/formal_curves.json, vocab_decode.json}`
- 代码：`demo/{app.py, data_loader.py, components.py, plots.py}`
- 脚本：`scripts/export_demo_{examples,trajectories,compatibility,metrics}.py`、
  `scripts/verify_demo_trajectories.py`
- 文档：`README/demo.md`、`protocol/errata_v4.2_eval_provenance.md`、本报告
