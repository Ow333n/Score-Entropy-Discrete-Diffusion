# Demo Asset Audit（Phase A）

> 日期：2026-10-04。目标：把已完成研究做成 4-Tab Gradio Demo（PRECOMPUTED MODE 优先）。
> 本审计只盘点与设计，不做 UI、不做任何训练。

---

## A. 已有可直接复用资产（无需任何重算）

### A1. Checkpoints（inference 用）

| 阶段 | 路径 | 权重 |
|---|---|---|
| Pretrained | HF `louaaron/sedd-small`（本地缓存，eval CLI 直接加载） | — |
| SFT formal s1 | `exp_local/regime_a/formal-vanilla-s1-191414/checkpoint_10200.pth` | EMA（RL init 同源）+ raw |
| SFT formal s2 | `exp_local/regime_a/formal-vanilla-s2-204215/checkpoint_10200.pth` | EMA + raw |
| RL-500 | `exp_local/regime_a/rlpilot-185545/checkpoint_step500.pth` | raw + EMA + optimizer |
| RL 中间 | 同目录 `eval_snapshot_step{50,100,250,500}.pth` | raw + EMA |
| P1 系列 | `exp_local/regime_a/p1-s1-020240/`、`p1-s2-110355/`（checkpoint_1020/5100 等） | raw + EMA |

### A2. Compatibility metrics（全部含 per-sample × 500，含 buckets）

| 数据 | 文件 | 关键值 |
|---|---|---|
| Pretrained CPI | `results/pretrained/cpi.json` | CPI_abs=0.3281, local_ce=4.2812, **per_sample{delta, delta_abs, local_ce, token_acc}×500** |
| Pretrained OG | `results/pretrained/order_gap.json` | OG_raw≈10.24, **per_sample{Q_by_path: l2r/r2l/random_0/1/2/confidence}** |
| SFT s1-10200 | `results/vanilla/cpi_s1_10200.json`、`og_s1_10200.json`、`g1_s1_10200.json` | CPI_abs=0.2891, OG=8.7557（与 step0 复算 0.2871/8.7475 差 ~2e-3/8e-3，G0 容忍内） |
| P1 dense 曲线 | `results/p1_dense/`（s1/s2 × {50,100,250,500,750,1020,2500} × raw/ema 的 cpi/og/g1） | **Dashboard Plot 1/2 的 SFT 段直接可用** |
| RL post-run | `results/rl_pilot/{cpi,og}_step{50,100,250,500}_{raw,ema}.json`（16 个） | per_sample + buckets 齐全 |
| RL step0 | `exp_local/regime_a/rlpilot-185545/step0_{cpi,order_gap}.json` | 同口径 |
| LR probe 汇总 | `reports/lr_probe_summary.json` | 三 LR 证据表 |

**重要发现**：CPI 的 `per_sample['delta']` 就是**逐 pair 的 δ_swap**（500 个 sample 各一个 pair 的四项组合值），且 `per_sample['local_ce']` 是**合法 per-sample masked-token NLL** —— Tab 1 的 per-example NLL、Tab 3 的 per-pair δ 数值**全部已有**，三阶段对齐（pretrained/SFT/RL-500 同一 manifest 同样本）。

### A3. Task metrics（RL 段）

| 数据 | 文件 |
|---|---|
| 5 个评估点（nll raw/ema、sampled64/greedy mean + per-sample 64 数组、gate stats） | `exp_local/regime_a/rlpilot-185545/eval_point_step{0,50,100,250,500}.json` |
| 训练曲线（reward/A 统计/显存，每 10 步） | 同目录 `learning_curve.csv` |
| run metadata / eval 子集 IDs | `run_metadata.json`、`eval_subset_ids.json` |

### A4. Frozen manifest（Demo 样本与 Tab 3 pair 的合法来源）

`manifests/regime_a_eval_v1.jsonl`（500 条：x0 / initial_state / sigma / span / i,j,a,b / paths{l2r,…}）+ SHA 侧车。

---

## B. 缺失、需要 inference-only export 的部分

| # | 缺失物 | 用途 | 导出方式（全部 inference-only，禁 backward） |
|---|---|---|---|
| 1 | **Trajectory 关键帧**（8–12 个 timestep 的 tokens / mask_positions / newly_revealed / sigma） | Tab 2 slider | 用冻结 `rollout_chunk` 同源代码 + 关键帧记录，3 阶段 × N 样本，固定 seed，deterministic |
| 2 | **最终重建文本**（sampled + greedy 的 final tokens）与 per-example M0 reward | Tab 1 三栏输出 | 同一次导出取 final 状态 + m0_reward；greedy 用 `rl/eval_formal.py` 的 argmax 变体（同 seed 同 J） |
| 3 | **Pair 四项分解**（log p(a|C)、log p(b|C,a)、log p(b|C)、log p(a|C,b)） | Tab 3 左右栏展示 | 逐字复用 frozen `compatibility/cpi.py evaluate_delta_swap_batch` 内部计算，仅额外捕获中间量；**导出的 δ 必须与 formal per_sample['delta'] 逐位对拍**（不是新 CPI 实现） |
| 4 | Semantic mismatch 定性案例（3–5 个，找不到就报"未找到"） | Tab 1 limitation 卡片 | 人工离线标注（不引入 LLM judge） |
| 5 | 三阶段对齐的 **curated 指标 JSON**（正式曲线只读副本） | Tab 4 图 + 卡片 | 纯 JSON 组装（从 A2/A3 文件），零 GPU |

**不需要**：任何 formal aggregate 的重算、任何新 scientific metric、任何训练。

---

## C. Demo 数据结构设计

```
demo_assets/
├── examples.json                    # 10–20 样本：manifest index、gt、span、initial corruption、
│                                    #   M0、selection_reason、per-stage {sampled_text, greedy_text,
│                                    #   m0_reward_s, m0_reward_g, local_ce(nll), delta_abs}
├── trajectories/
│   └── traj_{idx:03d}_{stage}.json  # stage ∈ {pretrained, sft, rl500}
│                                    # {sample_id, stage, seed, model_ref, ckpt_ref, export_git_head,
│                                    #  keyframes: [{k, step, sigma, tokens, mask_positions,
│                                    #               newly_revealed}]×10,
│                                    #  final_sampled, final_greedy, m0_reward_s, m0_reward_g}
├── compatibility_examples/
│   ├── pair_{n}.json                # {sample_id, context, token_a, token_b, per-stage:
│   │                                #   {order_ab: {logp_a_C, logp_b_Ca, sum_AB},
│   │                                #    order_ba: {logp_b_C, logp_a_Cb, sum_BA},
│   │                                #    delta, delta_formal_match: true}}   # 必须与 formal 逐位一致
│   └── og_example_{n}.json          # 从现有 OG JSON 提取：per-stage Q_by_path（6 paths）+ order_gap
├── metrics/
│   └── formal_curves.json           # {compatibility: {pretrained, sft:{s1/s2 系列}, rl:{0..500}},
│                                    #  rl_task: {step→{nll,sampled64,greedy}}, lr_probe: {...}}
└── figures/                         # 可选：静态 PNG 备份（第一版用 plots.py 实时渲染，可留空）
```

```
demo/
├── app.py            # Gradio Blocks，4 Tabs（+可选 Tab5），launch 入口
├── data_loader.py    # 只读 JSON（precomputed mode 不 import torch → 无 GPU 也能启动）
├── components.py     # 各 Tab 的 UI 构建
└── plots.py          # matplotlib 从 metrics/formal_curves.json 画图（分图，不混量纲）

scripts/
├── export_demo_examples.py        # Phase B：数据驱动选样（覆盖 easy/hard/SFT↑/RL≈/mismatch/fail）
├── export_demo_trajectories.py    # Phase C：关键帧 + 最终文本导出（含 determinism/NaN/长度/
│                                  #          final==saved output 验证）
├── export_demo_compatibility.py   # Phase C：pair 四项捕获（逐位对拍 formal δ）+ OG Q_by_path 提取
└── export_demo_metrics.py         # Phase C：formal_curves.json 组装（纯 JSON）
```

Demo-specific 逻辑**不进** `training/`、`rl/`（`rl/eval_formal.py` 的 greedy 变体只被 export 脚本 import，不改动）。

---

## D. 预计 GPU 时间（全部 inference-only，无 backward）

| 项 | 量 | 估算 |
|---|---|---|
| Trajectory 导出（20 样本 × 3 阶段 × {sampled+greedy}） | 30 chunk-rollouts（chunk=4，128 步） | ~2–3 min |
| Pair 四项捕获（5 pairs × 3 阶段 × 4 次 forward） | ~60 次 forward | ~2–3 min |
| 验证（determinism 重跑、对拍） | 少量重跑 | ~3–5 min |
| **合计** | | **~10 min**（GPU 空闲时集中执行） |

---

## E. 预计开发时间

| Phase | 内容 | 估算 |
|---|---|---|
| B | 选样 + examples.json（数据驱动 + 人工复核 reason） | 0.5–1 h |
| C | 三个 export 脚本 + 运行 + 验证 | 1–1.5 h（+10 min GPU） |
| D | Gradio 4 Tabs（app/data_loader/components/plots） | **3–5 h**（最大项） |
| E | acceptance 10 项 + 修复 | 0.5–1 h |
| — | README/demo.md | 0.5 h |
| **合计** | | **~6–9 h** |

---

## F. 文件树草案（新增）

```
demo/                        # Gradio 应用（4 文件）
demo_assets/                 # 预计算资产（examples/trajectories/compatibility_examples/metrics/figures）
scripts/export_demo_*.py     # 4 个 export 脚本
README/demo.md               # 启动说明 + provenance + limitations
reports/demo_asset_audit.md  # 本文件
```

不新增其他顶层目录；不动 `training/`、`rl/`、`protocol/`、`results/`。

---

## G. 风险与约束（写死）

1. **禁 cherry-pick**：选样按数据驱动标准（per-sample local_ce/token_acc/delta 的分布覆盖），覆盖"RL 不变/轻微变化/失败 case"；selection_reason 落盘可审计。
2. **数值纪律**：所有展示数值必须来自 A 节文件或 export 产物；pair 四项的 δ 必须与 formal per_sample 逐位一致；Tab 4 卡片数值与正式报告一致（acceptance 第 5/6/7 项）。
3. **措辞纪律**：RL-1 task = "approximately stable / weak improvement signal"；RL-1 compatibility = "no detectable change"；禁止"RL improves model significantly"。
4. **PRECOMPUTED MODE 优先**：第一版 Demo 不跑任何 live inference；LIVE MODE 为可选后置项，不做为 blocker。
5. **Trajectory 确定性**：固定 seed + 同一代码路径 → 重复导出逐位一致（acceptance 验证）。
