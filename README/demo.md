# Demo：Post-Training Reveal-Order Compatibility in Masked Diffusion Language Models

## 1. Demo purpose

展示课程项目的完整研究链：masked diffusion language model（SEDD-small）如何从 MASK 状态
多步恢复文本；为什么 token reveal order 会影响 conditional predictions（CPI）；
不同完整 reveal path 的似然差异（OrderGap）；以及 Vanilla SFT 与 RL-1 post-training
如何改变这些结构。Demo 为研究展示/教学用途，不是聊天机器人。

## 2. Four panels

| Tab | 内容 |
|---|---|
| 1. Masked Diffusion Generation | 10–20 个固定样本：Ground Truth / Initial Corruption + Pretrained / SFT / RL-500 三阶段的 sampled / greedy 重建文本、M0 exact-token reward、Local masked-token CE |
| 2. Reveal Trajectory Viewer | 128-step reverse rollout 的 ~10 个关键帧（按 mask-count 变化选取）：reverse step / σ / mask count / 新揭示位置高亮 |
| 3. Reveal Order / Compatibility Lab | frozen manifest pair 的 A→B 与 B→A 四项分解（log p(a\|C)、log p(b\|C,a)、log p(b\|C)、log p(a\|C,b)）、δ 与 \|δ\| 三阶段对比；CPI / OrderGap 定义与 aggregate cards；OrderGap 6-path Q_by_path 示例 |
| 4. Training Story Dashboard | Pretrained → SFT → RL-1 流程、CPI/OrderGap 三阶段卡片、RL task 曲线（Formal64 NLL / sampled64 / greedy vs step 0–500，分图）、科学解读与 provenance 说明 |

## 3. Data provenance

- 所有展示数值来自**预计算资产**（`demo_assets/`），来源字段记录在每个资产 JSON 内
  （source_checkpoint / source_manifest / manifest_sha256 / evaluation_git_head /
  evaluation_code_path / historical 对比）。
- **Compatibility 数值为 current-code harmonized 口径**：Pretrained / SFT / RL-500 全部由
  当前冻结评估代码复算（CPI 为单进程原子生成：pair 四项与 per-sample δ 来自同一批张量，
  逐位一致）。
- 历史早期 formal 结果（pretrained CPI_abs 0.3281 / OrderGap 10.2445、SFT 0.2891 / 8.7557）
  由当时未提交的评估代码生成，无法 bit-level 复现；已保留在
  `results/pretrained/`、`results/vanilla/` 与 `protocol/errata_v4.2_eval_provenance.md`，
  仅作 provenance 参考，**不是 Demo 主链数字**。
- RL 系列（rlpilot-185545 step0–500）全部由当前代码产生，内部一致。

## 4. Which outputs are precomputed

全部。`demo_assets/` 包含：

- `examples.json`：15 个样本（9 代表性 + 6 curated case studies），选择规则与 reason 落盘
- `trajectories/traj_s*.json`：15 样本 × 4 stages × 2 modes 的关键帧与最终输出
  （inference-only 导出，无 backward；sampled 固定 seed=10000+index，greedy 确定性；
  跨进程逐位一致已验证）
- `compatibility_examples/pair_*.json`：pair 四项分解（δ 与 harmonized per-sample 逐位一致）
- `compatibility_examples/og_example_*.json`：OrderGap Q_by_path（直接取自 formal 结果）
- `metrics/harmonized/*.json`：三阶段 current-code CPI/OG 资产（含 per_sample × 500 与 buckets）
- `metrics/formal_curves.json`：三阶段链 + RL task 曲线 + LR probe 汇总
- `vocab_decode.json`：GPT2 词表解码表（Demo 不 import transformers/torch）

Demo 启动**不加载 checkpoint、不 import torch、不要求 CUDA**。

## 5. How to launch

```bash
cd Score-Entropy-Discrete-Diffusion
.venv/bin/python demo/app.py
# 打开 http://127.0.0.1:7860
```

（依赖：gradio 6.x、matplotlib。已通过 `uv pip install --python .venv/bin/python gradio` 安装。）

## 6. GPU requirement

**PRECOMPUTED MODE：无 GPU 要求。** 只有重新生成 demo 资产时需要 GPU
（`scripts/export_demo_*.py`，inference-only，总 GPU 时间 ~10–20 min）。

## 7. Known limitations

- Demo 只覆盖 15 个固定样本（9 代表性 + 6 curated case studies）；case study 明确标注
  "Illustrative case study"，不代表统计总体。
- RL-500 的 EMA 仅在 Tab 1/2 作为 secondary 可选展示；主链为 RAW。
- Trajectory 展示 ~10 个关键帧而非全部 128 帧；帧选取规则（|Δmask_count| 最大节点）
  记录在导出脚本中。
- 跨进程 bf16 GEMM 存在低概率数值漂移（实测 ~3e-3 量级，见
  `protocol/errata_v4.2_eval_provenance.md`）；Demo 资产采用单进程原子生成规避对拍问题。

## 8. Exact-token reward limitation

M0 reward = exact-token 匹配率。语义等价输出（如 GT "article" vs 输出 "story"）会被记为
0 分。真实案例（s069，人工离线标注，无 LLM judge）已在 Tab 1 以
"Known limitation of exact-token reward" 卡片展示。

## 9. Scientific result summary（current-code harmonized 口径）

| 阶段 | CPI_abs | OrderGap_raw |
|---|---|---|
| Pretrained | 0.3301 | 10.2246 |
| SFT (s1-10200 EMA) | 0.2871 | 8.7475 |
| RL-500 (RAW) | 0.2852 | 8.7442 |

- **SFT：clear compatibility attenuation**（CPI −13%，OrderGap −14%，方向与量级和冻结
  v4.1 发现一致）。
- **RL-1：no detectable additional change**（step500 vs step0 paired bootstrap CI 含 0）。
- RL task metrics（Formal64, step0→500）：NLL 7.0490→7.0342、sampled64 0.2562→0.2619、
  greedy 0.3365→0.3380 —— approximately stable / weak improvement signal
  （**不是** significant improvement）。
- 解读（protocol v1.0 §40 Scenario B 最接近）：SFT 是观察到的主要 compatibility
  restructuring 阶段；当前 pure on-policy REINFORCE + K=1 的 short-horizon RL 基本保持
  SFT 后的 reveal-order structure。task signal 弱时，优先按 estimator /
  credit-assignment efficiency 的潜在限制解读。
