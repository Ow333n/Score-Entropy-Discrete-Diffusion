"""Post-Training Reveal-Order Compatibility Demo（Gradio，PRECOMPUTED MODE）。

不 import torch / transformers，不加载 checkpoint，不要求 CUDA。
只读取 demo_assets/*.json。

启动: .venv/bin/python demo/app.py    （默认 127.0.0.1:7860）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from data_loader import (load_examples, example_by_index, load_trajectory,
                         load_curves, load_vocab)
from components import (tab1_html, tab2_html, tab3_pair_html, compat_cards_html,
                        og_example_html, example_choices, PROV_FOOTER_ZH,
                        seq_html)
from plots import plot_compat_chain, plot_rl_task

VOCAB = load_vocab()
CURVES = load_curves()


def _ex_from_choice(choice):
    idx = int(choice.split(" ")[0][1:])
    return example_by_index(idx)


def render_tab1(choice, show_ema):
    ex = _ex_from_choice(choice)
    return tab1_html(ex, show_ema, VOCAB)


def render_tab2(choice, stage_label, mode, kf_i):
    ex = _ex_from_choice(choice)
    stage_key = dict(
        (label, key) for key, label in
        [("pretrained", "Pretrained"), ("sft", "SFT (s1-10200 EMA)"),
         ("rl_raw", "RL-500 (RAW)"), ("rl_ema", "RL-500 (EMA, secondary)")]
    )[stage_label]
    return tab2_html(ex["manifest_index"], stage_key, mode, int(kf_i), VOCAB)


def render_tab3(choice):
    ex = _ex_from_choice(choice)
    return tab3_pair_html(ex, VOCAB)


def render_og(idx_choice):
    idx = int(idx_choice)
    return og_example_html(idx)


def build_app():
    choices = example_choices()
    og_choices = [c for c in choices if "cs3" in c or "cs5" in c]
    og_choices = [c.split(" ")[0][1:] for c in og_choices]

    # 自定义卡片样式：颜色/边框全部走 Gradio 主题变量，深浅两种模式下
    # 自动与整站风格一致（深色模式=深底浅字，浅色模式=浅底深字）。
    CUSTOM_CSS = """
.gt-card {
  background: var(--body-background-fill);
  color: var(--body-text-color);
  border: 1px solid var(--border-color-primary);
  border-radius: 6px;
  padding: 10px;
  margin: 4px;
  overflow-wrap: anywhere;
}
.gt-card-note {
  margin: 8px 4px;
  border: 1px solid var(--border-color-accent);
}
"""

    with gr.Blocks(title="离散扩散语言模型后训练与 Reveal Order 可视化",
                   css=CUSTOM_CSS) as app:
        gr.Markdown("""
# 离散扩散语言模型后训练与 Reveal Order 可视化

**从 Masked Denoising、SFT 到 RL 的结构变化分析**

这个 Demo 展示三个问题：
1. 离散扩散语言模型（Diffusion Language Model, dLLM）如何通过多步反向扩散（Reverse
   Diffusion）从 Mask 状态恢复文本；
2. 为什么不同 token 揭示顺序（Reveal Order）会影响条件预测；
3. 监督微调（SFT）与强化学习（RL）后训练如何改变这种 reveal-order 敏感性。

**当前核心结果**：Vanilla SFT 明显降低 CPI 与 OrderGap；RL-1 在任务学习信号较弱的情况下，
基本保持 SFT 后的 compatibility 结构。
""")

        # ---------------- TAB 1: 生成结果对比 ----------------
        with gr.Tab("1. 生成结果对比"):
            with gr.Row():
                ex_sel1 = gr.Dropdown(choices=choices, value=choices[0],
                                      label="样本（代表性 / 案例研究）")
                show_ema = gr.Checkbox(label="显示 RL-500 EMA（辅助口径）", value=False)
            out1 = gr.HTML()
            ex_sel1.change(render_tab1, [ex_sel1, show_ema], out1)
            show_ema.change(render_tab1, [ex_sel1, show_ema], out1)
            app.load(render_tab1, [ex_sel1, show_ema], out1)
            gr.Markdown(
                "说明：三阶段 = 预训练模型（louaaron/sedd-small）/ SFT（formal s1 "
                "checkpoint_10200 EMA）/ RL-500（formal pilot RAW，EMA 可选）。"
                "**局部 Mask Token CE** 标注为 compatibility sample 口径，"
                "**不是** Formal64 NLL（两者评估样本集不同）。")

        # ---------------- TAB 2: 反向扩散轨迹 ----------------
        with gr.Tab("2. 反向扩散轨迹"):
            with gr.Row():
                ex_sel2 = gr.Dropdown(choices=choices, value=choices[0],
                                      label="样本")
                stage_sel = gr.Dropdown(
                    choices=["预训练模型", "SFT (s1-10200 EMA)", "RL-500 (RAW)",
                             "RL-500 (EMA, 辅助口径)"],
                    value="RL-500 (RAW)", label="阶段")
                mode_sel = gr.Radio(choices=["sampled", "greedy"], value="greedy",
                                    label="解码方式")
            kf_slider = gr.Slider(0, 9, step=1, value=0,
                                  label="关键帧（按 mask 数量变化最大的节点选取 ~10 帧）")
            out2 = gr.HTML()
            for c in (ex_sel2, stage_sel, mode_sel, kf_slider):
                c.change(render_tab2, [ex_sel2, stage_sel, mode_sel, kf_slider], out2)
            app.load(render_tab2, [ex_sel2, stage_sel, mode_sel, kf_slider], out2)
            gr.Markdown(
                "SEDD 从部分 Mask 的状态出发，通过多步反向扩散逐渐恢复 token。与自回归模型"
                "（Autoregressive, AR）严格从左到右生成不同，dLLM 的 token 可以按照更灵活"
                "的顺序被揭示——模型一次 forward 可以同时为多个 MASK 位置给出预测，但采样器"
                "会在多个 reverse steps 中随机决定哪些位置在当前 step 被 reveal"
                "（parallel prediction + iterative revealing），不是一次性把所有 token 定死。"
                "真实采样器共 128 个 reverse step；本页展示按 mask 数量变化最大的节点选取的"
                " ~10 个关键帧（含初始与 denoiser 收尾）。"
                "<span style='background-color:#a5d6a7'>绿色</span> = 相对上一帧新揭示的 token，"
                "灰色 [MASK] = 仍未揭示。σ<0.05 尾段保留完整采样展开（rollout），但策略梯度"
                "（Policy Gradient）的 timestep 采样只在 σ≥0.05 safe region 进行（协议 v1.0 §25）。")

        # ---------------- TAB 3: Reveal Order 一致性分析 ----------------
        with gr.Tab("3. Reveal Order 一致性分析"):
            ex_sel3 = gr.Dropdown(choices=choices, value=choices[0],
                                  label="Pair（来自 frozen manifest）")
            out3 = gr.HTML()
            ex_sel3.change(render_tab3, [ex_sel3], out3)
            app.load(render_tab3, [ex_sel3], out3)
            cards3 = gr.HTML(compat_cards_html())
            gr.Markdown("### OrderGap 路径示例（Q_by_path，直接取自 formal 结果）")
            og_sel = gr.Dropdown(choices=og_choices, value=og_choices[0],
                                 label="样本（cs3 高顺序敏感性 / cs5 RL 近似不变）")
            out3b = gr.HTML()
            og_sel.change(render_og, [og_sel], out3b)
            app.load(render_og, [og_sel], out3b)
            gr.Markdown(PROV_FOOTER_ZH)

        # ---------------- TAB 4: 训练与研究结果 ----------------
        with gr.Tab("4. 训练与研究结果"):
            gr.Markdown("""
### 预训练模型 → Vanilla SFT → RL-1

| 阶段 | CPI_abs | OrderGap_raw |
|---|---|---|
| 预训练模型 | 0.3301 | 10.2246 |
| SFT（s1-10200 EMA） | 0.2871 | 8.7475 |
| RL-500（RAW） | 0.2852 | 8.7442 |

**SFT：reveal-order sensitivity 明显下降** · **RL-1：无可检测的进一步结构变化**

（数值为 current-code harmonized 口径；历史旧代码值见 provenance 说明。）
""")
            gr.Plot(plot_compat_chain(CURVES), label="三阶段 Compatibility（分图，不混量纲）")
            gr.Markdown("""
### RL 任务指标（Formal64，step 0 → 500，RAW）

| step | Formal64 NLL | sampled64 reward | greedy reward |
|---|---|---|---|
| 0 | 7.0490 | 0.2562 | 0.3365 |
| 50 | 7.0100 | 0.2585 | 0.3373 |
| 100 | 7.0069 | 0.2573 | 0.3293 |
| 250 | 7.0477 | 0.2579 | 0.3277 |
| 500 | 7.0342 | 0.2619 | 0.3380 |

**approximately stable / weak improvement signal（近似平稳 / 弱提升信号）**
——不表述为 significant improvement（显著提升）。
""")
            gr.Plot(plot_rl_task(CURVES), label="RL 任务指标随 step 变化（分图）")
            gr.Markdown("""
### 科学解读

- **SFT**：compatibility attenuation（CPI 0.330→0.287，OrderGap 10.22→8.75）
- **RL-1**：在任务学习信号较弱的情况下，compatibility 结构基本保持不变 —— 当前 pure
  on-policy REINFORCE + K=1 的 short-horizon RL 没有产生明确 task-learning signal，
  因此 compatibility 也没有出现可检测的进一步变化（paired bootstrap CI 含 0）。
""")
            gr.Markdown(PROV_FOOTER_ZH)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True,
               theme=gr.themes.Base())
