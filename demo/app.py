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
                        og_example_html, example_choices, PROV_FOOTER_ZH, PROV_FOOTER_EN,
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

    with gr.Blocks(title="Reveal-Order Compatibility in Masked Diffusion LMs") as app:
        gr.Markdown("""
# Post-Training Reveal-Order Compatibility in Masked Diffusion Language Models

**从 MASK 恢复、Reveal Order 到 SFT / RL 后的结构变化**

这个 Demo 展示三个问题：
1. masked diffusion language model 如何通过多步 reverse denoising 恢复文本；
2. 为什么不同 token reveal order 会影响 conditional predictions；
3. SFT 与 RL post-training 如何改变这种 reveal-order sensitivity。

**当前核心结果**：Vanilla SFT 明显降低 CPI 与 OrderGap；RL-1 在 task learning signal
较弱的情况下，基本保持 SFT 后的 compatibility structure。
""")

        # ---------------- TAB 1: Generation ----------------
        with gr.Tab("1. Masked Diffusion Generation"):
            with gr.Row():
                ex_sel1 = gr.Dropdown(choices=choices, value=choices[0],
                                      label="Example（代表性 / 案例研究）")
                show_ema = gr.Checkbox(label="显示 RL-500 EMA（secondary）", value=False)
            out1 = gr.HTML()
            ex_sel1.change(render_tab1, [ex_sel1, show_ema], out1)
            show_ema.change(render_tab1, [ex_sel1, show_ema], out1)
            app.load(render_tab1, [ex_sel1, show_ema], out1)
            gr.Markdown(
                "说明：三阶段 = Pretrained（louaaron/sedd-small）/ SFT（formal s1 "
                "checkpoint_10200 EMA）/ RL-500（formal pilot RAW，EMA 可选）。"
                "**Local masked-token CE** 标注为 compatibility sample 口径，"
                "**不是** Formal64 NLL（两者 evaluation population 不同）。")

        # ---------------- TAB 2: Trajectory ----------------
        with gr.Tab("2. Reveal Trajectory Viewer"):
            with gr.Row():
                ex_sel2 = gr.Dropdown(choices=choices, value=choices[0],
                                      label="Example")
                stage_sel = gr.Dropdown(
                    choices=["Pretrained", "SFT (s1-10200 EMA)", "RL-500 (RAW)",
                             "RL-500 (EMA, secondary)"],
                    value="RL-500 (RAW)", label="Stage")
                mode_sel = gr.Radio(choices=["sampled", "greedy"], value="greedy",
                                    label="Decoding mode")
            kf_slider = gr.Slider(0, 9, step=1, value=0,
                                  label="Keyframe（按 |Δmask_count| 选取的 ~10 个关键帧）")
            out2 = gr.HTML()
            for c in (ex_sel2, stage_sel, mode_sel, kf_slider):
                c.change(render_tab2, [ex_sel2, stage_sel, mode_sel, kf_slider], out2)
            app.load(render_tab2, [ex_sel2, stage_sel, mode_sel, kf_slider], out2)
            gr.Markdown(
                "真实 sampler 为 128 reverse steps；本页展示按 mask-count 变化最大的 "
                "节点选取的 ~10 个关键帧（含初始与 denoiser 收尾）。"
                "<span style='background-color:#a5d6a7'>绿色</span> = 相对上一帧新揭示的位置，"
                "灰色 [MASK] = 仍未揭示。σ<0.05 尾段保留完整 rollout，但 PG timestep 采样"
                "（K=1）只在 σ≥0.05 safe region 进行（协议 v1.0 §25）。")

        # ---------------- TAB 3: Compatibility Lab ----------------
        with gr.Tab("3. Reveal Order / Compatibility Lab"):
            ex_sel3 = gr.Dropdown(choices=choices, value=choices[0],
                                  label="Pair（来自 frozen manifest）")
            out3 = gr.HTML()
            ex_sel3.change(render_tab3, [ex_sel3], out3)
            app.load(render_tab3, [ex_sel3], out3)
            cards3 = gr.HTML(compat_cards_html())
            gr.Markdown("### OrderGap 路径示例（Q_by_path，直接取自 formal 结果）")
            og_sel = gr.Dropdown(choices=og_choices, value=og_choices[0],
                                 label="Sample（cs3 unusual sensitivity / cs5 RL unchanged）")
            out3b = gr.HTML()
            og_sel.change(render_og, [og_sel], out3b)
            app.load(render_og, [og_sel], out3b)
            gr.Markdown(PROV_FOOTER_ZH)
            gr.Markdown(PROV_FOOTER_EN)

        # ---------------- TAB 4: Dashboard ----------------
        with gr.Tab("4. Training Story Dashboard"):
            gr.Markdown("""
### Pretrained → Vanilla SFT → RL-1

| 阶段 | CPI_abs | OrderGap_raw |
|---|---|---|
| Pretrained | 0.3301 | 10.2246 |
| SFT (s1-10200 EMA) | 0.2871 | 8.7475 |
| RL-500 (RAW) | 0.2852 | 8.7442 |

**SFT: clear attenuation** · **RL-1: no detectable additional change**

（数值为 current-code harmonized 口径；历史旧代码值见 provenance 说明。）
""")
            gr.Plot(plot_compat_chain(CURVES), label="Compatibility 三阶段（分图，不混量纲）")
            gr.Markdown("""
### RL task metrics（Formal64, step 0 → 500, RAW）

| step | Formal64 NLL | sampled64 reward | greedy reward |
|---|---|---|---|
| 0 | 7.0490 | 0.2562 | 0.3365 |
| 50 | 7.0100 | 0.2585 | 0.3373 |
| 100 | 7.0069 | 0.2573 | 0.3293 |
| 250 | 7.0477 | 0.2579 | 0.3277 |
| 500 | 7.0342 | 0.2619 | 0.3380 |

**approximately stable / weak improvement signal**（禁止表述为 significant improvement）。
""")
            gr.Plot(plot_rl_task(CURVES), label="RL task metrics vs step（分图）")
            gr.Markdown("""
### Scientific interpretation

- **SFT**: compatibility attenuation（CPI 0.330→0.287，OrderGap 10.22→8.75）
- **RL-1**: compatibility preserved under weak-learning RL —— 当前 pure on-policy
  REINFORCE + K=1 的 short-horizon RL 没有产生明确 task-learning signal，因此
  compatibility 也没有出现可检测的进一步变化（paired bootstrap CI 含 0）。
""")
            gr.Markdown(PROV_FOOTER_ZH)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True,
               theme=gr.themes.Base())
