"""Demo UI 渲染函数（纯 Python/HTML，无 torch）。中文主文案。"""

PROV_FOOTER_ZH = ("<small>数据来源说明：Demo 中 预训练→SFT→RL 的 compatibility 数值统一由当前冻结评估"
                  "代码复算（单进程原子生成，pair 四项与 per-sample δ 逐位一致），以保证三阶段完全同口径。"
                  "项目早期 formal 结果由当时未提交的评估代码生成，无法 bit-level 复现；历史记录已保留"
                  "（protocol/errata_v4.2_eval_provenance.md），聚合差异很小，不改变主要定性结论。</small>")


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def esc_attr(s):
    return esc(s).replace('"', "&quot;")


MASK_TOKEN_ID = 50257


def _tok_text(t, vocab):
    """token id → 展示文本（vocab_decode.json 已预解码，无内部 BPE marker）。"""
    s = vocab.get(str(t))
    if s is None:
        return "[MASK]" if t == MASK_TOKEN_ID else "<?>"
    return s


def _tok_html(t, vocab, cls=None, title=None, strip_leading_space=False):
    text = esc(_tok_text(t, vocab)).replace("\n", "<br>")
    if strip_leading_space and text.startswith(" "):
        text = text[1:]  # GPT-2 decode 约定：首 token 前导空格不显示
    if cls:
        tattr = f' title="{esc_attr(title)}"' if title else ""
        return f'<span class="{cls}"{tattr}>{text}</span>'
    return text


def gt_seq_html(tokens, m0set, vocab):
    """Ground Truth 序列：M0 位置蓝色高亮（初始需要重建的位置）。"""
    return "".join(_tok_html(t, vocab, "m0-token" if p in m0set else None,
                             strip_leading_space=(p == 0))
                   for p, t in enumerate(tokens))


def initial_seq_html(gt_tokens, m0set, vocab):
    """初始 Mask 状态：M0 位置渲染为紫色 MASK badge，其余为 GT token。"""
    return "".join('<span class="mask-badge">[MASK]</span>' if p in m0set
                   else _tok_html(t, vocab, strip_leading_space=(p == 0))
                   for p, t in enumerate(gt_tokens))


def pred_seq_html(final_tokens, gt_tokens, m0set, vocab):
    """模型输出序列（token-level exact comparison）：
    M0 位置：绿 = pred_token_id == gt_token_id，红 = 不等；
    非 M0 位置：橙 = 与 GT 不一致（理论上不应发生，逻辑保留）。"""
    out = []
    for p, t in enumerate(final_tokens):
        if p in m0set:
            if t == gt_tokens[p]:
                out.append(_tok_html(t, vocab, "correct-token",
                                     strip_leading_space=(p == 0)))
            else:
                out.append(_tok_html(
                    t, vocab, "wrong-token",
                    f"GT: {_tok_text(gt_tokens[p], vocab)} | Pred: {_tok_text(t, vocab)}",
                    strip_leading_space=(p == 0)))
        else:
            if t != gt_tokens[p]:
                out.append(_tok_html(
                    t, vocab, "changed-token",
                    f"非 M0 位置变化 GT: {_tok_text(gt_tokens[p], vocab)} | Pred: {_tok_text(t, vocab)}",
                    strip_leading_space=(p == 0)))
            else:
                out.append(_tok_html(t, vocab, strip_leading_space=(p == 0)))
    return "".join(out)


LEGEND_HTML = """
<div class="gt-card" style="padding:8px 10px;">
  <span class="m0-token">蓝：初始 M0（需重建）</span>
  <span class="mask-badge">紫：MASK</span>
  <span class="correct-token">绿：M0 重建正确</span>
  <span class="wrong-token">红：M0 重建错误（悬停看 GT/Pred）</span>
  <span class="changed-token">橙：非 M0 位置变化</span>
</div>
"""


def seq_html(tokens, mask_positions=None, newly=None, vocab=None):
    """tokens → 带高亮的 HTML。mask 灰色；新揭示位置绿色背景。"""
    from data_loader import decode_tokens
    mset = set(mask_positions or [])
    nset = set(newly or [])
    out = []
    for p, t in enumerate(tokens):
        if p in nset:
            out.append(f'<span style="background-color:#a5d6a7;color:#111;border-radius:3px;padding:0 1px;">'
                       f'{esc(decode_tokens([t], mask_positions=None, vocab=vocab))}</span>')
        elif p in mset:
            out.append('<span style="color:#9e9e9e;">[MASK]</span>')
        else:
            out.append(esc(decode_tokens([t], vocab=vocab)))
    return "".join(out)


def example_choices():
    from data_loader import load_examples
    out = []
    for e in load_examples():
        typ = "代表性" if e["category"] == "representative" else "案例研究"
        out.append(f's{e["manifest_index"]:03d} [{typ}] {e["selection_reason"][:52]}')
    return out


def tab1_html(ex, show_ema, vocab):
    from data_loader import (load_trajectory, load_harmonized,
                             load_manifest_tokens)
    traj = load_trajectory(ex["manifest_index"])
    mt = load_manifest_tokens()[ex["manifest_index"]]
    gt_tokens = mt["x0"]
    m0set = set(mt["initial_masked_positions"])
    stages = [("pretrained", "预训练模型"),
              ("sft", "SFT（s1-10200 EMA）"),
              ("rl_raw", "RL-500（RAW，主口径）")]
    if show_ema:
        stages.append(("rl_ema", "RL-500（EMA，辅助口径）"))
    harmonized = {s: load_harmonized(s, "cpi") for s in
                  ["pretrained", "sft", "rl500"]}
    harmonized_key = {"pretrained": "pretrained", "sft": "sft",
                      "rl_raw": "rl500", "rl_ema": "rl500"}
    cols = []
    for key, label in stages:
        sv = traj["stages"][key]["modes"]
        hk = harmonized_key.get(key)
        local_ce = harmonized[hk]["per_sample"]["local_ce"][ex["manifest_index"]] \
            if hk in harmonized else None
        ce_html = (f"<b>局部 Mask Token CE</b>（compatibility sample 口径）：{local_ce:.3f}"
                   if local_ce is not None else "")
        cols.append(f"""
<div class="gt-card">
  <b style="font-size:15px;">{label}</b><br>
  <b>Sampled 输出</b>（M0 精确重建率 {sv['sampled']['m0_reward']:.3f}）：<br>
  <div class="token-seq">{pred_seq_html(sv['sampled']['final_tokens'], gt_tokens, m0set, vocab)}</div><br>
  <b>Greedy 输出</b>（M0 精确重建率 {sv['greedy']['m0_reward']:.3f}）：<br>
  <div class="token-seq">{pred_seq_html(sv['greedy']['final_tokens'], gt_tokens, m0set, vocab)}</div><br>
  {ce_html}
</div>""")
    span_info = (f"目标 span：[{ex['span_start']}, {ex['span_end']}) · σ₀={ex['sigma']} · "
                 f"|M0|={len(m0set)} · 高亮为 token-level exact comparison"
                 f"（pred_token_id == gt_token_id 才算正确）")
    html = LEGEND_HTML + f"""
<div class="gt-card">
  <b>标准答案</b>（Ground Truth）：<br>
  <div class="token-seq">{gt_seq_html(gt_tokens, m0set, vocab)}</div><br>
  <b>初始 Mask 状态</b>（Initial Mask）：<br>
  <div class="token-seq">{initial_seq_html(gt_tokens, m0set, vocab)}</div><br>
  <small>{span_info}</small>
</div>
<div style="display:flex;gap:6px;flex-wrap:wrap;">{''.join(cols)}</div>
"""
    if ex.get("semantic_status") == "confirmed_real_case":
        ann = ex["semantic_annotation"]
        judgements = "<br>".join(f"&nbsp;&nbsp;{k}：<b>{v}</b>" for k, v in ann["judgements"].items())
        html += f"""
<div class="gt-card gt-card-note">
  <b>exact-token reward 的已知局限</b>（真实案例，人工离线标注）<br>
  标准答案 → RL greedy 输出中的 mismatch：article→story、fall→spring<br>
  {judgements}<br>
  <small>{esc(ann['note'])}</small>
</div>"""
    return html


def tab2_html(ex_idx, stage_key, mode, kf_i, vocab):
    from data_loader import load_trajectory
    traj = load_trajectory(ex_idx)
    mv = traj["stages"][stage_key]["modes"][mode]
    kfs = mv["keyframes"]
    kf_i = max(0, min(kf_i, len(kfs) - 1))
    f = kfs[kf_i]
    label = ("denoiser 收尾" if f["reverse_step"] == 128
             else f"Reverse Step {f['reverse_step']} / 128")
    final_note = ""
    if kf_i == len(kfs) - 1:
        final_note = (f"<br><b>最终重建结果</b>（{mode}）："
                      f"<span style='font-family:monospace;'>{esc(mv['final_text'])}</span>"
                      f"（M0 精确重建率 {mv['m0_reward']:.3f}）")
    newly = f.get("newly_revealed_positions", [])
    return f"""
<div style="border:1px solid #ddd;border-radius:6px;padding:12px;">
  <b>{label}</b> · σ = {f['sigma']:.4f} · 当前 MASK 数量 = {f['mask_count']}
  · <span style="background-color:#a5d6a7;color:#111;padding:0 4px;">本帧新揭示 {len(newly)} 个 token</span><br><br>
  <div style="font-family:monospace;font-size:16px;line-height:1.9;word-break:break-word;">
    {seq_html(f['tokens'], f['mask_positions'], newly, vocab)}
  </div>
  {final_note}
</div>"""


def tab3_pair_html(ex, vocab):
    from data_loader import load_pair
    stages = [("pretrained", "预训练模型"), ("sft", "SFT"), ("rl_raw", "RL-500")]
    idx = ex["manifest_index"]
    cards = []
    for key, label in stages:
        p = load_pair(key, idx)
        hist = ""
        if p.get("historical_delta_if_available") is not None:
            hist = (f"（历史旧代码值 {p['historical_delta_if_available']:.4f}，"
                    f"仅 provenance 参考，非主展示）")
        cards.append(f"""
<div style="border:1px solid #ddd;border-radius:6px;padding:10px;margin:4px;min-width:260px;">
  <b>{label}</b><br>
  <table style="font-size:13px;border-collapse:collapse;">
    <tr><td>顺序 A→B：</td><td>log p(a|C) = {p['logp_a_C']:.3f}</td></tr>
    <tr><td></td><td>log p(b|C,a) = {p['logp_b_Ca']:.3f}</td></tr>
    <tr><td></td><td><b>sum_AB = {p['sum_AB']:.3f}</b></td></tr>
    <tr><td>顺序 B→A：</td><td>log p(b|C) = {p['logp_b_C']:.3f}</td></tr>
    <tr><td></td><td>log p(a|C,b) = {p['logp_a_Cb']:.3f}</td></tr>
    <tr><td></td><td><b>sum_BA = {p['sum_BA']:.3f}</b></td></tr>
  </table>
  <b>δ = sum_AB − sum_BA = {p['delta']:+.4f}</b> &nbsp; |δ| = {abs(p['delta']):.4f}<br>
  <small>{hist}</small>
</div>""")
    return f"""
<div style="display:flex;gap:6px;flex-wrap:wrap;">{''.join(cards)}</div>
<p><b>大白话</b>：|δ| 越接近 0，说明交换两个 token 的揭示顺序后，模型给出的条件概率越一致；
|δ| 越大，reveal order 对条件预测的影响越大。</p>
<p>本组 pair 来自 frozen manifest 样本 s{idx:03d}（位置 i={ex['i']}, j={ex['j']}，
token a={ex['a']}, b={ex['b']}，σ={ex['sigma']}）。四项由当前冻结
<code>evaluate_delta_swap_batch</code> 同源逻辑捕获，δ 与 harmonized per-sample δ 逐位一致。</p>
"""


def compat_cards_html():
    from data_loader import load_harmonized
    stages = [("pretrained", "预训练模型"), ("sft", "SFT"), ("rl500", "RL-500")]
    cpi, og = [], []
    for key, label in stages:
        c = load_harmonized(key, "cpi")["summary"]
        o = load_harmonized(key, "og")["current"]
        cpi.append(f"<b>{label}</b>：CPI_abs = {c['cpi_abs']:.4f}")
        og.append(f"<b>{label}</b>：OrderGap_raw = {o['order_gap_raw']:.4f}")
    return f"""
<div style="display:flex;gap:8px;flex-wrap:wrap;">
  <div style="border:1px solid #ddd;border-radius:6px;padding:10px;">
    <b>CPI（局部两-token reveal-order 不一致程度）</b><br>{'<br>'.join(cpi)}<br>
    <small>CPI = E|δ|，δ = log p(a|C) + log p(b|C,a) − log p(b|C) − log p(a|C,b)。
    衡量局部两个 token 交换揭示顺序后，模型条件预测的不一致程度。</small>
  </div>
  <div style="border:1px solid #ddd;border-radius:6px;padding:10px;">
    <b>OrderGap（完整 reveal path 的全局顺序敏感性）</b><br>{'<br>'.join(og)}<br>
    <small>OrderGap = max_π Q_π(x) − min_π Q_π(x)，Q_π(x) = Σ_k log p(x_{{i_k}}|C, x_{{i_&lt;k}})。
    衡量同一个目标序列在不同完整 reveal path 下的概率差异。</small>
  </div>
</div>
<p><b>SFT：reveal-order sensitivity 明显下降</b> · <b>RL-1：no detectable additional change
（无可检测的进一步结构变化）</b></p>
"""


def og_example_html(idx):
    from data_loader import load_og_example
    d = load_og_example(idx)
    rows = []
    for key, label in [("pretrained", "预训练模型"), ("sft", "SFT"), ("rl_raw", "RL-500")]:
        sv = d["stages"][key]
        q = " &nbsp; ".join(f"{k}: {v:.2f}" for k, v in sv["q_by_path"].items())
        rows.append(f"<b>{label}</b>（OrderGap = {sv['order_gap']:.2f}）：{q}")
    return ("<div style='border:1px solid #ddd;border-radius:6px;padding:10px;'>"
            + "<br>".join(rows) + "</div>")
