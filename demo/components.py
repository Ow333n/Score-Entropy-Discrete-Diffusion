"""Demo UI 渲染函数（纯 Python/HTML，无 torch）。"""

PROV_FOOTER_ZH = ("<small>数据来源说明：Demo 中 Pretrained→SFT→RL 的 compatibility 数值统一由当前冻结评估"
                  "代码复算（单进程原子生成，pair 四项与 per-sample δ 逐位一致），以保证三阶段完全同口径。"
                  "项目早期 formal 结果由当时未提交的评估代码生成，无法 bit-level 复现；历史记录已保留"
                  "（protocol/errata_v4.2_eval_provenance.md），聚合差异很小，不改变主要定性结论。</small>")
PROV_FOOTER_EN = ("<small>Provenance: demo compatibility values are recomputed with the current frozen "
                  "evaluation harness for a consistent Pretrained→SFT→RL comparison. Historical early-run "
                  "aggregates are preserved separately; their exact evaluation implementation was not "
                  "committed and is therefore not bit-reproducible. The small aggregate discrepancies do "
                  "not alter the qualitative conclusions.</small>")


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def seq_html(tokens, mask_positions=None, newly=None, vocab=None):
    """tokens → 带高亮的 HTML。mask 灰色；newly revealed 绿色背景。"""
    from data_loader import decode_tokens
    mset = set(mask_positions or [])
    nset = set(newly or [])
    out = []
    for p, t in enumerate(tokens):
        if p in nset:
            out.append(f'<span style="background-color:#a5d6a7;border-radius:3px;padding:0 1px;">'
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
    from data_loader import load_trajectory, load_harmonized
    traj = load_trajectory(ex["manifest_index"])
    stages = [("pretrained", "Pretrained"),
              ("sft", "SFT (s1-10200 EMA)"),
              ("rl_raw", "RL-500 (RAW, primary)")]
    if show_ema:
        stages.append(("rl_ema", "RL-500 (EMA, secondary)"))
    harmonized = {s: load_harmonized(s, "cpi") for s in
                  ["pretrained", "sft", "rl500"]}
    cols = []
    for key, label in stages:
        sv = traj["stages"][key]["modes"]
        local_ce = harmonized[key]["per_sample"]["local_ce"][ex["manifest_index"]] \
            if key in harmonized else None
        ce_html = (f"<b>Local masked-token CE</b> (compatibility sample): {local_ce:.3f}"
                   if local_ce is not None else "")
        cols.append(f"""
<div style="border:1px solid #ddd;border-radius:6px;padding:10px;margin:4px;">
  <b style="font-size:15px;">{label}</b><br>
  <b>sampled</b>（M0 reward {sv['sampled']['m0_reward']:.3f}）:<br>
  <span style="font-family:monospace;">{esc(sv['sampled']['final_text'])}</span><br><br>
  <b>greedy</b>（M0 reward {sv['greedy']['m0_reward']:.3f}）:<br>
  <span style="font-family:monospace;">{esc(sv['greedy']['final_text'])}</span><br><br>
  {ce_html}
</div>""")
    span_info = (f"target span: [{ex['span_start']}, {ex['span_end']}) · σ₀={ex['sigma']} · "
                 f"|M0|={len(ex['m0_positions'])}")
    html = f"""
<div style="border:1px solid #bbb;border-radius:6px;padding:10px;margin:4px;background:#fafafa;">
  <b>Ground Truth</b>:<br>
  <span style="font-family:monospace;">{esc(ex['gt_text'])}</span><br><br>
  <b>Initial Corruption</b>:<br>
  <span style="font-family:monospace;">{esc(ex['initial_state_text'])}</span><br>
  <small>{span_info}</small>
</div>
<div style="display:flex;gap:6px;flex-wrap:wrap;">{''.join(cols)}</div>
"""
    if ex.get("semantic_status") == "confirmed_real_case":
        ann = ex["semantic_annotation"]
        judgements = "<br>".join(f"&nbsp;&nbsp;{k}: <b>{v}</b>" for k, v in ann["judgements"].items())
        html += f"""
<div style="border:1px solid #e0b97c;border-radius:6px;padding:10px;margin:8px 4px;background:#fff8ec;">
  <b>Known limitation of exact-token reward</b>（真实案例，人工离线标注）<br>
  GT → RL greedy 输出中的 mismatch：{', '.join(f"'{k}'→'{v}'" for k, v in
        [("article", "story"), ("fall", "spring")])}<br>
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
             else f"reverse step {f['reverse_step']} / 128")
    final_note = ""
    if kf_i == len(kfs) - 1:
        final_note = (f"<br><b>Final reconstruction</b>（{mode}）: "
                      f"<span style='font-family:monospace;'>{esc(mv['final_text'])}</span>"
                      f"（M0 reward {mv['m0_reward']:.3f}）")
    newly = f.get("newly_revealed_positions", [])
    return f"""
<div style="border:1px solid #ddd;border-radius:6px;padding:12px;">
  <b>{label}</b> · σ = {f['sigma']:.4f} · masked = {f['mask_count']}
  · <span style="background-color:#a5d6a7;padding:0 4px;">新揭示 {len(newly)} 个位置</span><br><br>
  <div style="font-family:monospace;font-size:16px;line-height:1.9;word-break:break-word;">
    {seq_html(f['tokens'], f['mask_positions'], newly, vocab)}
  </div>
  {final_note}
</div>"""


def tab3_pair_html(ex, vocab):
    from data_loader import load_pair
    stages = [("pretrained", "Pretrained"), ("sft", "SFT"), ("rl_raw", "RL-500")]
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
    <tr><td>Order A→B:</td><td>log p(a|C) = {p['logp_a_C']:.3f}</td></tr>
    <tr><td></td><td>log p(b|C,a) = {p['logp_b_Ca']:.3f}</td></tr>
    <tr><td></td><td><b>sum_AB = {p['sum_AB']:.3f}</b></td></tr>
    <tr><td>Order B→A:</td><td>log p(b|C) = {p['logp_b_C']:.3f}</td></tr>
    <tr><td></td><td>log p(a|C,b) = {p['logp_a_Cb']:.3f}</td></tr>
    <tr><td></td><td><b>sum_BA = {p['sum_BA']:.3f}</b></td></tr>
  </table>
  <b>δ = sum_AB − sum_BA = {p['delta']:+.4f}</b> &nbsp; |δ| = {abs(p['delta']):.4f}<br>
  <small>{hist}</small>
</div>""")
    return f"""
<div style="display:flex;gap:6px;flex-wrap:wrap;">{''.join(cards)}</div>
<p><b>大白话</b>：|δ| 越接近 0，说明模型越不在乎这两个 token 谁先被揭开；
|δ| 越大，reveal order 对 conditional prediction 的影响越大。</p>
<p>本组 pair 来自 frozen manifest 样本 s{idx:03d}（位置 i={ex['i']}, j={ex['j']}，
token a={ex['a']}, b={ex['b']}，σ={ex['sigma']}）。四项由当前冻结
<code>evaluate_delta_swap_batch</code> 同源逻辑捕获，δ 与 harmonized per-sample δ 逐位一致。</p>
"""


def compat_cards_html():
    from data_loader import load_harmonized
    stages = [("pretrained", "Pretrained"), ("sft", "SFT"), ("rl500", "RL-500")]
    cpi, og = [], []
    for key, label in stages:
        c = load_harmonized(key, "cpi")["summary"]
        o = load_harmonized(key, "og")["current"]
        cpi.append(f"<b>{label}</b>: CPI_abs = {c['cpi_abs']:.4f}")
        og.append(f"<b>{label}</b>: OrderGap_raw = {o['order_gap_raw']:.4f}")
    return f"""
<div style="display:flex;gap:8px;flex-wrap:wrap;">
  <div style="border:1px solid #ddd;border-radius:6px;padding:10px;">
    <b>CPI（条件预测不兼容性）</b><br>{'<br>'.join(cpi)}<br>
    <small>CPI = E|δ|，δ = log p(a|C) + log p(b|C,a) − log p(b|C) − log p(a|C,b)。
    随机挑两个 token 交换 reveal 顺序，看模型自己的 conditional predictions 矛盾有多大。</small>
  </div>
  <div style="border:1px solid #ddd;border-radius:6px;padding:10px;">
    <b>OrderGap（路径敏感性）</b><br>{'<br>'.join(og)}<br>
    <small>OrderGap = max_π Q_π(x) − min_π Q_π(x)，Q_π(x) = Σ_k log p(x_{{i_k}}|C, x_{{i_&lt;k}})。
    给同一个答案换很多条完整 reveal path，看模型最喜欢和最不喜欢的路径差多少。</small>
  </div>
</div>
<p><b>SFT: clear attenuation</b> · <b>RL-1: no detectable additional change</b></p>
"""


def og_example_html(idx):
    from data_loader import load_og_example
    d = load_og_example(idx)
    rows = []
    for key, label in [("pretrained", "Pretrained"), ("sft", "SFT"), ("rl_raw", "RL-500")]:
        sv = d["stages"][key]
        q = " &nbsp; ".join(f"{k}: {v:.2f}" for k, v in sv["q_by_path"].items())
        rows.append(f"<b>{label}</b>（OrderGap = {sv['order_gap']:.2f}）: {q}")
    return ("<div style='border:1px solid #ddd;border-radius:6px;padding:10px;'>"
            + "<br>".join(rows) + "</div>")
