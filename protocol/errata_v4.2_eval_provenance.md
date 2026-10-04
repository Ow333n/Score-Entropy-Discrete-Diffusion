# Formal Evaluation Provenance Record（append-only）

> 主题：pretrained / Vanilla SFT 历史 formal CPI / OrderGap 结果与当前 frozen
> evaluation code 的 bit-reproducibility 差异。本文件为 append-only 记录，
> 不改动任何 frozen 协议或结果文件。

## Record #1（2026-10-04，Demo Tab 3 pair 对拍失败排查）

### 现象

Demo 阶段 Tab 3 pair four-term decomposition 导出（逐字复用当前 frozen
`compatibility/cpi.py evaluate_delta_swap_batch` 内部计算）与历史 formal
per_sample['delta'] 对拍失败：

- sample s375, pretrained: exported delta = −0.6094 vs stored = −0.5625（差 0.047）

### 排查过程（全部为事实）

1. 排除了导出代码问题：直接用 frozen 函数 `evaluate_delta_swap_batch` 在相同
   （checkpoint, manifest 样本, sigma, torch 版本）上复算，结果 = −0.6094，
   与导出一致，与 stored 不一致 → 差异不在导出代码。
2. 历史结果文件的 `git_commit` = **84dc4b4**（Day 2-3 commit）。经查该 commit
   的 git 树中**不存在** `evaluation/` 与 `compatibility/` 代码 —— 当时的
   评估代码是未提交版本，未进入 git 历史，无法复现。
3. 当前 frozen 代码（de69bb9 起）在相同输入上的聚合复算：

| metric | stored（84dc4b4 时代代码） | current frozen code | diff |
|---|---|---|---|
| pretrained CPI_abs | 0.3281 | 0.3301 | 0.0020 |
| SFT s1-10200 CPI_abs | 0.2891 | 0.2871 | 0.0020 |
| pretrained OrderGap_raw | 10.2445 | 10.2246 | 0.0198 |
| SFT s1-10200 OrderGap_raw | 8.7557 | 8.7475 | 0.0081 |
| local_ce（两阶段） | 4.2812 / 3.8750 | 4.2812 / 3.8750 | **逐位相同** |

4. per-sample delta：mean|diff| = 0.0335，max|diff| = 0.1875（旧代码的 delta
   计算与 frozen 代码系统性不同；local_ce 逐位相同说明模型前向路径未变，
   差异只可能在 quartet/delta 计算细节）。

### 结论

- 历史 formal 数值由 **84dc4b4 时代的未提交评估代码**产生；该代码不在 git 中，
  **不可 bit-reproduce**。
- 聚合差异幅度（CPI ~2e-3、OG ~8e-3–2e-2）落在既有 G0 tolerance 级别，
  **不改变任何 frozen scientific finding**（SFT attenuation 0.328→~0.287、
  10.24→~8.75 全部成立）。
- RL 系列（rlpilot step0–500，本周产出）全部由 current frozen code 产生，
  内部完全一致；pilot step0（0.2871 / 8.7475）即 current-code SFT 基线。

### 对 Demo 的处置（待用户裁定）

- 候选 1：Demo 三阶段链统一改用 current-code 复算值（pretrained 复算已落
  /tmp/compat_diag/，SFT 用 pilot step0 文件），Tab 3 对拍即可逐位通过；
  历史 frozen 数值保留在 provenance 注释中。
- 候选 2：保留历史数值为主链，Tab 3 只展示存储的 per-sample delta（无四项分解）。
