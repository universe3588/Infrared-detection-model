# 实验数据梳理（2026-11-08）

对应稿件：`bare_jrnl_new_sample4.tex`。原始数值见 `../pd_fa_eval/` 与 `../deploy_rk3588/outputs/`。

## 1. 已齐（可关机）

| 项 | 状态 | 落点 |
|----|------|------|
| DNA/UIU/MSH 师生 IoU | 齐 | 表 `tab:main_iou` |
| SCTrans v3 师生 IoU | 齐 | 同上（均值 0.7907→0.7858） |
| 统一 Pd/Fa（32 组） | 齐 | 表 `tab:main_pdfa` |
| DNA fair re-eval | 齐 | 与主表 IoU 一致，已写入正文 |
| 四骨干 RK3588 资源 | 齐 | 表 `tab:resource`（含 SCTrans） |

## 2. 精度结论（一句话）

压缩后 **IoU / Pd 未系统性下降**；DNA 学生 IoU 明显上升（结构正则+区域损失）；SCTrans 均值略降 0.49pp（IRSTD/SIRST-V2 拖累）；Pd 四骨干均值学生≥教师；Fa 多数持平，IRSTD 上个别升高可接受。

### IoU 主表均值

| 骨干 | 教师 | 学生 | Δ |
|------|------|------|---|
| DNA | 0.7250 | 0.7696 | +4.46pp |
| UIU | 0.7804 | 0.7819 | +0.15pp |
| MSH | 0.7142 | 0.7140 | −0.02pp |
| SCTrans | 0.7907 | 0.7858 | −0.49pp |

### Pd 均值（统一脚本）

| 骨干 | 教师 | 学生 |
|------|------|------|
| DNA | 0.9372 | 0.9599 |
| UIU | 0.9560 | 0.9658 |
| MSH | 0.9498 | 0.9526 |
| SCTrans | 0.9506 | 0.9690 |

## 3. 板端结论（NUAA@256）

| 骨干 | Params | FLOPs | 延迟 |
|------|--------|-------|------|
| DNA | −75.8% | −53.6% | −11.7% |
| UIU | −72.5% | −57.1% | −38.6% |
| MSH | −78.2% | −61.5% | −21.9% |
| SCTrans | −49.5% | −34.2% | −14.3% |

共性：**FLOPs↓ 不等于 延迟等比↓**（DNA / SCTrans 最明显）。SCTrans 压缩比更弱，因 Transformer 层未剪。

## 4. DNA 消融（已跑完，已填表）

来源：`Infrared-Small-Target-Detection-master/compress_runs/.../ablation_stable_*` + `ablation_repair_*`。

**蒸馏 Group A**

| 设置 | NUAA | NUDT |
|------|------|------|
| A1 不重训 | 0.0081 | 0.2183 |
| A2 仅 GT | 0.7454 | 0.9149 |
| A3 +region | 0.7551 | 0.9180 |
| A4 +KD | 0.7188（复现）/ 主表 0.7387 | 0.9184（λ_kd=0.5；λ_kd=1 曾崩） |

**准则 Group B（NUAA）**

| 准则 | IoU |
|------|-----|
| random | 0.7247 |
| l1 | 0.7595（修复后；首次崩） |
| task | 0.7374 |
| contrast | 0.7408 |
| full I | 0.7188 / 主表 0.7387 |

仍缺（可选）：准则各变体的统一脚本 Pd/Fa；α/配额敏感性曲线。

## 5. 仍缺（不挡关机；投稿增强）

- P1：Pareto / 定性图；准则 Pd/Fa（可选）
- P2：敲除曲线、外部基线 H1、INT8、功耗计、失败案例

## 6. 口径备忘

- 主表 **IoU**：各骨干原 compress / 训练评测管线（UIU=rescale320 SoftIoU 日志）。
- 表 **Pd/Fa**：`scripts/eval_pd_fa_unified.py`，thr=0.5，质心<3；旁证 IoU 为全局像素 IoU，**勿替换** UIU 主表 IoU。
- SCTrans 配额 v3：`p=(0.05,0.15,0.25,0.35)`，蒸馏 350 epoch，`λ_region=0.75`。
