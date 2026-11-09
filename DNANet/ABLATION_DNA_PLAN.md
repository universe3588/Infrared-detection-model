# DNANet 消融实验方案（更稳版）

## 策略

| 数据集 | Group A 蒸馏恢复 | Group B 准则对照 | 目的 |
|--------|------------------|------------------|------|
| **NUAA-SIRST** | ✓ | ✓ | 填论文 `tab:distill` + `tab:criterion` |
| **NUDT-SIRST** | ✓ | ✗ | 专门回应合成集上学生 **0.92（+7.4 pp）** |
| IRSTD / SIRST-V2 | ✗ | ✗ | 主表已有；不做全消融以控时间 |

教师 fair 复测参考：NUAA **0.7171**；NUDT **0.8487**（学生完整配方约 **0.9226**）。

---

## 一键运行

```bash
cd Infrared-Small-Target-Detection-master
sed -i 's/\r$//' run_dna_ablation_stable.sh
nohup bash run_dna_ablation_stable.sh > logs/ablation_stable_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

可选：

```bash
SKIP_B=1 bash run_dna_ablation_stable.sh      # 只要两组 A（更省，不填准则表）
SKIP_NUDT=1 bash run_dna_ablation_stable.sh   # 只要 NUAA
```

产物：`compress_runs/ablation_stable_<时间戳>/`

```
ablation_stable_*/
  NUAA/A_distill/     # A1–A4
  NUAA/B_criterion/   # random/l1/task/contrast（full 用 A0）
  NUDT/A_distill/     # A1–A4
  summary_ablation.txt
```

粗算：NUAA（A≈3×200 + B≈4×200）+ NUDT（A≈3×200）≈ **10 次蒸馏量级**，大约 **一夜～一天半**（视 GPU；NUDT 单次通常比 NUAA 久）。

---

## Group A（NUAA & NUDT 相同结构）

同一 `keep_masks` + `student_init`，只改损失：

| ID | 设置 | λ_seg / λ_region / λ_kd |
|----|------|-------------------------|
| A1 | 剪后不重训 | eval `student_init` |
| A2 | 仅 GT SoftIoU | 1 / 0 / 0 |
| A3 | 分割 + 区域 | 1 / 0.5 / 0 |
| A4 | 完整（本文） | 1 / 0.5 / 1 |

**NUDT 重点看：** A2 是否已远超教师；A3 相对 A2 抬了多少——用来判断「0.92 是结构正则还是 region/KD 刷分」。

---

## Group B（仅 NUAA）

同压缩率 `0.2,0.3,0.4,0.5,0.6` + 完整蒸馏：

| 准则 | `--prune_criterion` |
|------|---------------------|
| 随机 | `random` |
| ℓ1 | `l1` |
| 仅 I_task | `task` |
| 仅 I_contrast | `contrast` |
| 本文 I | `full`（= NUAA A4） |

---

## 填表

- **tab:distill**：优先填 **NUAA**；正文可加一句 NUDT 同趋势/差异（附录可放 NUDT 全行）。
- **tab:criterion**：只填 **NUAA**。
- 讨论 DNA 暴涨：NUAA 给机制，**NUDT 的 A 组**给「合成集高分是否合理」的证据。
