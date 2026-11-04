# UIU-Net 可迁移压缩训练说明

与 DNANet 同一套协议，落在本仓库：

1. **P1 敏感性评分**（区域 SoftIoU 权重梯度 + 局部对比）
2. **P2 结构化剪枝**（按 4 档 `out_ch` 共掩码；`mid_ch` 按比例缩放）
3. **P3 蒸馏重训**（多侧输出 BCE + 区域 SoftIoU + 输出 KD）

**不改 UIU-Net 嵌套 RSU / AsymBi 拓扑**，只压缩各尺度输出通道宽度。

Stage 宽度分组（教师默认）：

| stage | 宽度 | 覆盖 |
|-------|------|------|
| 0 | 64 | stage1 / stage2d / stage1d / side1,2 |
| 1 | 128 | stage2 / stage3d / side3 / fuse2 |
| 2 | 256 | stage3 / stage4d / side4 / fuse3 |
| 3 | 512 | stage4/5/6 / stage5d / side5,6 / fuse4,5 |

---

## 0. 教师权重

全量训练产物示例：

```
saved_models/uiunet_irstd/<DATASET>_<stamp>/uiunet_best.pth.tar
```

---

## 1. 一键流水线（评分 → 剪枝 → 蒸馏）

```bash
cd IEEE_TIP_UIU-Net-main

python compress_train.py --mode pipeline \
  --teacher_ckpt saved_models/uiunet_irstd/NUAA-SIRST_20261030_151749/uiunet_best.pth.tar \
  --dataset NUAA-SIRST \
  --dataset_dir ../RDANet-main/datasets \
  --epochs 200 \
  --batch_size 3 \
  --prune_rates 0.20,0.30,0.40,0.50 \
  --work_dir ./compress_runs
```

Windows PowerShell 把 `\` 行续换成 `` ` ``，或写成一行。

输出目录：`compress_runs/<DATASET>_pipeline_<时间戳>/`

| 文件 | 含义 |
|------|------|
| `scores.pt` | 各 stage 通道重要性 |
| `keep_masks.pt` | 保留通道下标 + 学生宽度配置 |
| `student_init.pth.tar` | 剪枝后、蒸馏前学生 |
| `student_best.pth.tar` | 蒸馏过程最佳学生 |
| `distill_log.txt` | loss / IoU 日志 |
| `args.json` | 本次超参 |

四数据集可跑：

```bash
bash run_compress_all.sh
```

（先改脚本里的教师 ckpt 路径。）

---

## 2. 分步跑

```bash
python compress_train.py --mode score --teacher_ckpt TEACHER.pth.tar --dataset NUAA-SIRST
python compress_train.py --mode prune --teacher_ckpt TEACHER.pth.tar --keep_path xxx/keep_masks.pt --dataset NUAA-SIRST
python compress_train.py --mode distill --teacher_ckpt TEACHER.pth.tar \
  --keep_path xxx/keep_masks.pt --student_ckpt xxx/student_init.pth.tar --epochs 200
```

---

## 3. 关键超参

| 参数 | 默认 | 说明 |
|------|------|------|
| `--prune_rates` | `0.2,0.3,0.4,0.5` | stage0→3（浅少深多） |
| `--alpha` | `0.7` | 任务梯度 vs 对比项 |
| `--lambda_seg/kd/region` | `1/1/0.5` | 蒸馏损失 |
| `--lr` | `1e-3` | 与 UIU 教师 Adam 对齐 |
| `--epochs` | `200` | 蒸馏轮数 |
| `--score_batches` | `50` | 估分 batch 数 |

---

## 4. 与 DNANet 对比时注意

- UIU 为 **4 组宽度**；DNANet 为 5 组 `nb_filter`。剪枝率字符串长度不同。
- UIU 输出已是 **sigmoid 概率**；蒸馏 KD / SoftIoU 不再二次 sigmoid。
- 学生 ckpt 含 `enc_outs/dec_outs/mids_*`，后续导出 ONNX 时需按这些宽度重建网络。
