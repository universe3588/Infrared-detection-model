# MSHNet 可迁移压缩训练说明

与 DNANet / UIU-Net 同一套协议：

1. **P1 敏感性评分**（区域 SoftIoU 权重梯度 + 局部对比）
2. **P2 结构化剪枝**（5 档 `param_channels` 共掩码；重建学生并切片拷贝）
3. **P3 蒸馏重训**（多尺度 SoftIoU + 区域 SoftIoU + 输出 KD）

**不改** U 型拓扑 / ResNet+CBAM；只压缩通道宽度。

教师默认宽度：`[16, 32, 64, 128, 256]`

---

## 0. 教师权重

全量训练产物：

```
weight/MSHNet-<DATASET>-<stamp>/weight.pkl
```

四数据集应已存在（NUAA / NUDT / IRSTD-1k / SIRST-V2）。

---

## 1. 一键流水线

```bash
cd MSHNet-main

python compress_train.py --mode pipeline \
  --teacher_ckpt weight/MSHNet-NUAA-SIRST-2026-11-01-21-56-52/weight.pkl \
  --dataset NUAA-SIRST \
  --dataset_dir ../RDANet-main/datasets \
  --epochs 200 \
  --batch_size 4 \
  --prune_rates 0.20,0.30,0.40,0.50,0.60 \
  --work_dir ./compress_runs
```

四数据集：

```bash
bash run_compress_all.sh 2>&1 | tee compress_all.log
```

或仓库根目录：

```bash
bash scripts/run_mshnet_compress_overnight.sh
```

输出：`compress_runs/<DATASET>_pipeline_<时间戳>/`

| 文件 | 含义 |
|------|------|
| `scores.pt` | 各 stage 通道重要性 |
| `keep_masks.pt` | 保留通道下标 + 学生宽度 |
| `student_init.pth.tar` | 剪枝后、蒸馏前 |
| `student_best.pth.tar` | 蒸馏最佳（含 `param_channels`） |
| `distill_log.txt` | loss / IoU |
| `args.json` | 超参 |

---

## 2. 关键超参

| 参数 | 默认 | 说明 |
|------|------|------|
| `--prune_rates` | `0.2,0.3,0.4,0.5,0.6` | stage0→4 |
| `--epochs` | `200` | 蒸馏轮数 |
| `--lr` | `0.05` | Adagrad，对齐教师 |
| `--lambda_seg/kd/region` | `1/1/0.5` | 蒸馏损失 |

---

## 3. 与 DNANet / UIU 对比注意

- MSHNet 为 **5 组宽度**（同 DNANet）；UIU 为 4 组。
- MSHNet 输出为 **logits**（KD 用 sigmoid MSE）；UIU 教师前向已是概率。
- 学生 ckpt 含 `param_channels`，后续导出 ONNX / 板端对比时按该字段重建。
