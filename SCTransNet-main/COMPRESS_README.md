# SCTransNet 可迁移压缩训练说明

与 DNANet / UIU / MSHNet **同一协议**（方案 A）：

1. **P1 敏感性评分**（区域 SoftIoU 梯度 + 局部对比）
2. **P2 结构化剪枝**（4 档 CNN 宽度共掩码；`KV_size=sum(channels)` 联动）
3. **P3 蒸馏重训**（多头 BCE + 区域 SoftIoU + 输出 KD）

**不改** Transformer 层数/头数与 U 型拓扑；只压通道。

教师默认宽度：`[32, 64, 128, 256]`（`base_channel=32`，`KV_size=480`）

---

## 0. 教师权重

```
log/<DATASET>/SCTransNet_*_best.pth.tar
```

四数据集应已有（NUAA / NUDT / IRSTD-1k / SIRST-V2）。

---

## 1. 一键流水线

```bash
cd SCTransNet-main

python compress_train.py --mode pipeline \
  --teacher_ckpt log/NUAA-SIRST/SCTransNet_781_best.pth.tar \
  --dataset NUAA-SIRST \
  --dataset_dir ../RDANet-main/datasets \
  --epochs 200 \
  --batch_size 8 \
  --prune_rates 0.20,0.30,0.40,0.50 \
  --work_dir ./compress_runs
```

四数据集：

```bash
bash run_compress_all.sh 2>&1 | tee compress_all.log
```

或仓库根目录过夜：

```bash
bash scripts/run_sctrans_compress_overnight.sh
```

产物：`compress_runs/<DATASET>_pipeline_<时间戳>/`

| 文件 | 含义 |
|------|------|
| `scores.pt` / `keep_masks.pt` | 评分与保留通道 |
| `student_init.pth.tar` | 剪枝后 |
| `student_best.pth.tar` | 蒸馏最佳（含 `channel_num`） |
| `distill_log.txt` / `args.json` | 日志与超参 |

---

## 2. 关键超参

| 参数 | 默认 | 说明 |
|------|------|------|
| `--prune_rates` | `0.2,0.3,0.4,0.5` | 4 档（keep 强制偶数） |
| `--epochs` | `200` | 蒸馏轮数 |
| `--lr` | `1e-3` | Adam，对齐教师 |
| `--batch_size` | `8` | 教师+学生同卡可按显存改 |

---

## 3. 注意

- 输出已是 **sigmoid 概率**；KD / SoftIoU 不再二次 sigmoid。
- Attention depthwise 要求通道为偶数，评分后 `choose_keep_indices(force_even=True)`。
- Transformer 权重 best-effort 切片；切不动的层重初始化，靠蒸馏恢复。
