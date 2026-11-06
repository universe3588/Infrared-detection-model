# SCTransNet 可迁移压缩训练说明

与 DNANet / UIU / MSHNet **同一协议**（方案 A）：

1. **P1 敏感性评分**（区域 SoftIoU 梯度 + 局部对比）
2. **P2 结构化剪枝**（4 档 CNN 宽度共掩码；`KV_size=sum(channels)` 联动）
3. **P3 蒸馏重训**（多头 BCE + 区域 SoftIoU + 输出 KD）

**不改** Transformer 层数/头数与 U 型拓扑；只压通道。  
DNA / UIU / MSH **无需因本骨干调参而重训**。

教师默认宽度：`[32, 64, 128, 256]`（`base_channel=32`，`KV_size=480`）

---

## 为何 SCTrans 默认配额更保守？

首轮用与四尺度 CNN 相同的 `0.20,0.30,0.40,0.50` 时：

| 数据集 | 教师 | 学生 | Δ |
|--------|------|------|---|
| NUAA | 0.7765 | 0.7584 | −1.8 pp（可接受） |
| NUDT | 0.9444 | 0.9562 | +1.2 pp |
| IRSTD-1k | 0.6680 | 0.6146 | **−5.3 pp** |
| SIRST-V2 | 0.7735 | 0.7097 | **−6.4 pp** |

Transformer 不参与剪枝时，同等 CNN 剪枝率对难集过猛。论文叙事上这仍是式(1)预算 $B$ 的 `{p_s}` 实例化（骨干相关、浅层保护），**不是换协议**。

**v2 默认（仅 SCTrans）：**

| 参数 | 首轮 | v2 |
|------|------|-----|
| `prune_rates` | 0.20,0.30,0.40,0.50 | **0.10,0.20,0.30,0.40** |
| `epochs` | 200 | **300** |
| `lr` | 1e-3 | **5e-4** |
| `lambda_region` | 0.5 | **0.75** |

预期通道约 `[28, 52, 90, 154]`（偶数对齐后以日志为准），参数压缩会弱于首轮 ~67%，但精度应更接近教师。

---

## 1. 推荐：只重跑难集

```bash
cd SCTransNet-main
bash run_compress_hardsets.sh
# 或
DATASETS="IRSTD-1k SIRST-V2" bash run_compress_all.sh 2>&1 | tee compress_hardsets_v2.log
```

AutoDL 过夜（默认难集）：

```bash
bash scripts/run_sctrans_compress_overnight.sh
# 四集全量：FULL=1 bash scripts/run_sctrans_compress_overnight.sh
```

---

## 2. 单数据集 / 四数据集

```bash
python compress_train.py --mode pipeline \
  --teacher_ckpt log/IRSTD-1k/SCTransNet_*_best.pth.tar \
  --dataset IRSTD-1k \
  --dataset_dir ../RDANet-main/datasets \
  --epochs 300 --lr 5e-4 \
  --prune_rates 0.10,0.20,0.30,0.40 \
  --lambda_region 0.75 \
  --work_dir ./compress_runs
```

```bash
bash run_compress_all.sh 2>&1 | tee compress_all_v2.log
```

产物：`compress_runs/<DATASET>_pipeline_<时间戳>/`

| 文件 | 含义 |
|------|------|
| `scores.pt` / `keep_masks.pt` | 评分与保留通道 |
| `student_init.pth.tar` | 剪枝后 |
| `student_best.pth.tar` | 蒸馏最佳（含 `channel_num`） |
| `distill_log.txt` / `args.json` | 日志与超参 |

---

## 3. 关键超参

| 参数 | v2 默认 | 说明 |
|------|---------|------|
| `--prune_rates` | `0.1,0.2,0.3,0.4` | 4 档；keep 强制偶数 |
| `--epochs` | `300` | 蒸馏轮数 |
| `--lr` | `5e-4` | Adam + Cosine |
| `--lambda_region` | `0.75` | 强化小目标区域 |
| `--batch_size` | `8` | 按显存调整 |

若难集仍掉 >3 pp，可再试更保守：`PRUNE_RATES=0.05,0.15,0.25,0.35`（仍不改框架）。

---

## 4. 注意

- 输出已是 **sigmoid 概率**；KD / SoftIoU 不再二次 sigmoid。
- Attention depthwise 要求通道为偶数，评分后 `choose_keep_indices(force_even=True)`。
- Transformer 权重 best-effort 切片；切不动的层重初始化，靠蒸馏恢复。
- 首轮权重可保留作消融对照（激进配额 vs 保守配额）。
