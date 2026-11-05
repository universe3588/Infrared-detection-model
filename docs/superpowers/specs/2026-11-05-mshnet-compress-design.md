# MSHNet 可迁移压缩设计（已确认）

日期：2026-11-05  
范围：仅 MSHNet（SCTransNet 延后）

## 目标

与 DNANet / UIU-Net 同一协议，对已训教师做结构化通道剪枝 + 蒸馏，四数据集全量跑通，产物可后续 RK3588 导出。

## 协议

1. **P1** 敏感性评分：区域 SoftIoU 梯度 + 局部对比，按 stage 共掩码  
2. **P2** 结构化剪枝：缩小 `param_channels=[16,32,64,128,256]`，重建学生并切片拷贝权重  
3. **P3** 蒸馏：SoftIoU（多尺度）+ Region SoftIoU + 输出 KD  

不改 U 型拓扑与 ResNet+CBAM 结构。

## 超参默认

- `prune_rates=0.20,0.30,0.40,0.50,0.60`
- `epochs=200`，`lr=0.05` Adagrad，`base/crop=256`
- `lambda_seg/kd/region=1/1/0.5`

## 教师权重

`MSHNet-main/weight/MSHNet-<DS>-*/weight.pkl`（四数据集已有）

## 产物

`compress_runs/<DS>_pipeline_<stamp>/`：`scores.pt`、`keep_masks.pt`、`student_init/best.pth.tar`、`distill_log.txt`、`args.json`
