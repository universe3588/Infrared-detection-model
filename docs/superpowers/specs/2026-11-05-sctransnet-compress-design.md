# SCTransNet 可迁移压缩设计

日期：2026-11-05  
对齐：DNANet / UIU / MSHNet（P1→P2→P3）

## 范围（方案 A）

- **只压缩** 4 档 CNN 宽度：教师 `[32, 64, 128, 256]`（`base_channel=32`）
- Transformer **层数/头数不变**；`KV_size = sum(channel_num)` 随宽度联动
- 不改 SCTB / U 型拓扑

## 协议

1. P1：Res_block.conv2 梯度 + 特征对比，按 stage 共掩码  
2. P2：重建学生 `channel_num`，切片拷贝 CNN；Transformer 能切片则切片，否则重初始化由蒸馏恢复  
3. P3：多输出 BCE（同教师）+ Region SoftIoU + 输出 KD（概率 MSE）

## 默认超参

- **v1（首轮）** `prune_rates=0.20,0.30,0.40,0.50`：难集 IRSTD/SIRST-V2 掉约 5–6 pp  
- **v2（推荐）** `prune_rates=0.10,0.20,0.30,0.40`，`epochs=300`，`lr=5e-4`，`lambda_region=0.75`  
  理由：Transformer 不剪时同等 CNN 配额过猛；仍属预算 `{p_s}` 实例化，协议不变  
- keep 强制偶数以兼容 depthwise groups；`batch=8`，`patch=256`  
- 教师：`log/<DS>/SCTransNet_*_best.pth.tar`  
- 难集优先：`bash run_compress_hardsets.sh`

## 产物

`compress_runs/<DS>_pipeline_<stamp>/student_best.pth.tar`（含 `channel_num`）
