# SCTransNet 教师 vs 学生（NUAA，256，RK3588）

> 填完 `complexity_nuaa.csv` 与板端 `resource/resource_bench.csv` 后更新本表。  
> 输入为 **1 通道**；DNA/UIU/MSH 为 3 通道，跨骨干延迟对比时请注明。

## 1. 参数量 / FLOPs（`complexity_nuaa.csv`）

| 指标 | 教师 | 学生 | 变化 |
|------|------|------|------|
| Params (M) |  |  |  |
| FLOPs (≈2×MACs) |  |  |  |
| 通道 | [32,64,128,256] | [30,54,96,166]（v3） |  |

## 2. RKNN 体积

|  | 教师 | 学生 | 变化 |
|--|------|------|------|
| RKNN (MB) |  |  |  |

## 3. 板端（`resource_bench`，warmup20 / iters80）

| 指标 | 教师 | 学生 | 变化 |
|------|------|------|------|
| 延迟 mean (ms) |  |  |  |
| RSS mean (MB) |  |  |  |
| NPU busy (ms/帧) |  |  |  |

## 4. 精度备忘（v3，非板端）

|  | IoU |
|--|-----|
| 教师（训练日志 mIoU） | 0.7765 |
| 学生 v3 | 0.7849 |
