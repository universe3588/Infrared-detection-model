# UIU-Net 教师 vs 学生：资源补充测量（NUAA, 256）

> 填表方式对齐 DNANet `outputs/resource/COMPARE_RESOURCE.md`。  
> 跑完 `run_uiunet_export_autodl.sh` + 板端 `resource_bench.py` 后把数字填入。

## 0. 精度（训练日志，非正式板端复测）

| | Teacher | Student |
|--|---------|---------|
| Best IoU (NUAA) | 0.7686 | 0.7741 |

## 1. 参数量 / FLOPs（`measure_uiunet_complexity.py`）

| 指标 | 教师 | 学生 | 变化 |
|------|------|------|------|
| Params | _M | _M | _% |
| FLOPs (≈2×MACs) | _G | _G | _% |

数据文件：`complexity_nuaa.csv`

## 2. RKNN 体积

| 指标 | 教师 | 学生 | 变化 |
|------|------|------|------|
| RKNN 体积 | _MB | _MB | _% |

## 3. 板端进程内存 + NPU 忙碌估计（`resource_bench.py`，建议 80 iters）

| 指标 | 教师 | 学生 | 变化 |
|------|------|------|------|
| 延迟 mean | _ms | _ms | _% |
| 进程 RSS mean | _MB | _MB | _% |
| 估计 NPU 忙碌 ms/帧 | _ | _ | _ |
| 单位帧能耗 | — | — | 未测（需功率计） |

数据文件：`resource/resource_bench.csv`

## 4. 结论（可直接支撑论文表述）

- （待填）Params / FLOPs / RKNN 体积变化  
- （待填）墙钟延迟与 NPU busy 是否同 DNANet 一样「算力腰斩 ≠ 时延腰斩」  
- 能耗本轮不测
