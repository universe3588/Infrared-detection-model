# MSHNet 教师 vs 学生：资源补充测量（NUAA, 256）

对齐 DNANet / UIU `COMPARE_RESOURCE.md` 指标列。

## 0. 精度（训练日志）

| | Teacher | Student |
|--|---------|---------|
| Best IoU (NUAA) | 0.6909 | 0.6959 |

四数据集均值 IoU 教师 0.7142 / 学生 0.7140（≈持平）。

## 1. 参数量 / FLOPs

| 指标 | 教师 | 学生 | 变化 |
|------|------|------|------|
| Params | _M | _M | _% |
| FLOPs (≈2×MACs) | _G | _G | _% |

数据：`complexity_nuaa.csv`

## 2. RKNN 体积

| 指标 | 教师 | 学生 | 变化 |
|------|------|------|------|
| RKNN 体积 | _MB | _MB | _% |

## 3. 板端资源（`resource_bench.py`，80 iters）

| 指标 | 教师 | 学生 | 变化 |
|------|------|------|------|
| 延迟 mean | _ms | _ms | _% |
| 进程 RSS mean | _MB | _MB | _% |
| 估计 NPU 忙碌 ms/帧 | _ | _ | _ |
| 单位帧能耗 | — | — | 未测 |

## 4. 结论

- （待填）算力 / 体积 / 延迟 / RSS / NPU busy  
- 精度未降（见上表）
