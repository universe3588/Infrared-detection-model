# DNANet 教师 vs 学生：资源补充测量（NUAA, 256）

测量时间：2026-11-01。条件：RK3588、librknnrt 2.5.0、drv 0.7.2、core_mask=7。

## 1. 参数量 / FLOPs（ONNX 图，本机 `measure_onnx_complexity.py`）

| | 教师 | 学生 | 相对教师 |
|--|------|------|----------|
| Params | 4.688 M | 1.134 M | **−75.8%** |
| MACs | 13.927 G | 6.461 G | **−53.6%** |
| FLOPs (≈2×MACs) | 27.854 G | 12.921 G | **−53.6%** |
| ONNX 体积 | 18.02 MB | 4.46 MB | −75.2% |
| RKNN 体积 | 14.61 MB | 8.00 MB | −45.2% |

## 2. 板端进程内存 + NPU 忙碌估计（`board/resource_bench.py`，80 iters）

| | 教师 | 学生 | 相对教师 |
|--|------|------|----------|
| latency mean | 1033.7 ms | 913.1 ms | **−11.7%** |
| FPS | 0.97 | 1.10 | +13.4% |
| RSS mean | 90677 kB | 80486 kB | **−11.2%** |
| RSS peak | 90736 kB | 80552 kB | **−11.2%** |
| RSS−idle | 7425 kB | 6654 kB | −10.4% |
| NPU avg load | 24.1% | 28.0% | +3.9 pt |
| est busy ms/frame | 249.3 | 255.5 | ≈持平 |

说明：`est_busy ≈ latency × mean(NPU%)`，是忙碌时间启发式，**不是功耗计**。单位帧能耗本次无法测（板端无可靠功率节点）。

## 3. 结论（可写论文）

- **算力侧资源确实下降**：Params −76%、FLOPs −54%。
- **板端墙钟只降约 12%**，与「体积/算力砍半 ≠ 时延砍半」一致。
- **进程 RSS 约降 11%**（比 MemAvailable 有说服力，但仍是十兆字节量级）。
- **估计 NPU 忙碌时间基本持平**：学生利用率略高、帧时略短，乘积接近 → 不宜说「NPU 更闲」。
- **能耗**：需外接功率计或板厂工具；本次未测。

原始数据：`complexity_nuaa_onnx.csv`、`resource_bench.csv`。
