# Ex04 板端对照（DNA 学生 vs L2SK vs RepISD）

条件：RK3588、librknnrt、`core_mask=7`、warmup20 / iters80、输入 **256×256**。

| 模型 | 通道 | Params(M) | FLOPs(G) | RKNN(MB) | 延迟 mean(ms) | RSS mean(MB) | NPU busy(ms/帧) |
|------|------|-----------|----------|----------|---------------|--------------|-----------------|
| DNA 学生 | 3 | 1.13 | 12.92 | （填） | （填 / 或沿用 913） | （填） | （填） |
| L2SKNet-UNet | **1** | （complexity.csv） | （填） | （填） | （填） | （填） | （填） |
| RepISD | 3 | （complexity.csv） | （填） | （填） | （填） | （填） | （填） |

原始：`complexity.csv`、`resource_bench.csv`。

注：L2SK 单通道与 DNA/RepISD 三通道不可直接把 FLOPs 当成同输入公平；Params 仍可作同量级对照主轴。
