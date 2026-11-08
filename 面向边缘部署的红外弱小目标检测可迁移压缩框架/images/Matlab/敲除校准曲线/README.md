# 敲除校准曲线 / 消融对照图（Ex08）

本目录仅存放该主题的代码、CSV 与出图结果。

```
敲除校准曲线/
├── data/           # criterion / distill / knockout CSV
├── figures/        # PNG/PDF/FIG
├── plot_criterion_bar.m
├── plot_distill_bar.m
├── plot_knockout_curves.m
└── run_all_figures.m
```

```matlab
cd('.../images/Matlab/敲除校准曲线')
run_all_figures
```

- `fig_criterion_bar` / `fig_distill_bar`：实测消融
- `fig_knockout_curves`：示意敲除曲线（默认 SCHEMATIC 水印）
