# MATLAB 出图目录规范

每个主题一个子文件夹，内含独立的 `data/` 与 `figures/`，避免互相覆盖、便于辨认。

```
images/Matlab/
├── README.md
├── 敲除校准曲线/          # Ex08 + 消融柱状图
│   ├── data/
│   ├── figures/
│   ├── plot_*.m
│   └── run_all_figures.m
└── 精度-复杂度Pareto曲线/  # Fig.2 Ex06
    ├── data/
    ├── figures/
    ├── plot_pareto.m
    └── run_all_figures.m
```

## 运行

```matlab
cd('.../images/Matlab/精度-复杂度Pareto曲线'); run_all_figures
cd('.../images/Matlab/敲除校准曲线'); run_all_figures
```
