# ADB 部署 UIU-Net 教师 vs 学生（对齐 DNANet 资源对比表）

目标指标（与 DNANet `COMPARE_RESOURCE.md` 同列）：

| 指标 | 来源 |
|------|------|
| Params / FLOPs | AutoDL `measure_uiunet_complexity.py`（或 ONNX 版） |
| RKNN 体积 | `*.rknn` 文件大小 |
| 延迟 mean | 板端 `resource_bench.py` |
| 进程 RSS mean | 同上 |
| 估计 NPU 忙碌 ms/帧 | `latency × mean(NPU%)` |
| 单位帧能耗 | 本次仍不测（需功率计） |

默认协议：**NUAA-SIRST，input 256×256，FP16 RKNN，core_mask=7**（与 DNANet 板端一致）。

---

## 0. 前置检查

本机 `compress_runs` 若只有 `distill_log.txt` / `args.json`、**没有** `student_best.pth.tar`，必须先在 AutoDL 确认权重存在：

```bash
ls -lh IEEE_TIP_UIU-Net-main/compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar
ls -lh IEEE_TIP_UIU-Net-main/saved_models/uiunet_irstd/NUAA-SIRST_*/uiunet_best.pth.tar
```

---

## 1. AutoDL：导出 ONNX + 测 Params/FLOPs（+ 可选转 RKNN）

```bash
cd ~/autodl-tmp/Infrared-detection-model
git pull   # 拉下本目录新增脚本
sed -i 's/\r$//' deploy_rk3588/run_uiunet_export_autodl.sh
bash deploy_rk3588/run_uiunet_export_autodl.sh
```

产物：

```
deploy_rk3588/outputs/uiunet/
  teacher_nuaa.onnx / student_nuaa.onnx
  complexity_nuaa.csv
  teacher_nuaa.rknn / student_nuaa.rknn   # 若已装 toolkit2
```

若 AutoDL 无 toolkit2：只下 ONNX，在另一台有 `rknn-toolkit2` 的机器执行：

```bash
cd deploy_rk3588
python convert_rknn.py --onnx outputs/uiunet/teacher_nuaa.onnx --out outputs/uiunet/teacher_nuaa.rknn
python convert_rknn.py --onnx outputs/uiunet/student_nuaa.onnx --out outputs/uiunet/student_nuaa.rknn
```

---

## 2. 本机：ADB 推到 RK3588

```powershell
cd "D:\Desktop\红外检测模型\deploy_rk3588"

adb shell "mkdir -p /userdata/uiunet_bench"
adb push outputs\uiunet\teacher_nuaa.rknn /userdata/uiunet_bench/
adb push outputs\uiunet\student_nuaa.rknn /userdata/uiunet_bench/
adb push board\rknn_ctypes_bench.py /userdata/uiunet_bench/
adb push board\resource_bench.py /userdata/uiunet_bench/
adb push board\collect_system_stats.sh /userdata/uiunet_bench/
```

若板端 `librknnrt.so` 不在默认路径，按 DNANet 部署时同样方式放到 `/usr/lib` 或改脚本 `--lib`。

---

## 3. 板端资源测量（教师 / 学生各一遍）

```powershell
adb shell "cd /userdata/uiunet_bench && python3 resource_bench.py --rknn teacher_nuaa.rknn --tag teacher --warmup 20 --iters 80 --csv resource_bench.csv"
adb shell "cd /userdata/uiunet_bench && python3 resource_bench.py --rknn student_nuaa.rknn --tag student --warmup 20 --iters 80 --csv resource_bench.csv"
```

拉回：

```powershell
mkdir -p outputs\uiunet\resource
adb pull /userdata/uiunet_bench/resource_bench.csv .\outputs\uiunet\resource\
adb pull /userdata/uiunet_bench/resource_bench_teacher_npu_samples.txt .\outputs\uiunet\resource\
adb pull /userdata/uiunet_bench/resource_bench_student_npu_samples.txt .\outputs\uiunet\resource\
```

（采样文件名以板端实际打印为准。）

---

## 4. 填表模板

把 `complexity_nuaa.csv` + `resource_bench.csv` + 两个 `.rknn` 体积填入：

`outputs/uiunet/COMPARE_RESOURCE.md`

结论写法对齐 DNANet：算力是否下降、墙钟延迟是否同比例下降、NPU busy 是否近似持平。

---

## 精度对照（日志，非正式板端）

| | Teacher best IoU | Student best IoU |
|--|------------------|------------------|
| NUAA-SIRST | 0.7686 | 0.7741 |

资源对比与精度结论分开写；板端表只报资源列。
