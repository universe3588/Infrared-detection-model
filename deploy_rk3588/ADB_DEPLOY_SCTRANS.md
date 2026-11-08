# ADB 部署 SCTransNet 教师 vs 学生（对齐 DNA/UIU/MSH 资源表）

协议：**NUAA-SIRST，input 256×256，FP16 RKNN，core_mask=7**  
输入：**1 通道灰度**（与 SCTrans 训练一致；DNA/UIU/MSH 为 3 通道，表中注明即可）

指标：Params / FLOPs / RKNN 体积 / 延迟 mean / RSS mean / 估计 NPU 忙碌 ms/帧

---

## 0. 前置：学生权重

服务器上需有 v3 学生，例如：

```bash
ls -lh SCTransNet-main/compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar
# ckpt 内应含 channel_num≈[30,54,96,166]
```

若本地有、服务器没有，先把该 `student_best.pth.tar` 传到对应 `compress_runs/.../`。

教师：

```bash
ls -lh SCTransNet-main/log/NUAA-SIRST/SCTransNet_*_best.pth.tar
```

---

## 1. AutoDL：导出 ONNX + 测 Params/FLOPs（+ 可选转 RKNN）

```bash
cd ~/autodl-tmp/Infrared-detection-model
# 同步本仓库 deploy_rk3588 改动后：
sed -i 's/\r$//' deploy_rk3588/run_sctrans_export_autodl.sh
bash deploy_rk3588/run_sctrans_export_autodl.sh
```

产物：`deploy_rk3588/outputs/sctrans/{teacher,student}_nuaa.{onnx,rknn}` + `complexity_nuaa.csv`

若 AutoDL 无 rknn-toolkit2，只生成 ONNX；本机再转：

```bash
cd deploy_rk3588
python convert_rknn.py --onnx outputs/sctrans/teacher_nuaa.onnx --out outputs/sctrans/teacher_nuaa.rknn --mean 0 --std 255
python convert_rknn.py --onnx outputs/sctrans/student_nuaa.onnx --out outputs/sctrans/student_nuaa.rknn --mean 0 --std 255
```

---

## 2. 本机 ADB 上板

```powershell
cd "D:\Desktop\红外检测模型\deploy_rk3588"

adb shell "mkdir -p /userdata/sctrans_bench"
adb push outputs\sctrans\teacher_nuaa.rknn /userdata/sctrans_bench/
adb push outputs\sctrans\student_nuaa.rknn /userdata/sctrans_bench/
adb push board\rknn_ctypes_bench.py /userdata/sctrans_bench/
adb push board\resource_bench.py /userdata/sctrans_bench/

adb shell "cd /userdata/sctrans_bench && python3 resource_bench.py --rknn teacher_nuaa.rknn --tag teacher --warmup 20 --iters 80 --csv resource_bench.csv"
adb shell "cd /userdata/sctrans_bench && python3 resource_bench.py --rknn student_nuaa.rknn --tag student --warmup 20 --iters 80 --csv resource_bench.csv"

mkdir outputs\sctrans\resource -Force
adb pull /userdata/sctrans_bench/resource_bench.csv .\outputs\sctrans\resource\
```

填表：`outputs/sctrans/COMPARE_RESOURCE.md`

---

## 3. 常见问题

- **通道**：导出默认 `--in_channels 1`；转 RKNN 必须 `--mean 0 --std 255`（单通道）。
- **学生 missing keys 过多**：检查 ckpt 是否含 `channel_num`，或手动  
  `python export_sctrans_onnx.py --ckpt ... --channels 30,54,96,166`
- **Transformer 算子转 RKNN 失败**：把完整报错贴出来；必要时先用 ONNXRuntime 板外对照，再针对性改 opset / 简化导出。
