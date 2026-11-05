# ADB 部署 MSHNet 教师 vs 学生（对齐 DNANet / UIU 资源表）

协议：**NUAA-SIRST，input 256×256，FP16 RKNN，core_mask=7**

指标：Params / FLOPs / RKNN 体积 / 延迟 mean / RSS mean / 估计 NPU 忙碌 ms/帧

---

## 1. AutoDL 导出

```bash
cd ~/autodl-tmp/Infrared-detection-model
git pull
sed -i 's/\r$//' deploy_rk3588/run_mshnet_export_autodl.sh
# 确认学生权重存在
ls -lh MSHNet-main/compress_runs/NUAA-SIRST_pipeline_*/student_best.pth.tar
bash deploy_rk3588/run_mshnet_export_autodl.sh
```

产物：`deploy_rk3588/outputs/mshnet/{teacher,student}_nuaa.{onnx,rknn}` + `complexity_nuaa.csv`

---

## 2. 本机 ADB 上板

把 `outputs/mshnet/*.rknn` 拉到本机后：

```powershell
cd "D:\Desktop\红外检测模型\deploy_rk3588"

adb shell "mkdir -p /userdata/mshnet_bench"
adb push outputs\mshnet\teacher_nuaa.rknn /userdata/mshnet_bench/
adb push outputs\mshnet\student_nuaa.rknn /userdata/mshnet_bench/
adb push board\rknn_ctypes_bench.py /userdata/mshnet_bench/
adb push board\resource_bench.py /userdata/mshnet_bench/

adb shell "cd /userdata/mshnet_bench && python3 resource_bench.py --rknn teacher_nuaa.rknn --tag teacher --warmup 20 --iters 80 --csv resource_bench.csv"
adb shell "cd /userdata/mshnet_bench && python3 resource_bench.py --rknn student_nuaa.rknn --tag student --warmup 20 --iters 80 --csv resource_bench.csv"

mkdir outputs\mshnet\resource -Force
adb pull /userdata/mshnet_bench/resource_bench.csv .\outputs\mshnet\resource\
```

填表：`outputs/mshnet/COMPARE_RESOURCE.md`
