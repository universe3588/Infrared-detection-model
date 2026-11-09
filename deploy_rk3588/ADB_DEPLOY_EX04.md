# Ex04：L2SK / RepISD → RK3588 同协议板端表

与 DNA 学生对照条件：**256×256**、`core_mask=7`、warmup20 / iters80、`board/resource_bench.py`。

| 模型 | 输入 | Params 锚点 |
|------|------|-------------|
| DNA 学生（已有） | 3×256×256 | **1.13M** |
| L2SKNet-UNet | **1×256×256**（原生灰度） | ~0.90M |
| RepISD（可选） | 3×256×256 | 导出时打印 |

> 表注必须写明：L2SK 为单通道；DNA/RepISD 为三通道（与 SCTrans 脚注同风格）。

---

## A. AutoDL：统一口径复测 IoU/Pd/Fa

```bash
cd ~/autodl-tmp/Infrared-detection-model
git pull   # 拉到含 scripts/eval_l2sk_unified.py 的提交

# 确认权重
ls "Lightweight model comparison/L2SKNet/log/"*/L2SKNet_UNet/best_miou.pth.tar

pip install scikit-image tqdm pillow -q

python scripts/eval_l2sk_unified.py \
  --l2sk-root "Lightweight model comparison/L2SKNet" \
  --data-root RDANet-main/datasets \
  --device cuda:0 \
  --thr 0.5

# 结果
cat results/ex04_baselines/l2sk_unified_latest.csv
```

只跑某一集：

```bash
python scripts/eval_l2sk_unified.py --datasets NUAA-SIRST --device cuda:0
```

---

## B. AutoDL：导出 ONNX（+ 有 toolkit2 则转 RKNN）

```bash
cd ~/autodl-tmp/Infrared-detection-model/deploy_rk3588
bash run_ex04_export_autodl.sh
```

产物：`outputs/ex04/l2sk_unet_nuaa.onnx`、`repisd_nuaa.onnx`，以及（若有 toolkit2）对应 `.rknn`。

无 toolkit2 时，在装有 rknn-toolkit2 的环境：

```bash
python convert_rknn.py --onnx outputs/ex04/l2sk_unet_nuaa.onnx \
  --out outputs/ex04/l2sk_unet_nuaa.rknn --mean 0 --std 255
python convert_rknn.py --onnx outputs/ex04/repisd_nuaa.onnx \
  --out outputs/ex04/repisd_nuaa.rknn --mean 0 0 0 --std 255 255 255
```

把 `outputs/ex04/*.rknn` 拷回本机同路径。

---

## C. 本机 ADB → RK3588（你已连板）

```powershell
cd "d:\Desktop\红外检测模型\deploy_rk3588"

adb devices
adb shell "mkdir -p /userdata/ex04_bench"

adb push outputs\ex04\l2sk_unet_nuaa.rknn /userdata/ex04_bench/
adb push outputs\ex04\repisd_nuaa.rknn /userdata/ex04_bench/
# 可选：已有 DNA 学生 rknn 一并对比
adb push outputs\student_nuaa.rknn /userdata/ex04_bench/

adb push board\resource_bench.py /userdata/ex04_bench/
adb push board\rknn_ctypes_bench.py /userdata/ex04_bench/
```

板端测（`resource_bench` 会从 RKNN 读入维度；L2SK 应为 1 通道，DNA/RepISD 为 3 通道）：

```powershell
adb shell "cd /userdata/ex04_bench && python3 resource_bench.py --rknn l2sk_unet_nuaa.rknn --tag l2sk --warmup 20 --iters 80 --csv resource_bench.csv --core_mask 7"

adb shell "cd /userdata/ex04_bench && python3 resource_bench.py --rknn repisd_nuaa.rknn --tag repisd --warmup 20 --iters 80 --csv resource_bench.csv --core_mask 7"

adb shell "cd /userdata/ex04_bench && python3 resource_bench.py --rknn student_nuaa.rknn --tag dna_student --warmup 20 --iters 80 --csv resource_bench.csv --core_mask 7"
```

拉回：

```powershell
adb pull /userdata/ex04_bench/resource_bench.csv outputs\ex04\
```

填表模板：`outputs/ex04/COMPARE_RESOURCE.md`。

---

## D. SAGCP

无公开 PDF 绝对表时，继续只用摘要相对量（见 `results/ex04_baselines/sagcp_paper_cite.md`）。你若拿到 PDF，把 Table 数字填进该文件即可。
