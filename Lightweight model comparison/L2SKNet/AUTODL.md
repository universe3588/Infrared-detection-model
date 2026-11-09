# L2SKNet @ AutoDL（Ex04 从零轻量基线）

本目录已准备好服务器训练脚本，**不必在本机 RTX3060 上跑**。

## 0. 上传到 AutoDL

建议目录（示例）：

```text
/root/autodl-tmp/
  L2SKNet/                 # 本仓库
  datasets/                # 或 RDANet-main/datasets
    NUAA-SIRST/
    NUDT-SIRST/
    IRSTD-1k/              # 注意小写 k；脚本会映射到 IRSTD-1K
    SIRST-V2/
```

每个数据集需含：`images/` `masks/` `trainval.txt` `test.txt`（RDANet 布局即可）。

## 1. 环境

```bash
cd /root/autodl-tmp/L2SKNet
# 使用 AutoDL 自带 PyTorch+CUDA 镜像即可；不必死钉 requirements.txt 里的 torch==1.8
pip install -r requirements.txt --upgrade --no-deps 2>/dev/null || true
pip install einops thop scikit-image scipy opencv-python-headless tqdm Pillow matplotlib
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. 接数据

```bash
python prepare_data.py --rda-root /root/autodl-tmp/datasets
# 若数据集在 RDANet-main 下：
# python prepare_data.py --rda-root /root/autodl-tmp/RDANet-main/datasets
```

会生成：

```text
data/NUAA-SIRST/{images,masks,img_idx}
data/NUDT-SIRST/...
data/IRSTD-1K/...
data/SIRST-V2/...
```

`images/masks` 为软链；`img_idx` 为划分列表。

## 3. 训练（推荐）

主模型用论文里的 **L2SKNet_UNet**（轻、好对照）；需要更强再开 FPN。

```bash
# 冒烟 2 epoch
python train_ex04.py --model_names L2SKNet_UNet --dataset_names NUAA-SIRST \
  --batchSize 8 --nEpochs 2 --threads 4 --device cuda:0

# Ex04 四数据集正式训（400 epoch，官方默认）
python train_ex04.py --model_names L2SKNet_UNet \
  --dataset_names NUAA-SIRST NUDT-SIRST IRSTD-1K SIRST-V2 \
  --batchSize 8 --nEpochs 400 --threads 4 --device cuda:0
```

显存吃紧（尤其 IRSTD-1K 原图 512）：

```bash
python train_ex04.py --model_names L2SKNet_UNet --dataset_names IRSTD-1K --batchSize 2
```

权重输出：

```text
log/<dataset>/L2SKNet_UNet/best_miou.pth.tar
log/<dataset>/L2SKNet_UNet/last.pth.tar
log/<dataset>_L2SKNet_UNet_<time>.txt
```

一键脚本：

```bash
bash run_autodl.sh                 # 四数据集 UNet
bash run_autodl.sh FPN             # 改用 FPN
bash run_autodl.sh UNet NUAA-SIRST # 只跑 NUAA
```

## 4. 官方评测（可选）

```bash
# 官方 test.py 按 epoch 号取 ckpt；我们存的是 best_miou.pth.tar
# 可先复制一份：
cp log/NUAA-SIRST/L2SKNet_UNet/best_miou.pth.tar log/NUAA-SIRST/L2SKNet_UNet/400.pth.tar
python test.py --model_names L2SKNet_UNet --dataset_names NUAA-SIRST --test_epo 400
python cal_metrics.py --model_names L2SKNet_UNet --dataset_names NUAA-SIRST
```

训完后也可把 `best_miou.pth.tar` 拷回本机，用统一 `scripts/eval_pd_fa_unified.py` / Ex04 脚本做四集横评。

## 5. 与前面四骨干训练的对应关系（备忘）

| 骨干 | 入口 | 数据根 | 典型命令要点 |
|------|------|--------|----------------|
| DNA-Net | `DNANet/compress_train.py` | `RDANet-main/datasets` | `--mode pipeline --teacher_ckpt ... --epochs 200` |
| UIU-Net | `UIUNet/compress_train.py` | 同上 | rescale320/crop288；`--epochs 200` |
| MSHNet | `MSHNet/compress_train.py` | 同上 | `--dataset_dir ../RDANet-main/datasets --epochs 200` |
| SCTrans | `SCTransNet/compress_train.py` | 同上 | `--epochs 350 --prune_rates 0.05,0.15,0.25,0.35` |
| **L2SK（本仓库）** | `train_ex04.py` | `prepare_data.py` → `./data` | **从零训**，无教师；400 epoch Adam |

共同点：四骨干压缩管线都是「教师 → 评分剪枝 → 蒸馏」；L2SK 是 **从零轻量结构**，用作 H1 同预算对照，不是压缩学生。

## 6. 建议优先级

1. 先 `NUAA-SIRST` 跑通看 IoU  
2. 再 `NUDT-SIRST` / `IRSTD-1K` / `SIRST-V2`  
3. 需要论文主模型数字再补 `L2SKNet_FPN`
