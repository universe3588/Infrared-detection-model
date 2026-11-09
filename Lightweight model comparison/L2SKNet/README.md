# L2SKNet
**[IEEE TGRS] Implementation of our paper "Saliency at the Helm: Steering Infrared Small Target Detection with Learnable Kernels".** [**Paper**](https://ieeexplore.ieee.org/document/10813615)

<div align="center">
  <img src="https://github.com/user-attachments/assets/2d449c88-529c-4c75-bcc2-fab154f21380" alt="image" width="700"/>
</div>



<p align="center"> Highlighting our domain-aware LLSKM, unfolding the 'Center substracts Neighbors' pattern.</p>

## Requirements
- **Python 3.8**
- **Windows10, Ubuntu18.04 or higher**
- **NVDIA GeForce RTX 3090**
- **pytorch 1.8.0 or higher**
- **More details from requirements.txt** 

## Datasets

**We used the NUDT-SIRST, IRSTD-1K and sirst-aug for both training and test. Three datasets can be found and downloaded in:** [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection), [IRSTD-1K](https://github.com/RuiZhang97/ISNet), [SIRST-Aug](https://github.com/Tianfang-Zhang/AGPCNet). 

**Please first download these datasets and place the 3 datasets to the folder `./data/`.** 



* **The dataset in our project has the following structure:**
```
├──./data/
│    ├── sirst_aug
│    │    ├── images
│    │    │    ├── 000000.png
│    │    │    ├── 000001.png
│    │    │    ├── ...
│    │    ├── img_idx
│    │    │    ├── test.txt
│    │    │    ├── train.txt
│    │    ├── masks
│    │    │    ├── 000000_mask.png
│    │    │    ├── 000001_mask.png
│    │    │    ├── ...
│    ├── ...
```
<br>

## Commands for Training
* **Install the environment according to** `requirements.txt` **.**

* **Enter the repo, and run** `train_device0.py` **to perform network training:**
```bash
$ python train_device0.py --model_names L2SKNet_FPN --dataset_names NUDT-SIRST IRSTD-1K SIRST-aug
```
* **The** `model_name` **in our code corresponds to the model name in our paper as follows:**

  `L2SKNet_FPN` for L2SKNet-FPN; 

  `L2SKNet_UNet` for L2SKNet-UNet; 

  `L2SKNet_1D_FPN` for L2SKNet-FPN*;

  `L2SKNet_1D_UNet` for L2SKNet-UNet*.

  Note: The 'Recip' version is on the way out.
* **Checkpoints and Logs will be saved to** `./log/`**, and** `./log/` **has the following structure:**
```
├──./log/
│    ├── [dataset_name]
│    │   ├── [model_name]
│    │   │    ├── 1.pth.tar
│    │   │    ├── 2.pth.tar
│    │   │    ├── ...
│    ├── [dataset_name]_[model_name]_[time].txt
```
## Commands for Evaluate your own results
* **Run** `test.py` **to generate file of the format .mat and .png (`--test_epo 200` means test with the 200th epoch model):**
```bash
$ python test.py --model_names L2SKNet_FPN --dataset_names NUDT-SIRST IRSTD-1K SIRST-aug --test_epo 200
```
* **The file generated will be saved to** `./result/` **that has the following structure**:
```
├──./result/
│    ├── [dataset_name]
│    │   ├── img
│    │   │    ├── [model_name]
│    │   │    │    ├── 000000.png
│    │   │    │    ├── 000001.png
│    │   │    │    ├── ...
│    │   ├── mat
│    │   │    ├── [model_name]
│    │   │    │    ├── 000000.mat
│    │   │    │    ├── 000001.mat
│    │   │    │    ├── ...
```
* **Run** `cal_metrics.py` **for direct evaluation**:
```bash
$ python cal_metrics.py --model_names L2SKNet_FPN --dataset_names NUDT-SIRST IRSTD-1K SIRST-aug
```
* **The file generated will be saved to** `./result/` **that has the following structure**:
```
├──./result/
│    ├── [dataset_name]_[model_name]_[time].txt
│    ├── [dataset_name]_[model_name].mat
```

## Commands for parameters/FLOPs and runtimes calculation
* **Run** `t_models.py` **for parameters and FLOPs calculation:**
```bash
$ python t_models.py
```
* **Run** `t_time.py` **for runtimes calculation:**
```bash
$ python t_time.py
```

## Acknowledgement
We extend our sincere gratitude to Xinyi Ying and colleagues for their outstanding toolbox, [BasicIRSTD (Ver. July 24, 2023)](https://github.com/XinyiYing/BasicIRSTD). Additionally, we would like to thank [Luping Zhang](https://github.com/lupingzhang) for his invaluable contributions to this repository.

## Contact
For any questions regarding this paper or the code, please feel free to reach out to [wufengyi98@163.com](wufengyi98@163.com).

## Citation
```
@ARTICLE{Wu_2024_TGRS,
    author    = {Wu, Fengyi and Liu, Anran and Zhang, Tianfang and Zhang, Luping and Luo, Junhai and Peng, Zhenming},
    title     = {Saliency at the Helm: Steering Infrared Small Target Detection with Learnable Kernels},
    booktitle = {IEEE Transactions on Geoscience and Remote Sensing},
    year      = {2024},
    doi       = {10.1109/TGRS.2024.3521947}
}
```
