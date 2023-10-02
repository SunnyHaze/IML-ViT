# IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer
![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch) 
![last commit](https://img.shields.io/github/last-commit/Sunnyhaze/IML-ViT)
![GitHub](https://img.shields.io/github/license/Sunnyhaze/IML-ViT?logo=license)
![](https://img.shields.io/github/repo-size/sunnyhaze/IML-ViT?color=green)
![](https://img.shields.io/github/stars/sunnyhaze/IML-ViT)
[![Ask Me Anything !](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/Sunnyhaze) 

This repo contains an official PyTorch implementation of our paper: [IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer.](http://arxiv.org/abs/2307.14863)

![overview](./images/overview.png)


## News 
- **2023.10.03** 🎉🎉 Our related work that applies Contrastive learning on the image manipulation task to solve data insufficiency problem, **NCL-IML**, is accepted by ***ICCV2023***!🎉🎉
  - [[paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Pre-Training-Free_Image_Manipulation_Localization_through_Non-Mutually_Exclusive_Contrastive_Learning_ICCV_2023_paper.html)][[code](https://github.com/Knightzjz/NCL-IML)]

## Environment
Ubuntu LTS 20.04.1

CUDA 11.7 + cudnn 8.4.0

Python 3.8

PyTorch 1.11

## Requirements
You should install the packages in [requirements.txt](./requirements.txt) with `pip install -r requirements.txt` first.

## Quick Start
### A simple Demo
Currently, You can follow the tutorial to experience the running pipeline of IML-ViT.
- Step 1: You should download the pre-trained IML-ViT weights from [Google Drive](https://drive.google.com/file/d/1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7/view?usp=sharing) or [Baidu NetDisk](https://pan.baidu.com/s/1V-l1C6jCLBQTobrJcXDl7g?pwd=s835) and place it as `./checkpoints/iml-vit_checkpoint.pth`
- Step 2: You can follow the instructions in [Demo.ipynb](./Demo.ipynb) to see how we pad the images and inference with the IML-ViT. 

### Training
This part will be released after careful proofreading of our code. Before that, you could prepare your own training code with 
our released model like this:
```python
  model = iml_vit_model.iml_vit_model(
      vit_pretrain_path = args.vit_pretrain_path,
      edge_lambda = args.edge_lambda
  )
```
`vit_pretrain_path` is the path for MAE pre-trained ViT weights, you can download it in this [guide](./pretrained-weights/mae_download_page.md).

We have prepared the [Naive IML transforms class](./utils/iml_transforms.py) and [edge mask generator class](./utils/edge_generator.py). You can also directly call them by using `json_dataset` or `mani_dataset` in [./utils/datasets.py](./utils/datasets.py).
#### IML Dataset class
- We defined two types of Dataset class
  - `json_dataset`, which gets input image and corresponding ground truth from a JSON file with a protocol like this:
    ```
    [
        [
          "/Dataset/CASIAv1/Tp/image1.jpg",
          "/Dataset/CASIAv2/Gt/image1.jpg"
        ],
        [
          "/Dataset/CASIAv1/Tp/image2.jpg",
          "Negative"
        ],
        ......
    ]
    ```
    where "Negative" represents a totally black ground truth that doesn't needs a path(all authentic)
  - `mani_dataset` which loads images and ground truth pairs automatically from a directory having sub-directories named `Tp` (for input images) and `Gt` |(for ground truths). This class will generate the pairs using the sorted `os.listdir()` function. 
- These datasets will do **zero-padding** automatically. Standard augmentation methods like ImageNet normalization will also be added.
- Both datasets can generate `edge_mask` when specifying the `edge_width` parameter. Then, this dataset will return 3 objects (image, GT, edge mask) while only 2 objects when `edge_width=None`.
- For inference, returning the actual shape of the original image is crucial. You can set `if_return_shape=True` to get this value. 

## Citation
```
@misc{ma2023imlvit,
      title={IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer}, 
      author={Xiaochen Ma and Bo Du and Zhuohang Jiang and Ahmed Y. Al Hammadi and Jizhe Zhou},
      year={2023},
      eprint={2307.14863},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

****
<div align="center"> <a href="https://info.flagcounter.com/9Etf"><img src="https://s11.flagcounter.com/countxl/9Etf/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_12/viewers_0/labels_1/pageviews_1/flags_0/percent_0/" alt="Flag Counter" border="0"></a> </div>
