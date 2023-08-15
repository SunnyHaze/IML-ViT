# IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer

--------

This repo contains an official PyTorch implementation of our paper: [IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer.](http://arxiv.org/abs/2307.14863)

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
