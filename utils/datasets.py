# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import json
import os
from PIL import Image
import cv2

# Augmentation library
import albumentations as albu
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


if __name__ == "__main__":
    from edge_generator import EdgeGenerator
    from iml_transforms import get_albu_transforms

else:
    from .edge_generator import EdgeGenerator
    from .iml_transforms import get_albu_transforms


def pil_loader(path: str) -> Image.Image:
    """PIL image loader

    Args:
        path (str): image path

    Returns:
        Image.Image: PIL image (after np.array(x) becomes [0,255] int8)
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """denormalize image with mean and std
    """
    image = image.clone().detach().cpu()
    image = image * torch.tensor(std).view(3, 1, 1)
    image = image + torch.tensor(mean).view(3, 1, 1)
    return image

class base_dataset(Dataset):
    def _init_dataset_path(self, path):
        tp_path = None # Tampered image
        gt_path = None # Ground truth
        return tp_path, gt_path
        
    def __init__(self, path, output_size = 1024 ,transform = None, edge_width = None, if_return_name = False, if_return_shape = False, if_return_type = False) -> None:
        super().__init__()
        self.tp_path, self.gt_path = self._init_dataset_path(path)
        if self.tp_path == None:
            raise NotImplementedError
        self.transform = transform
        self.edge_generator = None if edge_width is None else EdgeGenerator(edge_width)
        self.padding_transform =  get_albu_transforms(type_='pad', outputsize=output_size)
        self.if_return_name = if_return_name
        self.if_return_shape = if_return_shape
        self.if_return_type = if_return_type
    def __getitem__(self, index):
        output_list = []
        tp_path = self.tp_path[index]
        gt_path = self.gt_path[index]
        
        tp_img = pil_loader(tp_path)
        tp_shape = tp_img.size
        # if "negative" then gt is a image with all 0
        if gt_path != "Negative":
            gt_img = pil_loader(gt_path)
            gt_shape = gt_img.size
        else:
            temp = np.array(tp_img)
            gt_img = np.zeros((temp.shape[0], temp.shape[1], 3))
            gt_shape = (temp.shape[1], temp.shape[0])
  
        assert tp_shape == gt_shape, "tp and gt image shape must be the same, but got {} and {}".format(tp_shape, gt_shape)
        
        tp_img = np.array(tp_img) # H W C
        gt_img = np.array(gt_img) # H W C
        
        # Do augmentations
        if self.transform != None:
            res_dict = self.transform(image = tp_img, mask = gt_img)
            tp_img = res_dict['image']
            gt_img = res_dict['mask']
        
        
        gt_img =  (np.mean(gt_img, axis = 2, keepdims = True)  > 127.5 ) * 1.0 # fuse the 3 channels to 1 channel, and make it binary(0 or 1)
        gt_img =  gt_img.transpose(2,0,1)[0] # H W C -> C H W -> H W
        masks_list = [gt_img]
        if self.edge_generator != None: # if need to generate broaden edge mask
            broaden_gt_img = self.edge_generator(gt_img)[0][0] # B C H W -> H W
            masks_list.append(broaden_gt_img)
        # Do padings
        res_dict = self.padding_transform(image = tp_img, masks = masks_list)
        
        tp_img = res_dict['image']
        gt_img = res_dict['masks'][0].unsqueeze(0) # H W -> 1 H W        
        output_list.append(tp_img)
        output_list.append(gt_img)
        
        if self.edge_generator != None:
            output_list.append(res_dict['masks'][1].unsqueeze(0)) # H W -> 1 H W
        if self.if_return_name:
            basenae = os.path.basename(tp_path)
            output_list.append(basenae)
        
        if self.if_return_shape:
            tp_shape = (tp_shape[1], tp_shape[0]) # swap for correct order
            tp_shape = torch.tensor(tp_shape)
            output_list.append(tp_shape)
            
        if self.if_return_type:
            gt_type = True if torch.max(gt_img) != 0 else False
            output_list.append(gt_type)
    
        return output_list
    def __len__(self):
        return len(self.tp_path)
    
class mani_dataset(base_dataset):
    def _init_dataset_path(self, path):
        path = path
        tp_dir = os.path.join(path, 'Tp')
        gt_dir = os.path.join(path, 'Gt')
        tp_list = os.listdir(tp_dir)
        gt_list = os.listdir(gt_dir)
        # Use sort mathod to keep order, to make sure the order is the same as the order in the tp_list and gt_list
        tp_list.sort()
        gt_list.sort()
        t_tp_list = [os.path.join(path, 'Tp', tp_list[index]) for index in range(len(tp_list))]
        t_gt_list = [os.path.join(path, 'Gt', gt_list[index]) for index in range(len(gt_list))]
        return t_tp_list, t_gt_list
    
class json_dataset(base_dataset):
    """ init from a json file, which contains all the images path
        file is organized as:
            [["./Tp/6.jpg", "./Gt/6.jpg"],
                ["./Tp/7.jpg", "./Gt/7.jpg"],
                ["./Tp/8.jpg", "Negative"],
                ......
            ]
        if path is "Neagative" then the image is negative sample, which means ground truths is a totally black image.
        
    Args:
        path (_type_): _description_
        transform_albu (_type_, optional): _description_. Defaults to None.
        mask_edge_generator (_type_, optional): _description_. Defaults to None.
        if_return_shape
    """
    def _init_dataset_path(self, path):
        images = json.load(open(path, 'r'))
        tp_list = []
        gt_list = []
        for record in images:
            tp_list.append(record[0])
            gt_list.append(record[1])
        return tp_list, gt_list


"""
Code below is for testing
"""

if __name__ == "__main__":
    transform = get_albu_transforms('train')
    data = mani_dataset(r'G:\Datasets\IML_Datasets_revised\CASIA2.0', edge_width=5, transform=transform, if_return_shape=True)
    # data = huge_dataset('./path.json', mask_edge_generator=5, transform_albu=transform)
    d = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    cnt = 0
    for sample in d:
        print(sample)
        image, mask, b_mask, shape = sample

        print(image.shape)
        print(mask.shape)
        print(b_mask.shape)
        print(shape)
        # print(sample.shape)
        import matplotlib.pyplot as plt
        plt.subplot(1,4,1)
        image = denormalize(image[0])
        # image = image[0]
        plt.imshow(image.permute(1,2,0))
        plt.subplot(1,4,2)
        plt.imshow(mask[0][0])
        plt.subplot(1,4,3)
        plt.imshow(b_mask[0][0])
        plt.show()
        
        cnt += 1
        if cnt == 10:
            break
    exit(0)
    
    path = r"/home/psdz/Datasets/CASIA2.0_revised"          # GPU-server dir
    # path = r"G:\Datasets\CASIA2.0_revised\CASIA2.0_revised" # local dir
    # path = r"/home/psdz/Datasets"
    transform_train = transforms.Compose([
            # transforms.RandomResizedCrop((224,224), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.Resize((224,224), interpolation=3),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([
            # transforms.RandomResizedCrop((224,224), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.Resize((224,224), interpolation=3),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    ])
    
    albu_transform_train = albu.Compose([
        albu.RandomResizedCrop(224,224, scale=(0.2, 1.0), interpolation=3), # 3 is bicubic
        albu.HorizontalFlip(),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])       
    import matplotlib.pyplot as plt
    data = huge_dataset("/home/psdz/Datasets/train.json", transform_albu= albu_transform_train)
    image, mask = data[0:10]
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.savefig("/home/psdz/Datasets/train.png")

    exit(0)
    data = mani_dataset(path)
    print("start validation")
    fault = dataset_validation(data)
    import json
    json.dump(fault, open("fault.json", "w"))
    print("end validation")
    exit(0)

    data_transform = mani_dataset(path, transform_train=transform_train, transform_mask=transform_test)

    data_albu_transform = mani_dataset(path,transform_albu=albu_transform_train)

    img1, mask1 = data_transform[0]
    img2, mask2 = data_albu_transform[0]
    print(img1)
    print(img2)
    print(mask1)
    print(mask2)
    exit(0)

    # albumentations dataset testing
    data = mani_dataset(path, transform_albu=albu_transform_train)
    
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    loader = DataLoader(data, batch_size=4, shuffle=True)
    for i in loader:
        
        # print(i)
        # print(i[0].shape)
        # print(i[1].shape)
        from matplotlib import pyplot as plt
        plt.subplot(1,2,1)
        plt.imshow(i[0][0].numpy().transpose((1,2,0)) / 2  + 0.5)
        plt.subplot(1,2,2)
        plt.imshow(i[1][0].numpy().transpose((1,2,0)))
        # name= input("enter name: ")
        plt.show()
        # plt.savefig(f"{name}.png")