
import os
import sys
import pickle

import matplotlib.pyplot
from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

class CESM(Dataset):
    def __init__(self, base_dir=None, labeled_num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.targets = []
        # self.split = split
        self.transform = transform
        dir = os.listdir(self._base_dir)
        dir.sort()
        for name in dir:
            image = os.path.join(self._base_dir, name)
            self.sample_list.append(image)


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(case, 'r')

        data = h5f['X'][:].astype(np.float32)


        label = h5f['Y'][()]



        # data_DCE = cv2.resize(data_DCE, (320, 320))
        # data_T2 = cv2.resize(data_T2, (320, 320))
        # csv = cv2.resize(csv , (320, 320))



        seed = np.random.randint(255)
        if self.transform:
            torch.manual_seed(seed)
            # print(seed )
            data = Image.fromarray(np.uint8(data.transpose(0,1, 2  )*255))
            torch.manual_seed(seed)
            data = self.transform(data)
            torch.manual_seed(seed)
            # 转换标签为 PyTorch 张量
            label = torch.tensor(label, dtype=torch.long)

            # torch.manual_seed(seed)
            # ENHANCE = Image.fromarray(np.uint8(255 * csv))




        sample = {'data': data, 'label': label}




        return sample
# 示例用法
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor()  # 将 PIL 图像转换为 PyTorch 张量
    ])

    dataset = CESM(base_dir=r"C:\Users\Administrator\Desktop\LJJ\train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i, sample in enumerate(dataloader):
        print(f"批次 {i + 1}")
        print(f"数据形状: {sample['data'].shape}")
        print(f"标签: {sample['label']}")