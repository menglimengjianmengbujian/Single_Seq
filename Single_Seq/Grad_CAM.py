import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from models.resnet50_2 import CustomResNet50 as model1
import torch.nn as nn
from dataset import CESM
from torch.utils.data import DataLoader

"""
修改path1为训练好的模型路径
将需要进行CAM可视化的图片保存到base_dir文件夹目录下，并修改base_dir的文件夹路径
show_cam_on_image函数的544行代码调节heatmap和调节原图的透明度
"""

def main():

    net = model1(3, num_classes=2, freeze_weights=False)
    path1 = r'E:\pycharmproject\LJJ\checkpoint\Resnet50\Wednesday_05_June_2024_19h_13m_21s\Resnet50-31-best.pth'
    net.load_state_dict(torch.load(path1))

    model = net
    target_layers = [net.model.layer4[-1]]
    CESMdata2 = CESM(base_dir=r'C:\Users\Administrator\Desktop\临时\预测',
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                     ]))

    CESM_10_test_l = DataLoader(CESMdata2, batch_size=1, shuffle=False, drop_last=True,
                                pin_memory=torch.cuda.is_available())




    for i, x in enumerate(CESM_10_test_l):
        data = x['data']

        input_tensor=data
        data=data.squeeze(0).numpy()
        data=np.transpose(data, (1, 2, 0))
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = x['label']  # tabby, tabby cat
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]


        visualization = show_cam_on_image(data ,
                                          grayscale_cam,
                                          use_rgb=True)
        # plt.imshow(visualization)
        # plt.show()
        cv2.imshow('Image', data)
        cv2.waitKey(0)
        cv2.imshow('Image', visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
