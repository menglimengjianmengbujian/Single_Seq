import torch
import torch.nn as nn
from torchvision import models


class CustomResNet50(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, freeze_weights=True):
        super(CustomResNet50, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # 加载预训练的ResNet-50模型
        self.resnet1 = models.resnet50(pretrained=True)
        self.resnet2 = models.resnet50(pretrained=True)
        self.resnet3 = models.resnet50(pretrained=True)

        # 修改输入层，适应单通道输入
        self.resnet1.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet2.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet3.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 冻结参数
        if freeze_weights:
            for param in self.resnet1.parameters():
                param.requires_grad = False
            for param in self.resnet2.parameters():
                param.requires_grad = False
            for param in self.resnet3.parameters():
                param.requires_grad = False

        # 修改最后的全连接层，适应二分类任务
        num_features = self.resnet1.fc.in_features
        self.resnet1.fc = nn.Identity()  # 移除最后一层，保留特征
        self.resnet2.fc = nn.Identity()  # 移除最后一层，保留特征
        self.resnet3.fc = nn.Identity()  # 移除最后一层，保留特征

        # 新的全连接层
        self.fc = nn.Linear(num_features * 3, num_classes)

    def forward(self, x):
        # 将输入拆分为三个通道
        x1 = x[:, 0:1, :, :]  # 第一个通道
        x2 = x[:, 1:2, :, :]  # 第二个通道
        x3 = x[:, 2:3, :, :]  # 第三个通道

        # 通过三个ResNet-50模型
        x1 = self.resnet1(x1)
        x2 = self.resnet2(x2)
        x3 = self.resnet3(x3)

        # 连接三个输出
        x = torch.cat((x1, x2, x3), dim=1)

        # 最后的全连接层
        x = self.fc(x)
        return x


# 示例：如何使用自定义的ResNet-50模型
if __name__ == "__main__":
    # 创建模型实例
    model = CustomResNet50(in_channels=1, num_classes=2, freeze_weights=True)

    # 打印所有参数的requires_grad属性
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    # 创建一个示例输入
    example_input = torch.randn(10, 3, 224, 224)  # 批大小为10，3个通道，224x224的图像

    # 获取模型输出
    output = model(example_input)

    # 打印输出形状
    print("输出形状:", output.shape)
