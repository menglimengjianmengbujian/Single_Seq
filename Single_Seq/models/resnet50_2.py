import torch
import torch.nn as nn
from torchvision import models


class CustomResNet50(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, freeze_weights=True):
        super(CustomResNet50, self).__init__()
        self.num_classes = num_classes
        # 加载预训练的ResNet-50模型
        resnet = models.resnet50(pretrained=True)

        # 修改第一层卷积层，适应不同的输入通道数
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 可以设置是否冻结权重，默认为冻结权重，如果为False，则所有权重都会被更新
        if freeze_weights:
            for name, param in resnet.named_parameters():
                param.requires_grad = False
                # print(name, param.requires_grad)
        else:
            for name, param in resnet.named_parameters():
                param.requires_grad = True

        # 修改最后的全连接层，适应二分类任务
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes)

        # 将修改后的模型赋值给自定义的ResNet-50网络
        self.model = resnet

    def forward(self, x):
        return self.model(x)


# 示例：如何使用自定义的ResNet-50模型
if __name__ == "__main__":
    # 创建模型实例
    model = CustomResNet50(in_channels=1, num_classes=2, freeze_weights=False)

    # 打印所有参数的requires_grad属性
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    # 打印模型结构
    # print(model)

    # 创建一个示例输入
    example_input = torch.randn(12, 1, 224, 224)  # 改为1通道输入

    # 获取模型输出
    output = model(example_input)

    # 打印输出形状
    print("输出形状:", output.shape)

