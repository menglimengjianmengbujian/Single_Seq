import torch
import torch.nn as nn
from torchvision import models

class CustomConvNeXtTiny(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, freeze_weights=True):
        super(CustomConvNeXtTiny, self).__init__()
        self.num_classes = num_classes
        # 加载预训练的ConvNeXt-Tiny模型
        convnext = models.convnext_tiny(pretrained=True)

        # 可以设置是否冻结权重，默认为冻结权重，如果为False，则所有权重都会被更新
        if freeze_weights:
            for name, param in convnext.named_parameters():
                param.requires_grad = False
                # print(name, param.requires_grad)
        else:
            for name, param in convnext.named_parameters():
                param.requires_grad = True

        # 修改最后的全连接层，适应二分类任务
        num_features = convnext.classifier[2].in_features
        convnext.classifier[2] = nn.Linear(num_features, num_classes)

        # 将修改后的模型赋值给自定义的ConvNeXt-Tiny网络
        self.model = convnext

    def forward(self, x):
        return self.model(x)

# 示例：如何使用自定义的ConvNeXt-Tiny模型
if __name__ == "__main__":
    # 创建模型实例
    model = CustomConvNeXtTiny(in_channels=3, num_classes=2, freeze_weights=True)

    # 打印所有参数的requires_grad属性
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    # 打印模型结构
    # print(model)

    # 创建一个示例输入
    example_input = torch.randn(12, 3, 224, 224)

    # 获取模型输出
    output = model(example_input)

    # 打印输出形状
    print("输出形状:", output.shape)
