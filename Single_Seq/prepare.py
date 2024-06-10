import numpy as np

# 文件地址
X_path = r"C:\Users\Administrator\Desktop\LJJ\X_data.npy"

# 加载数据
data = np.load(X_path)

# 打印原始数据形状
print(f"原始数据形状: {data.shape}")

# 移除维度
X = np.squeeze(data, axis=1)

# 打印修改后的数据形状
print(f"修改后的数据形状: {X.shape}")

# 文件地址
Y_path = r"C:\Users\Administrator\Desktop\LJJ\y_data.npy"

# 加载数据
Y = np.load(Y_path)

# 打印原始数据形状
print(f"原始数据形状: {Y.shape}")

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure

# 假设 img 是你的图像数据，形状为 (224, 224, 3)
img = X[0]  # 示例数据

# 分离三个通道
channel_1 = img[:, :, 0]
channel_2 = img[:, :, 1]
channel_3 = img[:, :, 2]

# 对每个通道进行直方图均衡化
channel_1_eq = exposure.equalize_hist(channel_1)
channel_2_eq = exposure.equalize_hist(channel_2)
channel_3_eq = exposure.equalize_hist(channel_3)

# 创建一个包含3个子图的图形来显示处理后的图像
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 绘制第一个通道的处理后图像
axs[0].imshow(channel_1_eq, cmap='gray')
axs[0].set_title('Channel 1 Equalized')
axs[0].axis('off')  # 不显示坐标轴

# 绘制第二个通道的处理后图像
axs[1].imshow(channel_2_eq, cmap='gray')
axs[1].set_title('Channel 2 Equalized')
axs[1].axis('off')  # 不显示坐标轴

# 绘制第三个通道的处理后图像
axs[2].imshow(channel_3_eq, cmap='gray')
axs[2].set_title('Channel 3 Equalized')
axs[2].axis('off')  # 不显示坐标轴

# 显示图形
plt.show()

# 创建一个包含3个子图的图形来显示处理后的像素值直方图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# 绘制第一个通道的处理后像素值直方图
axs[0].hist(channel_1_eq.flatten(), bins=256, color='r', alpha=0.6)
axs[0].set_title('Channel 1 Equalized Histogram')
axs[0].set_xlim([0, 1])  # 根据数据范围设置横坐标范围
axs[0].set_xlabel('Pixel Intensity')
axs[0].set_ylabel('Frequency')

# 绘制第二个通道的处理后像素值直方图
axs[1].hist(channel_2_eq.flatten(), bins=256, color='g', alpha=0.6)
axs[1].set_title('Channel 2 Equalized Histogram')
axs[1].set_xlim([0, 1])  # 根据数据范围设置横坐标范围
axs[1].set_xlabel('Pixel Intensity')
axs[1].set_ylabel('Frequency')

# 绘制第三个通道的处理后像素值直方图
axs[2].hist(channel_3_eq.flatten(), bins=256, color='b', alpha=0.6)
axs[2].set_title('Channel 3 Equalized Histogram')
axs[2].set_xlim([0, 1])  # 根据数据范围设置横坐标范围
axs[2].set_xlabel('Pixel Intensity')
axs[2].set_ylabel('Frequency')

# 显示图形
plt.show()

import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split
from skimage import exposure

# 示例数据
X = X
Y = Y

# 创建保存文件的目录
base_dir = r"C:\Users\Administrator\Desktop\临时"
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 按照7:1:2的比例划分数据集
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_temp, Y_temp, test_size=0.125, random_state=42)


# 定义处理图像的函数
def equalize_hist_image(img):
    equalized_img = np.zeros_like(img)
    for i in range(img.shape[2]):
        equalized_img[:, :, i] = exposure.equalize_hist(img[:, :, i])
    return equalized_img


# 保存数据集到相应的文件夹
def save_h5_files(X, Y, save_dir, prefix):
    for i, (x, y) in enumerate(zip(X, Y)):
        # 对图像进行直方图均衡化处理
        x = equalize_hist_image(x)

        file_path = os.path.join(save_dir, f"{prefix}_{i}.h5")
        with h5py.File(file_path, 'w') as h5f:
            h5f.create_dataset('X', data=x)
            h5f.create_dataset('Y', data=y)
        print(f"数据 {prefix}_{i} 已保存到 {file_path}")


# 保存训练集
save_h5_files(X_train, Y_train, train_dir, 'train')
# 保存验证集
save_h5_files(X_valid, Y_valid, valid_dir, 'valid')
# 保存测试集
save_h5_files(X_test, Y_test, test_dir, 'test')

print("所有数据集已保存完成。")