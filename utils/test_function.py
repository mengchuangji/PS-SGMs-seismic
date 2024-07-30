import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch
import numpy as np

# 生成一些示例数据
data = torch.randn(16, 1, 32, 32)  # 16张单通道的32x32图像

# 将数据规范化到 [-1, 1] 范围
normalized_data = (data - data.min()) / (data.max() - data.min()) * 2 - 1

# 使用 make_grid 创建图像网格
grid = make_grid(normalized_data, nrow=4, padding=2, pad_value=1.0)  # 设置 pad_value 为 1.0（白色）

# 将图像网格转换为 NumPy 数组
image = transforms.ToPILImage()(grid)
image_array = np.array(image)

# 获取图像网格的范围
extent = [0, image_array.shape[1], 0, image_array.shape[0]]

# 显示图像网格
plt.imshow(image_array, extent=extent, cmap='jet', norm=plt.Normalize(-1, 1))

# 添加 colorbar
cbar = plt.colorbar()
cbar.set_label('Colorbar Label')

# 隐藏坐标轴
plt.axis('off')

plt.show()