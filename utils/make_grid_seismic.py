from torchvision.utils import make_grid, save_image
from utils.make_grid_h import make_grid_h
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# # 创建一些单通道数据示例
# data = torch.randn(16, 1, 32, 32)  # 8张单通道的 32x32 图像
# data =data/abs(data).max()

from datasets.seis_mat import SeisMatTrainDataset, SeisMatValidationDataset
from torch.utils.data import DataLoader
# 'D:\\datasets\\seismic\\marmousi\\f35_s256_o128'
dataset = SeisMatValidationDataset(path='/home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128',
                                   patch_size=128, pin_memory=True)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True,
                        num_workers=0, drop_last=True)
data_iter = iter(dataloader)
samples_ = next(data_iter)
data = samples_['H']

# 使用torchvision.utils.make_grid创建图像网格
grid_img = torchvision.utils.make_grid(data,nrow=10,padding=2)

# 将[-1,1]范围映射到[0,1]范围
# grid_img = (grid_img + 1) / 2

# 转换为numpy数组，并将通道移动到最后一个维度
np_img = grid_img.numpy().squeeze()  # 去除单通道的维度
np_img = np_img.transpose((1, 2, 0))

# 显示图像和色标
plt.gcf().set_size_inches(15,15)
plt.imshow(np_img[:,:,0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
# plt.colorbar()  # 添加色标
plt.axis('off')  # 关闭坐标轴
plt.tight_layout()

# 保存图像
# plt.imsave('grid_image.png', np_img, cmap='seismic')
plt.savefig('./grid_image.jpg', format='jpg', dpi=300)
plt.show()