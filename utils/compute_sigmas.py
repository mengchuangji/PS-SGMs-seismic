import numpy as np
import torch
def get_sigmas(sigma_begin,sigma_end,num_classes):
    sigmas = torch.tensor(
        np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                           num_classes))).float()
    return sigmas

sigmas_th = get_sigmas(sigma_begin=23,sigma_end=0.01,num_classes=500)
sigmas = sigmas_th.numpy()
# 找到最接近给定值的元素的索引
index = np.abs(sigmas - 0.2).argmin()
print("最接近的值的索引:", index)
print('done!')

values = np.array([0.2, 0.1, 0.05])  # 给定的值数组
# 初始化最小差距和最接近值的索引
min_diff = np.inf
nearest_indices = []

# 查找最接近值的索引
for value in values:
    index = np.abs(sigmas - value).argmin()
    nearest_indices.append(index)

print("最接近值的索引:", nearest_indices)
