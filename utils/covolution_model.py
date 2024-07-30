import numpy as np
from scipy.signal import convolve,convolve2d

import numpy as np
import matplotlib.pyplot as plt
def plot(img,dpi,figsize):

    plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    # plt.title('fake_noisy')
    # plt.colorbar()
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.show()
def plot_cmap(img,dpi,figsize,data_range,cmap,cbar=False):

    plt.figure(dpi=dpi, figsize=figsize)
    plt.imshow(img, vmin=data_range[0], vmax=data_range[1], cmap=cmap)
    # plt.title('fake_noisy')
    if cbar:
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')
    plt.tight_layout()
    plt.show()

def zero_phase_wavelet(f, dt):
    """
    zero_phase_wavelet: zero phase wavelet of central frequency f.

    Args:
    f: central frequency in Hz (f << 1 / (2*dt))
    dt: sampling interval in sec

    Returns:
    w: the Ricker wavelet
    tw: axis
    """
    nw = 2.2 / f / dt
    nw = 2 * np.floor(nw / 2) + 1
    nc = np.floor(nw / 2)
    w = np.zeros(int(nw))
    k = np.arange(1, int(nw) + 1)
    alpha = (nc - k + 1) * f * dt * np.pi
    beta = alpha ** 2
    A = 1
    b = 1
    w = A * np.cos(alpha * 2) * np.exp(-beta * 4 / (b ** 2))
    if np.ndim(w) == 2:
        w = np.squeeze(w)
    tw = -(nc + 1 - np.arange(1, int(nw) + 1)) * dt
    return w, tw
def ricker(f, dt):
    """
    ricker: Ricker wavelet of central frequency f.

    Args:
    f: central frequency in Hz (f << 1 / (2*dt))
    dt: sampling interval in sec

    Returns:
    w: the Ricker wavelet
    tw: axis
    """
    nw = 2.2 / f / dt
    nw = 2 * np.floor(nw / 2) + 1
    nc = np.floor(nw / 2)
    w = np.zeros(int(nw))
    k = np.arange(1, int(nw) + 1)
    alpha = (nc - k + 1) * f * dt * np.pi
    beta = alpha ** 2
    w = (1 - beta * 2) * np.exp(-beta)
    tw = -(nc + 1 - np.arange(1, int(nw) + 1)) * dt
    return w, tw

dt=0.001 #时间采样间隔/s
fn=35#
wavelet,dw= ricker(fn,dt)
plt.plot(dw, wavelet)
plt.show()
# 假设 vp 和 R 是 NumPy 数组，wavelet 是一个 1D 数组
import scipy.io as sio
R=sio.loadmat('/home/shendi_mcj/datasets/seismic/marmousi/Marmousi2_R.mat')['R']
# plot(R, dpi=300, figsize=(6, 3))
plot_cmap(R[750:750+128,7200:7200+128], 300, (3.7, 3), data_range=[-1, 1], cmap=plt.cm.seismic, cbar=True)

# v1 版本
# # 计算 seismic_original
# L = len(wavelet) // 2
# signal = np.zeros_like(R)  # 创建一个与 R 相同形状的数组
# for j in range(R.shape[1]):
#     signal[:, j] = convolve(R[:, j], wavelet, mode='full')[L:-L]

# v2 版本
# 计算 seismic_original
L = len(wavelet) // 2
W = np.zeros((len(wavelet) + len(R) - 1, len(R)))

for i in range(len(W[0])):
    W[i:i+len(wavelet), i] = wavelet

W = W[L:len(W)-L, :]
signal = np.dot(W, R)

# signal= convolve2d(R, np.tile(wavelet, (1, R.shape[1])), mode='same', boundary='wrap')
plot_cmap(signal[750:750+128,7200:7200+128], 300, (3.7, 3), data_range=[-1, 1], cmap=plt.cm.seismic, cbar=True)


# import torch
#
# # 假设 diag_elements 是一个列表，每个元素都是一个方阵（托普利兹矩阵）
# # 假设每个方阵都有相同的大小
# # 假设 diag_elements 的长度为 n
# diag_elements = [torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]
#
# # 将列表转换为张量
# diag_tensor = torch.stack(diag_elements, dim=0)
#
# # 创建一个对角矩阵，每个对角元素用对应的方阵替换
# diag_matrix = torch.diag_embed(diag_tensor)
#
# # 创建双重托普利兹矩阵
# doubly_toeplitz_matrix = torch.zeros((diag_matrix.shape[0] * diag_matrix.shape[2],
#                                       diag_matrix.shape[1] * diag_matrix.shape[2]))
#
# # 将每个对角元素（方阵）放置到双重托普利兹矩阵的正确位置
# for i in range(diag_matrix.shape[2]):
#     start_row = i * diag_matrix.shape[0]
#     start_col = i * diag_matrix.shape[1]
#     end_row = start_row + diag_matrix.shape[0]
#     end_col = start_col + diag_matrix.shape[1]
#     doubly_toeplitz_matrix[start_row:end_row, start_col:end_col] = diag_matrix[:, :, i]
#
# # 输出双重托普利兹矩阵
# print(doubly_toeplitz_matrix)
#
