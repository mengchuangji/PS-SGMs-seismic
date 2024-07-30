import numpy as np
from scipy.io import loadmat
from math import floor

def MonoPao():
    zz = loadmat('.\MonoPaoSigma.mat')['data']
    zz = np.sqrt(zz)
    # print(np.mean(zz))
    print("MonoPao.median",np.median(zz))
    print("MonoPao.max",zz.max())
    print("MonoPao.min",zz.min())
    return zz

def Gauss(sigma):
    sigma0=sigma/255
    sigma1=np.ones((256,256))*sigma0
    return sigma1

def Panke100_228_19_147Sigma():
    zz = loadmat('.\PankeSigma100_228_19_147.mat')['data']
    zz = np.sqrt(zz)
    print("Panke100_228_19_147",np.median(zz))
    print("Panke100_228_19_147",zz.max())
    print("Panke100_228_19_147",zz.min())
    return zz
def gaussian_kernel():
    H = 128
    W = 128
    center = [64, 64]
    scale = 32
    centerH = center[0]
    centerW = center[1]
    XX, YY = np.meshgrid(np.arange(W), np.arange(H))
    ZZ = 1./(2*np.pi*scale**2) * np.exp( (-(XX-centerH)**2-(YY-centerW)**2)/(2*scale**2) )
    return  ZZ

def peaks(n):
    '''
    Implementation the peak function of matlab.
    '''
    X = np.linspace(-3, 3, n)
    Y = np.linspace(-3, 3, n)
    [XX, YY] = np.meshgrid(X, Y)
    ZZ = 3 * (1-XX)**2 * np.exp(-XX**2 - (YY+1)**2) \
            - 10 * (XX/5.0 - XX**3 -YY**5) * np.exp(-XX**2-YY**2) - 1/3.0 * np.exp(-(XX+1)**2 - YY**2)
    return ZZ

def generate_gauss_kernel_mix(H, W):
    '''
    Generate a H x W mixture Gaussian kernel with mean (center) and std (scale).
    Input:
        H, W: interger
        center: mean value of x axis and y axis
        scale: float value
    '''
    pch_size = 32
    K_H = floor(H / pch_size)
    K_W = floor(W / pch_size)
    K = K_H * K_W
    # prob = np.random.dirichlet(np.ones((K,)), size=1).reshape((1,1,K))
    centerW = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_W = np.arange(K_W) * pch_size
    centerW += ind_W.reshape((1, -1))
    centerW = centerW.reshape((1,1,K)).astype(np.float32)
    centerH = np.random.uniform(low=0, high=pch_size, size=(K_H, K_W))
    ind_H = np.arange(K_H) * pch_size
    centerH += ind_H.reshape((-1, 1))
    centerH = centerH.reshape((1,1,K)).astype(np.float32)
    scale = np.random.uniform(low=pch_size/2, high=pch_size, size=(1,1,K))
    scale = scale.astype(np.float32)
    XX, YY = np.meshgrid(np.arange(0, W), np.arange(0,H))
    XX = XX[:, :, np.newaxis].astype(np.float32)
    YY = YY[:, :, np.newaxis].astype(np.float32)
    ZZ = 1./(2*np.pi*scale**2) * np.exp( (-(XX-centerW)**2-(YY-centerH)**2)/(2*scale**2) )
    # ZZ *= prob
    # out = ZZ.sum(axis=2, keepdims=False)
    out = ZZ.sum(axis=2, keepdims=False) / K

    return out

def sincos_kernel():
    # Nips Version
    [xx, yy] = np.meshgrid(np.linspace(1, 10, 256), np.linspace(1, 20, 256))
    # [xx, yy] = np.meshgrid(np.linspace(1, 10, 256), np.linspace(-10, 15, 256))
    zz = np.sin(xx) + np.cos(yy)
    return zz
