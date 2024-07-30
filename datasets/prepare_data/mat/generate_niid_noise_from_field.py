#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Chuangji Meng 2019-05-05 14:44:04

# import sys
# sys.path.append('./')
import numpy as np
import cv2
from skimage import img_as_float
from seis_utils import generate_gauss_kernel_mix, peaks, sincos_kernel
import h5py as h5
from pathlib import Path
import pdb
import argparse
from scipy.io import loadmat
H5PY_DEFAULT_READONLY=1
# base_path = Path('test_data')
base_path=Path('../../seismic/overthrust')

seed = 10000

np.random.seed(seed)

def MonoPao():
    zz = loadmat("../../seismic/NoiseLevelMap/MonoPaoSigma.mat")['data']
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
    zz = loadmat("../../seismic/NoiseLevelMap/PankeSigma100_228_19_147.mat")['data']
    zz = np.sqrt(zz)
    print("Panke100_228_19_147",np.median(zz))
    print("Panke100_228_19_147",zz.max())
    print("Panke100_228_19_147",zz.min())
    return zz


kernels = [Gauss(sigma=75),sincos_kernel(), sincos_kernel(), Panke100_228_19_147Sigma(), MonoPao()]
dep_U = 4
# sigma_max = 75/255.0
# sigma_min = 10/255.0

from datasets.prepare_data.mat.bia2small_mat import generate_patch_from_mat
test_im_list = generate_patch_from_mat(dir="../../seismic/overthrust/", pch_size=128, stride=[24,64])
im_list = test_im_list.astype(np.float32)
# sigma_max = 75 / 255.0
# sigma_min = 10 / 255.0
data_name='overthrust'
noise_dir = base_path / 'noise_niid'
if not noise_dir.is_dir():
    noise_dir.mkdir()



for jj, sigma in enumerate(kernels):
    print('Case {:d} of Dataset {:s}: {:d} images'.format(jj + 1, data_name, len(im_list)))
    if jj==1:
         sigma_max = 75 / 255.0
         sigma_min = 10 / 255.0
         sigma = sigma_min + (sigma - sigma.min()) / (sigma.max() - sigma.min()) * (sigma_max - sigma_min)
         h5_path = noise_dir.joinpath(data_name + '_niid_case' + str(jj + 1) + '.hdf5')
         if h5_path.exists():
             h5_path.unlink()
    elif jj==2:
         sigma_max = 100 / 255.0
         sigma_min = 10 / 255.0
         sigma = sigma_min + (sigma - sigma.min()) / (sigma.max() - sigma.min()) * (sigma_max - sigma_min)
         h5_path = noise_dir.joinpath(data_name + '_niid_case' + str(jj + 1) + '.hdf5')
         if h5_path.exists():
             h5_path.unlink()
    else:
         sigma=sigma*1.0
         h5_path = noise_dir.joinpath(data_name + '_niid_case' + str(jj + 1) + '.hdf5')
         if h5_path.exists():
             h5_path.unlink()

    with h5.File(h5_path) as h5_file:
        for ii, im_gt in enumerate(im_list):
            H, W, C = im_gt.shape
            H -= int(H % pow(2, dep_U))
            W -= int(W % pow(2, dep_U))
            im_gt = img_as_float(im_gt[:H, :W, ])
            sigma = cv2.resize(sigma, (W, H))
            sigma = sigma.astype(np.float32)
            noise = np.random.randn(H, W, C) * np.expand_dims(sigma, 2)
            noise = noise.astype(np.float32)
            data = np.concatenate((noise, sigma[:, :, np.newaxis]), axis=2)
            h5_file.create_dataset(name=str(ii), dtype=data.dtype,
                                   shape=data.shape, data=data)
