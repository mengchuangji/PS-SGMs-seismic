#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Chuangji Meng

from datasets.get_patch import *
import numpy as np
import h5py as h5
from options import set_opts
# import sys
# sys.path.append('E:\VIRI\mycode\\toBGP')

args = set_opts()
cle_data_dir='../../seismic/fielddata/train/clean'
print('=> Generating patch samples')
args.cle_data_dir=cle_data_dir
args.path_h5='../../seismic/marmousi'
path_h5 = os.path.join(args.path_h5, 'small_seismic_train_from_mat.hdf5')
args.patch_size = (32, 32)
args.stride = (32, 32)

from datasets.prepare_data.mat.bia2small_mat import generate_patch_from_mat,generate_patch_from_noisy_mat
cle_data_list= generate_patch_from_mat(dir="../../seismic/marmousi", pch_size=32,
                                            stride=[24, 24])
# im_list = test_im_list.astype(np.float32)
ori_data_list =generate_patch_from_noisy_mat(dir="../../seismic/marmousi", pch_size=32,
                                            stride=[24, 24],sigma=75)
# ori_max=ori_data_list.max()



num_patch = 0
with h5.File(path_h5, 'w') as h5_file:
    for ii in range(len(ori_data_list)):
        if (ii+1) % 10 == 0:
            print('    The {:d} original images'.format(ii+1))

        pch_noisy = ori_data_list[ii]#/ori_max
        pch_gt = cle_data_list[ii]#/ori_max
        pch_imgs = np.concatenate((pch_noisy, pch_gt), axis=2)
        h5_file.create_dataset(name=str(num_patch), shape=pch_imgs.shape,
                               dtype=pch_imgs.dtype, data=pch_imgs)
        num_patch += 1
print('Total {:d} small images in training set'.format(num_patch))
print('Finish!\n')