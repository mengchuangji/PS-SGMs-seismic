#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Chuangji Meng 2021-5-13

from datasets.get_patch import *
import numpy as np
import h5py as h5
from options import set_opts
args = set_opts()
ori_data_dir='/home/shendi_mcj/datasets/seismic/fielddata/train/clean'
cle_data_dir='/home/shendi_mcj/datasets/seismic/fielddata/train/clean'
print('=> Generating patch samples')
args.ori_data_dir=ori_data_dir
args.cle_data_dir=cle_data_dir
args.path_h5='/home/shendi_mcj/datasets/seismic/fielddata'
path_h5 = os.path.join(args.path_h5, 'small_seismic_train.hdf5')
args.patch_size = (32, 32)
args.stride = (32, 32)

ori_data_list = datagenerator(data_dir=args.ori_data_dir, patch_size=args.patch_size, stride=args.stride,
                              train_data_num=args.train_data_num,
                              download=args.download, datasets=args.datasets, aug_times=args.aug_times,
                              scales=args.scales,
                              verbose=args.verbose, jump=args.jump, agc=False)
ori_max=ori_data_list.max()
cle_data_list = datagenerator(data_dir=args.cle_data_dir, patch_size=args.patch_size, stride=args.stride,
                             train_data_num=args.train_data_num,
                             download=args.download, datasets=args.datasets, aug_times=args.aug_times,
                             scales=args.scales,
                             verbose=args.verbose, jump=args.jump, agc=False)


num_patch = 0
with h5.File(path_h5, 'w') as h5_file:
    for ii in range(len(ori_data_list)):
        if (ii+1) % 10 == 0:
            print('    The {:d} original images'.format(ii+1))

        pch_noisy = ori_data_list[ii]/ori_max
        pch_gt = cle_data_list[ii]/ori_max
        pch_imgs = np.concatenate((pch_noisy, pch_gt), axis=2)
        h5_file.create_dataset(name=str(num_patch), shape=pch_imgs.shape,
                               dtype=pch_imgs.dtype, data=pch_imgs)
        num_patch += 1
print('Total {:d} small images in training set'.format(num_patch))
print('Finish!\n')