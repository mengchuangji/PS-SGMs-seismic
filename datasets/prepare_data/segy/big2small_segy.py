#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

import cv2
from scipy.io import loadmat
import glob
import numpy as np
import segyio
from datasets.gain import gain
import random

def generate_patch_from_segy(dir,pch_size,stride):
    file_list = glob.glob(dir + '/*.s*gy')  # get name list of all .s*gy files
    file_list = sorted(file_list)  # mcj
    pch_size = pch_size
    stride = stride
    num_patch = 0
    patches=[]
    for i in range(len(file_list)):
        f = segyio.open(file_list[i], ignore_geometry=True)
        f.mmap()  # mmap将一个文件或者其它对象映射进内存，加快读取速度
        data = np.asarray([np.copy(x) for x in f.trace[:,:]]).T[:,:] #[:,:19500]
        print(data.max())
        print(data.min())
        # data =data/abs(data).max()
        H = data.shape[0]
        W = data.shape[1]
        ind_H = list(range(0, H - pch_size[0] + 1, stride[0]))
        if ind_H[-1] < H - pch_size[0]:
            ind_H.append(H - pch_size[0])
        ind_W = list(range(0, W - pch_size[1] + 1, stride[1]))
        if ind_W[-1] < W - pch_size[1]:
            ind_W.append(W - pch_size[1])
        for start_H in ind_H:
            for start_W in ind_W:
                patch= data[start_H:start_H + pch_size[0], start_W:start_W + pch_size[1], ]
                patches.append(patch)
                num_patch += 1

    print('Total {:d} small patches'.format(num_patch))
    print('Finish!\n')
    patches=np.array(patches)
    return patches[:, :, :, np.newaxis]

def generate_patch_from_segy_1by1(dir,pch_size,stride,jump=1,agc=False,train_data_num=float('inf'),aug_times=[],scales = []):
    '''
    Args:
        aug_time(list): Corresponding function data_aug, if aug_time=[],mean don`t use the aug
        scales(list): data scaling; default scales = [],mean that the data don`t perform scaling,
                      if perform scaling, you can set scales=[0.9,0.8,...]
    '''
    file_list = glob.glob(dir + '/*.s*gy')  # get name list of all .s*gy files
    file_list = sorted(file_list)  # mcj
    pch_size = pch_size
    stride = stride
    num_patch = 0
    patches=[]
    for i in range(len(file_list)):
        f = segyio.open(file_list[i], ignore_geometry=True)
        f.mmap()  # mmap将一个文件或者其它对象映射进内存，加快读取速度
        # sourceX = np.asarray([np.copy(x) for x in f.trace[:]]).T
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        # print(sourceX.max())
        # print(sourceX.min())
        print('正在处理第'+str(i+1)+'个数据集...')
        trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
        shot_num = len(set(sourceX))  # shot number
        len_shot = trace_num // shot_num  # The length of the data in each shot data
        temp=np.asarray([np.copy(x) for x in f.trace[:]]).T
        temp = temp[[not np.all(temp[i] == 0) for i in range(temp.shape[0])], :]
        temp = temp[:, [not np.all(temp[:, i] == 0) for i in range(temp.shape[1])]]
        len_shot=temp.shape[1]//shot_num
        if len_shot < pch_size[0]:
            len_shot =temp.shape[1]   #len(sourceX)
            shot_num=1
        for j in range(0, shot_num, jump):
            # data = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
            # data = np.asarray([np.copy(x) for x in f.trace[0:5000]]).T # mcj 剖面分割
            data =temp[:, j * len_shot:(j + 1) * len_shot]

            if agc:
                # if random.randint(0, 1) == 1:
                #     data = gain(data, 0.002, 'agc', 0.05, 1)
                #     print('正在agc第' + str(j + 1) + '炮...')
                # else:
                #     data = data / abs(data).max()
                #     print('正在没用agc处理第' + str(j + 1) + '炮...')
                data = gain(data, 0.002, 'agc', 0.05, 1)
                print('正在agc第' + str(j + 1) + '炮...')
            else:
                data = data / abs(data).max()
                print('正在处理第' + str(j + 1) + '炮...')
            # 不用 scales 及 data augmentation
            H = data.shape[0]
            W = data.shape[1]
            ind_H = list(range(0, H - pch_size[0] + 1, stride[0]))
            if ind_H[-1] < H - pch_size[0]:
                ind_H.append(H - pch_size[0])
            ind_W = list(range(0, W - pch_size[1] + 1, stride[1]))
            if ind_W[-1] < W - pch_size[1]:
                ind_W.append(W - pch_size[1])
            for start_H in ind_H:
                for start_W in ind_W:
                    patch= data[start_H:start_H + pch_size[0], start_W:start_W + pch_size[1], ]
                    # if not np.all(patch == 0):
                    if not np.any(patch == 0):
                        patch=patch/abs(patch).max() #每一块归一化
                        patches.append(patch)
                        num_patch += 1
                    if len(patches) >= train_data_num:
                        f.close()
                        patches = np.expand_dims(patches, axis=3)
                        print(str(len(patches)) + ' ' + 'training data finished')
                        print('Total {:d} small patches'.format(num_patch))
                        print('Finish!\n')
                        return patches
            # # read data
            # h, w = data.shape
            # p_h, p_w = pch_size
            # s_h, s_w = stride
            # patches = []
            # for s in scales:
            #     h_scaled, w_scaled = int(h * s), int(w * s)
            #     data_scaled = cv2.resize(data, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
            #     for i in range(0, h_scaled - p_h + 1, s_h):
            #         for j in range(0, w_scaled - p_w + 1, s_w):
            #             x = data_scaled[i:i + p_h, j:j + p_w]
            #             if sum(sum(x)) != 0 and x.std() > 1e-5 and x.shape == pch_size:
            #                 num_patch += 1
            #                 patches.append(x)
            #                 if num_patch >= train_data_num:
            #                     return patches
            #                 for k in range(0, aug_times):
            #                     from data.util import  data_augmentation as data_aug
            #                     x_aug = data_aug(x, mode=np.random.randint(0, 8))
            #                     num_patch += 1
            #                     patches.append(x_aug)
            #                     if num_patch >= train_data_num:
            #                         f.close()
            #                         patches = np.expand_dims(patches, axis=3)
            #                         print(str(len(patches)) + ' ' + 'training data finished')
            #                         print('Total {:d} small patches'.format(num_patch))
            #                         print('Finish!\n')
            #                         return patches
        f.close()
    print('Total {:d} small patches'.format(num_patch))
    print('Finish!\n')
    patches=np.array(patches)
    return patches[:, :, :, np.newaxis]


def generate_patch_from_single_segy_file_1by1(file,pch_size,stride,jump=1,agc=False,train_data_num=float('inf'),aug_times=[],scales = []):
    '''
    Args:
        aug_time(list): Corresponding function data_aug, if aug_time=[],mean don`t use the aug
        scales(list): data scaling; default scales = [],mean that the data don`t perform scaling,
                      if perform scaling, you can set scales=[0.9,0.8,...]
    '''

    pch_size = pch_size
    stride = stride
    num_patch = 0
    patches=[]

    f = segyio.open(file, ignore_geometry=True)
    f.mmap()  # mmap将一个文件或者其它对象映射进内存，加快读取速度
    # sourceX = np.asarray([np.copy(x) for x in f.trace[:]]).T
    sourceX = f.attributes(segyio.TraceField.SourceX)[:]
    # print(sourceX.max())
    # print(sourceX.min())
    print('正在从' + str(os.path.split(file)[-1])+'生成patch')
    trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
    shot_num = len(set(sourceX))  # shot number
    len_shot = trace_num // shot_num  # The length of the data in each shot data
    temp = np.asarray([np.copy(x) for x in f.trace[:]]).T
    temp = temp[[not np.all(temp[i] == 0) for i in range(temp.shape[0])], :]
    temp = temp[:, [not np.all(temp[:, i] == 0) for i in range(temp.shape[1])]]
    len_shot = temp.shape[1] // shot_num
    if len_shot < pch_size[0]:
        len_shot = temp.shape[1]  # len(sourceX)
        shot_num = 1
    for j in range(0, shot_num, jump):
        # data = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
        # data = np.asarray([np.copy(x) for x in f.trace[0:5000]]).T # mcj 剖面分割
        data = temp[:, j * len_shot:(j + 1) * len_shot]

        if agc:
            if random.randint(0, 1) == 1:
                data = gain(data, 0.002, 'agc', 0.05, 1)
                print('正在agc第' + str(j + 1) + '炮...')
            else:
                data = data / abs(data).max()
                print('正在没用agc处理第' + str(j + 1) + '炮...')
        else:
            data = data / abs(data).max()
            print('正在处理第' + str(j + 1) + '炮...')
        # 不用 scales 及 data augmentation
        H = data.shape[0]
        W = data.shape[1]
        ind_H = list(range(0, H - pch_size[0] + 1, stride[0]))
        if ind_H[-1] < H - pch_size[0]:
            ind_H.append(H - pch_size[0])
        ind_W = list(range(0, W - pch_size[1] + 1, stride[1]))
        if ind_W[-1] < W - pch_size[1]:
            ind_W.append(W - pch_size[1])
        for start_H in ind_H:
            for start_W in ind_W:
                patch = data[start_H:start_H + pch_size[0], start_W:start_W + pch_size[1], ]
                # if not np.all(patch == 0):
                if not np.any(patch == 0):
                    patches.append(patch)
                    num_patch += 1
                if len(patches) >= train_data_num:
                    f.close()
                    patches = np.expand_dims(patches, axis=3)
                    print(str(len(patches)) + ' ' + 'training data finished')
                    print('Total {:d} small patches'.format(num_patch))
                    print('Finish!\n')
                    return patches
        # # read data
        # h, w = data.shape
        # p_h, p_w = pch_size
        # s_h, s_w = stride
        # patches = []
        # for s in scales:
        #     h_scaled, w_scaled = int(h * s), int(w * s)
        #     data_scaled = cv2.resize(data, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
        #     for i in range(0, h_scaled - p_h + 1, s_h):
        #         for j in range(0, w_scaled - p_w + 1, s_w):
        #             x = data_scaled[i:i + p_h, j:j + p_w]
        #             if sum(sum(x)) != 0 and x.std() > 1e-5 and x.shape == pch_size:
        #                 num_patch += 1
        #                 patches.append(x)
        #                 if num_patch >= train_data_num:
        #                     return patches
        #                 for k in range(0, aug_times):
        #                     from data.util import  data_augmentation as data_aug
        #                     x_aug = data_aug(x, mode=np.random.randint(0, 8))
        #                     num_patch += 1
        #                     patches.append(x_aug)
        #                     if num_patch >= train_data_num:
        #                         f.close()
        #                         patches = np.expand_dims(patches, axis=3)
        #                         print(str(len(patches)) + ' ' + 'training data finished')
        #                         print('Total {:d} small patches'.format(num_patch))
        #                         print('Finish!\n')
        #                         return patches
    f.close()
    print('Total {:d} small patches'.format(num_patch))
    print('Finish!\n')
    patches=np.array(patches)
    return patches[:, :, :, np.newaxis]






def generate_patch_from_noisy_mat(dir,pch_size,stride,sigma=75):
    file_list = glob.glob(dir + '/*.mat')  # get name list of all .mat files
    file_list = sorted(file_list)  # mcj
    pch_size = pch_size
    stride = stride
    sigma=sigma
    num_patch = 0
    patches=[]
    for i in range(len(file_list)):
        aa=loadmat(file_list[i])
        keys=aa.keys()
        data = loadmat(file_list[i])[list(keys)[3]]
        print(data.max())
        print(data.min())
        data =data/abs(data).max()
        data=data+np.random.normal(0,sigma/255,data.shape)
        H = data.shape[0]
        W = data.shape[1]
        ind_H = list(range(0, H - pch_size + 1, stride[0]))
        if ind_H[-1] < H - pch_size:
            ind_H.append(H - pch_size)
        ind_W = list(range(0, W - pch_size + 1, stride[1]))
        if ind_W[-1] < W - pch_size:
            ind_W.append(W - pch_size)
        for start_H in ind_H:
            for start_W in ind_W:
                patch= data[start_H:start_H + pch_size, start_W:start_W + pch_size, ]
                patches.append(patch)
                num_patch += 1

    print('Total {:d} small images in training set'.format(num_patch))
    print('Finish!\n')
    patches=np.array(patches)
    return patches[:, :, :, np.newaxis]




if __name__ == '__main__':
    # train_im_list = generate_patch_from_mat(dir="/home/shendi_mcj/datasets/seismic/marmousiShot", pch_size=32, stride=[24, 24])
    # train_im_list = generate_patch_from_noisy_mat(dir="/home/shendi_mcj/datasets/seismic/marmousi/marmousi20", pch_size=32,
    #                                         stride=[24, 24],sigma=75)
    open_segy_dir = '/home/shendi_mcj/datasets/seismic/train'
    open_im_list = generate_patch_from_segy_1by1(dir=open_segy_dir, pch_size=(128, 128), stride=(64, 64), jump=2,
                                                 agc=False, train_data_num=100000, aug_times=[], scales=[1])
    print("haha")