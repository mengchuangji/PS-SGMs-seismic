#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:51:11

import torch
import h5py as h5
import random
import cv2
import os
import numpy as np
import torch.utils.data as uData
from skimage import img_as_float64 as img_as_float
from .data_tools import sigma_estimate, random_augmentation, gaussian_kernel
from . import BaseDataSetH5, BaseDataSetImg

# Benchmardk Datasets: Renoir and SIDD
class BenchmarkTrain(BaseDataSetH5):
    def __init__(self, h5_file, length, pch_size=(128,128), radius=5, eps2=1e-6, noise_estimate=True):
        super(BenchmarkTrain, self).__init__(h5_file, length)
        self.win = 2*radius + 1
        self.sigma_spatial = radius
        self.noise_estimate = noise_estimate
        self.eps2 = eps2
        self.pch_size = pch_size

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[ind_im]]
            im_gt, im_noisy = self.crop_patch(imgs_sets)
        # im_gt = img_as_float(im_gt) #mcj  数据变为了float64
        # im_noisy = img_as_float(im_noisy)

        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        if self.noise_estimate:
            sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))
        eps2 = torch.tensor([self.eps2], dtype=torch.float32).reshape((1,1,1))

        if self.noise_estimate:
            sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2, 0, 1)))
            return im_noisy, im_gt, sigma2_map_est, eps2
        else:
            return im_noisy, im_gt


class BenchmarkTest(BaseDataSetH5):
    def __getitem__(self, index):
        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            C2 = imgs_sets.shape[2]
            C = int(C2/2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])
        # im_gt = img_as_float(im_gt)  #mcj  数据变为了float64
        # im_noisy = img_as_float(im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt

class SimulateTrain_VINonIID_Cl_Ne_sigma(BaseDataSetH5):
    def __init__(self, CL_h5_file, length, NE_h5_file, pch_size=128, radius=5, noise_estimate=True, noise_mode='S', sigma=75):
        super(SimulateTrain_VINonIID_Cl_Ne_sigma, self).__init__(CL_h5_file, length)
        self.win = 2*radius + 1
        self.sigma_spatial = radius
        self.noise_estimate = noise_estimate
        self.pch_size = pch_size
        self.sigma_min = 0
        self.sigma_max = sigma
        self.NE_h5_file = NE_h5_file
        self.noise_mode = noise_mode

        with h5.File(self.NE_h5_file, 'r') as h5_file:
            self.NE_keys = list(h5_file.keys())
            self.NE_num_images = len(self.NE_keys)

    # def crop_patch(self, CL, NE):
    #     H, W, C = CL.shape
    #     ind_H = random.randint(0, H-self.pch_size)
    #     ind_W = random.randint(0, W - self.pch_size)
    #     im_noise = np.array(NE[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, :])
    #     im_gt = np.array(CL[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, :])
    #     return im_gt, im_noise
    def crop_patch(self, img):
        H, W, C = img.shape
        ind_H = random.randint(0, H-self.pch_size)
        ind_W = random.randint(0, W - self.pch_size)
        img = np.array(img[ind_H:ind_H + self.pch_size, ind_W:ind_W + self.pch_size, :])
        return img

    def generate_sigma(self):
        pch_size = self.pch_size
        center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        scale = random.uniform(pch_size/4, pch_size/4*3)
        kernel = gaussian_kernel(pch_size, pch_size, center, scale)
        up = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        down = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        if up < down:
            up, down = down, up
        up += 5/255.0
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min()) *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :, np.newaxis]

    def __getitem__(self, index):
        num_images = self.num_images
        ind_gt= random.randint(0, num_images-1)
        ne_num_images = self.NE_num_images
        ind_ne = random.randint(0, ne_num_images - 1)

        with h5.File(self.h5_path, 'r') as h5_file:
            im_gt= np.array(h5_file[self.keys[ind_gt]])
            # while np.all(im_gt == 0):
            #     ind_gt = random.randint(0, num_images - 1)
            #     im_gt = np.array(h5_file[self.keys[ind_gt]])
            #     if not np.all(im_gt == 0):
            #         break

        with h5.File(self.NE_h5_file, 'r') as h5_file:
            im_noise = np.array(h5_file[self.NE_keys[ind_ne]])

        im_gt = self.crop_patch(im_gt)
        C = im_gt.shape[2]

        im_noise = self.crop_patch(im_noise)
        im_noise_max = abs(im_noise).max()

        # generate sigmaMap
        sigma_map = self.generate_sigma()

        if self.noise_mode == 'S':
            noise = np.random.normal(0, self.sigma_max / 255, im_gt.shape)
        if self.noise_mode == 'B':
            # noiseL_B = [0, 75]
            stdN = np.random.uniform(self.sigma_min, self.sigma_max)
            noise = np.random.normal(0, stdN / 255, im_gt.shape)
        if self.noise_mode == 'GB':
            # generate sigmaMap
            sigma_map = self.generate_sigma()
            # generate noise
            noise = torch.randn(im_gt.shape).numpy() * sigma_map
        if self.noise_mode == 'SYN_S':
            noise = (im_noise/im_noise_max)*(self.sigma_max / 255)
        if self.noise_mode == 'SYN_B':
            stdN = np.random.uniform(self.sigma_min, self.sigma_max)
            noise = (im_noise / im_noise_max) * (stdN / 255)
        if self.noise_mode == 'SYN_GB':
            # generate noise
            noise = (im_noise/im_noise_max) * sigma_map

        im_noisy= im_gt + noise

        # data augmentation
        im_gt, im_noisy, sigma_map = random_augmentation(im_gt, im_noisy, sigma_map)
        if self.noise_estimate:
            sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial) #CxHxW
            sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2,0,1)))
            # Groundtruth SigmaMap
            sigma2_map_gt = np.tile(np.square(sigma_map), (1, 1, C))
            sigma2_map_gt = np.where(sigma2_map_gt<1e-10, 1e-10, sigma2_map_gt)
            sigma2_map_gt = torch.from_numpy(sigma2_map_gt.transpose((2, 0, 1)))
        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1))).type(torch.float32)
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1))).type(torch.float32)


        if self.noise_estimate:
            return im_noisy, im_gt, sigma2_map_est, sigma2_map_gt
        else:
            return im_noisy, im_gt


# Simulation Datasets:
class SimulateTrain(BaseDataSetImg):
    def __init__(self, im_list, length,  pch_size=(128,128), radius=5, noise_estimate=True):
        super(SimulateTrain, self).__init__(im_list, length,  pch_size)
        self.win = 2*radius + 1
        self.sigma_spatial = radius
        self.noise_estimate = noise_estimate
        self.num_images = len(im_list)
        self.sigma_min = 0
        self.sigma_max = 75

    def generate_sigma(self):
        pch_size = self.pch_size[0]
        center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        scale = random.uniform(pch_size/4, pch_size/4*3)
        kernel = gaussian_kernel(pch_size, pch_size, center, scale)
        up = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        down = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        if up < down:
            up, down = down, up
        up += 5/255.0
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min())  *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :, np.newaxis]

    def __getitem__(self, index):
        pch_size = self.pch_size[0]
        ind_im = random.randint(0, self.num_images-1)

        im_ori = self.im_list[ind_im]

        im_gt = self.crop_patch(im_ori)  #mcj
        # im_gt = img_as_float(self.crop_patch(im_ori))

        C = im_gt.shape[2]

        # generate sigmaMap
        sigma_map = self.generate_sigma()

        # generate noise
        noise = torch.randn(im_gt.shape).numpy() * sigma_map
        # im_noisy = im_gt + noise.astype(np.float32)
        im_noisy = im_gt + noise

        im_gt, im_noisy, sigma_map = random_augmentation(im_gt, im_noisy, sigma_map)

        if self.noise_estimate:
            sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial) #CxHxW
            sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2,0,1)))
            # Groundtruth SigmaMap
            sigma2_map_gt = np.tile(np.square(sigma_map), (1, 1, C))
            sigma2_map_gt = np.where(sigma2_map_gt<1e-10, 1e-10, sigma2_map_gt)
            sigma2_map_gt = torch.from_numpy(sigma2_map_gt.transpose((2, 0, 1)))
        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        if self.noise_estimate:
            return im_noisy, im_gt, sigma2_map_est, sigma2_map_gt
        else:
            return im_noisy, im_gt

class SimulateTrain_sigma(BaseDataSetImg):
    def __init__(self, im_list, length,  pch_size=(128,128), radius=5, noise_estimate=True,sigma=125):
        super(SimulateTrain_sigma, self).__init__(im_list, length,  pch_size)
        self.win = 2*radius + 1
        self.sigma_spatial = radius
        self.noise_estimate = noise_estimate
        self.num_images = len(im_list)
        self.sigma_min = 0
        self.sigma_max = sigma

    def generate_sigma(self):
        pch_size = self.pch_size[0]
        center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        scale = random.uniform(pch_size/4, pch_size/4*3)
        kernel = gaussian_kernel(pch_size, pch_size, center, scale)
        up = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        down = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        if up < down:
            up, down = down, up
        up += 5/255.0
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min()) *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :, np.newaxis]

    def __getitem__(self, index):
        pch_size = self.pch_size[0]
        ind_im = random.randint(0, self.num_images-1)

        im_ori = self.im_list[ind_im]

        im_gt = self.crop_patch(im_ori)  #mcj
        # im_gt = img_as_float(self.crop_patch(im_ori))

        C = im_gt.shape[2]

        # generate sigmaMap
        sigma_map = self.generate_sigma() #*abs(im_gt).max()#mcj revise

        # generate noise
        noise = torch.randn(im_gt.shape).numpy() * sigma_map
        # im_noisy = im_gt + noise.astype(np.float32)
        im_noisy = im_gt + noise

        im_gt, im_noisy, sigma_map = random_augmentation(im_gt, im_noisy, sigma_map)

        if self.noise_estimate:
            sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial) #CxHxW
            sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2,0,1)))
            # Groundtruth SigmaMap
            sigma2_map_gt = np.tile(np.square(sigma_map), (1, 1, C))
            sigma2_map_gt = np.where(sigma2_map_gt<1e-10, 1e-10, sigma2_map_gt)
            sigma2_map_gt = torch.from_numpy(sigma2_map_gt.transpose((2, 0, 1)))
        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        if self.noise_estimate:
            return im_noisy, im_gt, sigma2_map_est, sigma2_map_gt
        else:
            return im_noisy, im_gt

class SimulateTrain_Gauss(BaseDataSetImg):
    def __init__(self, im_list, length,  pch_size=(128,128), radius=5, sigma=125):
        super(SimulateTrain_Gauss, self).__init__(im_list, length,  pch_size)
        self.win = 2*radius + 1
        self.sigma_spatial = radius
        self.num_images = len(im_list)
        self.sigma = sigma

    def __getitem__(self, index):
        pch_size = self.pch_size[0]
        ind_im = random.randint(0, self.num_images-1)

        im_ori = self.im_list[ind_im]

        im_gt = self.crop_patch(im_ori)  #mcj
        # im_gt = img_as_float(self.crop_patch(im_ori))

        C = im_gt.shape[2]

        # generate noise
        noise = torch.randn(im_gt.shape).numpy() * self.sigma/255
        # im_noisy = im_gt + noise.astype(np.float32)
        im_noisy = im_gt + noise

        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt




class SimulateTest(uData.Dataset):
    def __init__(self, im_list, h5_path):
        super(SimulateTest, self).__init__()
        self.im_list = im_list
        self.h5_path = h5_path

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        im_gt = self.im_list[index]
        # im_key = os.path.basename(self.im_list[index]).split('.')[0]
        im_key = str(index)
        C = im_gt.shape[2]

        with h5.File(self.h5_path, 'r') as h5_file:
            noise = np.array(h5_file[im_key][:,:,:C])
        H, W, _ = noise.shape

        # im_gt = img_as_float(im_gt[:H, :W])#mcj
        im_gt = im_gt[:H, :W]

        im_noisy = im_gt + noise

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1))).type(torch.float32)# mcj float32
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1))).type(torch.float32)

        return im_noisy, im_gt



# load Datasets:
class SeisNpyTrainDataset(BaseDataSetImg):
    def __init__(self, im_list, length,  pch_size=(128,128)):
        super(SeisNpyTrainDataset, self).__init__(im_list, length,  pch_size)
        self.num_images = len(im_list)

    def __getitem__(self, index):
        pch_size = self.pch_size[0]
        ind_im = random.randint(0, self.num_images-1)

        im_ori = self.im_list[ind_im]

        im_gt = self.crop_patch(im_ori)  #mcj
        # im_gt = img_as_float(self.crop_patch(im_ori))
        C = im_gt.shape[2]
        im_gt= random_augmentation(im_gt)
        im_gt = torch.from_numpy(im_gt[0].transpose((2, 0, 1)))
        return {'H': im_gt}

class SeisNpyTrainDataset_noAug(BaseDataSetImg):
    def __init__(self, im_list, length,  pch_size=(128,128)):
        super(SeisNpyTrainDataset_noAug, self).__init__(im_list, length,  pch_size)
        self.num_images = len(im_list)

    def __getitem__(self, index):
        pch_size = self.pch_size[0]
        # ind_im = random.randint(0, self.num_images-1)
        # im_ori = self.im_list[ind_im]
        im_ori = self.im_list[index]
        im_gt = self.crop_patch(im_ori)  #mcj
        # im_gt = img_as_float(self.crop_patch(im_ori))
        C = im_gt.shape[2]
        # im_gt= random_augmentation(im_gt)
        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        return {'H': im_gt}
