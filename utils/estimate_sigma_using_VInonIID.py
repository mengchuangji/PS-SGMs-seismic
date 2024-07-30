# -*- coding: utf-8 -*-
import argparse
import random
import os, time, datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from scipy.io import loadmat
from skimage import img_as_ubyte
from skimage.io import imread, imsave
# from skimage.measure import compare_ssim
import scipy.io as io


import segyio
from utils.networks import VDN, UNet, DnCNN
from seis_utils.utils import load_state_dict_cpu
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fielddata', type=str, help='directory of test dataset')
    parser.add_argument('--sigma', default=75, type=float, help='noise level')
    parser.add_argument('--agc', default=False, type=bool, help='Agc operation of the data,True or False')
    parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'DnCNN_sigma50'), help='directory of the model')
    parser.add_argument('--model_name', default='model.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results_denoise', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()




def show(x,y,sigma2,x_max):
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(12,30))
    plt.subplot(141)
    plt.imshow(x,vmin=-1,vmax=1,cmap='gray')
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(142)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=-1,vmax=1,cmap='gray')
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(143)
    noise= x-y
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=-1,vmax=1,cmap='gray')
    plt.title('noise')


    plt.subplot(144)
    # x_ = gain(x_, 0.004, 'agc', 0.05, 1)
    plt.imshow(sigma2)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.title('sigma2')
    plt.colorbar(shrink=0.5)
    plt.tight_layout()
    plt.show()
def showsigma(sigma2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,9))

    plt.imshow(sigma2, vmin=0,vmax=1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('sigma2')
    plt.colorbar(shrink= 0.8)
    plt.show()



def readsegy(data_dir, file,j):
        filename = os.path.join(data_dir, file)
        with segyio.open(filename, 'r', ignore_geometry=True) as f:
            f.mmap()
            sourceX = f.attributes(segyio.TraceField.SourceX)[:]
            trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
            shot_num = int(float(trace_num / 224))# 224 787
            len_shot = trace_num // shot_num  # The length of the data in each shot data
            data = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
            # data = data/data.max()
            # data = data  # 先不做归一化处理
            x = data[:, :]
            f.close()
            return x



def readsegy_all(data_dir, file):
    filename = os.path.join(data_dir, file)
    with segyio.open(filename, 'r', ignore_geometry=True) as f:
        f.mmap()
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
        data = np.asarray([np.copy(x) for x in f.trace[:trace_num]]).T
        f.close()
        return data



class VDN(nn.Module):
    def __init__(self, in_channels, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(VDN, self).__init__()
        # #VDN Unet
        self.DNet = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, slope=slope)
        # self.DNet = DnCNN(in_channels, in_channels * 2, dep=17, num_filters=64, slope=slope)
        # self.DNet = DnCNN_R(in_channels, in_channels*2, dep=17, num_filters=64, slope=slope)
        self.SNet = DnCNN(in_channels, in_channels*2, dep=dep_S, num_filters=64, slope=slope)

    def forward(self, x, mode='train'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x)
            return phi_sigma

use_gpu = True
C = 1
dep_U = 4
# # clip bound
log_max = math.log(1e4)
log_min = math.log(1e-8)

def estimate_sigma_using_VInonIID(x_):
    checkpoint = torch.load('/home/shendi_mcj/code/Reproducible/snips_torch-main/utils/TrainedModel/Non-IID-Unet/model_state_10')
    # checkpoint = torch.load(r'E:\VIRI\mycode\Reproducible\snips_torch-main\utils\TrainedModel\Non-IID-Unet\model_state_10')
    model = VDN(C, dep_U=dep_U, wf=64)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint)
    else:
        load_state_dict_cpu(model, checkpoint)
    model.eval()
    with torch.autograd.set_grad_enabled(False):
        phi_Z = model(x_, 'test')
        err = phi_Z.cpu().numpy()
        phi_sigma = model(x_, 'sigma')
        phi_sigma.clamp_(min=log_min, max=log_max)
        phi_sigma = phi_sigma  # /phi_sigma.max()
        log_alpha = phi_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = phi_sigma[:, C:, ]
        beta = torch.exp(log_beta)

        from seis_utils.PadUnet import PadUnet
        padunet = PadUnet(x_, dep_U=5)
        x_pad = padunet.pad()
        phi_Z_pad = model(x_pad, 'test')
        phi_Z = padunet.pad_inverse(phi_Z_pad)
        err = phi_Z.cpu().numpy()
        phi_sigma_pad = model(x_pad, 'sigma')
        phi_sigma = padunet.pad_inverse(phi_sigma_pad)
        phi_sigma.clamp_(min=log_min, max=log_max)
        phi_sigma = phi_sigma  # /phi_sigma.max()
        log_alpha = phi_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = phi_sigma[:, C:, ]
        beta = torch.exp(log_beta)
        sigma2 = beta / (alpha + 1)
        sigma2 = sigma2.cpu().numpy().squeeze()
        sigma = np.sqrt(sigma2)
        print("sigma.min:", sigma.min(), "sigma.median:", np.median(sigma), "sigma.max:", sigma.max())
        sigma_list = {'median':np.median(sigma), 'min':sigma.min(), 'max':sigma.max()}
        return sigma_list, sigma


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)


    data_dir = '/home/shendi_mcj/datasets/seismic/fielddata/'
    # im = 'PANKE-INline443'
    # im = '03-MonoNoiAtten-16_DYN_L1901-s11857.sgy'  00-L120.sgy
    original = readsegy_all(data_dir, '00-L120.sgy')
    x = original

    #############################
    x_max=max(abs(original.max()),abs(original.min()))
    x=x/x_max
    x=x/abs(x).max()
    x_ = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])
    torch.cuda.synchronize()
    start_time = time.time()
    if use_gpu:
        x_ = x_.cuda()
        print('Begin Testing on GPU')
    else:
        print('Begin Testing on CPU')
    # load the pretrained model
    print('Loading the Model')
    sigma_dict=estimate_sigma_using_VInonIID(x_)

    # VI-non-IID Unet
    checkpoint = torch.load('./TrainedModel/Non-IID-Unet/model_state_10')
    model = VDN(C, dep_U=dep_U, wf=64)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint)
    else:
        load_state_dict_cpu(model, checkpoint)

    model.eval()
    with torch.autograd.set_grad_enabled(False):
        phi_Z = model(x_, 'test')
        err = phi_Z.cpu().numpy()
        phi_sigma = model(x_, 'sigma')
        phi_sigma.clamp_(min=log_min, max=log_max)
        phi_sigma=phi_sigma#/phi_sigma.max()
        log_alpha = phi_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = phi_sigma[:, C:, ]
        beta = torch.exp(log_beta)

        from seis_utils.PadUnet import PadUnet
        padunet = PadUnet(x_, dep_U=5)
        x_pad = padunet.pad()
        phi_Z_pad = model(x_pad, 'test')
        phi_Z = padunet.pad_inverse(phi_Z_pad)
        err = phi_Z.cpu().numpy()
        phi_sigma_pad = model(x_pad, 'sigma')
        phi_sigma = padunet.pad_inverse(phi_sigma_pad)
        phi_sigma.clamp_(min=log_min, max=log_max)
        phi_sigma = phi_sigma  # /phi_sigma.max()
        log_alpha = phi_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = phi_sigma[:, C:, ]
        beta = torch.exp(log_beta)

        sigma2 = beta / (alpha + 1)
        sigma2 = sigma2.cpu().numpy().squeeze()
        sigma=np.sqrt(sigma2)
        print("Estimated sigmamap, sigma.min:",sigma.min(),"sigma.median:",np.median(sigma),"sigma.max:",sigma.max())
    if use_gpu:
        x_ = x_.cpu().numpy()
    else:
        x_ = x_.numpy()
    denoised = x_ - err[:, :C, ]
    denoised = denoised.squeeze()
    # sigma2 = sigma2.squeeze()
    elapsed_time = time.time() - start_time
    print(' elapsed_time: %2.4f second' % (elapsed_time))
    no=err[:, :C, ].squeeze()
    show(x, denoised, np.sqrt(sigma2), x_max)
    showsigma(sigma2)
    print('done')
