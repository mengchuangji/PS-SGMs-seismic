#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-22 22:07:08

import torch
import torch.nn as nn
from torch.autograd import Function as autoF
from scipy.special import gammaln
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio #as compare_psnr
from skimage.metrics import structural_similarity #as  compare_ssim
from skimage import img_as_ubyte
import numpy as np
import sys
from math import floor
import scipy

def ssim_index(im1, im2):
    '''
    Input:
        im1, im2: np.uint8 format
    '''
    if im1.ndim == 2:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                    use_sample_covariance=False, multichannel=False)
    elif im1.ndim == 3:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                     use_sample_covariance=False, multichannel=True)
    else:
        sys.exit('Please input the corrected images')
    return out

def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

def batch_PSNR(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=255)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += ssim_index(Iclean[i,:,:,:].transpose((1,2,0)), Img[i,:,:,:].transpose((1,2,0)))
    return (SSIM/Img.shape[0])

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

def capacity_cal(net):
    out = 0
    for param in net.parameters():
        out += param.numel()*4/1024/1024
    # print('Networks Parameters: {:.2f}M'.format(out))
    return out

class LogGamma(autoF):
    '''
    Implement of the logarithm of gamma Function.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            input_np = input.detach().cpu().numpy()
        else:
            input_np = input.detach().numpy()
        out = gammaln(input_np)
        out = torch.from_numpy(out).to(device=input.device).type(dtype=input.dtype)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.digamma(input) * grad_output

        return grad_input

def load_state_dict_cpu(net, state_dict0):
    state_dict1 = net.state_dict()
    for name, value in state_dict1.items():
        assert 'module.'+name in state_dict0
        state_dict1[name] = state_dict0['module.'+name]
    net.load_state_dict(state_dict1)

class PadVINonIID:
    '''
    im: N x C x H x W torch tensor
    dep_U: depth of UNet
    '''
    def __init__(self, im, dep_U, mode='reflect'):
        self.im_old = im
        self.dep_U = dep_U
        self.mode = mode
        self.H_old = im.shape[2]
        self.W_old = im.shape[3]

    def pad(self):
        # lenU = 2 ** self.dep_U
        # padH = 0 if ((self.H_old % lenU) == 0) else ((self.H_old//lenU+1)* lenU-self.H_old)
        # padW = 0 if ((self.W_old % lenU) == 0) else ((self.W_old//lenU+1)* lenU-self.W_old)
        # padding = (0, padW, 0, padH)
        # import torch.nn.functional as F
        # out = F.pad(self.im_old, pad=padding, mode=self.mode)
        # return out

        lenU = 2 ** (self.dep_U-1)
        padH = 0 if ((self.H_old % lenU) == 0) else (lenU - (self.H_old % lenU))
        padW = 0 if ((self.W_old % lenU) == 0) else (lenU - (self.W_old % lenU))
        padding = (0, padW, 0, padH)
        import torch.nn.functional as F
        out = F.pad(self.im_old, pad=padding, mode=self.mode)
        return out

    def pad_inverse(self, im_new):
        return im_new[:, :, :self.H_old, :self.W_old]


def plot_spectrum(shot, dt, title='', fmax=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ps = np.sum(np.abs(np.fft.fft(shot)) ** 2, axis=-2); freqs = np.fft.fftfreq(len(ps), dt); idx = np.argsort(freqs)
    causal = int(len(ps) // 2); freqs, ps = freqs[idx], ps[idx]; freqs = freqs[-causal:]; ps = ps[-causal:]; freqs = freqs[freqs < (fmax if fmax else np.max(freqs))];
    n = len(freqs);
    ax.plot(freqs[:n], ps[:n], label=title);
    ax.set_xlabel('Frequency (Hz)');
    ax.set_ylabel('Gain'); ax.grid(True);


def zero_below_freq(dat, fhi, dt, disable=False, reverse=False):
    """ Input zeros into frequency spectrum of data below or above specified frequency.

    Args:
        dat(np.ndarray): 2D array [noffsets, ntimes]
        fhi(float): threshold frequency, Hz
        dt(float): temporal sampling, sec
        disable(bool): do nothing, return input data
        reverse(bool): when True, set zeros above fhi, otherwise below
    """
    if disable:
        return dat

    h, w = dat.shape[-2:]
    dat_fx = np.fft.rfft(dat, w)
    ff = np.fft.rfftfreq(dat.shape[-1], d=dt)
    if not reverse:
        where_to_zero = np.where(ff < fhi)[0]
    else:
        where_to_zero = np.where(ff >= fhi)[0]
    dat_fx[..., where_to_zero] = 0. + 0. * 1j
    out = np.fft.irfft(dat_fx, w)
    return out

def butter_bandpass(flo=None, fhi=None, fs=None, order=8, btype='band'):
    """A component of `bandpass` function"""
    nyq = 0.5 * fs
    if btype == 'band':
        low = flo / nyq
        high = fhi / nyq
        lims = [low, high]
    elif btype == 'low':
        high = fhi / nyq
        lims = high
    elif btype == 'high':
        low = flo / nyq
        lims = low

    #b, a = scipy.signal.butter(order, lims, btype=btype)
    #return b, a

    sos = scipy.signal.butter(order, lims, btype=btype, output='sos')
    return sos


def bandpass(data, flo=None, fhi=None, dt=None, fs=None, order=4, btype='band', verbose=0, pad=(0, 8), upscale=0):
    """ Filter frequency content of 2D data in format [offset, time]

    Args:
        data (ndarray): [offset, time]
        flo (float): low coner frequency
        fhi (float): high corner frequency
        dt (float): sampling interval (introduced for back-compatibility). You can enter either one dt or fs
        fs (float): 1/dt, sampling frequency, Hz
        order:
        btype (str): band, high or low
            * band: limit from both left and right
            * high: limit from right only
            * low: limit from left only
        verbose (bool): print details

    Returns:
        ndarray
    """

    if not fs:
        fs = 1 / dt

    if isinstance(data, torch.Tensor):
        data = data.numpy()

    if upscale:
        no, nt = data.shape
        data = scipy.signal.resample(data, nt * upscale, axis=-1)
        fs = fs * upscale

    if pad:
        no, nt = data.shape
        tmp = np.zeros((no, nt + pad[0] + pad[1]))
        tmp[:, pad[0]:nt + pad[0]] = data
        data = tmp.copy()

    if verbose:
        print(f'Bandpass:\n\t{data.shape}\tflo={flo}\tfhi={fhi}\tfs={fs}')
    # b, a = butter_bandpass(flo, fhi, fs, order=order, btype=btype)
    # y = scipy.signal.filtfilt(b, a, data)

    sos = butter_bandpass(flo, fhi, fs, order=order, btype=btype)
    y = scipy.signal.sosfiltfilt(sos, data)

    if pad:
        y = y[:, pad[0]:-pad[1] if pad[1] else None]

    if upscale:
        y = y[:, ::upscale]
    return y

def get_spectrum(t, dt, phase=False):
    t_fft = np.fft.fft(t)
    if not phase:
        ps = np.sum(np.abs(t_fft) ** 2, axis=-2)
    else:
        ps = np.sum(np.arctan2(np.imag(t_fft), np.real(t_fft)), axis=-2)
    freqs = np.fft.fftfreq(len(ps), dt)
    idx = np.argsort(freqs)
    causal = int(len(ps) // 2)
    freqs, ps = freqs[idx], ps[idx]
    return freqs[-causal:], ps[-causal:] / t.shape[-2]

def get_max_value(martix):

    '''

    得到矩阵中每一列最大的值

    '''

    res_list=[]

    for j in range(len(martix[0])):

        one_list=[]

        for i in range(len(martix)):

            one_list.append(int(martix[i][j]))

            res_list.append(str(max(one_list)))

    return res_list

from scipy.signal import butter, lfilter

def butter_bandpass_(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass_(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
