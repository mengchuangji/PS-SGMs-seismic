import os
import cv2
import glob
import numpy as np
import scipy.io as sio

# cropSize = 512
# num = 50 #100
pch_size=256
stride=128


# Crop
GT_v = glob.glob(os.path.join('D:\datasets\seismic\marmousi', '*vp.mat'))
GT_den = glob.glob(os.path.join('D:\datasets\seismic\marmousi', '*den.mat'))


GT_v.sort()
GT_den.sort()

GT_v_max=(sio.loadmat(GT_v[0])['vp']).max()
GT_den_max=(sio.loadmat(GT_den[0])['den']).max()


out_dir = "D:\datasets\seismic\marmousi\\marmousi_vp_den_s256_o128"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(os.path.join(out_dir, 'CL')): #CL GT
    os.mkdir(os.path.join(out_dir, 'CL'))
# if not os.path.exists(os.path.join(out_dir, 'RN')): #RN Noisy
#     os.mkdir(os.path.join(out_dir, 'RN'))

num_patch = 0
for ii in range(len(GT_v)):
    if (ii + 1) % 10 == 0:
        print('    The {:d} original images'.format(ii + 1))

    im_gt_v = sio.loadmat(GT_v[ii])['vp']/GT_v_max  # [:, :, ::-1]
    im_gt_den = sio.loadmat(GT_den[ii])['den']/GT_den_max  # [:, :, ::-1]
    im_gt = np.stack((im_gt_v, im_gt_den), axis=0)

    C, H, W = im_gt.shape
    ind_H = list(range(0, H - pch_size + 1, stride))
    if ind_H[-1] < H - pch_size:
        ind_H.append(H - pch_size)
    ind_W = list(range(0, W - pch_size + 1, stride))
    if ind_W[-1] < W - pch_size:
        ind_W.append(W - pch_size)
    kk=0
    for start_H in ind_H:
        for start_W in ind_W:
            # pch_noisy = im_noisy[start_H:start_H + pch_size, start_W:start_W + pch_size,]
            pch_gt = im_gt[:, start_H:start_H + pch_size, start_W:start_W + pch_size]
            # sio.savemat(os.path.join(out_dir, 'CL', '%d_%d.mat' % (ii, kk)), {'data': np.expand_dims(pch_gt,axis=0)})
            sio.savemat(os.path.join(out_dir, 'CL', '%d_%d.mat' % (ii, kk)), {'data': pch_gt})
            # sio.savemat(os.path.join(out_dir, 'RN', '%d_%d.mat' % (ii, kk)), {'data': np.expand_dims(pch_noisy,axis=0)})
            kk+=1
            num_patch += 1
print('Total {:d} small images pairs'.format(num_patch))
print('Finish!\n')