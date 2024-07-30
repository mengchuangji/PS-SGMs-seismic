import numpy as np
import glob
import tqdm

import torch.nn.functional as F
import torch
import os
from torchvision.utils import make_grid, save_image
from utils.make_grid_h import make_grid_h
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from models.UNet import UNet
from datasets import get_dataset, data_transform, inverse_data_transform
from models import general_anneal_Langevin_dynamics,general_anneal_Langevin_dynamics_den,general_anneal_Langevin_dynamics_inp
from models import get_sigmas
from models.ema import EMAHelper
from filter_builder import get_custom_kernel
import scipy.io as sio
from seis_utils.localsimi import localsimi

__all__ = ['NCSNRunner']


def get_model(config):
    if config.data.dataset == 'CELEBA' or config.data.dataset == 'marmousi':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)
    elif config.data.dataset == 'marmousi_unet':
        return UNet(config).to(config.device)

class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def sample_general(self, score, samples, init_samples, sigma_0, sigmas, num_variations = 8, deg = 'sr4'):
        ## show stochastic variation ##
        stochastic_variations = torch.zeros((3 + num_variations) * self.config.sampling.batch_size, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)
        stochastic_variations_R = torch.zeros((3 + num_variations) * self.config.sampling.batch_size,
                                            self.config.data.channels, self.config.data.image_size,
                                            self.config.data.image_size)
        stochastic_variations_LS = torch.zeros((3 + num_variations) * self.config.sampling.batch_size,
                                              self.config.data.channels, self.config.data.image_size,
                                              self.config.data.image_size)



        # clean = samples.view(samples.shape[0], self.config.data.channels,
        #                               self.config.data.image_size,
        #                               self.config.data.image_size)
        # sample_gt = inverse_data_transform(self.config, clean)
        # stochastic_variations[0 : self.config.sampling.batch_size,:,:,:] = sample_gt

        img_dim = self.config.data.image_size ** 2
        image_size=self.config.data.image_size

        ## get degradation matrix ##
        H = 0
        if deg[:2] == 'cs':
            ## random with set singular values ##
            compress_by = int(deg[2:]) #4
            Vt = torch.rand(img_dim, img_dim).to(self.config.device)#[16384, 16384]
            Vt, _ = torch.qr(Vt, some=False) #torch.Size([16384, 16384])
            U = torch.rand(img_dim // compress_by, img_dim // compress_by).to(self.config.device) #[4096, 4096]
            U, _ = torch.qr(U, some=False)#[4096, 4096]
            S = torch.hstack((torch.eye(img_dim // compress_by), torch.zeros(img_dim // compress_by, (compress_by-1) * img_dim // compress_by))).to(self.config.device)#[4096, 16384]
            H = torch.matmul(U, torch.matmul(S, Vt))
        elif deg == 'den':
            ## denoise ##
            # H = torch.eye(img_dim).to(self.config.device)
            print('denosing')
        elif deg == 'syn':
            ## synthesis ##
            H = torch.zeros(img_dim).to(self.config.device)
            # stochastic_variations[0: self.config.sampling.batch_size, :, :, :] = torch.zeros(sample_gt.shape)
        elif deg == 'inp':

            ## crop ##
            H_0 = torch.eye(img_dim).to(self.config.device)
            # H = H_0[:-(self.config.data.image_size*20), :]
            H = H_0[:-(self.config.data.image_size * 10), :]

        elif deg == 'inp_v1':
            # M = torch.ones(self.config.data.image_size, self.config.data.image_size).to(self.config.device)
            # M[:, self.config.data.image_size // 2:] = 0

            # regular mask
            def regular_mask(image_size, a):
                n = image_size  # data.shape[-1]
                mask = torch.zeros((n, n)).to(self.config.device)
                for i in range(n):
                    if (i + 1) % a == 1:
                        mask[:, i] = 1
                    else:
                        mask[:, i] = 0
                return mask
            M=regular_mask(image_size=self.config.data.image_size,a=4)

            # # mask 1
            # M= torch.ones(self.config.data.image_size,self.config.data.image_size).to(self.config.device)
            # # 概率置零的概率（这里设置为 0.3）
            # prob = 0.5
            # # 生成概率矩阵（决定每列是否置零）
            # prob_tensor = torch.rand(M.shape[1]) < prob
            # # 将符合条件的列置零
            # M[:, prob_tensor] = 0

            # mask 2
            # 创建一个全为1的矩阵
            # M = torch.ones(self.config.data.image_size, self.config.data.image_size).to(self.config.device)
            # # 将后一半列的值置为0
            # # M[:, self.config.data.image_size // 2:] = 0
            # # M[:, 1*self.config.data.image_size // 8: 7*self.config.data.image_size // 8] = 0
            # M[:, 2 * self.config.data.image_size // 8: 6 * self.config.data.image_size // 8] = 0

            # mask 3
            # 加载 .npy 文件为 NumPy 数组
            # M = np.load('/home/shendi_mcj/code/Reproducible/snips_torch-main/inp_masks/text_mask_2.npy')
            # M[M<255]=0
            # M = M / M.max()
            # 将 NumPy 数组转换为 PyTorch 的 Tensor
            # M = torch.from_numpy(M).float().to(self.config.device)
            # M = torch.FloatTensor(M).to(self.config.device)

            def generate_sparse_matrix(rows, cols, min_consecutive_zeros):
                # 生成初始的全一矩阵
                ones_matrix = torch.ones((rows, cols))

                # 计算需要置为0的列的总数（占一半）
                num_zeros_cols = cols // 2

                # 确保至少有 min_consecutive_zeros 列为0
                start_col = torch.randint(0, cols - min_consecutive_zeros + 1, (1,))
                ones_matrix[:, start_col:start_col + min_consecutive_zeros] = 0
                num_zeros_cols -= min_consecutive_zeros

                # 随机选择其他列置为0，直到满足总数要求
                while num_zeros_cols > 0:
                    col_index = torch.randint(0, cols, (1,))
                    if torch.all(ones_matrix[:, col_index:col_index + min_consecutive_zeros] == 1):
                        ones_matrix[:, col_index:col_index + min_consecutive_zeros] = 0
                        num_zeros_cols -= min_consecutive_zeros

                return ones_matrix
            # M=generate_sparse_matrix(self.config.data.image_size,self.config.data.image_size,16).to(self.config.device)

            # fiexed mask
            # M_np=np.load('/home/shendi_mcj/code/Reproducible/snips_torch-main/inp_masks/mask_rnd_05.npy')
            # #'/home/shendi_mcj/code/Reproducible/snips_torch-main/inp_masks'
            # M = torch.from_numpy(M_np).float().to(self.config.device)

            # H_0 = torch.eye(img_dim).to(self.config.device) * torch.diag(M.flatten())
            # # 获取只有一个元素值为 1 的所有行
            # H = H_0[(H_0 == 1.).sum(dim=1) == 1.]

        elif deg == 'inp_v2':

            # # mask 1
            # M= torch.ones(self.config.data.image_size,self.config.data.image_size).to(self.config.device)
            # # 概率置零的概率（这里设置为 0.3）
            # prob = 0.0
            # # 生成概率矩阵（决定每列是否置零）
            # prob_tensor = torch.rand(M.shape[1]) < prob
            # # 将符合条件的列置零
            # M[:, prob_tensor] = 0

            #mask 2
            # 创建一个全为1的矩阵
            M = torch.ones(self.config.data.image_size, self.config.data.image_size).to(self.config.device)

            # 将后一半列的值置为0
            # M[:, self.config.data.image_size // 2:] = 0
            # M[:, 1*self.config.data.image_size // 8: 7*self.config.data.image_size // 8] = 0
            M[:, 2 * self.config.data.image_size // 8: 6 * self.config.data.image_size // 8] = 0

            # mask 3
            # # 加载 .npy 文件为 NumPy 数组
            # M = np.load('/home/shendi_mcj/code/Reproducible/snips_torch-main/inp_masks/text_mask_2.npy')
            # M[M < 255] = 0
            # M=M/M.max()
            # # 将 NumPy 数组转换为 PyTorch 的 Tensor
            # # M = torch.from_numpy(M).float().to(self.config.device)
            # M = torch.FloatTensor(M).to(self.config.device)

            H_0 = torch.eye(img_dim).to(self.config.device)*torch.diag(M.flatten())
            # 获取只有一个元素值为 1 的所有行
            H = H_0[(H_0 == 1.).sum(dim=1) == 1.]
        elif deg == 'deblur_uni':
            ## blur ##
            H = torch.from_numpy(get_custom_kernel(type="uniform", dim = self.config.data.image_size)).type(torch.FloatTensor).to(self.config.device)
        elif deg == 'deblur_gauss':
            ## blur ##
            H = torch.from_numpy(get_custom_kernel(type="gauss", dim = self.config.data.image_size)).type(torch.FloatTensor).to(self.config.device)
        elif deg[:2] == 'sr':
            ## downscale - super resolution ##
            blur_by = int(deg[2:])
            H = torch.zeros((img_dim // (blur_by**2), img_dim)).to(self.config.device)
            for i in range(self.config.data.image_size // blur_by):
                for j in range(self.config.data.image_size // blur_by):
                    for i_inc in range(blur_by):
                        for j_inc in range(blur_by):
                            H[i * self.config.data.image_size // blur_by + j, (blur_by*i + i_inc) * self.config.data.image_size + (blur_by*j + j_inc)] = (1/blur_by**2)
        else:
            print("ERROR: degradation type not supported")
            quit()

        if deg == 'inp_v1':
            y_0 = samples.view(samples.shape[0], self.config.data.channels,
                                                                       img_dim)*M.flatten()
            # y_0 = (samples + sigma_0 * torch.randn_like(samples)).view(samples.shape[0], self.config.data.channels,
            #                                                           img_dim)
            # torch.save((y_0).view(samples.shape[0], self.config.data.channels,
            #                                               img_dim), os.path.join(self.args.image_folder, "y_0.pt"))
            sio.savemat(os.path.join(self.args.image_folder, "y_0.mat"),
                        {'data': (y_0).view(samples.shape[0], self.config.data.channels,
                                                          image_size,image_size).cpu().squeeze().numpy()})

            pinv_y_0 = y_0.view(samples.shape[0] * self.config.data.channels,
                                                      img_dim, 1)
        elif deg == 'den':
            y_0 = samples.view(samples.shape[0], self.config.data.channels,
                               img_dim)
            # torch.save((y_0).view(samples.shape[0], self.config.data.channels,
            #                                     img_dim), os.path.join(self.args.image_folder, "y_0.pt"))
            sio.savemat(os.path.join(self.args.image_folder, "y_0.mat"),
                        {'data': (y_0).view(samples.shape[0], self.config.data.channels,
                                            image_size, image_size).cpu().squeeze().numpy()})

            pinv_y_0 = y_0.view(samples.shape[0] * self.config.data.channels,
                                img_dim, 1)
        elif deg=='inp_v2':
            # x_orig = samples #[8, 1, 128, 128]
            # x_orig = x_orig.to(self.config.device)
            # x_orig = data_transform(self.config, x_orig)
            # y_0 = H_funcs.H(x_orig)#[8, 8192]
            # y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
            # torch.save(y_0, os.path.join(self.args.image_folder, "y_0.pt"))
            # pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0]* self.config.data.channels, self.config.data.image_size*self.config.data.image_size,1)
            # y_0=y_0.view(y_0.shape[0], self.config.data.channels, -1)
            y_0 = torch.matmul(H, samples.view(samples.shape[0] * self.config.data.channels,
                                               img_dim, 1)).view(samples.shape[0], self.config.data.channels,
                                                                 H.shape[0])

            # y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
            # torch.save(y_0, os.path.join(self.args.image_folder, "y_0.pt"))
            sio.savemat(os.path.join(self.args.image_folder, "y_0.mat"),
                        {'data': (samples*M).view(samples.shape[0], self.config.data.channels,
                                            image_size, image_size).cpu().squeeze().numpy()})
            H_t = H.transpose(0, 1)
            H_cross = torch.matmul(H_t, torch.inverse(torch.matmul(H, H_t)))
            pinv_y_0 = torch.matmul(H_cross, y_0.view(samples.shape[0] * self.config.data.channels,
                                                      H.shape[0], 1))
        else:
                y_0 = torch.matmul(H, samples.view(samples.shape[0] * self.config.data.channels,
                                                   img_dim, 1)).view(samples.shape[0], self.config.data.channels,
                                                                     H.shape[0]) # (8,1,11264)
                # y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
                # torch.save(y_0, os.path.join(self.args.image_folder, "y_0.pt")) # (8,1,11264)
                sio.savemat(os.path.join(self.args.image_folder, "y_0.mat"),
                            {'data': (samples*M).view(samples.shape[0], self.config.data.channels,
                                                image_size, image_size).cpu().squeeze().numpy()})
                H_t = H.transpose(0, 1) #(16384,11264)
                H_cross = torch.matmul(H_t, torch.inverse(torch.matmul(H, H_t))) #(16384,11264)
                pinv_y_0 = torch.matmul(H_cross, y_0.view(samples.shape[0] * self.config.data.channels,
                                                          H.shape[0], 1)) #(8, 16384, 1)

        if deg == 'deblur_uni' or deg == 'deblur_gauss':
            pinv_y_0 = y_0

        sample_y_0 = inverse_data_transform(self.config, pinv_y_0.view(samples.shape[0], self.config.data.channels,
                                      self.config.data.image_size,
                                      self.config.data.image_size))


        if deg == 'inp_v2' or deg == 'inp_v1':
            stochastic_variations[0 * self.config.sampling.batch_size : 1 * self.config.sampling.batch_size,:,:,:] = sample_y_0#*M
        else:
            stochastic_variations[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :,:] = sample_y_0
        stochastic_variations_R[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :,
            :] = 0
        if deg == 'den':
            stochastic_variations_LS[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :,
                :] = 0

        if deg=='den':
            index = np.abs(sigmas - sigma_0).argmin()
            index_arr=range(index,len(sigmas),(len(sigmas)-index)//4)
            index_arr_len=len(index_arr)
            if index_arr[-1]==499:
                x_t_list_len= index_arr_len+1
            else:
                x_t_list_len = index_arr_len+1
        elif deg == 'inp_v2' or deg == 'inp_v1':
            x_t_list_len = 11 #11
        else:
            x_t_list_len = 11  # 11
        stochastic_variations_x_t = torch.zeros(
            ((1 + x_t_list_len) * num_variations) * self.config.sampling.batch_size,
            self.config.data.channels, self.config.data.image_size,
            self.config.data.image_size)

        ## apply SNIPS ##
        for i in range(num_variations):
            # all_samples, x_t_list = general_anneal_Langevin_dynamics(H, y_0, init_samples, score, sigmas,
            #                                self.config.sampling.n_steps_each,
            #                                self.config.sampling.step_lr, verbose=True,
            #                                final_only=self.config.sampling.final_only,
            #                                denoise=self.config.sampling.denoise, c_begin=0, sigma_0 = sigma_0)
            all_samples, x_t_list = general_anneal_Langevin_dynamics_den(y_0, init_samples, score, sigmas,
                                                           self.config.sampling.n_steps_each,
                                                           self.config.sampling.step_lr, verbose=True,
                                                           final_only=self.config.sampling.final_only,
                                                           denoise=self.config.sampling.denoise, c_begin=0,
                                                           sigma_0=sigma_0)
            # all_samples, x_t_list = general_anneal_Langevin_dynamics_inp(M, y_0, init_samples, score, sigmas,
            #                                                              self.config.sampling.n_steps_each,
            #                                                              self.config.sampling.step_lr, verbose=True,
            #                                                              final_only=self.config.sampling.final_only,
            #                                                              denoise=self.config.sampling.denoise,
            #                                                              c_begin=0,
            #                                                              sigma_0=sigma_0)

            sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                      self.config.data.image_size,
                                      self.config.data.image_size).to(self.config.device)
            stochastic_variations[(self.config.sampling.batch_size) * (i+1) : (self.config.sampling.batch_size) * (i+2),:,:,:] = inverse_data_transform(self.config, sample)
            # stochastic_variations_x_t[0 : self.config.sampling.batch_size, :, :, :] = sample_gt
            # stochastic_variations_x_t[1 * self.config.sampling.batch_size : 2 * self.config.sampling.batch_size, :, :, :] = sample_y_0
            if deg == 'inp_v2' or deg == 'inp_v1':
                sample_y_0[:,:,M<1]=-0.0
                stochastic_variations_x_t[i * (1+x_t_list_len): i * (1+x_t_list_len)+1 :, :,:] = sample_y_0
                stochastic_variations[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :,
                :] = sample_y_0
                stochastic_variations_R[(self.config.sampling.batch_size) * (i + 1): (self.config.sampling.batch_size) * (i + 2), :, :,
                :] = sample_y_0 - inverse_data_transform(self.config, sample)
                # print(i * (1 + x_t_list_len), 'hh')
            else:
                # stochastic_variations_x_t[i * self.config.sampling.batch_size: (i+1) * self.config.sampling.batch_size, :, :,
                # :] = sample_y_0
                stochastic_variations_x_t[i * (1+x_t_list_len): i * (1+x_t_list_len)+1 :, :,:] = sample_y_0
                stochastic_variations_R[(self.config.sampling.batch_size) * (i + 1): (self.config.sampling.batch_size) * (i + 2), :, :,
                :] = sample_y_0 - inverse_data_transform(self.config, sample)
                if deg == 'den':
                    LS=localsimi(inverse_data_transform(self.config, sample).cpu().squeeze().numpy(),
                                 (sample_y_0 - inverse_data_transform(self.config, sample)).cpu().squeeze().numpy(),
                                 rect=[5, 5, 1], niter=20, eps=0.0, verb=1).squeeze()[np.newaxis,np.newaxis, :]
                    energy_simi = np.sum(LS ** 2) / LS.size
                    print("energy_simi=", energy_simi)
                    LS=torch.from_numpy(LS).contiguous().type(torch.FloatTensor).to(sample_y_0.device)
                    stochastic_variations_LS[
                    (self.config.sampling.batch_size) * (i + 1): (self.config.sampling.batch_size) * (i + 2), :, :,
                    :] = LS
                print(i * (1+x_t_list_len),'hh')

            for j, x_t in enumerate(x_t_list):
                x_t = x_t.view(sample.shape[0], self.config.data.channels,
                                          self.config.data.image_size,
                                          self.config.data.image_size).to(self.config.device)
                stochastic_variations_x_t[
                (self.config.sampling.batch_size)* (j+1+(1+len(x_t_list))*i): (self.config.sampling.batch_size)* (j+2+(1+len(x_t_list))*i), :, :,
                :] = inverse_data_transform(self.config, x_t)
                # print(j+1+(1+len(x_t_list))*i)

        ## x_t evolution ##
        image_grid = make_grid(stochastic_variations_x_t, 1+x_t_list_len,padding=4)
        # save_image(image_grid, os.path.join(self.args.image_folder, 'stochastic_variation_x_t.png'))
        import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(15,15)
        # plt.gcf().set_size_inches(3*(2 + len(x_t_list)*num_variations) , 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_x_t.png'), dpi=300,bbox_inches='tight')
        if deg == 'inp_v2' or deg == 'inp_v1':
            sample_y_0[:,:,M < 1] = 0
            stochastic_variations_x_t[1 * self.config.sampling.batch_size: 2 * self.config.sampling.batch_size, :, :,
            :] = sample_y_0
        # torch.save(stochastic_variations_x_t, os.path.join(self.args.image_folder, "results_x_t.pt"))
        sio.savemat(os.path.join(self.args.image_folder, "results_x_t.mat"), {'data': stochastic_variations_x_t.cpu().squeeze().numpy()})

        # calculate mean and std ##
        runs = stochastic_variations[(self.config.sampling.batch_size) * (1) : (self.config.sampling.batch_size) * (1+num_variations),:,:,:]
        runs = runs.view(-1, self.config.sampling.batch_size, self.config.data.channels,
                          self.config.data.image_size,
                          self.config.data.image_size)
        stochastic_variations[(self.config.sampling.batch_size) * (-2) : (self.config.sampling.batch_size) * (-1),:,:,:] = torch.mean(runs, dim=0)
        stochastic_variations[(self.config.sampling.batch_size) * (-1) : ,:,:,:] = torch.std(runs, dim=0)
        stochastic_variations_R[(self.config.sampling.batch_size) * (-2): (self.config.sampling.batch_size) * (-1), :, :,
        :] = sample_y_0.cpu()-torch.mean(runs, dim=0)
        stochastic_variations_R[(self.config.sampling.batch_size) * (-1):, :, :, :] = 0
        if deg == 'den':
            LS = localsimi(torch.mean(runs, dim=0).cpu().squeeze().numpy(),
                           (sample_y_0.cpu() - torch.mean(runs, dim=0)).cpu().squeeze().numpy(), rect=[5, 5, 1],
                           niter=20, eps=0.0, verb=1).squeeze()[np.newaxis, np.newaxis, :]
            energy_simi = np.sum(LS ** 2) / LS.size
            print("energy_simi=", energy_simi)
            LS = torch.from_numpy(LS).contiguous().type(torch.FloatTensor).to(sample_y_0.device)
            stochastic_variations_LS[(self.config.sampling.batch_size) * (-2): (self.config.sampling.batch_size) * (-1), :,
            :,
            :] = LS
            stochastic_variations_LS[(self.config.sampling.batch_size) * (-1):, :, :, :] = 0

        ######### plot stochastic_variations ###############
        image_grid = make_grid_h(stochastic_variations,  self.config.sampling.batch_size,padding=8)
        # save_image(image_grid, os.path.join(self.args.image_folder, 'stochastic_variation.png'))
        # import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(10, 10)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation.png'), dpi=300,bbox_inches='tight')
        if deg == 'inp_v2' or deg == 'inp_v1':
            sample_y_0[:,:,M < 1] = 0
            stochastic_variations[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :,
            :] = sample_y_0
        # torch.save(stochastic_variations, os.path.join(self.args.image_folder, "results.pt"))
        sio.savemat(os.path.join(self.args.image_folder, "results.mat"),
                    {'data': stochastic_variations.cpu().squeeze().numpy()})

        ######### plot stochastic_variations_R (residual) ###############
        image_grid = make_grid_h(stochastic_variations_R, self.config.sampling.batch_size,padding=8)
        # save_image(image_grid, os.path.join(self.args.image_folder, 'stochastic_variation.png'))
        # import matplotlib.pyplot as plt
        plt.gcf().set_size_inches(10, 10)
        # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
        plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.seismic, vmin=-1, vmax=1)
        # plt.colorbar()  # 添加色标
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_R.png'), dpi=300, bbox_inches='tight')
        if deg == 'inp_v2' or deg == 'inp_v1':
            sample_y_0[:, :, M < 1] = 0
            stochastic_variations_R[0 * self.config.sampling.batch_size: 1 * self.config.sampling.batch_size, :, :,
            :] = sample_y_0
        # torch.save(stochastic_variations_R, os.path.join(self.args.image_folder, "results_R.pt"))
        sio.savemat(os.path.join(self.args.image_folder, "results_residual.mat"),
                    {'data': stochastic_variations_R.cpu().squeeze().numpy()})

        ######### plot stochastic_variations_LS (localsimi of denoised and noise) ###############
        if deg == 'den':
            image_grid = make_grid_h(stochastic_variations_LS, self.config.sampling.batch_size, padding=8,pad_value=1)
            # save_image(image_grid, os.path.join(self.args.image_folder, 'stochastic_variation.png'))
            # import matplotlib.pyplot as plt
            plt.gcf().set_size_inches(10, 10)
            # plt.gcf().set_size_inches(3*(4 + num_variations), 3*self.config.sampling.batch_size)  # 设置图像尺寸为 10x6
            plt.imshow(image_grid.numpy().squeeze().transpose((1, 2, 0))[:, :, 0], cmap=plt.cm.jet, vmin=0, vmax=1)
            # plt.colorbar()  # 添加色标
            plt.axis('off')  # 关闭坐标轴
            plt.savefig(os.path.join(self.args.image_folder, 'stochastic_variation_LS.png'), dpi=300, bbox_inches='tight')
            sio.savemat(os.path.join(self.args.image_folder, "results_localsimi.mat"),
                        {'data': stochastic_variations_LS.cpu().squeeze().numpy()})



        ## report PSNRs ##
        # clean = stochastic_variations[0 * self.config.sampling.batch_size : 1 * self.config.sampling.batch_size,:,:,:]
        # for i in range(num_variations):
        #     general = stochastic_variations[(2+i) * self.config.sampling.batch_size : (3+i) * self.config.sampling.batch_size,:,:,:]
        #     mse = torch.mean((general - clean) ** 2)
        #     instance_mse = ((general - clean) ** 2).view(general.shape[0], -1).mean(1)
        #     psnr = torch.mean(10 * torch.log10(1/instance_mse))
        #     print("MSE/PSNR of the general #%d: %f, %f" % (i, mse, psnr))
        # mean = stochastic_variations[(2+num_variations) * self.config.sampling.batch_size : (3+num_variations) * self.config.sampling.batch_size,:,:,:]
        # mse = torch.mean((mean - clean) ** 2)
        # instance_mse = ((mean - clean) ** 2).view(mean.shape[0], -1).mean(1)
        # psnr = torch.mean(10 * torch.log10(1/instance_mse))
        # print("MSE/PSNR of the mean: %f, %f" % (mse, psnr))
    

    def sample(self,obs):
        score, states = 0, 0
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path_model, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path_model, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        sigma_0 = self.args.sigma_0

        # dataset = get_dataset(self.args, self.config)
        # dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
        #                         num_workers=4)
        # from datasets.seis_mat import SeisMatTrainDataset, SeisMatValidationDataset
        # #'D:\\datasets\\seismic\\marmousi\\f35_s256_o128'
        # dataset = SeisMatValidationDataset(path='/home/shendi_mcj/datasets/seismic/marmousi/f35_s256_o128',
        #                                     patch_size=self.config.data.image_size, pin_memory=True)
        # dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
        #                          num_workers=0, drop_last=True)

        score.eval()

        # data_iter = iter(dataloader)
        # samples, _ = next(data_iter)

        # data_iter = iter(dataloader)
        # samples_ = next(data_iter)
        # samples = samples_['L']#/samples_['H'].max()

        samples = obs.to(self.config.device)
        samples = data_transform(self.config, samples)
        init_samples = torch.rand_like(samples)

        self.sample_general(score, samples, init_samples, sigma_0, sigmas, num_variations=self.args.num_variations, deg=self.args.degradation)