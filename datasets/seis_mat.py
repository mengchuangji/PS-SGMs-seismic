'''
We observe slightly better performance with training inputs in [0, 255] range than that in [0, 1],
so we follow AP-BSN that do not normalize the input image from [0, 255] to [0, 1].
'''
from datasets.base import BaseTrainDataset, dataset_path
import glob
import numpy as np
import os
from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset
import scipy

# sidd_path = os.path.join(dataset_path, 'SIDD')
mat_path = dataset_path

class SeisMatTrainDataset(BaseTrainDataset):
    def __init__(self, path, patch_size, pin_memory):
        super(SeisMatTrainDataset, self).__init__(path, patch_size, pin_memory)

    def __getitem__(self, index):
        index = index % len(self.img_paths)

        if self.pin_memory:
            img_L = self.imgs[index]['L']
            img_H = self.imgs[index]['H']
        else:
            img_path = self.img_paths[index]
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])

        # minner = img_L.min()
        # img_L = img_L - minner
        # img_H  = img_H - minner
        # maxer = img_L.max()
        # img_L = img_L / maxer
        # img_H  = img_H  / maxer

        img_L, img_H = self.crop(img_L, img_H)
        img_L, img_H = self.augment(img_L, img_H)

        img_L, img_H = np.float32(np.ascontiguousarray(img_L)), np.float32(np.ascontiguousarray(img_H))
        return {'L': img_L, 'H': img_H}

    def _get_img_paths(self, path):
        self.img_paths = []
        L_pattern = os.path.join(path, 'RN/*.mat')
        L_paths = sorted(glob.glob(L_pattern))
        for L_path in L_paths:
            self.img_paths.append({'L': L_path, 'H': L_path.replace('RN', 'CL')})
        self.img_paths = self.img_paths[int(0 / 40 * len(self.img_paths)):int(30 / 40 * len(self.img_paths))]#mcj
        # self.img_paths = self.img_paths[:]  # mcj
    def _open_images(self):
        self.imgs = []
        for img_path in self.img_paths:
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])
            self.imgs.append({'L': img_L, 'H': img_H})

    def _open_image(self, path):
        # img = Image.open(path)
        # img = np.asarray(img)
        # img = np.transpose(img, (2, 0, 1))
        img = np.array(scipy.io.loadmat(path)['data'])
        return img

class SeisMatTrainDataset_noisy(BaseTrainDataset):
    def __init__(self, path, patch_size, pin_memory):
        super(SeisMatTrainDataset_noisy, self).__init__(path, patch_size, pin_memory)

    def __getitem__(self, index):
        index = index % len(self.img_paths)

        if self.pin_memory:
            img_L = self.imgs[index]['L']
        else:
            img_path = self.img_paths[index]
            img_L = self._open_image(img_path['L'])

        img_L= self.crop(img_L)
        img_L= self.augment(img_L)

        img_L = np.float32(np.ascontiguousarray(img_L))
        return {'L': img_L}

    def _get_img_paths(self, path):
        self.img_paths = []
        L_pattern = os.path.join(path, 'RN/*.mat')
        L_paths = sorted(glob.glob(L_pattern))
        for L_path in L_paths:
            self.img_paths.append({'L': L_path})
        self.img_paths = self.img_paths[:int(30 / 40 * len(self.img_paths))]#mcj
        # self.img_paths = self.img_paths[int(39 / 40 * len(self.img_paths)):int(40 / 40 * len(self.img_paths))]  # mcj
        # self.img_paths = self.img_paths[:]  # mcj

    def _open_images(self):
        self.imgs = []
        for img_path in self.img_paths:
            img_L = self._open_image(img_path['L'])
            self.imgs.append({'L': img_L})

    def _open_image(self, path):
        img = np.array(scipy.io.loadmat(path)['data'])
        return img

class SeisMatValidationDataset(BaseTrainDataset):
    # def __init__(self):
    #     super(SeisMatValidationDataset, self).__init__(mat_path)
    def __init__(self, path, patch_size, pin_memory):
        super(SeisMatValidationDataset, self).__init__(path, patch_size, pin_memory)
        self._get_img_paths(path)
        self.n_data = len(self.img_paths)

    def __getitem__(self, index):
        index = index % len(self.img_paths)

        if self.pin_memory:
            img_L = self.imgs[index]['L']
            img_H = self.imgs[index]['H']
        else:
            img_path = self.img_paths[index]
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])

        # minner = img_L.min()
        # img_L = img_L - minner
        # img_H = img_H - minner
        # maxer = img_L.max()
        # img_L = img_L / maxer
        # img_H = img_H / maxer

        img_L, img_H = self.crop(img_L, img_H)
        # img_L, img_H = self.augment(img_L, img_H)

        img_L, img_H = np.float32(np.ascontiguousarray(img_L)), np.float32(np.ascontiguousarray(img_H))
        return {'L': img_L, 'H': img_H}


    def _get_img_paths(self, path):
        self.img_paths = []
        L_pattern = os.path.join(path, 'RN/*.mat')
        L_paths = sorted(glob.glob(L_pattern))
        for L_path in L_paths:
            self.img_paths.append({'L': L_path, 'H': L_path.replace('RN', 'CL')})
        # self.img_paths = self.img_paths[int(39 / 40 * len(self.img_paths)):]#mcj
        # self.img_paths = self.img_paths[int(0 / 40 * len(self.img_paths)):]  # mcj
        self.img_paths = self.img_paths[int(30 / 40 * len(self.img_paths)):]  # mcj
        # self.img_paths = self.img_paths[:]

    def _open_images(self):
        self.imgs = []
        for img_path in self.img_paths:
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])
            self.imgs.append({'L': img_L, 'H': img_H})

    def _open_image(self, path):
        # img = Image.open(path)
        # img = np.asarray(img)
        # img = np.transpose(img, (2, 0, 1))
        img = np.array(scipy.io.loadmat(path)['data'])
        return img

    def __len__(self):
        return self.n_data

class SeisMatValidationDataset1(BaseTrainDataset):
    # def __init__(self):
    #     super(SeisMatValidationDataset, self).__init__(mat_path)
    def __init__(self, path, patch_size, pin_memory):
        super(SeisMatValidationDataset1, self).__init__(path, patch_size, pin_memory)
        self._get_img_paths(path)
        self.n_data = len(self.img_paths)

    def __getitem__(self, index):
        index = index % len(self.img_paths)

        if self.pin_memory:
            img_L = self.imgs[index]['L']
            img_H = self.imgs[index]['H']
        else:
            img_path = self.img_paths[index]
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])


        img_L, img_H = self.crop(img_L, img_H)
        # img_L, img_H = self.augment(img_L, img_H)

        img_L, img_H = np.float32(np.ascontiguousarray(img_L)), np.float32(np.ascontiguousarray(img_H))
        return {'L': img_L, 'H': img_H}


    def _get_img_paths(self, path):
        self.img_paths = []
        L_pattern = os.path.join(path, 'RN/*.mat')
        L_paths = sorted(glob.glob(L_pattern))
        for L_path in L_paths:
            self.img_paths.append({'L': L_path, 'H': L_path.replace('RN', 'CL')})
        self.img_paths = self.img_paths[int(39 / 40 * len(self.img_paths)):]#mcj
        # self.img_paths = self.img_paths[int(0 / 40 * len(self.img_paths)):]  # mcj
        # self.img_paths = self.img_paths[int(30 / 40 * len(self.img_paths)):]  # mcj

    def _open_images(self):
        self.imgs = []
        for img_path in self.img_paths:
            img_L = self._open_image(img_path['L'])
            img_H = self._open_image(img_path['H'])
            self.imgs.append({'L': img_L, 'H': img_H})

    def _open_image(self, path):
        # img = Image.open(path)
        # img = np.asarray(img)
        # img = np.transpose(img, (2, 0, 1))
        img = np.array(scipy.io.loadmat(path)['data'])
        return img

    def __len__(self):
        return self.n_data


# class SIDDSrgbValidationDataset(Dataset):
#     def __init__(self):
#         super(SIDDSrgbValidationDataset, self).__init__()
#         self._open_images(sidd_path)
#         self.n = self.noisy_block.shape[0]
#         self.k = self.noisy_block.shape[1]
#
#     def __getitem__(self, index):
#         index_n = index // self.k
#         index_k = index % self.k
#
#         img_H = self.gt_block[index_n, index_k]
#         img_H = np.float32(img_H)
#         img_H = np.transpose(img_H, (2, 0, 1))
#
#         img_L = self.noisy_block[index_n, index_k]
#         img_L = np.float32(img_L)
#         img_L = np.transpose(img_L, (2, 0, 1))
#
#         return {'H':img_H, 'L':img_L}
#
#     def __len__(self):
#         return self.n * self.k
#
#     def _open_images(self, path):
#         mat = sio.loadmat(os.path.join(path, 'SIDD Validation Data and Ground Truth/ValidationNoisyBlocksSrgb.mat'))
#         self.noisy_block = mat['ValidationNoisyBlocksSrgb']
#         mat = sio.loadmat(os.path.join(path, 'SIDD Validation Data and Ground Truth/ValidationGtBlocksSrgb.mat'))
#         self.gt_block = mat['ValidationGtBlocksSrgb']
