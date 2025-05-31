from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch
import random
import os
import hdf5storage
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate
import torch.nn.functional as F
import h5py


class TrainDataset(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size

        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_list = data_list
        self.img_num = len(self.data_list)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):


        f = h5py.File(self.data_path + self.data_list[idx], 'r')
        hsi = f['hsi'][:]
        f.close()

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        random_h = random.randint(0,hsi.shape[1] - patch_size_h -1)
        random_w = random.randint(0,hsi.shape[2] - patch_size_w -1)
        output_hsi = hsi[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_paths = []
        self.patch_size = patch_size

        data_list = os.listdir(data_path)
        data_list.sort()
        for i in range(len(data_list)):

            self.data_paths.append(data_path + data_list[i])

        self.img_num = len(self.data_paths)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):

        f = h5py.File(self.data_paths[idx], 'r')
        hsi = f['hsi'][:]
        f.close()

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

       
        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        random_h = random.randint(0, hsi.shape[1] - patch_size_h -1)
        random_w = random.randint(0, hsi.shape[2] - patch_size_w -1)
        output_hsi = hsi[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num




class TrainDataset_HSIPOL(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size

        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_list = data_list
        self.img_num = len(self.data_list)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(2, 3))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, :, ::-1, :].copy()
        return img

    def __getitem__(self, idx):

        f = h5py.File(self.data_path + self.data_list[idx], 'r')
        hsi = f['hsi'][:]
        f.close()
        # print('hsi:', hsi.dtype, hsi.shape, hsi.max(), hsi.mean(), hsi.min())

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)
        # print('hsi:', hsi.dtype, hsi.shape, hsi.max(), hsi.mean(), hsi.min())

        random_h = random.randint(0, hsi.shape[2] - patch_size_h -1)
        random_w = random.randint(0, hsi.shape[3] - patch_size_w -1)
        output_hsi = hsi[:, :, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num

class ValidDataset_HSIPOL(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size

        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_list = data_list
        self.img_num = len(self.data_list)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(2, 3))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, :, ::-1, :].copy()
        return img

    def __getitem__(self, idx):

        f = h5py.File(self.data_path + self.data_list[idx], 'r')
        hsi = f['hsi'][:]
        f.close()
        # print('hsi:', hsi.dtype, hsi.shape, hsi.max(), hsi.mean(), hsi.min())

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)
        # print('hsi:', hsi.dtype, hsi.shape, hsi.max(), hsi.mean(), hsi.min())

        random_h = random.randint(0, hsi.shape[2] - patch_size_h -1)
        random_w = random.randint(0, hsi.shape[3] - patch_size_w -1)
        output_hsi = hsi[:, :, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num







class TrainDataset_change_rate(Dataset):
    def __init__(self, data_path, patch_size, change_rate, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size

        self.change_rate = change_rate

        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_list = data_list
        self.img_num = len(self.data_list)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):


        f = h5py.File(self.data_path + self.data_list[idx], 'r')
        hsi = f['hsi'][:]
        f.close()

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        random_h = random.randint(0,hsi.shape[1] - patch_size_h -1)
        random_w = random.randint(0,hsi.shape[2] - patch_size_w -1)
        output_hsi = hsi[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        # print('output_hsi', output_hsi.dtype, output_hsi.shape, output_hsi.max(), output_hsi.mean(), output_hsi.min()) 
        output_hsi = output_hsi * self.change_rate
        output_hsi = output_hsi.astype(np.float32)
        
        # print('output_hsi', output_hsi.dtype, output_hsi.shape, output_hsi.max(), output_hsi.mean(), output_hsi.min()) 

        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num

class ValidDataset_change_rate(Dataset):
    def __init__(self, data_path, patch_size, change_rate, arg=False):

        self.arg = arg
        self.data_paths = []
        self.patch_size = patch_size
        self.change_rate = change_rate

        data_list = os.listdir(data_path)
        data_list.sort()
        for i in range(len(data_list)):

            self.data_paths.append(data_path + data_list[i])

        self.img_num = len(self.data_paths)

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):

        f = h5py.File(self.data_paths[idx], 'r')
        hsi = f['hsi'][:]
        f.close()

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

       
        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            hsi = self.arguement(hsi, rotTimes, vFlip, hFlip)

        random_h = random.randint(0, hsi.shape[1] - patch_size_h -1)
        random_w = random.randint(0, hsi.shape[2] - patch_size_w -1)
        output_hsi = hsi[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]

        # print('output_hsi', output_hsi.dtype, output_hsi.shape, output_hsi.max(), output_hsi.mean(), output_hsi.min()) 
        output_hsi = output_hsi * self.change_rate

        output_hsi = output_hsi.astype(np.float32)
        output_hsi = output_hsi / output_hsi.max()

        return np.ascontiguousarray(output_hsi)

    def __len__(self):
        return self.img_num








class TrainDatasetPolsImages(Dataset):
    def __init__(self, data_path, patch_size, light_path, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size

        self.light = hdf5storage.loadmat(light_path)['spectra']
        print('self.light:', self.light.dtype, self.light.shape, self.light.max(), self.light.mean(), self.light.min())
        self.light_nums = self.light.shape[1]


        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_paths = []


        for i in range(len(data_list)):

            self.data_paths.append(data_path + data_list[i])

        self.img_num = len(self.data_paths)










        # for i in range(len(data_list)):

        #     bmp = cv2.imread(self.data_path + self.data_list[i])[:, :, 0]

        #     bmp = np.expand_dims(bmp, 0)

        #     _, h, w = bmp.shape
        #     bmp_pol = np.concatenate((bmp[:, 0:h:2, 0:w:2],  # 0
        #     bmp[:, 0:h:2, 1:w:2],  # 45
        #     bmp[:, 1:h:2, 1:w:2],  # 90
        #     bmp[:, 1:h:2, 0:w:2]), axis=0)


        #     # bmp_pol = bmp_pol[:, self.start_dir[0]:self.start_dir[0]+self.image_size[0], self.start_dir[1]:self.start_dir[1] + self.image_size[1]]
        #     mos = bmp_pol / bmp_pol.max()
        #     mos = mos.astype(np.float32)
        #     self.Pols_list.append(mos)
        #     # print('bmp:', bmp.dtype, bmp.shape, bmp.max(), bmp.mean(), bmp.min())
        #     # print('bmp_pol:', bmp_pol.dtype, bmp_pol.shape, bmp_pol.max(), bmp_pol.mean(), bmp_pol.min())
        #     # print('mos:', mos.dtype, mos.shape, mos.max(), mos.mean(), mos.min())
        #     # exit()
        # self.img_num = len(self.data_list)


    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):



        bmp = cv2.imread(self.data_paths[idx])[:, :, 0]

        bmp = np.expand_dims(bmp, 0)

        _, h, w = bmp.shape
        pols = np.concatenate((bmp[:, 0:h:2, 0:w:2],  # 0
        bmp[:, 0:h:2, 1:w:2],  # 45
        bmp[:, 1:h:2, 1:w:2],  # 90
        bmp[:, 1:h:2, 0:w:2]), axis=0)

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            pols = self.arguement(pols, rotTimes, vFlip, hFlip)

        random_h = random.randint(0, pols.shape[1] - patch_size_h -1)
        random_w = random.randint(0, pols.shape[2] - patch_size_w -1)
        output_pols = pols[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_pols = output_pols.astype(np.float32)
        output_pols = output_pols / output_pols.max()


        random_light = self.light[:, random.randint(0, self.light_nums -1)]
        random_light = random_light / random_light.max()



        # print('output_pols:', output_pols.dtype, output_pols.shape, output_pols.max(), output_pols.mean(), output_pols.min())
        

        return output_pols, random_light

    def __len__(self):
        return self.img_num






class TrainDatasetPolsImages_OneChannel(Dataset):
    def __init__(self, data_path, patch_size, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size



        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_list = data_list
        
        self.Pols_list = []

        for i in range(len(data_list)):

            bmp = cv2.imread(self.data_path + self.data_list[i])[:, :, 0]

            bmp = np.expand_dims(bmp, 0)

            # _, h, w = bmp.shape
            # bmp_pol = np.concatenate((bmp[:, 0:h:2, 0:w:2],  # 0
            # bmp[:, 0:h:2, 1:w:2],  # 45
            # bmp[:, 1:h:2, 1:w:2],  # 90
            # bmp[:, 1:h:2, 0:w:2]), axis=0)


            # bmp_pol = bmp_pol[:, self.start_dir[0]:self.start_dir[0]+self.image_size[0], self.start_dir[1]:self.start_dir[1] + self.image_size[1]]
            mos = bmp / bmp.max()
            mos = mos.astype(np.float32)
            self.Pols_list.append(mos)
            # print('bmp:', bmp.dtype, bmp.shape, bmp.max(), bmp.mean(), bmp.min())
            # print('bmp_pol:', bmp_pol.dtype, bmp_pol.shape, bmp_pol.max(), bmp_pol.mean(), bmp_pol.min())
            # print('mos:', mos.dtype, mos.shape, mos.max(), mos.mean(), mos.min())
            # exit()
        self.img_num = len(self.data_list)


    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):


        pols = self.Pols_list[idx]
        # print('pols:', pols.dtype, pols.shape, pols.max(), pols.mean(), pols.min())

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            pols = self.arguement(pols, rotTimes, vFlip, hFlip)

        random_h = random.randint(0, pols.shape[1] - patch_size_h -1)
        random_w = random.randint(0, pols.shape[2] - patch_size_w -1)
        output_pols = pols[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_pols = output_pols.astype(np.float32)
        output_pols = output_pols / output_pols.max()

        # print('output_pols:', output_pols.dtype, output_pols.shape, output_pols.max(), output_pols.mean(), output_pols.min())
        

        return np.ascontiguousarray(output_pols)

    def __len__(self):
        return self.img_num



class TrainDatasetPolsImages_1024(Dataset):
    def __init__(self, data_path, patch_size, light_path, arg=False):

        self.arg = arg
        self.data_path = data_path
        self.patch_size = patch_size

        self.light = hdf5storage.loadmat(light_path)['spectra']
        print('self.light:', self.light.dtype, self.light.shape, self.light.max(), self.light.mean(), self.light.min())
        self.light_nums = self.light.shape[1]



        data_list = os.listdir(data_path)
        data_list.sort()

        self.data_list = data_list
        
        self.Pols_list = []

        for i in range(len(data_list)):

            bmp = cv2.imread(self.data_path + self.data_list[i])[:, :, 0]

            bmp = np.expand_dims(bmp, 0)

            _, h, w = bmp.shape
            bmp_pol = np.concatenate((bmp[:, 0:h:2, 0:w:2],  # 0
            bmp[:, 0:h:2, 1:w:2],  # 45
            bmp[:, 1:h:2, 1:w:2],  # 90
            bmp[:, 1:h:2, 0:w:2]), axis=0)

            # bmp_pol = bmp_pol[:, self.start_dir[0]:self.start_dir[0]+self.image_size[0], self.start_dir[1]:self.start_dir[1] + self.image_size[1]]
            mos = bmp_pol / bmp_pol.max()
            mos = mos.astype(np.float32)
            self.Pols_list.append(mos)
            # print('bmp:', bmp.dtype, bmp.shape, bmp.max(), bmp.mean(), bmp.min())
            # print('bmp_pol:', bmp_pol.dtype, bmp_pol.shape, bmp_pol.max(), bmp_pol.mean(), bmp_pol.min())
            # print('mos:', mos.dtype, mos.shape, mos.max(), mos.mean(), mos.min())
            # exit()
        self.img_num = len(self.data_list)


    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):


        pols = self.Pols_list[idx]
        # print('pols:', pols.dtype, pols.shape, pols.max(), pols.mean(), pols.min())

        patch_size_h = self.patch_size[0]
        patch_size_w = self.patch_size[1]

        if self.arg:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            pols = self.arguement(pols, rotTimes, vFlip, hFlip)

        if pols.shape[1] == 1024:
            random_h = 0
        else:
            random_h = random.randint(0, pols.shape[1] - patch_size_h -1)

        if pols.shape[2] == 1024:
            random_w = 0
        else:
            random_w = random.randint(0, pols.shape[2] - patch_size_w -1)



        # random_h = random.randint(0, pols.shape[1] - patch_size_h -1)
        # random_w = random.randint(0, pols.shape[2] - patch_size_w -1)
        output_pols = pols[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
        output_pols = output_pols.astype(np.float32)
        output_pols = output_pols / output_pols.max()

        random_light = self.light[:, random.randint(0, self.light_nums -1)]
        random_light = random_light / random_light.max()

        # print('random_light:', random_light.dtype, random_light.shape, random_light.max(), random_light.mean(), random_light.min())



        # print('output_pols:', output_pols.dtype, output_pols.shape, output_pols.max(), output_pols.mean(), output_pols.min())
        

        return np.ascontiguousarray(output_pols), random_light

    def __len__(self):
        return self.img_num




class TestDataset_POL(Dataset):
    def __init__(self, data_path, data_list, start_dir, image_size, arg=False):

        self.arg = arg
        self.data_path = data_path

        self.start_dir = start_dir
        self.image_size = image_size

        self.data_list = data_list

        self.MOS_list = []

        for i in range(len(data_list)):

            bmp = cv2.imread(self.data_path + self.data_list[i])[:, :, 0]
            bmp = bmp[self.start_dir[0]:self.start_dir[0]+self.image_size[0], self.start_dir[1]:self.start_dir[1] + self.image_size[1]]
            bmp = bmp / bmp.max()
            bmp = bmp.astype(np.float32)
            mos = np.expand_dims(bmp, axis=0)
            self.MOS_list.append(mos)
            
        self.img_num = len(self.data_list)

    def __getitem__(self, idx):
        mos_name = self.data_list[idx]
        mos = self.MOS_list[idx]

        return np.ascontiguousarray(mos), mos_name

    def __len__(self):
        return self.img_num




class TestDataset_MOS(Dataset):
    def __init__(self, data_path, data_list, start_dir, image_size, arg=False):

        self.arg = arg
        self.data_path = data_path

        self.start_dir = start_dir
        self.image_size = image_size

        self.data_list = data_list

        self.MOS_list = []

        for i in range(len(data_list)):

            bmp = cv2.imread(self.data_path + self.data_list[i])[:, :, 0]

            bmp = np.expand_dims(bmp, 0)

            _, h, w = bmp.shape
            bmp_pol = np.concatenate((bmp[:, 0:h:2, 0:w:2],  # 0
            bmp[:, 0:h:2, 1:w:2],  # 45
            bmp[:, 1:h:2, 1:w:2],  # 90
            bmp[:, 1:h:2, 0:w:2]), axis=0)


            bmp_pol = bmp_pol[:, self.start_dir[0]:self.start_dir[0]+self.image_size[0], self.start_dir[1]:self.start_dir[1] + self.image_size[1]]
            mos = bmp_pol / bmp_pol.max()
            mos = mos.astype(np.float32)
            self.MOS_list.append(mos)
            print('bmp:', bmp.dtype, bmp.shape, bmp.max(), bmp.mean(), bmp.min())
            print('bmp_pol:', bmp_pol.dtype, bmp_pol.shape, bmp_pol.max(), bmp_pol.mean(), bmp_pol.min())
            print('mos:', mos.dtype, mos.shape, mos.max(), mos.mean(), mos.min())
            # exit()
        self.img_num = len(self.data_list)

    def __getitem__(self, idx):
        mos_name = self.data_list[idx]
        mos = self.MOS_list[idx]

        return np.ascontiguousarray(mos), mos_name

    def __len__(self):
        return self.img_num






class TestDataset_PolsImages(Dataset):
    def __init__(self, data_path, data_list, start_dir, image_size, arg=False):

        self.arg = arg
        self.data_path = data_path

        self.start_dir = start_dir
        self.image_size = image_size

        self.data_list = data_list

        self.MOS_list = []

        for i in range(len(data_list)):

            bmp = cv2.imread(self.data_path + self.data_list[i])[:, :, 0]

            bmp = np.expand_dims(bmp, 0)

            _, h, w = bmp.shape
            bmp_pol = np.concatenate((bmp[:, 0:h:2, 0:w:2],  # 0
            bmp[:, 0:h:2, 1:w:2],  # 45
            bmp[:, 1:h:2, 1:w:2],  # 90
            bmp[:, 1:h:2, 0:w:2]), axis=0)


            bmp_pol = bmp_pol[:, self.start_dir[0]:self.start_dir[0]+self.image_size[0], self.start_dir[1]:self.start_dir[1] + self.image_size[1]]
            mos = bmp_pol / bmp_pol.max()
            mos = mos.astype(np.float32)
            self.MOS_list.append(mos)
            print('bmp:', bmp.dtype, bmp.shape, bmp.max(), bmp.mean(), bmp.min())
            print('bmp_pol:', bmp_pol.dtype, bmp_pol.shape, bmp_pol.max(), bmp_pol.mean(), bmp_pol.min())
            print('mos:', mos.dtype, mos.shape, mos.max(), mos.mean(), mos.min())
            # exit()
        self.img_num = len(self.data_list)

    def __getitem__(self, idx):
        mos_name = self.data_list[idx]
        mos = self.MOS_list[idx]

        return np.ascontiguousarray(mos), mos_name

    def __len__(self):
        return self.img_num







#一次性读入所有偏振图像数据
# class TrainDatasetPolsImages(Dataset):
#     def __init__(self, data_path, patch_size, light_path, arg=False):

#         self.arg = arg
#         self.data_path = data_path
#         self.patch_size = patch_size

#         self.light = hdf5storage.loadmat(light_path)['spectra']
#         print('self.light:', self.light.dtype, self.light.shape, self.light.max(), self.light.mean(), self.light.min())
#         self.light_nums = self.light.shape[1]


#         data_list = os.listdir(data_path)
#         data_list.sort()

#         self.data_list = data_list
        
#         self.Pols_list = []

#         for i in range(len(data_list)):

#             bmp = cv2.imread(self.data_path + self.data_list[i])[:, :, 0]

#             bmp = np.expand_dims(bmp, 0)

#             _, h, w = bmp.shape
#             bmp_pol = np.concatenate((bmp[:, 0:h:2, 0:w:2],  # 0
#             bmp[:, 0:h:2, 1:w:2],  # 45
#             bmp[:, 1:h:2, 1:w:2],  # 90
#             bmp[:, 1:h:2, 0:w:2]), axis=0)


#             # bmp_pol = bmp_pol[:, self.start_dir[0]:self.start_dir[0]+self.image_size[0], self.start_dir[1]:self.start_dir[1] + self.image_size[1]]
#             mos = bmp_pol / bmp_pol.max()
#             mos = mos.astype(np.float32)
#             self.Pols_list.append(mos)
#             # print('bmp:', bmp.dtype, bmp.shape, bmp.max(), bmp.mean(), bmp.min())
#             # print('bmp_pol:', bmp_pol.dtype, bmp_pol.shape, bmp_pol.max(), bmp_pol.mean(), bmp_pol.min())
#             # print('mos:', mos.dtype, mos.shape, mos.max(), mos.mean(), mos.min())
#             # exit()
#         self.img_num = len(self.data_list)


#     def arguement(self, img, rotTimes, vFlip, hFlip):
#         # Random rotation
#         for j in range(rotTimes):
#             img = np.rot90(img.copy(), axes=(1, 2))
#         # Random vertical Flip
#         for j in range(vFlip):
#             img = img[:, :, ::-1].copy()
#         # Random horizontal Flip
#         for j in range(hFlip):
#             img = img[:, ::-1, :].copy()
#         return img

#     def __getitem__(self, idx):


#         pols = self.Pols_list[idx]
#         # print('pols:', pols.dtype, pols.shape, pols.max(), pols.mean(), pols.min())

#         patch_size_h = self.patch_size[0]
#         patch_size_w = self.patch_size[1]

#         if self.arg:
#             rotTimes = random.randint(0, 3)
#             vFlip = random.randint(0, 1)
#             hFlip = random.randint(0, 1)
#             pols = self.arguement(pols, rotTimes, vFlip, hFlip)

#         random_h = random.randint(0, pols.shape[1] - patch_size_h -1)
#         random_w = random.randint(0, pols.shape[2] - patch_size_w -1)
#         output_pols = pols[:, random_h:random_h+patch_size_h, random_w:random_w+patch_size_w]
#         output_pols = output_pols.astype(np.float32)
#         output_pols = output_pols / output_pols.max()


#         random_light = self.light[:, random.randint(0, self.light_nums -1)]
#         random_light = random_light / random_light.max()



#         # print('output_pols:', output_pols.dtype, output_pols.shape, output_pols.max(), output_pols.mean(), output_pols.min())
        

#         return output_pols, random_light

#     def __len__(self):
#         return self.img_num

