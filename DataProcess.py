import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

class Data_Process(object):
    def __init__(self):
        self.noise_sigma = 0
        self.hsi_max = []

    def add_noise(self, inputs, sigma):
        noise = torch.zeros_like(inputs)
        noise.normal_(0, sigma)
        noisy = inputs + noise
        noisy = torch.clamp(noisy, 0, 1.0)
        return noisy

    #Randomly extract sub-patches required for training from the original patch
    def get_random_mask_patches(self, mask, image_size, patch_size, batch_size):

        masks = []
        for i in range(batch_size):
            random_h = random.randint(0, image_size[0] - patch_size[0] -1)
            random_w = random.randint(0, image_size[1] - patch_size[1] -1)
            mask_patch = mask[:, random_h:random_h + patch_size[0], random_w:random_w + patch_size[1]]
            mask_patch = mask_patch / mask_patch.max()
            masks.append(mask_patch)
            
        mask_patches = torch.stack(masks, dim=0)
        return mask_patches
           
    #从偏振图像里面选择
    def get_random_mask_patches_pol(self, mask, image_size, patch_size, batch_size):

        masks = []
        for i in range(batch_size):
            random_h = random.randint(0, image_size[0] - patch_size[0] -1)
            random_w = random.randint(0, image_size[1] - patch_size[1] -1)
            # print('mask:', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())
            mask_patch = mask[random.randint(0, 3), :, random_h:random_h + patch_size[0], random_w:random_w + patch_size[1]]
            # mask_patch = mask_patch / mask_patch.max()
            # print('random_h, random_w', random_h, random_w)
            # print('random.randint(0, 3)', random.randint(0, 3))
            # print('mask_patch:', mask_patch.dtype, mask_patch.shape, mask_patch.max(), mask_patch.mean(), mask_patch.min())
            masks.append(mask_patch)
        mask_patches = torch.stack(masks, dim=0)
        return mask_patches       
    
    #从偏振图像里面选择
    def get_random_mask_patches_pol_together(self, mask, image_size, patch_size, batch_size):

        masks = []
        for i in range(batch_size):
            random_h = random.randint(0, image_size[0] - patch_size[0] -1)
            random_w = random.randint(0, image_size[1] - patch_size[1] -1)
            # print('mask:', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())
            mask_patch = mask[:, :, random_h:random_h + patch_size[0], random_w:random_w + patch_size[1]]
            # mask_patch = mask_patch / mask_patch.max()
            # print('random_h, random_w', random_h, random_w)
            # print('random.randint(0, 3)', random.randint(0, 3))
            # print('mask_patch:', mask_patch.dtype, mask_patch.shape, mask_patch.max(), mask_patch.mean(), mask_patch.min())
            masks.append(mask_patch)
        mask_patches = torch.stack(masks, dim=0)
        return mask_patches      

    #Forward model of snapshot hyperspectral imaging for generating input synthesized measurements from hyperspectral targets
    def get_mos_hsi(self, hsi, mask, sigma=0, mos_size=2048, hsi_input_size=512, hsi_target_size=512, init_div_rat=10):
        if not hsi_input_size == hsi_target_size:
            hsi_out = self.extend_spatial_resolution(hsi, extend_rate=hsi_target_size / hsi_input_size)
        else:
            hsi_out=hsi

        if not mos_size == hsi_input_size:
            hsi_expand = self.extend_spatial_resolution(hsi, extend_rate=mos_size / hsi_input_size)
        else:
            hsi_expand=hsi
        mos = torch.sum(hsi_expand * mask, dim=1).unsqueeze(1)
        mos_max = torch.max(mos.view(mos.shape[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)

        #normalize the input and target data using the adaptive variable
        output_hsi = hsi_out / mos_max * init_div_rat
        input_mos = mos / mos_max

        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)

        return input_mos, output_hsi


    #Forward model of snapshot hyperspectral imaging for generating input synthesized measurements from hyperspectral targets
    def get_mos_hsi_SR(self, hsi, mask, sigma=0, mos_size=2048, hsi_input_size=512, hsi_target_size=512, init_div_rat=10):
        if not hsi_input_size == hsi_target_size:
            hsi_out = self.extend_spatial_resolution(hsi, extend_rate=hsi_target_size / hsi_input_size)
        else:
            hsi_out=hsi

        if not mos_size == hsi_input_size:
            hsi_expand = self.extend_spatial_resolution(hsi, extend_rate=mos_size / hsi_input_size)
        else:
            hsi_expand=hsi
        mos = torch.sum(hsi_expand * mask, dim=1).unsqueeze(1)
        mos_max = torch.max(mos.view(mos.shape[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)

        #normalize the input and target data using the adaptive variable
        output_hsi = hsi_out / mos_max * init_div_rat
        input_mos = mos / mos_max

        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)

        return input_mos, output_hsi
    def get_mos_hsi_Norm(self, hsi, mask, sigma=0, mos_size=2048, hsi_input_size=512, hsi_target_size=512, init_div_rat=12):
        if not hsi_input_size == hsi_target_size:
            hsi_out = self.extend_spatial_resolution(hsi, extend_rate=hsi_target_size / hsi_input_size)
        else:
            hsi_out=hsi

        if not mos_size == hsi_input_size:
            hsi_expand = self.extend_spatial_resolution(hsi, extend_rate=mos_size / hsi_input_size)
        else:
            hsi_expand=hsi
        mos = torch.sum(hsi_expand * mask, dim=1).unsqueeze(1)
        mos_max = mos.max()

        # print('mos_max', mos_max.shape, mos_max)


        #normalize the input and target data using the adaptive variable
        output_hsi = hsi_out / mos_max * init_div_rat
        input_mos = mos / mos_max


        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)
        input_mos = input_mos / input_mos.max()

        return input_mos, output_hsi


    def get_mos_hsi_Norm_pols_togethor(self, hsi, mask, sigma=0, mos_size=2048, hsi_input_size=512, hsi_target_size=512, init_div_rat=12):

        
        
        if not hsi_input_size == hsi_target_size:
            
            hsi_out = self.extend_spatial_resolution(hsi, extend_rate=hsi_target_size / hsi_input_size)
        else:
            hsi_out=hsi
        if not mos_size == hsi_input_size:
            hsi_expand = self.extend_spatial_resolution(hsi, extend_rate=mos_size / hsi_input_size)
        else:
            hsi_expand=hsi


        mos = torch.sum(hsi_expand * mask, dim=2).unsqueeze(2)
        mos_max = mos.max()

        # print('mos_max', mos_max.shape, mos_max)


        #normalize the input and target data using the adaptive variable
        output_hsi = hsi_out / mos_max * init_div_rat
        input_mos = mos / mos_max


        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)
        input_mos = input_mos / input_mos.max()
        # print('input_mos', input_mos.shape, input_mos.max())

        return input_mos, output_hsi


    def get_mos_hsi_Norm_pols_togethor_SR(self, hsi, mask, sigma=0, mos_size=2048, hsi_input_size=512, hsi_target_size=512, init_div_rat=12):

        if not hsi_input_size == hsi_target_size:
            hsi_out = self.extend_spatial_resolution_pol(hsi, extend_rate=hsi_target_size / hsi_input_size)
        else:
            hsi_out=hsi
        if not mos_size == hsi_input_size:
            hsi_expand = self.extend_spatial_resolution_pol(hsi, extend_rate=mos_size / hsi_input_size)
        else:
            hsi_expand=hsi

        # print('hsi', hsi.shape, hsi.max())
        # print('hsi_expand', hsi_expand.shape, hsi_expand.max())
        # print('mask', mask.shape, mask.max())


        mos = torch.sum(hsi_expand * mask, dim=2).unsqueeze(2)
        mos_max = mos.max()

        # print('mos_max', mos_max.shape, mos_max)


        #normalize the input and target data using the adaptive variable
        output_hsi = hsi_out / mos_max * init_div_rat
        input_mos = mos / mos_max


        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)
        input_mos = input_mos / input_mos.max()
        # print('input_mos', input_mos.shape, input_mos.max())

        return input_mos, output_hsi
    




    def get_grays_from_HSI(self, hsi, css):

        grays = hsi * css

        grays = torch.sum(grays, 1)
        grays = grays / grays.max()
        grays = grays.unsqueeze(1)
        return grays






    def get_mos_pols_images(self, pols, mask, sigma=0, mos_size=2048, pols_input_size=512, pols_target_size=512, light_css=None, init_div_rat=6):
        if not pols_input_size == pols_target_size:
            pols_out = self.extend_spatial_resolution(pols, extend_rate=pols_target_size / pols_input_size)
        else:
            pols_out=pols

        if not mos_size == pols_input_size:
            pols_expand = self.extend_spatial_resolution(pols, extend_rate=mos_size / pols_input_size)
        else:
            pols_expand=pols

        mask = mask.unsqueeze(0)
        # print('mask:', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())
        # print('pols_expand:', pols_expand.dtype, pols_expand.shape, pols_expand.max(), pols_expand.mean(), pols_expand.min())
        # print('light_css:', light_css.dtype, light_css.shape, light_css.max(), light_css.mean(), light_css.min())


        pols_expand = pols_expand.unsqueeze(2)
        pols_expand = pols_expand.repeat(1, 1, 61, 1, 1)

        # print('pols_expand:', pols_expand.dtype, pols_expand.shape, pols_expand.max(), pols_expand.mean(), pols_expand.min())

        pols_expand = pols_expand * light_css
        # print('pols_expand2:', pols_expand.dtype, pols_expand.shape, pols_expand.max(), pols_expand.mean(), pols_expand.min())




        mos = torch.sum(pols_expand * mask, dim=2)
        # print('mos:', mos.dtype, mos.shape, mos.max(), mos.mean(), mos.min())
 
        mos_max = mos.max()

        mos_max = torch.max(mos.view(mos.shape[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # output_max = torch.max(pols_out.view(pols_out.shape[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)

        # print('mos_max', mos_max.shape, mos_max)
        #normalize the input and target data using the adaptive variable



        # output_pols = pols_out / output_max

        output_pols = pols_out / mos_max * init_div_rat
        input_mos = mos / mos_max


        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)
        input_mos = input_mos / input_mos.max()

        # print('input_mos:', input_mos.dtype, input_mos.shape, input_mos.max(), input_mos.mean(), input_mos.min())
        # print('output_pols:', output_pols.dtype, output_pols.shape, output_pols.max(), output_pols.mean(), output_pols.min())
 
        # exit()

        return input_mos, output_pols





    def get_mos_pols_images_one_channel(self, pols, mask, sigma=0, mos_size=2048, pols_input_size=512, pols_target_size=512, light_css=None, init_div_rat=15):
        if not pols_input_size == pols_target_size:
            pols_out = self.extend_spatial_resolution(pols, extend_rate=pols_target_size / pols_input_size)
        else:
            pols_out=pols

        if not mos_size == pols_input_size:
            pols_expand = self.extend_spatial_resolution(pols, extend_rate=mos_size / pols_input_size)
        else:
            pols_expand=pols



        # print('mask:', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())
        # print('pols_expand:', pols_expand.dtype, pols_expand.shape, pols_expand.max(), pols_expand.mean(), pols_expand.min())
        # print('light_css:', light_css.dtype, light_css.shape, light_css.max(), light_css.mean(), light_css.min())


        # print('pols_expand:', pols_expand.dtype, pols_expand.shape, pols_expand.max(), pols_expand.mean(), pols_expand.min())



        pols_expand = pols_expand.repeat(1, 61, 1, 1)
        # mos = torch.sum(pols_expand * mask, dim=1).unsqueeze(1)
        # mos_max = mos.max()


        pols_expand = pols_expand * light_css
        # print('pols_expand2:', pols_expand.dtype, pols_expand.shape, pols_expand.max(), pols_expand.mean(), pols_expand.min())

        mos = torch.sum(pols_expand * mask, dim=1).unsqueeze(1)
        # print('mos:', mos.dtype, mos.shape, mos.max(), mos.mean(), mos.min())
 
        mos_max = mos.max()

        # print('mos_max', mos_max.shape, mos_max)
        #normalize the input and target data using the adaptive variable
        output_pols = pols_out / mos_max * init_div_rat
        input_mos = mos / mos_max


        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)
        input_mos = input_mos / input_mos.max()

        # print('input_mos:', input_mos.dtype, input_mos.shape, input_mos.max(), input_mos.mean(), input_mos.min())
        # print('output_pols:', output_pols.dtype, output_pols.shape, output_pols.max(), output_pols.mean(), output_pols.min())
 


        # exit()

        return input_mos, output_pols



    def extend_spatial_resolution_pol(self, hsi, extend_rate):
        b, p, c, h, w = hsi.shape
        hsi = hsi.view(b*p, c, h, w)
        hsi_extend = torch.nn.functional.interpolate(hsi, recompute_scale_factor=True, scale_factor=extend_rate)
        h, w = hsi_extend.shape[-2:]
        # h = hsi.shape[-2]
        # w = hsi.shape[-1]
        hsi_extend = hsi_extend.view(b, p, c, h, w)


        return hsi_extend


    def extend_spatial_resolution(self, hsi, extend_rate):
        hsi_extend = torch.nn.functional.interpolate(hsi, recompute_scale_factor=True, scale_factor=extend_rate)
        return hsi_extend

class Image_Cut(object):
    def __init__(self, image_size, patch_size, stride):
        self.patch_size = patch_size
        self.stride = stride 
        self.image_size = image_size

        self.patch_number = []
        self.hsi_max = []

    def image2patch(self, image):
        '''
        image_size = C, H, W
        '''
        patch_size = self.patch_size
        stride = self.stride

        c, h, w = image.shape
        image = image.unsqueeze(0)
        range_h = np.arange(0, h-patch_size[0], stride)
        range_w = np.arange(0, w-patch_size[1], stride)

        range_h = np.append(range_h, h-patch_size[0])
        range_w = np.append(range_w, w-patch_size[1])
        patches = []
        for m in range_h:
            for n in range_w:
                patches.append(image[:, :, m : m + patch_size[0], n : n + patch_size[1]])

        return torch.cat(patches, 0)
    def patch2image(self, patches):

        patch_size = self.patch_size
        stride = self.stride
        c = patches.shape[1]
        h, w = self.image_size

        res = torch.zeros((c, h, w)).to(patches.device)
        weight = torch.zeros((c, h, w)).to(patches.device)

        range_h = np.arange(0, h-patch_size[0], stride)
        range_w = np.arange(0, w-patch_size[1], stride)


        range_h = np.append(range_h, h-patch_size[0])
        range_w = np.append(range_w, w-patch_size[1])

        index = 0

        for m in range_h:
            for n in range_w:
                res[:, m : m + patch_size[0], n : n + patch_size[1]] = res[:, m : m + patch_size[0], n : n + patch_size[1]] + patches[index, ...]

                weight[:, m : m + patch_size[0], n : n + patch_size[1]] = weight[:, m : m + patch_size[0], n : n + patch_size[1]] + 1
                index = index+1

        image = res / weight
        return image








