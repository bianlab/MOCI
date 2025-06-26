import hdf5storage
import torch
import argparse
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from getdataset import TestDataset_MOS
from my_utils import initialize_logger
import torch.utils.data
from architecture import model_generator
import numpy as np
import h5py
import random
import matplotlib.pyplot as plt
import spectral
import cv2


parser = argparse.ArgumentParser(description="Reconstruct hypersepctral images from measurements")
parser.add_argument("--method", type=str, default='psrnet', help='Model')
parser.add_argument("--gpu_id", type=str, default='2', help='path log files')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--mask_path", type=str, default='./MASK/mask_0108_pols_all.mat', help='path log files')
parser.add_argument("--start_dir", type=int, default=(0, 0), help="size of test image coordinate")
parser.add_argument("--image_size", type=int, default=(1024, 1224), help="size of test image")


parser.add_argument("--pretrained_model_path", type=str, default='./Model_zoo/psrnet_MOCI.pth', help='path log files')


parser.add_argument("--image_folder", type=str, default= './Test/Measurement/test_1/', help='path log files')
parser.add_argument("--save_folder", type=str, default= './Test/Measurement/test_1_HSIPOLS_0531/', help='path log files')


opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def HSI2RGB_function(bands, hsi):
    CIE1931 = np.array([[380, 0.0272, -0.0115, 0.9843],
    [385, 0.0268, -0.0114, 0.9846],
    [390, 0.0263, -0.0114, 0.9851],
    [395, 0.0256, -0.0113, 0.9857],
    [400, 0.0247, -0.0112, 0.9865],
    [405, 0.0237, -0.0111, 0.9874],
    [410, 0.0225, -0.0109, 0.9884],
    [415, 0.0207, -0.0104, 0.9897],
    [420, 0.0181, -0.0094, 0.9913],
    [425, 0.0142, -0.0076, 0.9934],
    [430, 0.0088, -0.0048, 0.9960],
    [435, 0.0012, -0.0007, 0.9995],
    [440, -0.0084, 0.0018, 1.0036],
    [445, -0.0213, 0.0120, 1.0093],
    [450, -0.0390, 0.0218, 1.0172],
    [455, -0.0618, 0.0345, 1.0273],
    [460, -0.0909, 0.0517, 1.0392],
    [465, -0.1281, 0.0762, 1.0519],
    [470, -0.1821, 0.1175, 1.0646],
    [475, -0.2584, 0.1840, 1.0744],
    [480, -0.3667, 0.2906, 1.0761],
    [485, -0.5200, 0.4568, 1.0632],
    [490, -0.7150, 0.6996, 1.0154],
    [495, -0.9459, 1.0247, 0.9212],
    [500, -1.1685, 1.3905, 0.7780],
    [505, -1.3182, 1.7195, 0.5987],
    [510, -1.3371, 1.9318, 0.4053],
    [515, -1.2076, 1.9699, 0.2377],
    [520, -0.9830, 1.8534, 0.1296],
    [525, -0.7386, 1.6662, 0.0724],
    [530, -0.5159, 1.4761, 0.0398],
    [535, -0.3304, 1.3105, 0.0199],
    [540, -0.1707, 1.1628, 0.0079],
    [545, -0.0293, 1.0282, 0.0011],
    [550, 0.0974, 0.9051, -0.0025],
    [555, 0.2121, 0.7919, -0.0040],
    [560, 0.3164, 0.6881, -0.0045],
    [565, 0.4112, 0.5932, -0.0044],
    [570, 0.4973, 0.5067, -0.0040],
    [575, 0.5751, 0.4283, -0.0034],
    [580, 0.6449, 0.3579, -0.0028],
    [585, 0.7071, 0.2952, -0.0023],
    [590, 0.7617, 0.2402, -0.0019],
    [595, 0.8087, 0.1928, -0.0015],
    [600, 0.8475, 0.1537, -0.0012],
    [605, 0.8800, 0.1209, -0.0009],
    [610, 0.9059, 0.0949, -0.0008],
    [615, 0.9265, 0.0741, -0.0006],
    [620, 0.9425, 0.0580, -0.0005],
    [625, 0.9550, 0.0454, -0.0004],
    [630, 0.9649, 0.0354, -0.0003],
    [635, 0.9730, 0.0272, -0.0002],
    [640, 0.9797, 0.0205, -0.0002],
    [645, 0.9850, 0.0152, -0.0002],
    [650, 0.9888, 0.0113, -0.0001],
    [655, 0.9918, 0.0083, -0.0001],
    [660, 0.9940, 0.0061, -0.0001],
    [665, 0.9954, 0.0047, -0.0001],
    [670, 0.9966, 0.0035, -0.0001],
    [675, 0.9975, 0.0025, 0.0000],
    [680, 0.9984, 0.0016, 0.0000],
    [685, 0.9991, 0.0009, 0.0000],
    [690, 0.9996, 0.0004, 0.0000],
    [695, 0.9999, 0.0001, 0.0000],
    [700, 1.0000, 0.0000, 0.0000],
    [705, 1.0000, 0.0000, 0.0000],
    [710, 1.0000, 0.0000, 0.0000],
    [715, 1.0000, 0.0000, 0.0000],
    [720, 1.0000, 0.0000, 0.0000],
    [725, 1.0000, 0.0000, 0.0000],
    [730, 1.0000, 0.0000, 0.0000],
    [735, 1.0000, 0.0000, 0.0000],
    [740, 1.0000, 0.0000, 0.0000],
    [745, 1.0000, 0.0000, 0.0000],
    [750, 1.0000, 0.0000, 0.0000],
    [755, 1.0000, 0.0000, 0.0000],
    [760, 1.0000, 0.0000, 0.0000],
    [765, 1.0000, 0.0000, 0.0000],
    [770, 1.0000, 0.0000, 0.0000],
    [775, 1.0000, 0.0000, 0.0000],
    [780, 1.0000, 0.0000, 0.0000]])
    select_index = []
    for i in range(len(bands)):
        index = np.where(CIE1931[:, 0]==bands[i])[0]
        select_index.append(index[0])
    select_index = np.array(select_index)
    select_cie = CIE1931[select_index, 1:]
    hsi_rgb = hsi[:len(bands), :, :]

    rgb = hsi_rgb.transpose(1, 2, 0) @ select_cie
    rgb = rgb / rgb.max()
    rgb = np.maximum(rgb, 0)
    return rgb

def white_balance_loops(rgb_R):
    rgb_R = rgb_R / rgb_R.max() * 255
    avg_color_per_channel = np.mean(rgb_R, axis=(0, 1))
    scaling_factors = avg_color_per_channel.max() / avg_color_per_channel
    rgb_R = (rgb_R * scaling_factors).clip(0, 255).astype(np.uint8)
    return rgb_R

def Get_AOP_DOP(pols):
    pols = pols / pols.max()
    pol_0 = pols[0, :, :]
    pol_45 = pols[1, :, :]
    pol_90 = pols[2, :, :]
    pol_135 = pols[3, :, :]

    S0 = pol_0 + pol_90
    S1 = pol_0 - pol_90
    S2 = pol_45 - pol_135
    S3 = 0

    aop = np.arctan((S2 + 1e-5) / (S1 + 1e-5)) + 0.5*np.pi

    dop = (np.sqrt((S1 ** 2 + S2 ** 2 + S3 ** 2)) + 1e-5) / (S0 + 1e-5)

    return dop, aop




def main():
    cudnn.benchmark = True

    mask_init = hdf5storage.loadmat(opt.mask_path)['mask']
    print('mask_init:', mask_init.shape)

    mask = mask_init[:, :, opt.start_dir[0]:opt.start_dir[0]+opt.image_size[0], opt.start_dir[1]:opt.start_dir[1] + opt.image_size[1]]
    mask = np.maximum(mask, 0)
    mask = mask / mask.max()
    mask = torch.from_numpy(mask)
    mask = mask.cuda()
    print('mask:', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())





    save_image_path = opt.save_folder + 'save_images/'
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    select_bands = np.arange(400, 1010, 10)
    pols_names = ['_0度_', '_45度_', '_90度_', '_135度_']

    css = hdf5storage.loadmat('/ssd/wzh/home_ssd/1_SpecPol/1_SpecPolCode/MASK/CA050.mat')['css']
    css = np.expand_dims(np.expand_dims(np.expand_dims(css, 0), 2), 3)
    print('css:', css.dtype, css.shape, css.max(), css.mean(), css.min())






    model = model_generator(opt.method, opt.pretrained_model_path)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    if torch.cuda.is_available():
        model.cuda()
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)

    test_list = os.listdir(opt.image_folder)
    test_list.sort()

    model.eval()


    for i in range(len(test_list)):

        mos_name = test_list[i]

        bmp = cv2.imread(opt.image_folder + mos_name)[:, :, 0]
  
        bmp = np.expand_dims(bmp, 0)
        bmp = bmp / bmp.max()
        bmp = bmp.astype(np.float32)
        _, h, w = bmp.shape
        MOS = np.concatenate((bmp[:, 0:h:2, 0:w:2],  # 0
        bmp[:, 0:h:2, 1:w:2],  # 45
        bmp[:, 1:h:2, 1:w:2],  # 90
        bmp[:, 1:h:2, 0:w:2]), axis=0)

        MOS = torch.from_numpy(MOS)
        MOS = MOS.unsqueeze(1)
        MOS = MOS[:, :, opt.start_dir[0]:opt.start_dir[0]+opt.image_size[0], opt.start_dir[1]:opt.start_dir[1] + opt.image_size[1]]
        MOS = MOS / MOS.max()
        MOS = MOS.unsqueeze(0)

        MOS_spec = MOS
        MOS_spec = MOS_spec.cuda()

        mask_patch = mask
        mask_patch = mask_patch / mask_patch.max()
        mask_patch = mask_patch.unsqueeze(0)

        with torch.no_grad():

            outputs_spec = model(MOS_spec, mask_patch, 'pols')

            outputs_spec = outputs_spec / outputs_spec.max()

            output_hsi = torch.maximum(outputs_spec, torch.tensor(0))
            output_hsi = output_hsi.squeeze()
            output_hsi = output_hsi.cpu().numpy()
            
            MOS = MOS.squeeze()
            input_mos = MOS.cpu().numpy()

        data_name = mos_name

        f = h5py.File(opt.save_folder + 'HSI_R_' + data_name[:-4] + '_pol_all.h5', 'w')
        f['mos'] = input_mos
        f['hsi_R'] = output_hsi
        f.close()    


if __name__ == '__main__':
    main()


