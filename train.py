import hdf5storage
import torch
import argparse
import os
import time
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from getdataset import TrainDataset, ValidDataset, TrainDataset_HSIPOL, ValidDataset_HSIPOL
from my_utils import AverageMeter, initialize_logger, save_checkpoint, Loss_RMSE, Loss_PSNR, Loss_TV, Loss_MRAE, Loss_SAM, get_boolean_by_ratio, Loss_AOP_DOP
from DataProcess import Data_Process
import torch.utils.data
from architecture import model_generator
import numpy as np
import random
from itertools import cycle



parser = argparse.ArgumentParser(description="Model training of HyperspecI-V1")
parser.add_argument("--method", type=str, default='psrnet', help='Model')

parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument("--end_epoch", type=int, default=200, help="number of epochs")
parser.add_argument("--epoch_sam_num", type=int, default=10000, help="per_epoch_iteration")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--gpu_id", type=str, default='0', help='select gpu')

parser.add_argument("--pretrained_model_path", type=str, default=None, help='pre-trained model path')

parser.add_argument("--sigma", type=float, default=(0, 0.2/255, 0.5/255, 1 / 255, 2/255, 3/255), help="Sigma of Gaussian Noise")

parser.add_argument("--mask_path", type=str, default='./MASK/mask.mat', help='path of calibrated sensing matrix')

parser.add_argument("--output_folder", type=str, default='./exp/Polarization_Spec_PSRNet_Fusion_MASK/mask_0108_pols_all_pols_0421_train256_alternate_freq2/', help='output path')
parser.add_argument("--start_dir", type=int, default=(0, 0), help="size of test image coordinate")
parser.add_argument("--image_size", type=int, default=(1024, 1024), help="size of test image")
parser.add_argument("--train_patch_size", type=int, default=(512, 512), help="size of patch")
parser.add_argument("--valid_patch_size", type=int, default=(512, 512), help="size of patch")
parser.add_argument("--mos_patch_size", type=int, default=(256, 256), help="size of patch")


parser.add_argument("--train_data_path_spec", type=str, default="/ssd/wzh/Dataset_HSI/HSI_Light_ALL/Train_Patch_650/", help='path datasets')
parser.add_argument("--valid_data_path_spec", type=str, default="/ssd/wzh/Dataset_HSI/HSI_Light_ALL/Valid_Patch_650/", help='path datasets')

parser.add_argument("--train_data_path_hsipol", type=str, default="/ssd/wzh/home_ssd/Dataset_HSI/Dataset_HSIPOL/HSIPOL_All/Train_650_650/", help='path datasets')
parser.add_argument("--valid_data_path_hsipol", type=str, default="/ssd/wzh/home_ssd/Dataset_HSI/Dataset_HSIPOL/HSIPOL_All/Valid_650_650/", help='path datasets')



opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id 


criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
criterion_mrae = Loss_MRAE()
criterion_sam = Loss_SAM()
criterion_tv = Loss_TV(TVLoss_weight=float(0.5))
criterion_aop_dop = Loss_AOP_DOP()
data_processing = Data_Process()


mask_init = hdf5storage.loadmat(opt.mask_path)['mask']
print('mask_init:', mask_init.shape)

mask = mask_init[:, :, opt.start_dir[0]:opt.start_dir[0]+opt.image_size[0], opt.start_dir[1]:opt.start_dir[1] + opt.image_size[1]]
mask = np.maximum(mask, 0)
mask = mask / mask.max()
mask = torch.from_numpy(mask)
mask = mask.cuda()
print('mask:', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())


def main():
    cudnn.benchmark = True

    #Load HSI
    print("\nloading spectral dataset ...")
    train_data_spec = TrainDataset(data_path=opt.train_data_path_spec, patch_size=opt.train_patch_size,  arg=True)
    print('len(train_data_spec):', len(train_data_spec))
    print(f"Iteration per epoch: {len(train_data_spec)}")
    val_data_spec = ValidDataset(data_path=opt.valid_data_path_spec, patch_size=opt.valid_patch_size, arg=True)
    print('len(val_data_spec):', len(val_data_spec))


    #Load HSI and polarizaiton data
    print("\nloading spectral dataset ...")
    train_data_hsipol = TrainDataset_HSIPOL(data_path=opt.train_data_path_hsipol, patch_size=opt.train_patch_size,  arg=True)
    print('len(train_data_spec):', len(train_data_spec))
    print(f"Iteration per epoch: {len(train_data_spec)}")
    val_data_hsipol = ValidDataset_HSIPOL(data_path=opt.valid_data_path_hsipol, patch_size=opt.valid_patch_size, arg=True)
    print('len(val_data_spec):', len(val_data_spec))


    output_path = opt.output_folder

    # iterations
    per_epoch_iteration = opt.epoch_sam_num // opt.batch_size
    total_iteration = per_epoch_iteration*opt.end_epoch

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = model_generator(opt.method, opt.pretrained_model_path)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    if torch.cuda.is_available():
        criterion_rmse.cuda()
        criterion_psnr.cuda()
        criterion_tv.cuda()
        criterion_mrae.cuda()
        
    start_epoch = 0
    iteration = start_epoch * per_epoch_iteration

    #opt.init_lr
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.init_lr,
                                 betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration - iteration, eta_min=1e-6)

    log_dir = os.path.join(output_path, 'train.log')
    logger = initialize_logger(log_dir)

    record_rmse_loss = 10000
    strat_time = time.time()
    losses_spec = AverageMeter()


    train_loader_spec = DataLoader(dataset=train_data_spec, batch_size=opt.batch_size, shuffle=True, num_workers=8,
                pin_memory=True, drop_last=True)
    val_loader_spec = DataLoader(dataset=val_data_spec, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    


    train_loader_hsipol = DataLoader(dataset=train_data_hsipol, batch_size=1, shuffle=True, num_workers=8,
                pin_memory=True, drop_last=True)
    val_loader_hsipol = DataLoader(dataset=val_data_hsipol, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    

    train_spec_iter = iter(train_loader_spec)
    train_hsipol_iter = iter(train_loader_hsipol)



    alternate_freq = 10  # Switch the dataset every 10 batches.
    spec_train_steps = 0
    hsipol_train_steps = 0

    while iteration < total_iteration:
        model.train()

        if iteration % (2 * alternate_freq) < alternate_freq:

            try:
                HSIs = next(train_spec_iter)
            except StopIteration:
                train_spec_iter = iter(train_loader_spec)  
                HSIs = next(train_spec_iter)

            
            HSIs = HSIs.cuda()
            mask_patch = data_processing.get_random_mask_patches_pol(mask=mask, image_size=opt.image_size, patch_size=opt.mos_patch_size, batch_size=opt.batch_size)
            #Generate the measurements using traning HSIs and selected sub-pattern

            inputs_HSIs, targets_HSIs = data_processing.get_mos_hsi_SR(hsi=HSIs, mask=mask_patch, sigma=opt.sigma, mos_size=opt.mos_patch_size[0], hsi_input_size=opt.train_patch_size[0], hsi_target_size=opt.train_patch_size[0])


            lr = optimizer.param_groups[0]['lr']
            outputs_spec = model(inputs_HSIs, mask_patch, 'spec')
            loss_rmse_spec = criterion_rmse(outputs_spec, targets_HSIs)
            loss_tv_spec = criterion_tv(outputs_spec, targets_HSIs) 
            loss_mrae_spec = criterion_mrae(outputs_spec, targets_HSIs) * 0.2

            loss_spec = loss_rmse_spec + loss_tv_spec + loss_mrae_spec 
            loss_all = loss_spec
            spec_train_steps += 1


        else:
      
            try:
                HSIPOLs = next(train_hsipol_iter) #Read training data from the dataloader
            except StopIteration:
                train_hsipol_iter = iter(train_loader_hsipol)  
                HSIPOLs = next(train_hsipol_iter)  

            HSIPOLs = HSIPOLs.cuda()

            mask_patch = data_processing.get_random_mask_patches_pol_together(mask=mask, image_size=opt.image_size, patch_size=opt.mos_patch_size, batch_size=1)
            #Generate the measurements using traning HSIs and selected sub-pattern

            inputs_HSIs, targets_HSIs = data_processing.get_mos_hsi_Norm_pols_togethor_SR(hsi=HSIPOLs, mask=mask_patch, sigma=opt.sigma, mos_size=opt.mos_patch_size[0], hsi_input_size=opt.train_patch_size[0], hsi_target_size=opt.train_patch_size[0])

            lr = optimizer.param_groups[0]['lr']

            outputs_hsipol = model(inputs_HSIs, mask_patch, 'pols')

            #Calculate the training loss in spatial-spectral dimensions
            loss_rmse_spec = criterion_rmse(outputs_hsipol, targets_HSIs)
            loss_mrae_spec = criterion_mrae(outputs_hsipol, targets_HSIs) * 0.2
            loss_spec = loss_rmse_spec + loss_mrae_spec 




            b, p, c, h, w = targets_HSIs.shape
            targets_HSIs = targets_HSIs.view(b*p, c, h, w)
            outputs_hsipol = outputs_hsipol.view(b*p, c, h, w)

            #Calculate the training loss in polarization dimension
            loss_dop, loss_aop, loss_S1, loss_S2 = criterion_aop_dop(outputs_hsipol.permute(1, 0, 2, 3), targets_HSIs.permute(1, 0, 2, 3))

            loss_pols = loss_S1 + loss_S2 + loss_dop*0.2 + loss_aop*0.2


            #Hybrid loss function
            loss_all = loss_spec + loss_pols


            hsipol_train_steps += 1


        loss_all.backward()
        optimizer.step() 
        optimizer.zero_grad() 
        scheduler.step() 
        losses_spec.update(loss_spec.data)



        iteration = iteration + 1


        if iteration % per_epoch_iteration == 0:
            epoch = iteration // per_epoch_iteration

            end_time = time.time()
            epoch_time = end_time - strat_time
            strat_time = time.time()
            psnr_loss_spec, rmse_loss_spec, mrae_loss_spec, loss_dop, loss_aop = Validate(val_loader_hsipol, model, mask)

            # Save model
            if torch.abs(record_rmse_loss - rmse_loss_spec) < 0.0001 or rmse_loss_spec < record_rmse_loss or iteration % 10000 == 0:
                print(f'Saving to {output_path}')
                save_checkpoint(output_path, (epoch), iteration, model, optimizer)
                if rmse_loss_spec < record_rmse_loss:
                    record_rmse_loss = rmse_loss_spec
            # print loss
            print(" Iter[%06d/%06d], Epoch[%06d], Time[%06d],  learning rate : %.9f, Train Loss_spec: %.9f,"
                    "Test RMSE_spec: %.9f, Test PSNR_spec: %.9f, Test MRAE_spec: %.9f, Test dop: %.9f, Test aop: %.9f"
                    % (iteration, total_iteration, epoch, epoch_time, lr, losses_spec.avg, rmse_loss_spec, psnr_loss_spec, mrae_loss_spec, loss_dop, loss_aop))

            logger.info(" Iter[%06d/%06d], Epoch[%06d], Time[%06d],  learning rate : %.9f, Train Loss_spec: %.9f,"
                    "Test RMSE_spec: %.9f, Test PSNR_spec: %.9f, Test MRAE_spec: %.9f, Test dop: %.9f, Test aop: %.9f"
                    % (iteration, total_iteration, epoch, epoch_time, lr, losses_spec.avg, rmse_loss_spec, psnr_loss_spec, mrae_loss_spec, loss_dop, loss_aop))
                
        
def Validate(val_loader_hsipol, model, mask):
    model.eval()

    losses_psnr_spec = AverageMeter()
    losses_rmse_spec = AverageMeter()
    losses_mrae_spec = AverageMeter()
    losses_dop = AverageMeter()
    losses_aop = AverageMeter()


    for i, (HSIPOLs) in enumerate(val_loader_hsipol):
        HSIPOLs = HSIPOLs.cuda()
        mask_patch = data_processing.get_random_mask_patches_pol_together(mask=mask, image_size=opt.image_size, patch_size=opt.mos_patch_size, batch_size=1)
        #Generate the measurements using traning HSIs and selected sub-pattern
        inputs_HSIs, targets_HSIs = data_processing.get_mos_hsi_Norm_pols_togethor_SR(hsi=HSIPOLs, mask=mask_patch, sigma=opt.sigma, mos_size=opt.mos_patch_size[0], hsi_input_size=opt.train_patch_size[0], hsi_target_size=opt.train_patch_size[0])
        


        with torch.no_grad():
  
            outputs_hsipol = model(inputs_HSIs, mask_patch, 'pols')

            b, p, c, h, w = targets_HSIs.shape
            targets_HSIs = targets_HSIs.view(b*p, c, h, w)
            outputs_hsipol = outputs_hsipol.view(b*p, c, h, w)
            loss_dop, loss_aop, loss_S1, loss_S2 = criterion_aop_dop(outputs_hsipol.permute(1, 0, 2, 3), targets_HSIs.permute(1, 0, 2, 3))

            loss_psnr_spec = criterion_psnr(outputs_hsipol, targets_HSIs)
            loss_rmse_spec = criterion_rmse(outputs_hsipol, targets_HSIs)
            loss_mrae_spec = criterion_mrae(outputs_hsipol, targets_HSIs)

            losses_psnr_spec.update(loss_psnr_spec.data)
            losses_rmse_spec.update(loss_rmse_spec.data)
            losses_mrae_spec.update(loss_mrae_spec.data)



            losses_dop.update(loss_dop.data)
            losses_aop.update(loss_aop.data)



    return losses_psnr_spec.avg, losses_rmse_spec.avg, losses_mrae_spec.avg, losses_dop.avg, losses_aop.avg


if __name__ == '__main__':
    main()


