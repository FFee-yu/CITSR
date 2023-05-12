import os
device_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = device_id
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from lib.dataset import DoubleDataset
from torch.utils.data import DataLoader
from lib import utilis
from lib.utilis import evaluate, wrap_image_plus, txt2list
from model.citsr import CITSR

def model_test(G, loader, nums_to_save, img_save_path, times):
    print('nums_to_save:{}'.format(nums_to_save))
    print('Eval model...')
    root_path = os.path.join(img_save_path, str(times))
    if not os.path.exists(root_path.split('.pth')[0]):
        os.makedirs(root_path.split('.pth')[0])
    save_path = os.path.join(root_path.split('.pth')[0], 'ssim_psnr.txt')
    ssims = list()
    psnrs = list()
    img_save_counter = 0
    with torch.no_grad():
        for index, (lr_img, hr_img) in enumerate(loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            gen, _ = G(lr_img)
            for i in range(gen.shape[0]):
                tensor_list = [lr_img[i,:,:,:], gen[i,:,:,:], hr_img[i,:,:,:]]
                wraped = wrap_image_plus(tensor_list)
                h = wraped.shape[0]
                gen_ = wraped[:,h:2*h,:]
                hr = wraped[:,2*h:,:]
                ssim, psnr = evaluate(gen_, hr)
                with open(save_path, 'a') as f:
                    f.write('img_num:{}  ssim:{:.2f}  psnr:{:.2f}\n'.format(img_save_counter, ssim, psnr))
                ssims.append(ssim)
                psnrs.append(psnr)
                if img_save_counter % nums_to_save == 0:
                    img_path = os.path.join(root_path.split('.pth')[0], f'{img_save_counter//nums_to_save}.tif')
                    cv2.imwrite(img_path, wraped)
                del wraped, h, gen_, hr
                img_save_counter += 1
            del gen, hr_img, lr_img
    avg_ssim = sum(ssims)/len(ssims)
    avg_psnr = sum(psnrs)/len(psnrs)
    return [avg_ssim,avg_psnr]

device_num = 1
device = torch.device('cuda')
method = 'mymodel'
title = 'my_3a1a1_tw33_2jl_mlp'
filePath = f'/mnt/sda2/zjy/{method}/checkpoints/{title}'
path_log = os.listdir(filePath)
path_log=[x for x in path_log if 'G' in os.path.splitext(os.path.basename(x))[0]]
path_log.sort(key = lambda x:int(x[2:-4]))
print(path_log)
log_num = len(path_log)

G = CITSR(upscale=2, img_size=128, patch_size=1,
                  window_size=8, img_range=1., depths=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                  embed_dim=60, num_heads=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], mlp_ratio=2).to(device)

all_path = '/mnt/sda2/zjy/'
lr_path = '/mnt/sda2/zjy/new_data/test/lr.txt'
hr_path = '/mnt/sda2/zjy/new_data/test/hr.txt'
test_img_log_path = f'/mnt/sda2/zjy/{method}/test_img/{title}/'
tensorboard_log = f'/mnt/sda2/zjy/{method}/tensorboard/{title}/'
utilis.path_checker(test_img_log_path)

Writer = SummaryWriter(tensorboard_log)
test_lr_path_list = txt2list(all_path, lr_path)
test_hr_path_list = txt2list(all_path, hr_path)
print(test_hr_path_list[151:152])
print(len(test_lr_path_list), len(test_hr_path_list))
test_set = DoubleDataset(test_lr_path_list, test_hr_path_list, brightness=0)
loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=3)

save_log = test_img_log_path + 'ssim_psnr.txt'
with open(save_log, 'a') as f:
        f.write('test_imgs:{}  nums_to_save:{}\n'.format(len(test_lr_path_list), int(0.02*len(test_lr_path_list))))
start1 = 34
for i in range(start1, log_num):
    print(path_log[i])
    G.load_state_dict(torch.load(filePath+'/'+path_log[i]))
    G.eval()
    ssim, psnr = model_test(G, loader, nums_to_save = int(0.02*len(test_lr_path_list)), 
                            img_save_path=test_img_log_path, times=path_log[i])
    with open(save_log, 'a') as f:
        f.write('modle_name:{}  ssim:{:.2f}  psnr:{:.2f}\n'.format(path_log[i], ssim, psnr))
    Writer.add_scalar('test_all/ssim', ssim, i)
    Writer.add_scalar('test_all/psnr', psnr, i)
Writer.close()
    