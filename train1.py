import os
device_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = device_id
import sys
import cv2
import torch
from torch import optim
from lib import utilis
from lib.dataset import DoubleDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize
from lib.utilis import wrap_image_plus, txt2list
from model.citsr import CITSR

torch.manual_seed(0) 
device_num = 1 
device = torch.device('cuda')

#================================
# config path
#================================
method = 'mymodel'
title = 'my_3a1a1_tw33_2jl_mlp'
all_path = '/mnt/sda2/zjy/'
lr_path = '/mnt/sda2/zjy/new_data/train/lr.txt'
hr_path = '/mnt/sda2/zjy/new_data/train/hr.txt'
tensorboard_log = f'/mnt/sda2/zjy/{method}/tensorboard/{title}/'
checkpoints_path = f'/mnt/sda2/zjy/{method}/checkpoints/{title}/'
train_img_log_path = f'/mnt/sda2/zjy/{method}/train_img/{title}/'

train_lr_path_list = txt2list(all_path, lr_path)
train_hr_path_list = txt2list(all_path, hr_path)
print(len(train_lr_path_list), len(train_hr_path_list))

#=================================
# model config
#=================================
stage1_epoch = 3
stage2_epoch = 3
stage3_epoch = 3
batch_size = 3
img_save_step_ratio = 0.02
model_save_step_ratio = 0.2
NUM_EPOCHS = stage1_epoch + stage2_epoch + stage3_epoch
lr = 8e-5
batch_size *= device_num    # batch size
num_workers = device_num*3

pretrain_flag = False
sample_nums = len(train_lr_path_list)
batch_nums = sample_nums//batch_size
model_save_step = int(batch_nums*model_save_step_ratio)
img_log_step = int(batch_nums*img_save_step_ratio)

#check path
utilis.path_checker(checkpoints_path)
utilis.path_checker(train_img_log_path)

train_set = DoubleDataset(train_lr_path_list, train_hr_path_list, brightness=0)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#load model
G = CITSR(upscale=2, img_size=128, patch_size=1,
                  window_size=8, img_range=1., depths=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                  embed_dim=60, num_heads=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], mlp_ratio=2).to(device)

#set cost function
l1_loss = torch.nn.L1Loss(reduction='mean')

# #load pretrain model
# if pretrain_flag:
#     G.load_state_dict(torch.load(G_path))

Writer = SummaryWriter(tensorboard_log)
########################
#1.train
########################
image_save_counter = 0
optimizer1 = optim.Adam(G.parameters(), lr=lr)
torch_resize = Resize([256,256])

if __name__ == '__main__':
    for epoch in range(stage1_epoch):
        G.train()
        for i, (img_lr, img_hr) in enumerate(train_loader):
            img_lr = img_lr.to(device)
            img_hr = img_hr.to(device)
            gen = G(img_lr, stage1=True)
            G.zero_grad()
            print(gen.shape)
            g_loss = l1_loss(gen, torch_resize(img_hr))
            g_loss.backward()              
            optimizer1.step()  
            print('train1 done')
            if i % img_log_step == 0:
                ssims = list()
                psnrs = list()
                for j in range(gen.shape[0]):
                    ssim, psnr = utilis.evaluate(utilis.tensor2image(gen[j]), utilis.tensor2image(img_hr[j]))
                    ssims.append(ssim)
                    psnrs.append(psnr)
                avg_ssim = sum(ssims)/len(ssims)
                avg_psnr = sum(psnrs)/len(psnrs)
                Writer.add_scalar('train1/loss', g_loss, image_save_counter)
                Writer.add_scalar('train1/lr', lr, image_save_counter)
                Writer.add_scalar('train1/ssim', avg_ssim, image_save_counter)
                Writer.add_scalar('train1/psnr', avg_psnr, image_save_counter)
                # save image log
                # cv2.imwrite((train_img_path+'stage1_'+str(epoch)+'_'+str(i//img_log_step)+'.tif'), utilis.tensor2image(gen[0]))
                tensor_list = [img_lr[0, :, :, :], gen[0, :, :, :], img_hr[0, :, :, :]]
                warped_image = wrap_image_plus(tensor_list)
                image_log_name = train_img_log_path+'stage1_'+str(epoch)+'_'+str(i//img_log_step)+'.tif'
                cv2.imwrite(image_log_name, warped_image)
                del tensor_list, warped_image
                image_save_counter += 1
            sys.stdout.write("\r[Epoch {}/{}] [Batch {}/{}] [G:{:.4f}]".format(epoch, NUM_EPOCHS, i, len(train_loader), g_loss.item()))
            sys.stdout.flush()
            if i%model_save_step == 0 and i!=0:
                torch.save(G.state_dict(), checkpoints_path+'G_%d_%d.pth' % (epoch, i//model_save_step))
                # torch.save(optimizer1.state_dict(), checkpoints_path+'OP_%d_%d.pth' % (epoch, i//model_save_step))
                lr = lr*0.6
                for param in optimizer1.param_groups:
                    param['lr'] = lr
                
    batch_size = 1
    batch_nums = sample_nums//batch_size
    model_save_step = int(batch_nums*model_save_step_ratio) 
    img_log_step = int(batch_nums*img_save_step_ratio) 
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    image_save_counter = 0
    lr = 1e-4
    netG_params = utilis.need_grad(False, G, 'stage1')
    optimizer2 = optim.Adam(netG_params,lr = lr)           
    for epoch in range(stage1_epoch, stage2_epoch+stage1_epoch):
        G.train()
        for i, (img_lr, img_hr) in enumerate(train_loader):
            img_lr = img_lr.to(device)
            img_hr = img_hr.to(device)
            gen, _ = G(img_lr)
            G.zero_grad()
            g_loss = l1_loss(gen, img_hr)
            g_loss.backward()
            optimizer2.step()
            print('train2 done')
            if i % img_log_step == 0:
                # G loss situation
                ssims = list()
                psnrs = list()
                for j in range(gen.shape[0]):
                    ssim, psnr = utilis.evaluate(utilis.tensor2image(gen[j]), utilis.tensor2image(img_hr[j]))
                    ssims.append(ssim)
                    psnrs.append(psnr)
                avg_ssim = sum(ssims)/len(ssims)
                avg_psnr = sum(psnrs)/len(psnrs)
                Writer.add_scalar('train2/loss', g_loss, image_save_counter)
                Writer.add_scalar('train2/lr', lr, image_save_counter)
                Writer.add_scalar('train2/ssim', avg_ssim, image_save_counter)
                Writer.add_scalar('train2/psnr', avg_psnr, image_save_counter)
                # save image log
                # cv2.imwrite((train_img_path+'stage2_'+str(epoch)+'_'+str(i//img_log_step)+'.tif'), utilis.tensor2image(gen[0]))
                tensor_list = [img_lr[0, :, :, :], gen[0, :, :, :], img_hr[0, :, :, :]]
                warped_image = wrap_image_plus(tensor_list)
                image_log_name = train_img_log_path+'stage2_'+str(epoch)+'_'+str(i//img_log_step)+'.tif'
                cv2.imwrite(image_log_name, warped_image)
                del tensor_list, warped_image
                image_save_counter += 1
            sys.stdout.write("\r[Epoch {}/{}] [Batch {}/{}] [G:{:.4f}]".format(epoch, NUM_EPOCHS, i, len(train_loader), g_loss.item()))
            sys.stdout.flush()
            if i%model_save_step == 0 and i!=0:
                torch.save(G.state_dict(), checkpoints_path+'G_%d_%d.pth' % (epoch, i//model_save_step))
                # torch.save(optimizer2.state_dict(), checkpoints_path+'OP_%d_%d.pth' % (epoch, i//model_save_step))
                lr = lr*0.5
                for param in optimizer2.param_groups:
                    param['lr'] = lr
      
    batch_size = 1
    batch_nums = sample_nums//batch_size
    model_save_step = int(batch_nums*model_save_step_ratio) 
    img_log_step = int(batch_nums*img_save_step_ratio) 
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    image_save_counter = 0
    lr = 8e-5
    G_params = utilis.need_grad(True, G, 'stage1')
    optimizer3 = optim.Adam(G_params, lr = lr)           
    for epoch in range(stage2_epoch+stage1_epoch, NUM_EPOCHS):
        G.train()
        for i, (img_lr, img_hr) in enumerate(train_loader):
            img_lr = img_lr.to(device)
            img_hr = img_hr.to(device)
            gen, gen_l = G(img_lr)
            print(gen.shape)
            G.zero_grad()
            loss_1 = l1_loss(gen, img_hr)
            loss_2 = l1_loss(gen_l, torch_resize(img_hr))
            g_loss = loss_1 + 0.1*loss_2
            g_loss.backward()
            optimizer3.step()
            print('train3 done')
            if i % img_log_step == 0:
                # G loss situation
                ssims = list()
                psnrs = list()
                for j in range(gen.shape[0]):
                    ssim, psnr = utilis.evaluate(utilis.tensor2image(gen[j]), utilis.tensor2image(img_hr[j]))
                    ssims.append(ssim)
                    psnrs.append(psnr)
                avg_ssim = sum(ssims)/len(ssims)
                avg_psnr = sum(psnrs)/len(psnrs)
                Writer.add_scalars('train3/loss', {'g_loss':g_loss, 'loss_h':loss_1, 'loss_l':loss_2}, image_save_counter)
                Writer.add_scalar('train3/lr', lr, image_save_counter)
                Writer.add_scalar('train3/ssim', avg_ssim, image_save_counter)
                Writer.add_scalar('train3/psnr', avg_psnr, image_save_counter)
                # save image log
                # cv2.imwrite((train_img_path+'stage2_'+str(epoch)+'_'+str(i//img_log_step)+'.tif'), utilis.tensor2image(gen[0]))
                tensor_list = [img_lr[0, :, :, :], gen[0, :, :, :], img_hr[0, :, :, :]]
                warped_image = wrap_image_plus(tensor_list)
                image_log_name = train_img_log_path+'stage3_'+str(epoch)+'_'+str(i//img_log_step)+'.tif'
                cv2.imwrite(image_log_name, warped_image)
                del tensor_list, warped_image
                image_save_counter += 1
            sys.stdout.write("\r[Epoch {}/{}] [Batch {}/{}] [G:{:.4f}]".format(epoch, NUM_EPOCHS, i, len(train_loader), g_loss.item()))
            sys.stdout.flush()
            if i%model_save_step == 0 and i!=0:
                torch.save(G.state_dict(), checkpoints_path+'G_%d_%d.pth' % (epoch, i//model_save_step))
                lr = lr*0.6
                for param in optimizer3.param_groups:
                    param['lr'] = lr              
    
    Writer.close()