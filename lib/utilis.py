import os
# import json
import random
import cv2
import numpy as np
from skimage.measure import compare_ssim, compare_psnr


def evaluate(lr, hr):
    """
    计算SSIM和PSNR
    return:[ssim,psnr]
    """
    h, w, _ = lr.shape
    hr = cv2.resize(hr, (w, h))
    ssim = compare_ssim(lr, hr, multichannel=True)
    psnr = compare_psnr(lr, hr)
    return ssim, psnr

def path_checker(path):
    """
    检查目录是否存在，不存在，则创建
    """
    if not os.path.isdir(path):
        os.makedirs(path)
        print('目录不存在，已创建...')
    else:
        print('目录已存在')


def read_split_data(root: str, val_rate: float = 0.2, name_list: str='lr.txt'):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    img_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root))]
    img_class.sort()
    print(img_class[0])

    train_img_path_list = []
    val_img_path_list = []
    every_class_num = []
    supported = [".tif", ".bmp"]
    for cla in img_class:
        # print(cla)
        cla_path = os.path.join(root, cla)
        imgs = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                if os.path.splitext(i)[-1] in supported]
        every_class_num.append(len(imgs))
        eval_path = random.sample(imgs, k=int(len(imgs) * val_rate))
        for img_path in imgs:
            if img_path in eval_path:
                with open('/mnt/sda2/zjy/new_data/test/'+name_list, 'a') as f:
                    img_path = img_path.split('/', 4)
                    f.write('{}\n'.format(img_path[4]))
                val_img_path_list.append(img_path)
            else:
                with open('/mnt/sda2/zjy/new_data/train/'+name_list, 'a') as f:
                    img_path = img_path.split('/', 4)
                    f.write('{}\n'.format(img_path[4]))
                train_img_path_list.append(img_path)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images were found in the train dataset.".format(len(train_img_path_list)))
    print("{} images were found in the test dataset.".format(len(val_img_path_list)))
    return train_img_path_list, val_img_path_list
    
    
def tensor2image (tensor):
    '''
    tensor_list: list，内在元素是 pytorch tensor [c,h,w]
    变化成 image 0-255 hwc BGR

    '''
    imgx = tensor.cpu().detach().numpy()        
    imgx = np.transpose(imgx, axes = (1,2,0))
    imgx = cv2.cvtColor(imgx,cv2.COLOR_RGB2BGR)
    imgx = np.uint8(imgx*255)
    return imgx


def need_grad(status,model,fix_name):
    '''
    锁定模型部分权值不更新
    '''
    for name,value in model.named_parameters():
        if fix_name in name:
            value.requires_grad = status
            if status:
                print(name,':has been released...')
            else:
                print(name,':has been fixed...')
    model_params = filter(lambda p: p.requires_grad,model.parameters() )
    return model_params

    

def wrap_image_plus(tensor_list,ycbcr2bgr = False):
    """
    tensor_list: list, 内在元素是 pytorch tensor [c,h,w]
    for convience
    4x,gen10x,10x,gen20x,20x
    """
    # 将tensor转为numpy并转换通道
    assert len(tensor_list[0].shape) == 3, "tensor 必须 为 CHW"
    img_list = []
    for tensor in tensor_list:
        img_x = tensor.cpu().detach().numpy()
        img_x = np.transpose(img_x,axes=(1,2,0))
        if ycbcr2bgr:
            img_x = cv2.cvtColor(img_x,cv2.COLOR_YCrCb2BGR)
        else:
            img_x = cv2.cvtColor(img_x,cv2.COLOR_RGB2BGR)
        img_list.append(img_x)
    h, w, _ = img_list[-1].shape
    for i, img in enumerate(img_list[:-1]):
        img = cv2.resize(img, (h, w))
        img_list[i] = img
    assemble_img = np.concatenate(img_list,axis = 1)
    assemble_img = np.uint8(assemble_img*255) #注意
    return assemble_img

def txt2list(all_path, txt_path):
    #---
    # 功能：读取只包含数字的txt文件，并转化为list形式
    # txt_path：txt的路径
    #---
    data_list = []
    with open(txt_path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")  # 去除末尾的换行符
        line = all_path + line
        data_list.append(line)
    return data_list


# if __name__ == '__main__':
#     root = "DATASET"
#     train_img_path, train_img_target, val_img_path, val_img_target=read_split_data(root)
#     111