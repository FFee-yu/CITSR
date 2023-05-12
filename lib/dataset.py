import random
import cv2
import numpy as np
import torch
from torch.utils import data


class DoubleDataset(data.Dataset):
    def __init__(self, image_lr_path:list, image_hr_path:list, crop = False, brightness=None):
        self.image_lr_path = image_lr_path
        self.image_hr_path = image_hr_path
        self.crop = crop
        self.brightness = brightness

    def __len__(self):
        return len(self.image_lr_path)

    def __getitem__(self, id):
        img_lr = cv2.imread(self.image_lr_path[id])
        img_hr = cv2.imread(self.image_hr_path[id])

        # BGR to RGB
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        if self.crop:
            #because input 4x is 256
            img_lr = img_lr[64:192, 64:192, :]
            img_hr = img_hr[256:768, 256:768, :]
        if self.brightness is not None:
            img_lr = cv2.cvtColor(img_lr, cv2.COLOR_RGB2HSV)
            img_hr = cv2.cvtColor(img_hr, cv2.COLOR_RGB2HSV)
            img_lr[..., 2] = img_lr[..., 2] + self.brightness
            img_hr[..., 2] = img_hr[..., 2] + self.brightness
            img_lr = cv2.cvtColor(img_lr, cv2.COLOR_HSV2RGB)
            img_hr = cv2.cvtColor(img_hr, cv2.COLOR_HSV2RGB)
        # HWC to CHW
        img_lr = np.transpose(img_lr, axes=(2, 0, 1)).astype(np.float32)/255
        img_hr = np.transpose(img_hr, axes=(2, 0, 1)).astype(np.float32)/255
        # numpy array to torch tensor
        img_lr = torch.from_numpy(img_lr)
        img_hr = torch.from_numpy(img_hr)
        return img_lr, img_hr






