import torch
import torch.utils.data as data
import cv2
import os
import os.path
import random
import glob
from datasets.utls import *




class MakeTestSet(data.Dataset):

    def __init__(self, root):
        self.root = root

        path = sorted(glob.glob(os.path.join(self.root, "*.jpg"))+glob.glob(os.path.join(self.root, "*.png"))+glob.glob(os.path.join(self.root, "*.bmp")))
        self.path_valid = path

    def __getitem__(self, item):
        
        ir_name=self.path_valid[item].split('input_vis')[0]+'/input_ir/'+self.path_valid[item].split('input_vis')[-1]
        img_bgr = cv2.imread(self.path_valid[item], cv2.IMREAD_COLOR)
        img_gray = cv2.imread(ir_name, cv2.IMREAD_COLOR)
        img_bgr_resize = cv2.resize(img_bgr,(128,128))
        img_gray_resize = cv2.resize(img_gray,(128,128))
        img_gray_resize = cv2.cvtColor(img_gray_resize,cv2.COLOR_BGR2GRAY)[...,None]/ 255
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        img_rgb = img_bgr[:, :, ::-1] / 255
        img_rgb_resize = img_bgr_resize[:, :, ::-1] / 255
        img_gray = img_gray[:, :, np.newaxis] / 255


        img_g = img_gray.transpose((2, 0, 1))  # C*W*H
        img_c = img_rgb.transpose((2, 0, 1))  # C*W*H
        img_g_resize = img_gray_resize.transpose((2, 0, 1))  # C*W*H
        img_c_resize = img_rgb_resize.transpose((2, 0, 1))  # C*W*H

        tensor_g = torch.from_numpy(img_g.astype(np.float32))
        tensor_c = torch.from_numpy(img_c.astype(np.float32))
        tensor_g_resize = torch.from_numpy(img_g_resize.astype(np.float32))
        tensor_c_resize = torch.from_numpy(img_c_resize.astype(np.float32))
        return tensor_g, tensor_c,tensor_g_resize, tensor_c_resize,ir_name.split('/')[-1]
    def __len__(self):
        return len(self.path_valid)