import torch
import torch.utils.data as data
import cv2
import os
import os.path
import random
import glob
from datasets.utls import *
import time

class Kodak24(data.Dataset):
    def __init__(self, root='/media/ruizhao/programs/datasets/Denoising/testset/Kodak24/'):
        self.root = root
        self.image_name = []
        for _, _, files in os.walk(self.root):
            for file in files:
                self.image_name.append(os.path.join(self.root, file))
            break
        self.image_name = sorted(self.image_name)

    def __getitem__(self, item):
        img_name = self.image_name[item]
        # print(img_name)
        # time.sleep(1000)
        img_bgr = cv2.imread(img_name, cv2.IMREAD_COLOR)

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        # img_name = img_name.split("/")[-1]

        img_c = img_bgr[:, :, ::-1] / 255  # BGR to RGB
        img_g = img_gray[:, :, np.newaxis] / 255

        h, w, _ = img_g.shape

        pad_x, pad_y = 512 - h, 512 - w

        img_g = np.pad(img_g, ((0, pad_x), (0, pad_y), (0, 0)))
        img_c = np.pad(img_c, ((0, pad_x), (0, pad_y), (0, 0)))

        img_c = img_c.transpose((2, 0, 1))  # C*W*H
        img_g = img_g.transpose((2, 0, 1))  # C*W*H

        img_c = torch.from_numpy(img_c.astype(np.float32))
        img_g = torch.from_numpy(img_g.astype(np.float32))
        return img_g, img_c, h, w

    def __len__(self):
        return len(self.image_name)


class MakeTrainSet(data.Dataset):
    def __init__(self, root,crop_size=128):
        self.root = root
        # print(self.root)
        # print("#################")
        path = sorted(glob.glob(os.path.join(self.root, "*.jpg"))+glob.glob(os.path.join(self.root, "*.png")))
        # self.path_train = path[3367:]
        self.path_train = path
        self.crop_size=crop_size

        # print(len(self.path_train))
        # self.path_train_select = random.sample(self.path_train, 8000)
        self.path_train_select = random.sample(self.path_train, len(self.path_train))

    def __getitem__(self, item):
        # print(self.path_train_select[item])
        ir_name=self.path_train_select[item].split('input_vis')[0]+'/input_ir/'+self.path_train_select[item].split('input_vis')[-1]
        gt = np.load(self.path_train_select[item].split('input_vis')[0]+'/shift/'+self.path_train_select[item].split('input_vis')[-1].split('.')[0]+'.npy')
        img_bgr = cv2.imread(self.path_train_select[item], cv2.IMREAD_COLOR)
        # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        img_gray = cv2.imread(ir_name, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        # print(img_gray.shape)
        img_rgb = img_bgr[:, :, ::-1] / 255
        img_gray = img_gray[:, :, np.newaxis] / 255
        # print(img_gray.shape)
        # time.sleep(1000)
        h, w, _ = img_gray.shape
        
        if h < self.crop_size:
            pad_x = self.crop_size - h
            img_gray = np.pad(img_gray, ((0, pad_x), (0, 0), (0, 0)))
            img_rgb = np.pad(img_rgb, ((0, pad_x), (0, 0), (0, 0)))
        if w < self.crop_size:
            pad_y = self.crop_size - w
            img_gray = np.pad(img_gray, ((0, 0), (0, pad_y), (0, 0)))
            img_rgb = np.pad(img_rgb, ((0, 0), (0, pad_y), (0, 0)))

        h, w, _ = img_gray.shape
        x, y = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
        img_gray = img_gray[x: x + self.crop_size, y: y + self.crop_size, :]
        img_rgb = img_rgb[x:x + self.crop_size, y: y + self.crop_size, :]

        mode = random.randint(0, 7)

        img_g = data_aug(img_gray, mode=mode).transpose((2, 0, 1))  # C*W*H
        img_c = data_aug(img_rgb, mode=mode).transpose((2, 0, 1))  # C*W*H

        tensor_g = torch.from_numpy(img_g.astype(np.float32))
        tensor_c = torch.from_numpy(img_c.astype(np.float32))
        gt=torch.from_numpy(gt)
        # print(tensor_g.shape)
        # print(tensor_c.shape)
        # time.sleep(100)
        return tensor_g, tensor_c, gt

    def __len__(self):
        return len(self.path_train_select)


class MakeValidSet(data.Dataset):

    def __init__(self, root):
        self.root = root

        path = sorted(glob.glob(os.path.join(self.root, "*.jpg"))+glob.glob(os.path.join(self.root, "*.png")))
        # self.path_valid = path[:3367]
        self.path_valid = path

    def __getitem__(self, item):
        
        ir_name=self.path_valid[item].split('input_vis')[0]+'/input_ir/'+self.path_valid[item].split('input_vis')[-1]
        gt = np.load(self.path_valid[item].split('input_vis')[0]+'/shift/'+self.path_valid[item].split('input_vis')[-1].split('.')[0]+'.npy')
        img_bgr = cv2.imread(self.path_valid[item], cv2.IMREAD_COLOR)
        # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        img_gray = cv2.imread(ir_name, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
        # print(img_gray.shape)
        img_rgb = img_bgr[:, :, ::-1] / 255
        img_gray = img_gray[:, :, np.newaxis] / 255
        # print(img_gray.shape)
        # time.sleep(1000)
        h, w, _ = img_gray.shape
        
        # if h < self.crop_size:
        #     pad_x = self.crop_size - h
        #     img_gray = np.pad(img_gray, ((0, pad_x), (0, 0), (0, 0)))
        #     img_rgb = np.pad(img_rgb, ((0, pad_x), (0, 0), (0, 0)))
        # if w < self.crop_size:
        #     pad_y = self.crop_size - w
        #     img_gray = np.pad(img_gray, ((0, 0), (0, pad_y), (0, 0)))
        #     img_rgb = np.pad(img_rgb, ((0, 0), (0, pad_y), (0, 0)))

        h, w, _ = img_gray.shape
        # x, y = random.randint(0, h - self.crop_size), random.randint(0, w - self.crop_size)
        # img_gray = img_gray[x: x + self.crop_size, y: y + self.crop_size, :]
        # img_rgb = img_rgb[x:x + self.crop_size, y: y + self.crop_size, :]

        mode = 0

        img_g = data_aug(img_gray, mode=mode).transpose((2, 0, 1))  # C*W*H
        img_c = data_aug(img_rgb, mode=mode).transpose((2, 0, 1))  # C*W*H

        tensor_g = torch.from_numpy(img_g.astype(np.float32))
        tensor_c = torch.from_numpy(img_c.astype(np.float32))
        gt=torch.from_numpy(gt)
        # print(tensor_g.shape)
        # print(tensor_c.shape)
        # time.sleep(100)
        return tensor_g, tensor_c, gt,ir_name.split('/')[-1]
    def __len__(self):
        return len(self.path_valid)
