import numpy as np
from hr4ir.pwc.tensordlt import solve_DLT
# from nemar.pwc.tf_spatial_transform import transform
from torchvision.transforms import Resize
from hr4ir.pwc.spatial_transform import Transform
import torch.nn.functional as F
import torch
import torch.nn as nn
# from torchvision.transforms import Resize
from hr4ir.attention import PositionEncodingSine, LocalFeatureTransformer
import time
from hr4ir.pwc.corr import AGCL as corr
# import kornia#.geometry.transform.get_perspective_transform as getH
# from kornia.geometry import transform
class feature_extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.resize=Resize((128,128))
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(64)
        # self.conv1=torch.nn.Sequential( nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3, padding=1),
        #                                 nn.BatchNorm2d(64),
        #                                 nn.ReLU(True)#,
        #                                 # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        #                                 # nn.BatchNorm2d(64),
        #                                 # nn.ReLU(True)
        #                                 )
        # self.feature.append(conv1)
        # self.maxpool1 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv1=torch.nn.Sequential( nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True)
                                        )
        # self.maxpool2 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv2=torch.nn.Sequential( nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True),
                                        nn.BatchNorm2d(128),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.ReLU(True))
        # self.maxpool3 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv3 = torch.nn.Sequential( nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True))
        self.resize1 = Resize([64,64])
        self.resize2 = Resize([32,32])
        self.resize3 = Resize([16,16])
        # end
    def forward(self, middle_1, middle_2, middle_3):
        # input=self.resize(input)
        # conv1 = self.conv1(input)
        # conv1m = self.maxpool1(conv1)
        middle_1=self.resize1(middle_1)
        middle_2=self.resize2(middle_2)
        middle_3=self.resize3(middle_3)
  
        conv1 = self.conv1(middle_1)
        conv2 = self.conv2(middle_2)
        conv3 = self.conv3(middle_3)
        return conv1,conv2,conv3

def resize_shift_final(offset,crop_size):
    offset_ = offset.clone()
    if isinstance(crop_size, tuple):
        offset_[:,0,:] = offset[:,0,:]*crop_size[0]/128.
        offset_[:,1,:] = offset[:,1,:]*crop_size[1]/128.
        offset_[:,2,:] = offset[:,2,:]*crop_size[0]/128.
        offset_[:,3,:] = offset[:,3,:]*crop_size[1]/128.
        offset_[:,4,:] = offset[:,4,:]*crop_size[0]/128.
        offset_[:,5,:] = offset[:,5,:]*crop_size[1]/128.
        offset_[:,6,:] = offset[:,6,:]*crop_size[0]/128.
        offset_[:,7,:] = offset[:,7,:]*crop_size[1]/128.
    else:
        offset_=offset*crop_size/128.
    return offset_

def cost_volume(c1, warp, search_range,pad,lrelu):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = pad(warp)
    _,c ,h, w = c1.shape
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice=padded_lvl[:, :,y:y+h, x:x+w]
            cost = torch.mean(c1 * slice, axis=1, keepdims=True)
            cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, axis=1)
    cost_vol = lrelu(cost_vol)

    return cost_vol

class H_estimator(torch.nn.Module):
    def __init__(self, batch_size,crop_size,device, is_training=1):
        super().__init__()
        self.crop_size=crop_size
        self.device=device
        self.feature = feature_extractor()
        # self.keep_prob = 0.5 if is_training==True else 1.0
        # print(is_training)
        # time.sleep(1000)
        # self.keep_prob = 0.5 if is_training==True else 0.0
        self.keep_prob = 0.5 if is_training==True else 0.0
        for m in self.feature.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        self.getoffset1 = torch.nn.Sequential(torch.nn.Linear(in_features = 16*16*36, out_features = 512),
                                            nn.ReLU(True),
                                            nn.Dropout(p = self.keep_prob),
                                            torch.nn.Linear(in_features = 512, out_features = 8))
        self.getoffset2 = torch.nn.Sequential(torch.nn.Linear(in_features = 32*32*36, out_features = 256),
                                            nn.ReLU(True),
                                            nn.Dropout(p = self.keep_prob),
                                            torch.nn.Linear(in_features = 256, out_features = 8))
        self.getoffset3 = torch.nn.Sequential(torch.nn.Linear(in_features = 64*64*36, out_features = 256),
                                            nn.ReLU(True),
                                            nn.Dropout(p = self.keep_prob),
                                            torch.nn.Linear(in_features = 256, out_features = 8))
        # self.M_tile_inv_128, self.M_tile_128 = self.to_transform_H(128, batch_size)
        self.M_tile_inv_32, self.M_tile_32 = self.to_transform_H(32, batch_size)
        self.M_tile_inv_16, self.M_tile_16 = self.to_transform_H(16, batch_size)
        self.M_tile_inv_64, self.M_tile_64 = self.to_transform_H(64, batch_size)
        # self.M_tile_inv_final, self.M_tile_final = self.to_transform_H(crop_size, batch_size)
        # print(crop_size)
        if isinstance(crop_size, tuple):
            self.M_tile_inv_final, self.M_tile_final = self.to_transform_H_rec(self.crop_size, batch_size)
        else:
            self.M_tile_inv_final, self.M_tile_final = self.to_transform_H(self.crop_size, batch_size)
        self.transform32 = Transform(32, 32,batch_size,self.device).cuda(device=self.device)
        self.transform64 = Transform(64, 64,batch_size,self.device).cuda(device=self.device)
        # self.transform128 = Transform(128,128,batch_size,self.device).cuda(device=self.device)
        if isinstance(crop_size, tuple):
            self.transform_final = Transform(crop_size[1],crop_size[0],batch_size,self.device).cuda(device=self.device)
        else:
            self.transform_final = Transform(crop_size,crop_size,batch_size,self.device).cuda(device=self.device)
        # self.transform_final = Transform(crop_size,crop_size,batch_size,self.device).cuda(device=self.device)
        self.transform_768 = Transform(768,768,batch_size,self.device).cuda(device=self.device)
        self.transform16 = Transform(16, 16,batch_size,self.device).cuda(device=self.device)
        self.ori= torch.from_numpy ( np.array([[[0.,0.],[1.,0.],[0.,1.],[1.,1.]]]) )
        for i in range(3):
            self.ori=torch.cat((self.ori,self.ori),0)
        self.ori=self.ori.cuda(device=self.device)
        # loftr
        self.self_att_fn = LocalFeatureTransformer(
            d_model=128, nhead=8, layer_names=["self"] * 1, attention="linear"
        )
        self.cross_att_fn = LocalFeatureTransformer(
            d_model=128, nhead=8, layer_names=["cross"] * 1, attention="linear"
        )
        self.search_num=9
        self.conv_offset_1_deform = nn.Conv2d(
            128, self.search_num * 2, kernel_size=3, stride=1, padding=1
        )
        self.conv_offset_2_deform = nn.Conv2d(
            128, self.search_num * 2, kernel_size=3, stride=1, padding=1
        )
        self.conv_offset_3_deform = nn.Conv2d(
            64, self.search_num * 2, kernel_size=3, stride=1, padding=1
        )
        
        self.pos_encoding_fn_small = PositionEncodingSine(
                d_model=128, max_shape=(16, 16)
            )

        
    def to_transform_H_rec(self,width_height, batch_size):
        width, height = width_height
        M = np.array([
            [width / 2.0, 0., width / 2.0],
            [0., height / 2.0, height / 2.0],
            [0., 0., 1.]
        ], dtype=np.float32)

        M_tensor = torch.from_numpy(M)
        M_tile = torch.unsqueeze(M_tensor, 0).repeat(batch_size, 1, 1)

        M_inv = np.linalg.inv(M)
        M_tensor_inv = torch.from_numpy(M_inv)
        M_tile_inv = torch.unsqueeze(M_tensor_inv, 0).repeat(batch_size, 1, 1)

        M_tile_inv = M_tile_inv.cuda(device=self.device)
        M_tile = M_tile.cuda(device=self.device)
        return M_tile_inv, M_tile
    def to_transform_H(self, patch_size, batch_size):            
        M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                    [0., patch_size / 2.0, patch_size / 2.0],
                    [0., 0., 1.]]).astype(np.float32)
        M_tensor = torch.from_numpy(M)
        M_tile = torch.unsqueeze(M_tensor, 0).repeat( [batch_size, 1, 1])
        M_inv = np.linalg.inv(M)
        M_tensor_inv = torch.from_numpy(M_inv)
        M_tile_inv = torch.unsqueeze(M_tensor_inv, 0).repeat([batch_size, 1, 1])
       
        M_tile_inv=M_tile_inv.cuda(device=self.device)
        M_tile=M_tile.cuda(device=self.device)
        return M_tile_inv, M_tile
    def forward(self, middle_1_f, middle_2_f, middle_3_f, middle_1_b, middle_2_b, middle_3_b,vis):
        batch_size = vis.shape[0]
        batch_size = middle_1_f.shape[0]
        feature1 = self.feature(middle_1_b, middle_2_b, middle_3_b)# [1, 64, 64, 64] [1, 128, 32, 32] [1, 128, 16, 16]
        feature2 = self.feature(middle_1_f, middle_2_f, middle_3_f)
        offset_1_deform = self.conv_offset_1_deform(feature1[-1])
        self.range_16 = 1
        offset_1_deform = self.range_16 * (torch.sigmoid(offset_1_deform) - 0.5) * 2.0 #[1, 18, 16, 16]
        
        offset_2_deform = self.conv_offset_2_deform(feature1[-2])
        self.range_32 = 1
        offset_2_deform = self.range_32 * (torch.sigmoid(offset_2_deform) - 0.5) * 2.0 #[1, 18, 32, 32]

        offset_3_deform = self.conv_offset_3_deform(feature1[-3])
        self.range_64 = 1
        offset_3_deform = self.range_64 * (torch.sigmoid(offset_3_deform) - 0.5) * 2.0 #[1, 18, 64, 64]
        # b, c, h, w
        x1_tmp = self.pos_encoding_fn_small(feature1[-1])
        
        x2_tmp = self.pos_encoding_fn_small(feature2[-1])
        
        # # b, h*w, c
        feature1_min = x1_tmp.permute(0, 2, 3, 1).reshape(x1_tmp.shape[0], x1_tmp.shape[2] * x1_tmp.shape[3], x1_tmp.shape[1])
        feature2_min = x2_tmp.permute(0, 2, 3, 1).reshape(x2_tmp.shape[0], x2_tmp.shape[2] * x2_tmp.shape[3], x2_tmp.shape[1])
        feature1_min, feature2_min = self.self_att_fn(feature1_min, feature2_min)
       
        feature1_min, feature2_min = [
                x.reshape(x.shape[0], 128 // 8, -1, x.shape[2]).permute(0, 3, 1, 2)
                for x in [feature1_min, feature2_min]]#(8,128,16,16)
        
        # ##############################  Regression Net 1 ##############################
        offset_1 = torch.zeros(batch_size, 8, 1).to(self.device)
        offset_2 = torch.zeros(batch_size, 8, 1).to(self.device)
        offset_3 = torch.zeros(batch_size, 8, 1).to(self.device)
        corr_fn_att_1 = corr(feature1_min, feature2_min, att = self.cross_att_fn)
        feature2_16_warp=None
        feature2_32_warp=None
        feature2_64_warp=None
        
        for itr in range(4):
            if itr % 4 == 0:
                small_patch = False
            else:
                small_patch = True

            out_corrs = corr_fn_att_1(offset_1_deform,  small_patch, feature2_16_warp)
            out_corrs_flatten = out_corrs.contiguous().view(out_corrs.shape[0],-1)
            offset_1 += torch.unsqueeze(self.getoffset1(out_corrs_flatten), 2)
            if itr<3:
                H16 = solve_DLT(offset_1 / 8., 16.)
                H16= torch.matmul(torch.matmul(self.M_tile_inv_16, H16), self.M_tile_16)
                feature2_16_warp = self.transform16(nn.functional.normalize(feature2[-1], dim=1, p=2), H16)
            else:
                H32 = solve_DLT(offset_1 / 4., 32.)
                H32= torch.matmul(torch.matmul(self.M_tile_inv_32, H32), self.M_tile_32)
                feature2_32_warp = self.transform32(nn.functional.normalize(feature2[-2], dim=1, p=2), H32)
        
        ############################  Regression Net 2 #####################################
        corr_fn_att_2 = corr(feature1[-2], feature2_32_warp)
        for itr in range(4):
            if itr % 4 == 0:
                small_patch = False
            else:
                small_patch = True
            out_corrs = corr_fn_att_2(offset_2_deform,  small_patch, feature2_32_warp)

            out_corrs_flatten = out_corrs.contiguous().view(out_corrs.shape[0],-1)

            offset_2 += torch.unsqueeze(self.getoffset2(out_corrs_flatten), 2)
            if itr<3:
                H32 = solve_DLT((offset_1+offset_2) / 4., 32.)
                H32= torch.matmul(torch.matmul(self.M_tile_inv_32, H32), self.M_tile_32)
                feature2_32_warp = self.transform32(nn.functional.normalize(feature2[-2], dim=1, p=2), H32)
            else:
                H64 = solve_DLT((offset_1+offset_2) / 2., 64.)
                H64 = torch.matmul(torch.matmul(self.M_tile_inv_64, H64), self.M_tile_64)
                feature2_64_warp = self.transform64(nn.functional.normalize(feature2[-3], dim=1, p=2), H64)
        ############################  Regression Net 3 #####################################
        corr_fn_att_3 = corr(feature1[-3], feature2_64_warp)
        for itr in range(4):
            if itr % 4 == 0:
                small_patch = False
            else:
                small_patch = True
            out_corrs = corr_fn_att_3(offset_3_deform,  small_patch, feature2_64_warp)
            out_corrs_flatten = out_corrs.contiguous().view(out_corrs.shape[0],-1)

            offset_3 += torch.unsqueeze(self.getoffset3(out_corrs_flatten), 2)
            if itr<3:
                H64 = solve_DLT((offset_1+offset_2+offset_3) / 2., 64.)
                H64= torch.matmul(torch.matmul(self.M_tile_inv_64, H64), self.M_tile_64)
                feature2_64_warp = self.transform64(nn.functional.normalize(feature2[-3], dim=1, p=2), H64)
        ###############################################################################
        # H = solve_DLT((offset_1 + offset_2 + offset_3)*self.crop_size/128., self.crop_size)
        H = solve_DLT(resize_shift_final((offset_1 + offset_2 + offset_3), self.crop_size), self.crop_size)

        H_mat = torch.matmul(torch.matmul(self.M_tile_inv_final, H), self.M_tile_final)

        warp2_vis = self.transform_final(vis, H_mat)
        return warp2_vis
