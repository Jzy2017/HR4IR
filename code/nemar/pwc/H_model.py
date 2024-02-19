import numpy as np
from nemar.pwc.tensordlt import solve_DLT
# from nemar.pwc.tf_spatial_transform import transform
from torchvision.transforms import Resize
from nemar.pwc.spatial_transform import Transform
import torch.nn.functional as F
import torch
import torch.nn as nn
# from torchvision.transforms import Resize
import time
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
        self.maxpool1 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv1=torch.nn.Sequential( nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True)
                                        )
        self.maxpool2 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv2=torch.nn.Sequential( nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True),
                                        nn.BatchNorm2d(128),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.ReLU(True))
        self.maxpool3 = torch.nn.MaxPool2d((2,2), stride=2)
        self.conv3 = torch.nn.Sequential( nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True))
        self.conv4 = torch.nn.Sequential( nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(True))
        self.resize1 = Resize([64,64])
        # self.resize2 = Resize([32,32])
        # self.resize3 = Resize([16,16])
        # end
    def forward(self, middle_1):
        middle_1=self.resize1(middle_1)
        # middle_2=self.resize2(middle_2)
        # middle_3=self.resize3(middle_3)
  
        # conv1 = self.conv1(middle_1)
        # conv2 = self.conv2(middle_2)
        # conv3 = self.conv3(middle_3)
        # return conv1,conv2,conv3
        conv1 = self.conv1(middle_1)
        # conv1m = self.maxpool1(conv1)
        conv2 = self.conv2(conv1)
        conv2m = self.maxpool1(conv2)
        conv3 = self.conv3(conv2m)
        conv3m = self.maxpool1(conv3)
        conv4 = self.conv4(conv3m)
        return conv1,conv2,conv3,conv4
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
class RegressionNet(torch.nn.Module):
    def __init__(self,is_training, search_range, warped=False):
        super().__init__()
        self.search_range=search_range
        self.warped=warped
        self.keep_prob = 0.5 if is_training==True else 1.0
        self.feature = []
        if self.search_range==16:
            self.stride=[1,1,1]
            self.out=[512,512,512]
            self.fully=1024
        elif self.search_range==8:
            self.stride=[1,1,2]
            self.out=[256,256,256]
            self.fully=512
        elif self.search_range==4:
            self.stride=[1,2,2]
            self.out=[128,128,128]
            self.fully=256
        self.pad=nn.ConstantPad2d(value=0,padding=[search_range, search_range, search_range, search_range])
        self.lrelu=nn.LeakyReLU(inplace=True)
        # self.conv0 =torch.nn.Sequential( nn.Conv2d(in_channels=256, out_channels=((2*self.search_range)+1)**2, kernel_size=3, stride=self.stride[0],padding=1),
        #                                 nn.BatchNorm2d(((2*self.search_range)+1)**2),
        #                                 nn.ReLU(True))
        self.conv1 =torch.nn.Sequential( nn.Conv2d(in_channels=((2*self.search_range)+1)**2, out_channels=self.out[0], kernel_size=3, stride=self.stride[0],padding=1),
                                        nn.BatchNorm2d(self.out[0]),
                                        nn.ReLU(True))
        # self.conv1 =torch.nn.Sequential( nn.Conv2d(in_channels=512,out_channels=self.out[0], kernel_size=3, stride=self.stride[0],padding=1),
        #                                 nn.ReLU(True))
        self.conv2 = torch.nn.Sequential( nn.Conv2d(in_channels=self.out[0], out_channels=self.out[1], kernel_size=3, stride=self.stride[1],padding=1),
                                        nn.BatchNorm2d(self.out[1]),
                                        nn.ReLU(True))
        self.conv3 = torch.nn.Sequential( nn.Conv2d(in_channels=self.out[1], out_channels=self.out[2], kernel_size=3, stride=self.stride[2],padding=1),
                                        nn.BatchNorm2d(self.out[2]),
                                        nn.ReLU(True))
        # Flatten dropout_conv4
        # self.getoffset = torch.nn.Sequential(torch.nn.Linear(in_features = 256*self.out[2], out_features = self.out[2]*2),
        #                                     nn.ReLU(True),
        #                                     nn.Dropout(p=self.keep_prob),
        #                                     torch.nn.Linear(in_features =self.out[2]*2, out_features = 8))
        self.getoffset = torch.nn.Sequential(torch.nn.Linear(in_features = 256*self.out[2], out_features = self.fully),
                                            nn.ReLU(True),
                                            nn.Dropout(p = self.keep_prob),
                                            torch.nn.Linear(in_features = self.fully, out_features = 8))
    # end

    def forward(self, feature1, feature2):
        # global_correlation=self.conv0(torch.cat((feature1,feature2),1))
        if not self.warped:
            global_correlation = cost_volume(nn.functional.normalize(feature1, dim=1, p=2), nn.functional.normalize(feature2,dim=1, p=2), self.search_range,self.pad,self.lrelu)  
        else:
            global_correlation = cost_volume(nn.functional.normalize(feature1, dim=1, p=2), feature2, self.search_range,self.pad,self.lrelu)
        conv1 = self.conv1(global_correlation)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        # print(conv3.shape)
        # time.sleep(1000)
        conv3_flatten = conv3.contiguous().view(conv3.shape[0],-1)
        offset=self.getoffset(conv3_flatten)
        # offset=torch.clamp(offset,-128.,128.)
        # print(torch.max(offset))
        # time.sleep(3)
        return offset

class H_estimator(torch.nn.Module):
    def __init__(self, batch_size,crop_size,device, is_training=1):
        super().__init__()
        self.crop_size=crop_size
        self.device=device
        self.feature = feature_extractor()
        self.Rnet1 = RegressionNet(is_training, 16, warped=False)#.cuda()
        self.Rnet2 = RegressionNet(is_training, 8, warped=True)#.cuda()
        self.Rnet3 = RegressionNet(is_training, 4, warped=True)#.cuda()
        for m in self.feature.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        for m in self.Rnet1.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        for m in self.Rnet2.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        for m in self.Rnet3.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        self.M_tile_inv_128, self.M_tile_128 = self.to_transform_H(128, batch_size)
        self.M_tile_inv_32, self.M_tile_32 = self.to_transform_H(32, batch_size)
        self.M_tile_inv_64, self.M_tile_64 = self.to_transform_H(64, batch_size)
        self.M_tile_inv_final, self.M_tile_final = self.to_transform_H(crop_size, batch_size)
        self.transform32 = Transform(32, 32,batch_size,self.device).cuda(device=self.device)
        self.transform64 = Transform(64, 64,batch_size,self.device).cuda(device=self.device)
        self.transform128 = Transform(128,128,batch_size,self.device).cuda(device=self.device)
        self.transform_final = Transform(crop_size,crop_size,batch_size,self.device).cuda(device=self.device)
        self.ori= torch.from_numpy ( np.array([[[0.,0.],[1.,0.],[0.,1.],[1.,1.]]]) )
        for i in range(3):
            self.ori=torch.cat((self.ori,self.ori),0)
        self.ori=self.ori.cuda(device=self.device)

        
        
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
    def forward(self, middle_1_f, middle_1_b, ir,vis,vis2ir):
        batch_size = ir.shape[0]
        # print(middle_1_f.shape)
        # print(middle_2_f.shape)
        # print(middle_3_f.shape)
        # print(middle_1_b.shape)
        # print(middle_2_b.shape)
        # print(middle_3_b.shape)
        # time.sleep(1000)
        ############### build_model ###################################
        # input1 = torch.nn.functional.interpolate(ir , [128,128])
        # input2 = torch.nn.functional.interpolate(vis, [128,128])
        # input1 = torch.unsqueeze(torch.mean(input1, axis=1),1)
        # input2 = torch.unsqueeze(torch.mean(input2, axis=1),1)
        batch_size = middle_1_f.shape[0]
        ##############################  feature_extractor ##############################       
        # feature1 = self.feature(input1.to(torch.float32))
        # feature2 = self.feature(input2.to(torch.float32))
        # feature1 = self.feature(middle_1_b, middle_2_b, middle_3_b)
        # feature2 = self.feature(middle_1_f, middle_2_f, middle_3_f)
        # print(middle_1_b.shape)
        # print("hahahahaha")
        # time.sleep(1000)
        feature1 = self.feature(middle_1_b)
        feature2 = self.feature(middle_1_f)
        ##############################  Regression Net 1 ##############################
        # print(feature1[-1].shape)
        # print(feature2[-1].shape)
        # print(feature2[-1].shape)
        # time.sleep(1000)
        net1_f = self.Rnet1(feature1[-1],feature2[-1])
        # net1_f = net1_f * 128.
        net1_f = torch.unsqueeze(net1_f, 2)#*128
        
        # H1 = self.solve_DLT(net1_f/4., 32.)
        H1 = solve_DLT(net1_f/4., 32.)
        H1 = torch.matmul(torch.matmul(self.M_tile_inv_32, H1), self.M_tile_32)
        # net1_f=net1_f.reshape(batch_size,4,2)
        # H1 = transform.get_perspective_transform(self.ori*32.,(((self.ori+net1_f)*32.).clamp(-16,16)).reshape(batch_size,4,2)) 
        # H1 = torch.matmul(torch.matmul(self.M_tile_inv_32, H1.float()), self.M_tile_32)



        feature2_warp = self.transform32(nn.functional.normalize(feature2[-2], dim=1, p=2), H1)
        # ##############################  Regression Net 2 ##############################
        net2_f = self.Rnet2(feature1[-2], feature2_warp)
        net2_f = torch.unsqueeze(net2_f, 2)#*128
        # net2_f=net2_f*128.
        # # print(torch.max(net1_f+net2_f))
        # # H2 = self.solve_DLT((net1_f+net2_f)/2., 64.)
        H2 = solve_DLT((net1_f+net2_f)/2., 64.)
        H2 = torch.matmul(torch.matmul(self.M_tile_inv_64, H2), self.M_tile_64)

        feature3_warp = self.transform64(nn.functional.normalize(feature2[-3], dim=1, p=2), H2)
        # ##############################  Regression Net 3 ##############################
        net3_f = self.Rnet3(feature1[-3], feature3_warp)
        net3_f = torch.unsqueeze(net3_f, 2)#*128
        # net3_f=net3_f*128.
        ###############################################################################
        
        # H1 = self.solve_DLT(net1_f, 128)
        # H2 = self.solve_DLT(net1_f + net2_f, 128)
        # H3 = self.solve_DLT(net1_f + net2_f + net3_f, 128)
        H1 = solve_DLT((net1_f)*self.crop_size/128., self.crop_size)
        H2 = solve_DLT((net1_f + net2_f)*self.crop_size/128., self.crop_size)
        H3 = solve_DLT((net1_f + net2_f + net3_f)*self.crop_size/128., self.crop_size)
 
        # H_final=solve_DLT(net1_f + net2_f + net3_f, self.crop_size)
        H1_mat = torch.matmul(torch.matmul(self.M_tile_inv_final, H1), self.M_tile_final)
        H2_mat = torch.matmul(torch.matmul(self.M_tile_inv_final, H2), self.M_tile_final)
        H3_mat = torch.matmul(torch.matmul(self.M_tile_inv_final, H3), self.M_tile_final)

  
        warp2_H1 = self.transform_final(vis2ir.repeat(1, 3, 1, 1), H1_mat)
        warp2_H2 = self.transform_final(vis2ir.repeat(1, 3, 1, 1), H2_mat)
        warp2_H3 = self.transform_final(vis2ir.repeat(1, 3, 1, 1), H3_mat)
        warp2_vis = self.transform_final(vis, H3_mat)
        warp2_vis2ir = self.transform_final(vis2ir, H3_mat)
        
        one = torch.ones_like(vis)
        one_warp_H1 = self.transform_final(one, H1_mat)*ir
        one_warp_H2 = self.transform_final(one, H2_mat)*ir
        one_warp_H3 = self.transform_final(one, H3_mat)*ir

        warp2=[warp2_H1, warp2_H2, warp2_H3,warp2_vis]
        warp2_fakeir=warp2_vis2ir
        warp1=[one_warp_H1, one_warp_H2, one_warp_H3]
        return net1_f*self.crop_size/128., (net1_f+net2_f)*self.crop_size/128., (net1_f+net2_f+net3_f)*self.crop_size/128., warp1, warp2, warp2_fakeir
        # return net1_f, net1_f, net1_f, warp2_H1, warp2_H1, warp2_H1, one_warp_H1, one_warp_H1, one_warp_H1,warp2_gt, one_warp_gt
