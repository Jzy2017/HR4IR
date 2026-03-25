import torch
import hr4ir.stn as stn
import hr4ir.pwc as pwc
from util.tb_visualizer import TensorboardVisualizer
from . import networks
from .base_model import BaseModel
from utlz import Quantization
import torch.nn as nn
quantize = Quantization()
#repeat
# loss_dist = nn.MSELoss(reduction="sum")
class HR4IRModel(BaseModel):
    """
    NeMAR: a neural multimodal adversarial image registration network.
    This class train a registration network and a geometry preserving translation network network. This is done
    using three networks:

    netT - A translation network that translates from modality A --to--> modality B (by default a
    netR - A registration network that applies geometric transformation to spatially align modality A --with--> modality B
    netD - Adversarial network that discriminates between fake an real images.

    Official implementation of:
    Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation paper
    https://arxiv.org/abs/2003.08073

    Inspired by the implementation of pix2pix:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Modify the command line."""
        parser = stn.modify_commandline_options(parser, is_train)
        if is_train:
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='Weight for the GAN loss.')
            parser.add_argument('--lambda_recon', type=float, default=100.0,
                                help='Weight for the L1 reconstruction loss.')
            parser.add_argument('--lambda_smooth', type=float, default=0.0, help='Regularization term used by the STN')
            parser.add_argument('--enable_tbvis', action='store_true',
                                help='Enable tensorboard visualizer (default : False)')
            parser.add_argument('--multi_resolution', type=int, default=1,
                                help='Use of multi-resolution discriminator.'
                                     '(if equals to 1 then no multi-resolution training is applied)')
            TensorboardVisualizer.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # Setup the visualizers
        self.train_stn = True
        # self.setup_visualizers()
        # if self.isTrain and opt.enable_tbvis:
        #     self.tb_visualizer = TensorboardVisualizer(self, ['netR', 'netT', 'netD'], self.loss_names, self.opt)
        # else:
        #     self.tb_visualizer = None
        self.define_networks()
        # if self.tb_visualizer is not None:
        #     print('Enabling Tensorboard Visualizer!')
        #     self.tb_visualizer.enable()
        # print(self.isTrain)
        if self.isTrain:
            # define loss functions
            self.loss_cons = ConsistencyLoss(device=str(self.device), img_shape=(opt.batch_size, 1, opt.crop_size, opt.crop_size), c_weight=1)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.restore_loss = nn.MSELoss(reduction="sum")
            self.setup_optimizers()



    def define_networks(self):
        opt = self.opt
        self.netR = pwc.H_estimator(opt.batch_size,self.opt.crop_size,opt.gpu_ids,self.isTrain)
        self.netR = self.netR.cuda(device=opt.gpu_ids)


    def reset_weights(self):
        opt = self.opt
        networks.init_weights(self.netT, opt.init_type, opt.init_gain)


    
    def set_invert_model(self, net):
        self.netT=net
        for p in self.netT.parameters():
            p.requires_grad = False
        self.netT.eval()

    

    def set_input(self, vis,ir,gt):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        if AtoB:
            self.real_A = vis.to(self.device)
            self.real_B = ir.to(self.device)
            self.gt = gt.to(self.device)
            # self.image_paths = input['A_paths']
        else:
            self.real_A = ir.to(self.device)
            self.real_B = vis.to(self.device)
            self.gt = gt.to(self.device)
            # self.image_paths = input['B_paths']
        # self.gt=(gt).to(self.device)
    def set_test_input(self, vis_ori,ir_ori,vis,ir):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        if AtoB:
            self.real_A = vis.to(self.device)
            self.real_B = ir.to(self.device)
            self.real_A_ori = vis_ori.to(self.device)
            self.real_B_ori = ir_ori.to(self.device)
            # self.image_paths = input['A_paths']
        else:
            self.real_A = ir.to(self.device)
            self.real_B = vis.to(self.device)
            self.real_A_ori = ir_ori.to(self.device)
            self.real_B_ori = vis_ori.to(self.device)


    # def forward(self):

    #     #过去了得到假红外
    #     with torch.no_grad():
    #         tensor_x, self.middle_1_f, self.middle_2_f, self.middle_3_f = self.netT(x=[self.real_A], rev=False)

    #         self.tensor_fake_B = tensor_x[:, [0], :, :]
    #         self.fake_B = quantize(self.tensor_fake_B)# 假红外

    #         #过去了再回来得到恢复后的原可见光
    #         self.tensor_z_ = torch.randn_like(tensor_x[:, 1:, :, :]).to(tensor_x[:, 1:, :, :].device)

    #         tensor_x_ = torch.cat((self.tensor_fake_B, self.tensor_z_), dim=1)
    #         self.tensor_back_A,_,_,_ = self.netT(x=[tensor_x_], rev=True)
    #         self.back_A = quantize(self.tensor_back_A) # 从假红外返回为原始的可见光图像

    #         #往回走得到假可见光
    #         print(self.real_B.max())
    #         tensor_realir_ = torch.cat((self.real_B, self.tensor_z_), dim=1)
    #         self.tensor_fake_A, self.middle_1_b, self.middle_2_b, self.middle_3_b = self.netT(x=[tensor_realir_], rev=True)
    #         # [1, 3, 128, 128] [1, 24, 64, 64] [1, 96, 32, 32] [1, 384, 16, 16]
            
    #         self.fake_A = quantize(self.tensor_fake_A)

    #     #得到配准后的可见光和配准后的假红外
    #     self.offset1, self.offset2, self.offset3, self.warp1,self.warp2, self.tensor_fake_RT_B= self.netR(self.middle_1_f, self.middle_2_f, self.middle_3_f, self.middle_1_b, self.middle_2_b, self.middle_3_b,\
    #                                                                                                self.real_B.repeat(1, 3, 1, 1),self.real_A,self.tensor_fake_B,self.gt)#用假红外input2和真红外进行估计仿射参数并变形得到warped_image
    #     self.registered_real_A = self.warp2[3]
    #     self.warp_gt = self.gt
    #     self.fake_RT_B = quantize(self.tensor_fake_RT_B)
    #     # self.stn_reg_term = reg_term
 
    #     #将配准后的可见光转化为假红外
    #     tensor_x2,_,_,_ = self.netT(x=[self.registered_real_A], rev=False)
    #     self.tensor_fake_TR_B = tensor_x2[:, [0], :, :]
    #     self.fake_TR_B = quantize(self.tensor_fake_TR_B)

    # def backward_T_and_R(self):
    #     """Calculate GAN and L1 loss for the translation and registration networks."""
    #     # 图像恢复loss
    #     self.loss_restore = self.restore_loss(self.tensor_back_A,self.real_A)

    #     #配准loss
    #     # self.loss_L1_RT_pwc = 32*self.criterionL1(self.warp1[0], self.warp2[0])+\
    #     #                     16*self.criterionL1(self.warp1[1], self.warp2[1])+\
    #     #                     8 *self.criterionL1(self.warp1[2], self.warp2[2])
    #     self.loss_L1_RT_pwc = 0
    #     self.loss_L2_gt = 0.1* self.criterionL2(self.offset1, self.gt)+\
    #                     0.05* self.criterionL2(self.offset2, self.gt)+\
    #                     0.01* self.criterionL2(self.offset3, self.gt)
     
        
    #     self.loss_GAN_TR=0
    #     self.loss_GAN_RT=0
    #     # loss = self.loss_L1_TR + self.loss_L1_RT + self.loss_GAN_TR + self.loss_GAN_RT +self.loss_L1_RT_pwc+self.loss_L2_gt+ 0.1*self.loss_fakeA + self.loss_L1_fakeA
    #     # loss = loss+ 0.5*(0.01*self.loss_invt_RT+0.01*self.loss_invt_TR+  self.loss_restore) 
    #     loss = self.loss_L2_gt
    #     loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(self.netT.parameters(), 10)
    #     return loss



    # def get_current_visuals(self):
    #     return self.real_A, self.real_B.repeat(1, 3, 1, 1), self.fake_B.repeat(1, 3, 1, 1), self.fake_A,self.fake_TR_B.repeat(1, 3, 1, 1),self.fake_RT_B.repeat(1, 3, 1, 1),self.registered_real_A,self.warp_gt,self.warp1,self.warp2,self.back_A



    def set_input_test(self, vis,ir, vis_resize,ir_resize):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        if AtoB:
            self.real_A = vis_resize.to(self.device)
            self.real_B = ir_resize.to(self.device)
            self.real_A_ = vis.to(self.device)
            self.real_B_ = ir.to(self.device)
        else:
            self.real_A = ir_resize.to(self.device)
            self.real_B = vis_resize.to(self.device)
            self.real_A_ = ir.to(self.device)
            self.real_B_ = vis.to(self.device)

        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        '''
        tensor_c=self.real_A
        tensor_g=self.real_B
        tensor_prg=self.fake_B

        '''
        with torch.no_grad():
            # self.netR.eval()
            tensor_x, self.middle_1_f, self.middle_2_f, self.middle_3_f = self.netT(x=[self.real_A], rev=False)
            
            self.fake_B = tensor_x[:, [0], :, :]
            tensor_z = tensor_x[:, 1:, :, :]
            

            self.tensor_z_ = torch.randn_like(tensor_z).to(tensor_z.device)


            tensor_x_ = torch.cat((self.fake_B, self.tensor_z_), dim=1)
            self.tensor_y,_,_,_ = self.netT(x=[tensor_x_], rev=True)

            tensor_realir_ = torch.cat((self.real_B, self.tensor_z_), dim=1)
            self.tensor_fake_A, self.middle_1_b, self.middle_2_b, self.middle_3_b = self.netT(x=[tensor_realir_], rev=True)

            self.warp2= self.netR(self.middle_1_f, self.middle_2_f, self.middle_3_f, \
                 self.middle_1_b, self.middle_2_b, self.middle_3_b, self.real_A_ori)
        return self.warp2

        
    # def get_current_visuals_test(self):
    #     return self.real_A, self.real_B.repeat(1, 3, 1, 1), quantize(self.tensor_fake_A),quantize(self.fake_B).repeat(1, 3, 1, 1), quantize(self.tensor_y),quantize(self.fake_RT_B).repeat(1, 3, 1, 1),self.registered_real_A,self.warp_gt,self.warp1,self.warp2,quantize(self.fake_B_ori)