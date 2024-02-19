from datasets.dataset import MakeTrainSet, Kodak24,MakeValidSet
from utlz import Quantization
from invertible_net import Inveritible_Decolorization
from datasets.utls import str2bool
import torch.utils.data as data
from criteria import ConsistencyLoss
from torch.optim import Adam
import os, sys, time, math
import torch, cv2
import logging
import argparse
from nemar import create_model
import numpy as np
import torch.nn as nn
from termcolor import colored
# from mmcv.utils import get_logger
from options.train_options import TrainOptions
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2
# from visdom import Visdom
#128
use_visdom=False

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main(args):
    beginner = args.beginner
    stride = args.stride
    device = args.device
    crop_size=args.crop_size
    batch_size = args.batch_size
    epoch = args.Epoch
    dataroot = args.root
    crop_size = args.crop_size
    lr = args.lr
    weight_decay = args.weight_decay
    model_path = "./models/"

    opt = TrainOptions().parse()  # get training options
    opt.gpu_ids=args.device
    opt.crop_size=crop_size
    opt.batch_size=batch_size
    opt.lr=lr
    # opt.crop_size=crop_size
    # print(opt)
    net = Inveritible_Decolorization()
    # optimizer = Adam(net.parameters(), weight_decay=weight_decay, betas=(0.5, 0.999), lr=lr)
    net.cuda(device=device)
    
    nemar_model = create_model(opt)  # create a model given opt.model and other options
    # print(dir(nemar_model))
    nemar_model.setup(opt)
    # visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = os.path.join(model_path, f'{timestamp}.log')
    # logger = get_logger(name='IDN', log_file=log_file, log_level=logging.INFO)
    # logger.info(f"batch size {batch_size}")

    TestSet = MakeValidSet(dataroot)
    TestLoader = data.DataLoader(TestSet, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)
    # ValidSet = Kodak24(dataroot)
    # ValidSe= MakeValidSet(dataroot)
    # ValidLoadert  = data.DataLoader(ValidSet, batch_size=8, num_workers=1, shuffle=False, pin_memory=True)

    # logger.info("# Training Samples: " + str(TestSet.__len__()) )
    if args.load_checkpoint:
        # logger.info("loading checkpoint...")
        net.load_state_dict(torch.load("./models/net_epoch_025.pth",map_location='cpu'))
        nemar_model.netR.load_state_dict(torch.load("./models/net_epoch_025_R.pth",map_location='cpu'))
        # nemar_model.netD.load_state_dict(torch.load("./models_new_8/net_epoch_023_D.pth"))
        print("######### load done ####################")
    # time.sleep(10000)

    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    
    # net.train(True)
    nemar_model.set_invert_model(net)
    quantize = Quantization()
    # loss_cons = ConsistencyLoss(device="cuda:"+str(device), img_shape=(batch_size, 1, 128, 128), c_weight=args.c_weight)
    # loss_cons = ConsistencyLoss(device="cuda:"+str(device), img_shape=(batch_size, 1, crop_size, crop_size), c_weight=args.c_weight)
    # loss_dist = nn.MSELoss(reduction="sum")

    tmp = filter(lambda x: x.requires_grad, net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', str(num))

    BestPSNR = 0.
    
    # use warmup to stablize the training, or it may not converage; set a smaller r_weight at the beginning may help training.
    net.eval()
    # nemar_model.eval()

    # print("hahah")
    for batch_idx, (tensor_g, tensor_c,gt,name) in enumerate(TestLoader):
        print(batch_idx)
        n, _, h, w = tensor_c.size()
        # print(tensor_c.shape)
        # print(name)
        # time.sleep(1000)
        tensor_g.requires_grad = False
        tensor_c.requires_grad = False
        nemar_model.set_input(tensor_c,tensor_g,gt)
        nemar_model.forward_test()
        

        # trainSamples += n

        
        warp2= nemar_model.get_current_visuals_test()

        warp2_ours = warp2[-1][0].detach().cpu().permute(1,2,0).numpy()*255
        # print(os.path.join('test/road/output/',name[0]))
        cv2.imwrite('test/road/output/'+name[0],warp2_ours)
 
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # For Dataset and Record
    parser.add_argument("--stride", type=int, default=20, help='the stride for saving models')
    parser.add_argument("--paired", type=str2bool, default=True, help="paired or unpaired")
    # parser.add_argument("--root", type=str, default="../../roadscene_ablation/input_vis",
    #                     help="data root")
    parser.add_argument("--root", type=str, default="test/road/input_vis",
                        help="data root")
    # For Training
    parser.add_argument("--load_checkpoint", type=str2bool, default=True)
    # 'models/net_epoch_160.pth'
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beginner", type=int, default=0)
    parser.add_argument('--Epoch', type=int, default=100)
    parser.add_argument("--c_weight",default=1,  type=float)
    parser.add_argument("--s_weight",default=1,  type=float)
    parser.add_argument("--r_weight",default=1, type=float)
    parser.add_argument("--g_weight",default=1, type=float)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--crop_size", type=int, default=256)
    
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    main(args)
