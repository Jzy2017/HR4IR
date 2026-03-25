from datasets.dataset import MakeTestSet
from invertible_net import Inveritible_Net
from datasets.utls import str2bool
import torch.utils.data as data
import torch, cv2
import argparse
from hr4ir import create_model
import numpy as np
from options.test_options import TestOptions
import cv2

def main(args):
    device = args.device
    crop_size=args.crop_size
    batch_size = args.batch_size
    dataroot = args.root

    opt = TestOptions().parse()  # get options
    opt.gpu_ids=args.device
    opt.crop_size=crop_size
    opt.isTrain=False
    opt.batch_size=batch_size
    # opt.lr=lr

    net = Inveritible_Net()
    net.cuda(device=device)
    
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)


    TestSet = MakeTestSet(dataroot)
    TestLoader = data.DataLoader(TestSet, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

    # net.load_state_dict(torch.load("./checkpoints/roadscene_T.pth",map_location='cpu'))
    # model.netR.load_state_dict(torch.load("./checkpoints/roadscene_R.pth",map_location='cpu'))
    net.load_state_dict(torch.load("./checkpoints/tno_T.pth",map_location='cpu'))
    model.netR.load_state_dict(torch.load("./checkpoints/tno_R.pth",map_location='cpu'))

    model.set_invert_model(net)

    tmp = filter(lambda x: x.requires_grad, net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', str(num))



    for _, (tensor_g_ori, tensor_c_ori,tensor_g, tensor_c,name) in enumerate(TestLoader):
        print(name[0])
        tensor_g.requires_grad = False
        tensor_c.requires_grad = False
        model.set_test_input(tensor_c_ori,tensor_g_ori,tensor_c,tensor_g)
        warp2 = model.forward()[0].detach().cpu().permute(1,2,0).numpy()*255
        warp2 = cv2.cvtColor(warp2,cv2.COLOR_RGB2BGR)

        cv2.imwrite('results/roadscene/'  + name[0].split('.')[0] + '.png',warp2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # For Dataset and Record
    parser.add_argument("--stride", type=int, default=20, help='the stride for saving models')
    parser.add_argument("--paired", type=str2bool, default=True, help="paired or unpaired")
    parser.add_argument("--root", type=str, default="benchmarks/roadscene_dataset/input_vis",
                        help="data root")
    parser.add_argument("--load_checkpoint", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beginner", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=320, help='float(height=width) or tuple(width, height)')
    parser.add_argument('--device', type=int, default=2)
    args = parser.parse_args()

    main(args)
