import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2


from lib.model import IDFNet
from utils.dataloader_0626 import My_test_dataset
import warnings
warnings.filterwarnings("ignore")

# CUDA_VISIBLE_DEVICES=0
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size default 352')
parser.add_argument('--pth_path', type=str, default='xx.pth')
opt = parser.parse_args()


for _data_name in ['ORSSD']:
    data_path = 'xxxx'
    save_path = 'xxxx'
    model = IDFNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = My_test_dataset(image_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1, res2, res3, res4 = model(image)

        res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        cv2.imwrite(save_path+name,res*255)
    print('finished!!')