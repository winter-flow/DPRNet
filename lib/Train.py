import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime

from lib.model import IDFNet
from utils.dataloader_0626 import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import cv2
import os
import cv2
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
import py_sod_metrics

import torch
import random
import numpy as np

# 设置随机数种子
# seed = 42
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import warnings
warnings.filterwarnings("ignore")
####
####CUDA_VISIBLE_DEVICES=0 python3 Train.py
####

from torch import nn
from torch.nn.modules.loss import _Loss

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

def val(model, epoch, save_path, writer,model_name):
    """
    validation function
    """
    # global best_mae, best_epoch
    save_path_pth = save_path
    global best_metric, best_epoch
    save_path = './results/' + model_name + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    mask_root = 'xxxx/'
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        test_loader = test_dataset(image_root=opt.test_path + '/Imgs/',
                            gt_root=opt.test_path + '/GT/',
                            testsize=opt.trainsize)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, res1, res2, res3, res4 = model(image)
            res = F.upsample(res[-1] + res1, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            cv2.imwrite(save_path+name,res*255)

            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mask_name_list = sorted(os.listdir(mask_root))
        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        M = MAE()
        for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(save_path, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            FM.step(pred=pred, gt=mask)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            M.step(pred=pred, gt=mask)

        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = M.get_results()["mae"]

        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max(),
        }

        metric = (results["Smeasure"] + results["wFmeasure"] + results["meanEm"])/3
        writer.add_scalar('metric', torch.tensor(metric), global_step=epoch)
        if epoch == 1:
            best_metric = metric
        else:
            # if mae < best_mae:
            if metric > best_metric:
                # best_mae = mae
                best_metric = metric
                best_epoch = epoch
                torch.save(model.state_dict(), save_path_pth + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))

        print('Epoch: {}, sm: {}, wFm: {}, meanE: {}, mae: {}, bestmetric: {}, bestEpoch: {}.'.format(epoch, results["Smeasure"],results["wFmeasure"],results["meanEm"], results["MAE"],best_metric, best_epoch))
        logging.info(
            '[Val Info]:Epoch:{} sm: {} wFm: {} meanE: {} mae: {} bestEpoch:{} bestmetric:{}'.format(epoch, results["Smeasure"],results["wFmeasure"],results["meanEm"], results["MAE"],best_epoch, best_metric))
        
  
def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [1]
    loss_P2_record = AvgMeter()
    # loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record ,loss_5_record= AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    loss_record, loss_1_record, loss_edge_record, loss_ip_record, loss_cp_record ,loss_hel_record, loss_c1_record= AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # print('this is i',i)
            # ---- data prepare ----
            images, gts, edges = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            edges = Variable(edges).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            predict0, predict1, predict2, predict3, predict4= model(images)
            # ---- loss function ----

            loss_1 = structure_loss(predict0, gts)
            loss_2 = structure_loss(predict1, gts)
            loss_3 = structure_loss(predict2, gts)
            loss_4 = structure_loss(predict3, gts)
            loss_5 = structure_loss(predict4, gts)

            loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5
            # ---- backward ----
            # loss.backward()
            loss.backward(retain_graph=True) 
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                loss_1_record.update(loss_1.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_1_record.show()))
            logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_1_record.show()))
    # save model
    save_path = opt.save_path
    if epoch % opt.epoch_save == 0:
        torch.save(model.state_dict(), save_path + str(epoch) + 'IDFNet-PVT.pth')
    
if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0
    ##################model_name#############################
    model_name = 'xxxx'

    ###############################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,default=200, help='epoch number')
    parser.add_argument('--lr', type=float,default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str,default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int,default=40, help='training batch size')
    parser.add_argument('--trainsize', type=int,default=384, help='training dataset size,candidate=352,704,1056')
    parser.add_argument('--clip', type=float,default=0.5, help='gradient clipping margin')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--load_pre', type=str, default='xxxx', help='train from checkpoints')
    parser.add_argument('--decay_rate', type=float,default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,default='xxxx',help='path to train dataset')
    parser.add_argument('--test_path', type=str,default='xxxx',help='path to testing dataset')
    parser.add_argument('--save_path', type=str,default='xxxx/'+model_name+'/')
    parser.add_argument('--epoch_save', type=int,default=1, help='every n epochs to save model')
    opt = parser.parse_args()


    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    logging.basicConfig(filename=opt.save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")


    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = IDFNet().cuda()

    # if (opt.load_pre is not None):
    #     model.load_pre(opt.load_pre)
    #     print('load model from ', opt.load_pre)

    if opt.load is not None:
        pretrained_dict=torch.load(opt.load)
        print('!!!!!!Sucefully load model from!!!!!! ', opt.load)
        load_matched_state_dict(model, pretrained_dict)

    print('model paramters',sum(p.numel() for p in model.parameters() if p.requires_grad))

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    writer = SummaryWriter(opt.save_path + 'summary')

    print("#" * 20, "Start Training", "#" * 20)
    best_mae = 1
    best_epoch = 0
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.save_path)
        if epoch % opt.epoch_save==0:
            val( model, epoch, opt.save_path, writer,model_name)