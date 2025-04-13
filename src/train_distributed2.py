#!/usr/bin/python3
# coding=utf-8

import sys
import datetime
import argparse
import os

import torch.utils
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
from FRINet_deit import FRINet
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from utils.lr_scheduler import LR_Scheduler
from utils.log import TBRecorder
import utils.metrics as Measure
import utils.distributed
from tqdm import tqdm
import torch.distributed as dist
import cv2
import numpy as np
torch.cuda.empty_cache()
# os.environ['MASTER_PORT'] = '29501'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

def get_coef(iter_percentage=0.5, method="cos", milestones=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = 0, 1

        ual_coef = 1.0
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            if method == "linear":
                ratio = (max_coef - min_coef) / (max_point - min_point)
                ual_coef = ratio * (iter_percentage - min_point)
            elif method == "cos":
                perc = (iter_percentage - min_point) / (max_point - min_point)
                normalized_coef = (1 - np.cos(perc * np.pi)) / 2
                ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
        return ual_coef

def window_partition(x,window_size):
    # input B C H W
    x = x.permute(0,2,3,1)
    B,H,W,C = x.shape
    x = x.view(B,H//window_size,window_size,W//window_size,window_size,C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C)
    return windows #B_ H_ W_ C

def bce_iou_loss(pred, mask,weight=None):
    size = pred.size()[2:]
    mask = F.interpolate(mask,size=size, mode='bilinear')
    if weight is not None:
        wbce = F.binary_cross_entropy_with_logits(pred, mask,weight=1+weight)
    else:
        wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def mse(pred_re,img_gray):
    img_gray= F.interpolate(img_gray,size=pred_re.size()[2:], mode='bilinear')
    return F.mse_loss(pred_re,img_gray)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=4, type=int)
    parser.add_argument('--savepath', default="/data3/YG/FRINet/code/model/CAMO", type=str)
    parser.add_argument('--datapath', default="/data3/YG/FRINet/code/data/CAMO/train", type=str)
    # parser.add_argument('--datapath', default="../data/TrainDataset", type=str)
    parser.add_argument('--checkpoint',default=None,type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epoch', default=250, type=int)
    parser.add_argument('--wd', default=0.0005, type=float)
    parser.add_argument('--fr', default=0.025, type=float)
    parser.add_argument('--syncBN', type=bool, default=True)

    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.parse_args()
    return parser.parse_args()

def main(tr_cfg, te_cfg, tr_dl, te_dl, tr_sampler, net, args):
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()
    MF = Measure.FbetaMeasure()
    ADAF = Measure.AdaptiveFbetaMeasure()

    paralist=[[{'params':[]}] for _ in range(14)]
    namelist=[[{'params':[]}] for _ in range(14)]
    para = sum(paralist,[])
    for name, param in net.named_parameters():
        if 'bkbone' in name and 'bkbone.blocks' not in name:
            namelist[13][0]['params'].append(name)
            paralist[13][0]['params'].append(param)
        elif 'bkbone.blocks.0' in name:
            paralist[12][0]['params'].append(param)
        elif 'bkbone.blocks.1' in name:
            paralist[11][0]['params'].append(param)
        elif 'bkbone.blocks.2' in name:
            paralist[10][0]['params'].append(param)
        elif 'bkbone.blocks.3' in name:
            paralist[9][0]['params'].append(param)
        elif 'bkbone.blocks.4' in name:
            paralist[8][0]['params'].append(param)
        elif 'bkbone.blocks.5' in name:
            paralist[7][0]['params'].append(param)
        elif 'bkbone.blocks.6' in name:
            paralist[6][0]['params'].append(param)
        elif 'bkbone.blocks.7' in name:
            paralist[5][0]['params'].append(param)
        elif 'bkbone.blocks.8' in name:
            paralist[4][0]['params'].append(param)
        elif 'bkbone.blocks.9' in name:
            paralist[3][0]['params'].append(param)
        elif 'bkbone.blocks.10' in name:
            paralist[2][0]['params'].append(param)
        elif 'bkbone.blocks.11' in name:
            paralist[1][0]['params'].append(param)
        else:
            paralist[0][0]['params'].append(param)

    optimizer = torch.optim.SGD(para, lr=tr_cfg.lr, momentum=tr_cfg.momen,
                                weight_decay=tr_cfg.decay, nesterov=True)

    scheduler = LR_Scheduler('cos',tr_cfg.lr,tr_cfg.epoch,len(tr_dl),warmup_epochs=20,fr=args.fr)
    # net = net.cuda()
    
    for epoch in range(tr_cfg.epoch):
        tr_sampler.set_epoch(epoch)
        if utils.distributed.is_main_process():
            tr_dl = tqdm(tr_dl)
        net.train()
        for step, (image, RFC, mask,_) in enumerate(tr_dl):
            image, RFC, mask = image.float().to(device),  RFC.float().to(device),mask.float().to(device)
            # print(image.shape) # torch.Size([4, 3, 384, 384])
            # print(RFC.shape) # torch.Size([4, 12, 384, 384])
            optimizer.zero_grad()
            scheduler(optimizer,step,epoch)
            
            p2,p3,p4,p5,re4,re8,re16= net(image,RFC)
            # print(p2.shape, p3.shape, p4.shape, p5.shape, re4.shape, re8.shape, re16.shape)
            prob_p2 = p2.sigmoid()
            prob_p3 = p3.sigmoid()
            prob_p4 = p4.sigmoid()
            prob_p5 = p5.sigmoid()
            ual_coef_p2 = get_coef(method="cos", milestones=(0, 1))
            ual_loss_p2 = ual_coef_p2 * (1 - (2 * prob_p2 - 1).abs().pow(2)).mean()
            ual_coef_p3 = get_coef(method="cos", milestones=(0, 1))
            ual_loss_p3 = ual_coef_p3 * (1 - (2 * prob_p3 - 1).abs().pow(2)).mean()
            ual_coef_p4 = get_coef(method="cos", milestones=(0, 1))
            ual_loss_p4 = ual_coef_p4 * (1 - (2 * prob_p4 - 1).abs().pow(2)).mean()
            ual_coef_p5 = get_coef(method="cos", milestones=(0, 1))
            ual_loss_p5 = ual_coef_p5 * (1 - (2 * prob_p5 - 1).abs().pow(2)).mean()

            ls2 = bce_iou_loss(p2,mask)
            ls3 = bce_iou_loss(p3,mask)
            ls4 = bce_iou_loss(p4,mask)
            ls5 = bce_iou_loss(p5,mask)
            ls_b = (mse(re4,image)+mse(re8,image)+mse(re16,image))*0.05
            loss = (ls2+ls3*0.5+ls4*0.25+ls5*0.125)+ls_b+(0.1*ual_coef_p2+0.1*ual_coef_p3+0.1*ual_coef_p4+0.1*ual_coef_p5)
                
            loss.backward()
            optimizer.step()
        
        dist.barrier()
        
        net.eval()
        with torch.no_grad():
            for step, (image, hf, mask, shape, name) in enumerate(te_dl):
                print(name)
                image = image.to(device).float()
                mask  = mask[0].cpu().numpy()*255
                hf    = hf.to(device).float()
                p = net(image,hf)
                # print(p[0].shape) # torch.Size([1, 1, 448, 448])
                out_resize   = F.interpolate(p[0],size=shape, mode='bilinear')
                pred   = torch.sigmoid(out_resize[0,0])
                pred  = (pred*255).detach().cpu().numpy()

                # res = (pred*255).detach().cpu().numpy()
                res = p[0].sigmoid().detach().squeeze().cpu().numpy()
                # print(res.shape, mask.shape)
                # print(res.shape, mask.shape)
                WFM.step(pred=res*255, gt=mask*255)
                SM.step(pred=res*255, gt=mask*255)
                EM.step(pred=res*255, gt=mask*255)
                MAE.step(pred=res*255, gt=mask*255)
                MF.step(pred=res*255, gt=mask*255)
                ADAF.step(pred=res*255, gt=mask*255)

            sm1 = SM.get_results()['sm'].round(3)
            adpem1 = EM.get_results()['em']['adp'].round(3)
            wfm1 = WFM.get_results()['wfm'].round(3)
            mae1 = MAE.get_results()['mae'].round(3)
            mean_f_beta = MF.get_results()['mean_fbeta'].round(3)
            ada_f_beta = ADAF.get_results()['ada_fbeta'].round(3)

            sm1 = utils.distributed.reduce_value(torch.tensor(sm1).to(device).to_dense(), average=True).item()
            adpem1 = utils.distributed.reduce_value(torch.tensor(adpem1).to(device).to_dense(), average=True).item()
            mean_f_beta = utils.distributed.reduce_value(torch.tensor(mean_f_beta).to(device).to_dense(), average=True).item()
            wfm1 = utils.distributed.reduce_value(torch.tensor(wfm1).to(device).to_dense(), average=True).item()
            mae1 = utils.distributed.reduce_value(torch.tensor(mae1).to(device).to_dense(), average=True).item()
            ada_f_beta = utils.distributed.reduce_value(torch.tensor(ada_f_beta).to(device).to_dense(), average=True).item()
            if args.rank == 0:
                with open("/data3/YG/FRINet/code/log/CAMO/output.txt", "a") as f:
                        f.write(f"mae1:{mae1} sm1:{sm1} adpem1:{adpem1} mean_f_beta:{mean_f_beta} ada_f_beta:{ada_f_beta} wm:{wfm1}\n")

                torch.save(net.state_dict(), tr_cfg.savepath + '/model-' + str(epoch + 1))
        # if args.rank == 0:
        #     net.eval()
        #     with torch.no_grad():
        #         for image, hf, mask, shape, name in te_dl:
        #             image = image.to(device).float()
        #             mask  = mask[0].cpu().numpy()*255
        #             hf    = hf.to(device).float()
        #             p = net(image,hf)
        #             # print(p[0].shape) # torch.Size([1, 1, 448, 448])
        #             out_resize   = F.interpolate(p[0],size=shape, mode='bilinear')
        #             pred   = torch.sigmoid(out_resize[0,0])
        #             pred  = (pred*255).detach().cpu().numpy()

        #             # res = (pred*255).detach().cpu().numpy()
        #             res = p[0].sigmoid().detach().squeeze().cpu().numpy()
        #             # print(res.shape, mask.shape)
        #             # print(res.shape, mask.shape)
        #             WFM.step(pred=res*255, gt=mask*255)
        #             SM.step(pred=res*255, gt=mask*255)
        #             EM.step(pred=res*255, gt=mask*255)
        #             MAE.step(pred=res*255, gt=mask*255)
        #             MF.step(pred=res*255, gt=mask*255)
        #             ADAF.step(pred=res*255, gt=mask*255)

        #         sm1 = SM.get_results()['sm'].round(3)
        #         adpem1 = EM.get_results()['em']['adp'].round(3)
        #         wfm1 = WFM.get_results()['wfm'].round(3)
        #         mae1 = MAE.get_results()['mae'].round(3)
        #         mean_f_beta = MF.get_results()['mean_fbeta'].round(3)
        #         ada_f_beta = ADAF.get_results()['ada_fbeta'].round(3)

        # sm1 = utils.distributed.reduce_value(torch.tensor(sm1).to(device).to_dense(), average=True).item()
        # adpem1 = utils.distributed.reduce_value(torch.tensor(adpem1).to(device).to_dense(), average=True).item()
        # mean_f_beta = utils.distributed.reduce_value(torch.tensor(mean_f_beta).to(device).to_dense(), average=True).item()
        # # wfm1 = utils.distributed.reduce_value(torch.tensor(wfm1).to(device).to_dense(), average=True).item()
        # mae1 = utils.distributed.reduce_value(torch.tensor(mae1).to(device).to_dense(), average=True).item()
        # ada_f_beta = utils.distributed.reduce_value(torch.tensor(ada_f_beta).to(device).to_dense(), average=True).item()
        
        dist.barrier()

    if args.rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    utils.distributed.cleanup()

if __name__ == '__main__': 
    args = parser()
    device = torch.device(args.device)
    
    args.lr *= args.world_size # 学习率要根据并行GPU的数量进行倍增
    utils.distributed.init_distributed_mode(args=args)

    tr_cfg = dataset.Config(datapath=args.datapath, savepath=args.savepath,mode='train', batch=args.batchsize, lr=args.lr, momen=0.9,
                         decay=args.wd, epoch=args.epoch, snapshot=args.checkpoint)
    tr_ds = dataset.Data(tr_cfg)
    te_cfg = dataset.Config(datapath='/data3/YG/FRINet/code/data/CAMO/test', mode='test')
    te_ds= dataset.Data(te_cfg)

    # 实现分配数据
    tr_sampler = torch.utils.data.distributed.DistributedSampler(tr_ds)
    te_sampler = torch.utils.data.distributed.DistributedSampler(te_ds)

    tr_batch_sampler = torch.utils.data.BatchSampler(tr_sampler, args.batchsize, drop_last=True)
    te_batch_sampler = torch.utils.data.BatchSampler(te_sampler, 1, drop_last=True)

    if args.rank == 0:
            print('<<=========Start enabling distributed training=========>>')
            
    tr_dl = DataLoader( tr_ds,
                        batch_sampler=tr_batch_sampler,
                        num_workers=8,
                        pin_memory=True,
                        collate_fn=tr_ds.collate
                    )

    # te_dl = DataLoader(te_ds, batch_size=1, shuffle=False, num_workers=0)
    te_dl = DataLoader( te_ds,
                        batch_sampler=te_batch_sampler,
                        num_workers=8,
                        pin_memory=True
                    )
    model = FRINet(tr_cfg).to(device)
    checkpoint_path = checkpoint_path = os.path.join('/data3/YG/FRINet/code/model/tmp', "initial_weights.pt")
    if args.rank == 0:
        torch.save(model.state_dict(), checkpoint_path)
    dist.barrier()
    # checkpoint_path = '/data3/YG/FRINet/code/model/2/model-150'
    # dist.barrier()

    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

    if args.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    main(tr_cfg, te_cfg, tr_dl, te_dl, tr_sampler, model, args)

    # main(dataset, FRINet)
