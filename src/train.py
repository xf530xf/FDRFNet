#!/usr/bin/python3
# coding=utf-8

import sys
import datetime
import argparse
import os
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
from FDRFNet_deit import FDRFNet
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from utils.lr_scheduler import LR_Scheduler
from utils.log import TBRecorder
import utils.metrics as Measure
import utils.distributed
from tqdm import tqdm

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
    parser.add_argument('--batchsize', default=8, type=int)
    parser.add_argument('--savepath', default="/data3/YG/FDRFNet/code/model", type=str)
    parser.add_argument('--datapath', default="/data3/YG/FDRFNet/code/data/COD10K/train", type=str)
    # parser.add_argument('--datapath', default="../data/TrainDataset", type=str)
    parser.add_argument('--checkpoint',default=None,type=str)
    parser.add_argument('--lr', default=0.06, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--wd', default=0.0005, type=float)
    parser.add_argument('--fr', default=0.025, type=float)
    parser.add_argument('--syncBN', type=bool, default=True)

    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.parse_args()
    return parser.parse_args()

# def main(Dataset, Network):
#     # dataset
#     args = parser()
#     print(torch.cuda.device_count())
#     cfg = Dataset.Config(datapath=args.datapath, savepath=args.savepath,mode='train', batch=args.batchsize, lr=args.lr, momen=0.9,
#                          decay=args.wd, epoch=args.epoch, snapshot=args.checkpoint)
#     data = Dataset.Data(cfg)
#     loader = torch.utils.data.DataLoader(data,
#                                          batch_size=args.batchsize,
#                                          shuffle=True,
#                                          num_workers=8,
#                                          pin_memory=True,
#                                          collate_fn=data.collate)

#     net = Network(cfg)
#     ids=[0]
#     net = nn.DataParallel(net,device_ids=ids)
#     tb = TBRecorder('./tb/')
#     net.train(True)
#     paralist=[[{'params':[]}] for _ in range(14)]
#     namelist=[[{'params':[]}] for _ in range(14)]
#     para = sum(paralist,[])
#     for name, param in net.named_parameters():
#         if 'bkbone' in name and 'bkbone.blocks' not in name:
#             namelist[13][0]['params'].append(name)
#             paralist[13][0]['params'].append(param)
#         elif 'bkbone.blocks.0' in name:
#             paralist[12][0]['params'].append(param)
#         elif 'bkbone.blocks.1' in name:
#             paralist[11][0]['params'].append(param)
#         elif 'bkbone.blocks.2' in name:
#             paralist[10][0]['params'].append(param)
#         elif 'bkbone.blocks.3' in name:
#             paralist[9][0]['params'].append(param)
#         elif 'bkbone.blocks.4' in name:
#             paralist[8][0]['params'].append(param)
#         elif 'bkbone.blocks.5' in name:
#             paralist[7][0]['params'].append(param)
#         elif 'bkbone.blocks.6' in name:
#             paralist[6][0]['params'].append(param)
#         elif 'bkbone.blocks.7' in name:
#             paralist[5][0]['params'].append(param)
#         elif 'bkbone.blocks.8' in name:
#             paralist[4][0]['params'].append(param)
#         elif 'bkbone.blocks.9' in name:
#             paralist[3][0]['params'].append(param)
#         elif 'bkbone.blocks.10' in name:
#             paralist[2][0]['params'].append(param)
#         elif 'bkbone.blocks.11' in name:
#             paralist[1][0]['params'].append(param)
#         else:
#             paralist[0][0]['params'].append(param)

#     optimizer = torch.optim.SGD(para, lr=cfg.lr, momentum=cfg.momen,
#                                 weight_decay=cfg.decay, nesterov=True)

#     scheduler = LR_Scheduler('cos',cfg.lr,cfg.epoch,len(loader),warmup_epochs=20,fr=args.fr)
#     net = net.cuda()
    
#     global_step = 0
#     scaler = amp.GradScaler(True)

#     for epoch in range(cfg.epoch):
#         net.train()

#         for step, (image, RFC, mask,_) in enumerate(loader):
#             image, RFC, mask = image.float().cuda(),  RFC.float().cuda(),mask.float().cuda()
#             optimizer.zero_grad()      
#             scheduler(optimizer,step,epoch)
#             with amp.autocast(True):
#                 p2,p3,p4,p5,re4,re8,re16= net(image,RFC)
#                 ls2 = bce_iou_loss(p2,mask)
#                 ls3 = bce_iou_loss(p3,mask)
#                 ls4 = bce_iou_loss(p4,mask)
#                 ls5 = bce_iou_loss(p5,mask)
#                 ls_b = (mse(re4,image)+mse(re8,image)+mse(re16,image))*0.05
#                 loss = (ls2+ls3*0.5+ls4*0.25+ls5*0.125)+ls_b
                
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()       
            
            
#             global_step += 1
#             if step % 60 == 0 :
#                 tb.record_curve('loss',{'ls2':ls2.item(),'ls3':ls3.item(),'ls4':ls4.item(),'ls5':ls5.item()},global_step)
#                 tb.record_curve('lr',{'lr':optimizer.param_groups[1]['lr']},global_step)
#                 print('%s | step:%d/%d | lr=%.6f  loss=%.6f ' % (datetime.datetime.now(), epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],loss.item()))
#                 print('%s |   ls2=%.6f ls3=%.6f ls4=%.6f ls5=%.6f ls_in=%.6f' % (datetime.datetime.now(),ls2.item(),ls3.item(),ls4.item(),ls5.item(),0))

#         if epoch==cfg.epoch-1:
#             if not os.path.exists(cfg.savepath):
#                 os.makedirs(cfg.savepath)
#             torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))


def main(tr_cfg, te_cfg, tr_dl, te_dl, net):
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()

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
    net = net.cuda()
    
    scaler = amp.GradScaler(True)

    for epoch in range(tr_cfg.epoch):
        net.train()

        for step, (image, RFC, mask,_) in enumerate(tqdm(tr_dl)):
            image, RFC, mask = image.float().cuda(),  RFC.float().cuda(),mask.float().cuda()
            optimizer.zero_grad()      
            scheduler(optimizer,step,epoch)
            with amp.autocast(True):
                p2,p3,p4,p5,re4,re8,re16= net(image,RFC)
                ls2 = bce_iou_loss(p2,mask)
                ls3 = bce_iou_loss(p3,mask)
                ls4 = bce_iou_loss(p4,mask)
                ls5 = bce_iou_loss(p5,mask)
                ls_b = (mse(re4,image)+mse(re8,image)+mse(re16,image))*0.05
                loss = (ls2+ls3*0.5+ls4*0.25+ls5*0.125)+ls_b
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()

        net.eval()
        for image, hf, mask, shape, name in te_dl:
            image = image.cuda().float()
            mask  = mask[0].cpu().numpy()*255
            hf    = hf.cuda().float()      
            p = net(image,hf)
            # print(p[0].shape) # torch.Size([1, 1, 448, 448])
            out_resize   = F.interpolate(p[0],size=shape, mode='bilinear')
            pred   = torch.sigmoid(out_resize[0,0])
            # pred  = (pred*255).cpu().numpy()

            # res = (pred*255).detach().cpu().numpy()
            res = p[0].sigmoid().detach().squeeze().cpu().numpy()
            # print(res.shape, mask.shape)
            # print(res)
            # print('*'*12)
            # print(mask)
            WFM.step(pred=res*255, gt=mask*255)
            SM.step(pred=res*255, gt=mask*255)
            EM.step(pred=res*255, gt=mask*255)
            MAE.step(pred=res*255, gt=mask*255)

        sm1 = SM.get_results()['sm'].round(3)
        adpem1 = EM.get_results()['em']['adp'].round(3)
        wfm1 = WFM.get_results()['wfm'].round(3)
        mae1 = MAE.get_results()['mae'].round(3)

        with open("/data3/YG/FDRFNet/code/log/output.txt", "a") as f:
            f.write(f"sm1:{sm1} adpem1:{adpem1} wfm1:{wfm1} mae1:{mae1}\n")

        torch.save(net.state_dict(), tr_cfg.savepath + '/model-' + str(epoch + 1))

if __name__ == '__main__': 
    args = parser()
    utils.distributed.init_distributed_mode(args=args)

    tr_cfg = dataset.Config(datapath=args.datapath, savepath=args.savepath,mode='train', batch=args.batchsize, lr=args.lr, momen=0.9,
                         decay=args.wd, epoch=args.epoch, snapshot=args.checkpoint) 
    tr_ds = dataset.Data(tr_cfg)
    te_cfg = dataset.Config(datapath='/data3/YG/FDRFNet/code/data/COD10K/test', mode='test')
    te_ds= dataset.Data(te_cfg)

    # 实现分配数据
    tr_sampler = torch.utils.data.distributed.DistributedSampler(tr_ds)
    te_sampler = torch.utils.data.distributed.DistributedSampler(te_ds)

    tr_batch_sampler = torch.utils.data.BatchSampler(tr_sampler, args.batchsize, drop_last=True)

    nw = 8

    if args.rank == 0:
            print('Using {} dataloader workers every process'.format(nw))
            
    tr_dl = DataLoader( tr_ds,
                        batch_size=args.batchsize,
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        collate_fn=tr_ds.collate
                    )

    te_dl = DataLoader(te_ds, batch_size=1, shuffle=False, num_workers=8)

    net = FDRFNet(tr_cfg)
    main(tr_cfg, te_cfg, tr_dl, te_dl, net)





    # main(dataset, FDRFNet)
