#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import dataset
from FDRFNet_deit import FDRFNet
import utils.metrics as Measure

class Test(object):
    def __init__(self, Dataset, Network, path, model):
        ## dataset
        self.model  = '/data3/YG/FDRFNet/code/model/CoCOD8K/'+model
        self.cfg    = Dataset.Config(datapath=path, snapshot=self.model, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()


    def save(self):
        res=[]
        with torch.no_grad(): 

            WFM = Measure.WeightedFmeasure()
            SM = Measure.Smeasure()
            EM = Measure.Emeasure()
            MAE = Measure.MAE()
            MF = Measure.FbetaMeasure()
            ADAF = Measure.AdaptiveFbetaMeasure()


            for image, hf, mask, shape, name in self.loader:
                print(image.shape)
                image = image.cuda().float()
                mask  = mask[0].cpu().numpy()*255
                hf    = hf.cuda().float()
                start = time.time()
                p = self.net(image,hf)
                end = time.time()
                res.append(end-start)
                out_resize   = F.interpolate(p[0],size=shape, mode='bilinear')
                pred   = torch.sigmoid(out_resize[0,0])
                pred  = (pred*255).cpu().numpy()

                res1 = p[0].sigmoid().detach().squeeze().cpu().numpy()


                WFM.step(pred=res1*255, gt=mask*255)
                SM.step(pred=res1*255, gt=mask*255)
                EM.step(pred=res1*255, gt=mask*255)
                MAE.step(pred=res1*255, gt=mask*255)
                MF.step(pred=res1*255, gt=mask*255)
                ADAF.step(pred=res1*255, gt=mask*255)

                sm1 = SM.get_results()['sm'].round(3)
                adpem1 = EM.get_results()['em']['adp'].round(3)
                wfm1 = WFM.get_results()['wfm'].round(3)
                mae1 = MAE.get_results()['mae'].round(3)
                mean_f_beta = MF.get_results()['mean_fbeta'].round(3)
                ada_f_beta = ADAF.get_results()['ada_fbeta'].round(3)

                with open("/data3/YG/FDRFNet/code/log/Cross/output-cocod8kalltest2.txt", "a") as f:
                    f.write(f"mae1:{mae1} sm1:{sm1} adpem1:{adpem1} mean_f_beta:{mean_f_beta} ada_f_beta:{ada_f_beta} wm:{wfm1}\n")


                # head  = '../result/'+self.model+'/'+ self.cfg.datapath.split('/')[-1]
                    
                head  = '/data3/YG/FDRFNet/code/pred/cocod8k-test2'
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))
            
        time_sum=0
        for i in res:
            time_sum += i
        print("FPS: %f "%(1.0/(time_sum/len(res))))

if __name__=='__main__':
    # dir = '../data/TestDataset/'
    # dir = '/data3/YG/FDRFNet/code/data/COD10K/test'
    dir = '/data3/YG/FDRFNet/code/data/CoCOD8K/test'
    # for path in ['COD10K','NC4K','CAMO']:
    t = Test(dataset,FDRFNet, dir,'model-100')
    # t = Test(dataset,FDRFNet, dir+path,'model-80')
    t.save()
