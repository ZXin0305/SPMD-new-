import sys
sys.path.append('/home/xuchengjun/ZXin/SPMD')
import enum
from re import S
import torch
from path import Path
import torchvision
from Config import config
from IPython import embed
from model.spmd import SPMD
import torch.optim as optim
from Dataset.CMU import dataset as dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import numpy as np
from lib.loss import MultiLossFactory
from lib.checkpoint import load_ck, save_ck, load_state
from torch.autograd import Variable
from datetime import datetime
from time import time
from lib.utils import nms,sum_features,to_cam_3d
from lib.association import Association
from lib.test_metric import joint_det_metrics
from lib.save_results import save_results
from lib.solver import make_optimizer, make_lr_scheduler
import argparse
from Dataset.dataloader import get_train_loader, get_test_loader


class Trainer(object):
    def __init__(self,cnf) -> None:
        self.cnf = cnf

        #init some values
        self.epoch = 0
        self.end_epoch = cnf.epoch
        self.iter = 0
        self.best_test_f1 = None

        #build the net
        self.model_path = Path(cnf.model_path)
        self.device = cnf.device
        self.gpu_ids = cnf.gpu_ids
        self.num_gpu = len(self.gpu_ids)  #default is 4 or 2, adjust iter settings
        self.net = SPMD(cnf)  #using SMAP's net

        #possibly load checkpoint
        checkpoint = None
        self.ck_path = Path(self.cnf.ck_path) / 'train.pth'
        if self.ck_path.exists():
            print(f'load checkpoint --> {self.ck_path}')
            checkpoint = torch.load(self.ck_path)
            load_state(self.net, checkpoint)

        self.dp = True
        if self.dp:
            print('using DP Mode ..')
            self.net.to(cnf.device)  #just load model to cuda
            self.net = torch.nn.DataParallel(self.net,device_ids = self.gpu_ids) #copy model to sever cards 

        #the optimizer
        self.checkpoint_period = self.cnf.checkpoint_period
        self.optimizer = make_optimizer(self.cnf, self.net,  self.num_gpu)
        self.scheduler = make_lr_scheduler(optimizer=self.optimizer)
        if not self.cnf.weights_only and checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.iter = checkpoint['iter']
            self.epoch = checkpoint['current_epoch']

        #init logging stuffs
        self.log_dir = Path(cnf.log_dir)
        self.sw = SummaryWriter(self.log_dir)

        #init dataset
        #CMU
        train_set = dataset.CMU_Dataset(cnf,mode='train')
        val_set = dataset.CMU_Dataset(cnf,mode='val')

        #init train/val loader
        # self.train_loader = get_train_loader(self.cnf)
        # self.val_loader = get_test_loader(self.cnf)

        self.train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.cnf.batch_size, num_workers=self.cnf.num_workers)
        self.val_loader = DataLoader(dataset=val_set, shuffle=False, batch_size=1, num_workers=0)

        self.train_epoch_len = len(self.train_loader)
        self.val_epoch_len = len(self.val_loader) 
        
        #cal loss
        self.cal_loss = MultiLossFactory()

    def train(self):
        """
        start to train the model
        """
        self.net.train()
        train_loss = []
        # mean_epoch_loss = []
        time_list = []
        global_st = time()
        iteration = self.iter

        for batch_data in (self.train_loader):
            start_time = time()
            
            img = batch_data['img'].cuda()
            gt_dict = {
                'center_maps':batch_data['center_map'].cuda(),
                'center_mask':batch_data['center_mask'].cuda(),
                'offset_maps':batch_data['offset_map'].cuda(),
                'offset_mask':batch_data['offset_mask'].cuda(),
                'depth_maps':batch_data['depth_map'].cuda(),
                'depth_mask':batch_data['depth_mask'].cuda(),
            }
            #using SMAP's net , return a dict
            pre_dict = self.net(img)  
            loss_dict = self.cal_loss.forward(pre_dict,gt_dict,self.cnf.stage_num,4, self.cnf.batch_size)
            total_loss = loss_dict['total_loss']

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            end_time = time()

            #store all iteration's loss
            train_loss.append(total_loss.data.item())  #.data可以获得该节点的值，Tensor类型,.item() --> 将torch中的值取出来

            time_list.append(self.train_epoch_len * (end_time - start_time) / 3600)

            print('\r[{}] Epoch: {} progerss: {} / {} Loss: {:0.8f} Center_l: {:0.6f} Offset_l: {:0.6f} Root_l: {:0.6f} Depth_l: {:0.6f} lr: {:0.6f} Total_t: {:0.2f}'.format(datetime.now().strftime("%m-%d@%H:%M"),
            self.epoch,iteration+1,self.train_epoch_len,
            np.mean(np.array(train_loss)),
            loss_dict['loss_center'] , loss_dict['loss_offset'],
            loss_dict['loss_root'] , loss_dict['loss_3d'],
            self.optimizer.param_groups[0]["lr"] / self.num_gpu,
            np.mean(np.array(time_list))
            ), end='')

            #every iter save the avg total loss
            # mean_epoch_loss.append(np.mean(np.array(train_loss)))

            #
            iteration += 1
            if iteration % self.cnf.checkpoint_period == 0 and iteration != 0:
                ck = {
                    'state_dict': self.net.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'iter':iteration,
                    'current_epoch':self.epoch
                }
                torch.save(ck, self.ck_path)
                torch.save(ck, self.model_path / f'{self.epoch}_{iteration}.pth')

            if iteration >= self.train_epoch_len:
                # save_results('total',mean_epoch_loss,self.epoch)
                # save_results('center',loss_center_list,self.epoch)
                # save_results('offset',loss_offset_list,self.epoch)
                # save_results('root',loss_root_list,self.epoch)
                # save_results('3d',loss_3d_list,self.epoch)

                break
        
        # the epoch loss
        mean_epoch_loss = np.mean(train_loss)
        self.sw.add_scalar(tag='train/loss',scalar_value=mean_epoch_loss,global_step=self.epoch)
        global_et = time()
        print(f'\nTime: {((global_et - global_st) / 3600):.2f} h \n')

    def run(self):
        for i in range(self.epoch,self.end_epoch):
            self.train()
            # self.test()
            torch.save(self.net.state_dict(),self.model_path / f'epoch_{self.epoch}.pth')
            self.epoch += 1
            self.scheduler.step()
            


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
    cnf = config.set_param()
    trainer = Trainer(cnf)

    trainer.run()
