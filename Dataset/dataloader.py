import math

import torch
import torchvision.transforms as transforms
from easydict import EasyDict as edict
from Dataset.CMU import dataset as dataset
from IPython import embed


def collect_fn_train(batch):

    transpose_list = list(zip(*batch))
    imgs = torch.stack(transpose_list[0], dim=0)
    center_maps = torch.stack(transpose_list[1], dim=0)
    center_mask = torch.stack(transpose_list[2], dim=0)
    offset_maps = torch.stack(transpose_list[3], dim=0)
    offset_mask = torch.stack(transpose_list[4], dim=0)
    depth_maps = torch.stack(transpose_list[5], dim=0)
    depth_mask = torch.stack(transpose_list[6], dim=0)

    return imgs, center_maps, center_mask, offset_maps, offset_mask, depth_maps, depth_mask

def collect_fn_val(batch):

    img = batch[0]
    cam_coors = batch[1]
    cam_info = batch[2]

    return img, cam_coors, cam_info    


def get_train_loader(cnf, is_shuffle=True, use_sampler=False):
    # global batch_size_train
    batch_size_train = cnf.batch_size 
    dataset_ = dataset.CMU_Dataset(cnf,mode='train')      

    if use_sampler:
        # -------- make samplers -------- #  
        if is_shuffle:
            sampler =  torch.utils.data.sampler.RandomSampler(dataset_)   #normal shuffle operation
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset_) #not use shuffle or dist

        #obtain total
        batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, batch_size_train, drop_last=False) 
    
        data_loader = torch.utils.data.DataLoader(
            dataset_, num_workers=cnf.num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collect_fn_train,
            )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset_, batch_size=batch_size_train,
            shuffle=True, num_workers=cnf.num_workers
        )

    return data_loader

def get_test_loader(cnf, use_sampler=False):
    # -------- get raw dataset interface -------- #

    dataset_ = dataset.CMU_Dataset(cnf,mode='val')

    if use_sampler:

        # -------- make samplers -------- #
        sampler = torch.utils.data.sampler.SequentialSampler(subset)
        batch_size = 1
        batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, batch_size, drop_last=False)

        # -------- make data_loader -------- #
        data_loader = torch.utils.data.DataLoader(
                dataset_, num_workers=cnf.num_workers,
                batch_sampler=batch_sampler,
                collate_fn=collect_fn_val,
            )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset_, batch_size=1, 
            shuffle=False, num_workers=cnf.num_workers
        )

    return data_loader