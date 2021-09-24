from math import e
from IPython.terminal.embed import embed
import numpy as np
from numpy.core.defchararray import array, center
import torch
import torch.nn as nn

class L2Loss(nn.Module):
    """
    using l2 loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, pre, gt, mask, batch_size):

        assert pre.size() == gt.size()  #(B,C,H,W),mask is too
        pre_ = pre.clone()
        gt_ = gt.clone()
        mask_ = mask.clone()

        loss = ((pre_ - gt_) ** 2) * mask_
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pre, gt, mask, batch_size):
        assert pre.size() == gt.size()
        pre_ = pre.clone()
        gt_ = gt.clone()
        mask_ = mask.clone()
        feature_chs = pre_.shape[1]

        loss = 0.0
        for i in range(batch_size):
            for j in range(feature_chs):
                nonzero_num = len(torch.nonzero(mask_[i,j]))
                tmp_loss = torch.sum(torch.abs(pre_[i,j] - gt_[i,j]) * mask_[i,j])
                if nonzero_num == 0:
                    nonzero_num = 1
                    # tmp_loss = 0
                loss = loss + (tmp_loss / nonzero_num)

        return loss / batch_size

class MultiLossFactory(nn.Module):
    """
    offset_map (B,C,h,w)
    offset_mask (B,C,h,w)
    joint_heatmap (B,C,h,w)
    joint_mask (B,C,h,w)
    depth_map (B,C,h,w)
    depth_mask (B,C,h,w)
    """
    def __init__(self):
        super().__init__()
        self.center_map_loss = L2Loss()
        self.offset_map_loss = L1Loss()
        self.reldep_loss = L2Loss()
        self.rootdep_loss = L1Loss()

    def forward(self,pre,gt,stage_num,out_up_blocks,batch_size):
        loss = 0.             #total
        loss_center = 0.      #center
        loss_offset = 0.      #offset
        loss_root = 0.        #root depth
        loss_3d = 0.          #depth
        pre_dict = pre
        gt_dict = gt

        gt_center_maps = gt_dict['center_maps']
        gt_center_mask = gt_dict['center_mask']
        gt_offset_maps = gt_dict['offset_maps']
        gt_offset_mask = gt_dict['offset_mask']
        gt_depth_maps = gt_dict['depth_maps']
        gt_depth_mask= gt_dict['depth_mask']
        
        for i in range(stage_num):
            
            #center map loss
            tmp_center_loss = self.center_map_loss(pre_dict['center_maps'][i],\
                                                        gt_center_maps,\
                                                        gt_center_mask, batch_size)
            #offset map loss
            tmp_offset_loss = self.offset_map_loss(pre_dict['offset_maps'][i],\
                                                    gt_offset_maps,\
                                                    gt_offset_mask,batch_size)
            #depth map
            #root-depth map
            tmp_root_loss = self.rootdep_loss(pre_dict['depth_maps'][i][:,0:1],\
                                                gt_depth_maps[:,0:1],\
                                                gt_depth_mask[:,0:1],batch_size)

            #rel-depth map
            tmp_rel_loss = self.reldep_loss(pre_dict['depth_maps'][i][:,1:],\
                                            gt_depth_maps[:,1:],\
                                            gt_depth_mask[:,1:],batch_size) 

            loss_center += tmp_center_loss
            loss_offset += tmp_offset_loss
            loss_root += tmp_root_loss
            loss_3d += tmp_rel_loss         

            tmp_loss = tmp_center_loss + tmp_offset_loss + \
                        tmp_root_loss + tmp_rel_loss
            
            loss += tmp_loss
                
        return dict(total_loss=loss,\
                        loss_center=loss_center / stage_num, loss_offset=loss_offset / stage_num,\
                        loss_root=loss_root / stage_num, loss_3d=loss_3d / stage_num)



        
