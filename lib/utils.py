import sys
import random
from numpy.core.fromnumeric import size
# from Dataset.pose import Pose
from re import L
from typing import ContextManager
from IPython import embed
import cv2
import math
import PIL
import json
import numpy as np
from numpy.core.defchararray import _join_dispatcher, center
import torch
from torch.nn.functional import hinge_embedding_loss
import torch.optim as optim
import math
from torchvision.transforms.functional import scale
from operator import itemgetter

sort_index = [1,2,0,7,8,9,13,14,15,10,11,12,16,17,18,3,4,5,6]
sort_index_flip = [1,2,0,10,11,12,16,17,18,7,8,9,13,14,15,5,6,3,4]

def imread(path):
    # with open(path, 'rb') as f:
    #     with PIL.Image.open(f) as img:
    #         # img = img.resize(size=(540,960))
    #         return img.convert('RGB')

    img = cv2.imread(path)
    assert img.shape[2] > 0
    return img

def read_json(path):
    with open(path,'rb') as file:
        data = json.load(file)
    return data


def prepare_centers(coors):  # coors --> pixel_coors
    """
    在这里要得到的是在像素平面中的中心点的位置
    """
    centers = []
    for coor in coors:
        center = coor[:2,2]  #center顺序 --> (x,y)
        _center = (int(center[0]),int(center[1]))
        centers.append(_center)
    return centers

def prepare_keypoints(ori_shape,out_shape,cam_coors,pixel_coors,dataset=None):
    """
    inplace operation
    像素坐标进行了scale
    """
    factory = ori_shape[0] / out_shape[0]  #shape顺序　--> (h,w) == (y,x)
    factorx = ori_shape[1] / out_shape[1]  #x,y方向上的缩放因子　--> 默认为4

    if dataset == 'cmu':
        # for cam_coor in cam_coors:
        #     cam_coor[[0,1],:] = cam_coor[[1,0],:]
        for pixel_coor in pixel_coors:
            pixel_coor[0,:] = (pixel_coor[0,:] / factorx + 0.5).astype(np.int32)  # x
            pixel_coor[1,:] = (pixel_coor[1,:] / factory + 0.5).astype(np.int32) # y
    else:
        pass
    return cam_coors , pixel_coors


def create_center_map(center_map,center,mask,sigma = 6,th=4.6052):
    """
    高斯分布:一小块一小块的进行高斯
    """
    center_x , center_y = int(center[0]) , int(center[1]) #(x,y)
    delta = math.sqrt(th * 2)

    height = center_map.shape[0] 
    width = center_map.shape[1]

    x0 = int(max(0,center_x - delta * sigma + 0.5))
    y0 = int(max(0,center_y - delta * sigma + 0.5))

    x1 = int(min(width, center_x + delta * sigma + 0.5))
    y1 = int(min(height, center_y + delta * sigma + 0.5))

    if x0 > width or x1 < 0 or x1 <= x0:
        return center_map,mask
    if y0 > height or y1 < 0 or y1 <= y0:
        return center_map,mask

    ## fast way
    arr_heat = center_map[y0:y1, x0:x1]  #　一整张图  center_map 只有一个channnel

    exp_factorx = 1 / 2.0 / sigma / sigma # (1/2) * (1/sigma^2)
    exp_factory = 1 / 2.0 / sigma / sigma
    x_vec = (np.arange(x0, x1) - center_x) ** 2
    y_vec = (np.arange(y0, y1) - center_y) ** 2
    arr_sumx = exp_factorx * x_vec
    arr_sumy = exp_factory * y_vec
    xv, yv = np.meshgrid(arr_sumx, arr_sumy)   #这一步是进行网格化
    # print(xv.shape,yv.shape)

    arr_sum = xv + yv
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    # mask_tmp = arr_exp.copy()
    # mask_tmp[arr_sum < th] = 1

    center_map[y0:y1, x0:x1] = np.maximum(arr_heat, arr_exp)
    # mask[y0:y1, x0:x1] = np.maximum(mask_tmp,mask[y0:y1, x0:x1])

    return center_map,mask

def create_heatmap(joint_heatmap,coor,th,mask,sigma):
    """
    joint heat map-->one joint type channel
    coor-->a joint
    """
    th = th
    delta = math.sqrt(th * 2)

    height = joint_heatmap.shape[0]
    width = joint_heatmap.shape[1]

    joint_x, joint_y = int(coor[0]), int(coor[1])
    
    x0 = int(max(0,joint_x - delta * sigma + 0.5))
    y0 = int(max(0,joint_y - delta * sigma + 0.5))
    x1 = int(min(width, joint_x + delta * sigma + 0.5))
    y1 = int(min(height, joint_y + delta * sigma + 0.5))

    arr_heat = joint_heatmap[y0:y1, x0:x1]
    exp_factorx = 1 / 2.0 / sigma / sigma
    exp_factory = 1 / 2.0 / sigma / sigma
    x_vec = (np.arange(x0, x1) - joint_x) ** 2
    y_vec = (np.arange(y0, y1) - joint_y) ** 2
    arr_sumx = exp_factorx * x_vec
    arr_sumy = exp_factory * y_vec
    xv, yv = np.meshgrid(arr_sumx, arr_sumy)
    arr_sum = xv + yv

    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0

    mask_tmp = arr_exp.copy()
    mask_tmp[arr_sum < th] = 1

    joint_heatmap[y0:y1, x0:x1] = np.maximum(arr_heat, arr_exp)
    mask[y0:y1, x0:x1] = np.maximum(mask_tmp,mask[y0:y1, x0:x1])
    return joint_heatmap, mask

def convert_coor_format():
    pass


def nms(max_people,heatmap,th = 0.5,need_confi = False):
    """
    heatmap: (h, w)  -- output_shape
    return: [(y,x),...]  a list
    """
    # heatmap = heatmap.cpu().numpy()
    heatmap[heatmap < th] = 0  #小于阈值的首先置零

    height , width = int(heatmap.shape[0]),int(heatmap.shape[1])
    map_left = np.zeros((height,width),dtype=np.float32)
    map_right = np.zeros((height,width),dtype=np.float32)
    map_up = np.zeros((height,width),dtype=np.float32)
    map_bottom = np.zeros((height,width),dtype=np.float32)

    map_weight = np.zeros((height,width)) 

    map_left[:,:-1] = heatmap[:,1:]
    map_right[:,1:] = heatmap[:,:-1]
    map_up[:-1,:] = heatmap[1:,:]
    map_bottom[1:,:] = heatmap[:-1,:]

    map_weight[heatmap >= map_left] = 1
    map_weight[heatmap >= map_right] += 1
    map_weight[heatmap >= map_up] += 1
    map_weight[heatmap >= map_bottom] += 1
    map_weight[heatmap >= th] += 1
    map_weight[map_weight != 5] = 0

    # peaks = np.argwhere(map_weight[1:(height - 1),1:(width - 1)] != 0)  #如果用torch --> torch.nonzero(..).cpu() 从2开始是排除在边缘的坐标点

    peaks = list(zip(np.nonzero(map_weight[:,:])[0], np.nonzero(map_weight[:,:])[1]))  # --> (y,x)
    # peaks = sorted(peaks, key=itemgetter(0))  #排序

    suppressed = np.zeros(len(peaks), np.uint8)
    keypoints_with_score = []

    #过滤距离较近的peak
    for i in range(len(peaks)):
        if suppressed[i] or heatmap[peaks[i][0],peaks[i][1]] < th:
            continue
        # if peaks[i][0] < 5 or peaks[i][0] > (height - 5):
        #     continue
        # elif peaks[i][1] < 5 or peaks[i][1] > (width - 5):
        #     continue
        for j in range(i+1,len(peaks)):
            if math.sqrt((peaks[i][0] - peaks[j][0]) ** 2 + (peaks[i][1] - peaks[j][1]) ** 2 ) < 20 or \
                          heatmap[peaks[j][0],peaks[j][1]] < th :
                suppressed[j] = 1

        keypoints_with_score.append((peaks[i][0],peaks[i][1],heatmap[peaks[i][0],peaks[i][1]]))  

    keypoints_with_score = sorted(keypoints_with_score, key=lambda x:x[2], reverse=True)
    return keypoints_with_score[:max_people]

def refine_joint():
    pass

def to_cam_3d(poses,cam):
    """
    poses: person_num * 3 * 19  3 -- (x,y,z)
    """
    cam_pose = []
    for pose in poses:
        tmp_pose = np.zeros(shape=pose.shape,dtype=np.float32)
        tmp_pose[0,:] = (pose[0,:] - cam[0,2]) * pose[2,:] / cam[0,0]  # x
        tmp_pose[1,:] = (pose[1,:] - cam[1,2]) * pose[2,:] / cam[1,1]  # y
        tmp_pose[2,:] = pose[2,:]

        cam_pose.append(tmp_pose)
    
    return cam_pose

def sum_features(features):
    stage_num = 3
    out_block = 4
    sum_feature = torch.zeros(size=features[2][3].shape).cuda()
    for i in range(stage_num):
        for j in range(out_block):
            sum_feature += features[i][j]
    
    return sum_feature / 3 / 4

def pad_width(img, min_dims, stride, pad_value=(0,0,0)):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))  # up
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))  # left
    pad.append(int(min_dims[0] - h - pad[0]))             # down
    pad.append(int(min_dims[1] - w - pad[1]))             # right
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)    
    return padded_img, pad

def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype = np.float32)
    img = (img - img_mean) * img_scale
    return img

def test_crop(img, cnf):
    img_shape = (img.shape[1], img.shape[0]) # x ,y                 
    crop_x = cnf.resize_x  # width 456 
    crop_y = cnf.resize_y  # height 256
    scale = min(crop_x / img_shape[0], crop_y / img_shape[1])  #返回的是最小值
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    
    pad = [0, 0, 0, 0]  # left, right, up, down
    # has resized
    center = np.array([img.shape[1]//2, img.shape[0]//2], dtype=np.int)
    
    if img.shape[1] < crop_x:    # pad left and right
        margin_l = (crop_x - img.shape[1]) // 2
        margin_r = crop_x - img.shape[1] - margin_l
        pad_l = np.ones((img.shape[0], margin_l, 3), dtype=np.uint8) * 128
        pad_r = np.ones((img.shape[0], margin_r, 3), dtype=np.uint8) * 128
        pad[0], pad[1] = margin_l, margin_r
        img = np.concatenate((pad_l, img, pad_r), axis=1)        #在1维进行拼接　也就是w
    elif img.shape[0] < crop_y:  # pad up and down
        margin_u = (crop_y - img.shape[0]) // 2
        margin_d = crop_y - img.shape[0] - margin_u
        pad_u = np.ones((margin_u, img.shape[1], 3), dtype=np.uint8) * 128
        pad_d = np.ones((margin_d, img.shape[1], 3), dtype=np.uint8) * 128
        pad[2], pad[3] = margin_u, margin_d
        img = np.concatenate((pad_u, img, pad_d), axis=0)       #在0维进行拼接　也就是h
    
    return img, pad


if __name__ == '__main__':
    center_map = np.zeros(shape=(512,832),dtype=np.float32)
    mask = np.zeros(shape=(512,832),dtype=np.uint8)
    center_1 = (1,1)
    center_2 = (3,10)  #传进去的center坐标是（x,y） ,但是如果是图片的话，和图片中的坐标是一致的，NMS之后出来的是(y,x) 所以最后要对x\y进行转换
    center_3 = (100,100)
    center_4 = (20,5)
    offset = []
    center_map,mask,a_list = create_center_map(center_map,center_1,mask,offset)
    center_map,mask,a_list = create_center_map(center_map,center_2,mask,offset)
    # center_map,mask = create_center_map(center_map,center_3,mask,offset)
    # center_map,mask = create_center_map(center_map,center_4,mask,offset)
    embed()
    center_map = torch.from_numpy(center_map).unsqueeze(0).unsqueeze(0)
    k = nms(center_map)