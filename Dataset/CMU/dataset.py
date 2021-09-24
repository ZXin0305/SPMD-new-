import os
import sys
from IPython.core.magic_arguments import real_name
sys.path.append('/home/xuchengjun/ZXin/SPMD')
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset , DataLoader
from torchvision import transforms
from path import Path
from IPython import embed
from Config import config
from lib.utils import imread,read_json,prepare_keypoints,prepare_centers
from Dataset.spm import SingleStageLabel
import matplotlib.pyplot as plt
import cv2
import random
from Dataset.ImageAugmentation import Scale, Rotate, CropPad,aug_flip,aug_crop,aug_trans,aug_rotate
from Dataset.CMU.getDataList import GetDataset
from Dataset.CMU.project import reproject, projectPoints
import copy

useful_train_dirs = ['170407_haggling_a1','160422_ultimatum1','160906_pizza1'] # '170221_haggling_b1','160422_ultimatum1','160906_pizza1'
useful_val_dirs = ['160422_ultimatum1','160906_pizza1'] 
useful_img_dirs_train = ['00_00','00_01','00_02','00_03','00_04','00_05','00_07','00_08','00_09'] #'00_01','00_02','00_03','00_04','00_05','00_07','00_08','00_09'
useful_img_dirs_val = ['00_16','00_30']
body_edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1

class CMU_Dataset(Dataset):

    def __init__(self,cnf,mode='train') -> None:   #cnf --> configuration
        super().__init__()

        self.cnf = cnf
        self.data_path = Path(cnf.data_path)
        self.sub_dirs = self.data_path.dirs()  #得到所有根路径下的子文件夹
        self.mode = mode
        self.get_list = GetDataset(self.sub_dirs,useful_train_dirs,useful_img_dirs_train,\
                                                          useful_val_dirs,useful_img_dirs_val)

        if self.mode == 'train':
            self.data_list = self.get_list.get_train_data(self.cnf.max_iter_train)  #include all json file
        elif self.mode == 'val':
            self.data_list = self.get_list.get_val_data(self.cnf.max_iter_val)


    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, idx):
        #load json file
        json_path = self.data_list[idx]
        json_file = read_json(json_path)

        #img path
        img_path = json_file['img_path']
        img = imread(img_path)   # read original img without resize
        
        cam_coors_ori = json_file['cam_coors']
        pixel_coors_ori = json_file['pixel_coors']
        skel_with_conf_ori = json_file['skel_with_conf'] 
        cam = json_file['cam'] 

        cam_coors_ori = np.array(cam_coors_ori)
        pixel_coors_ori = np.array(pixel_coors_ori)
        skel_with_conf_ori = np.array(skel_with_conf_ori)
        cam = np.matrix(cam)

        #create label
        img_shape = [json_file['img_height'],json_file['img_width']]
        crop_shape = (self.cnf.resize_y, self.cnf.resize_x)
        out_shape = (self.cnf.outh,self.cnf.outw)

        pixel_coors = copy.deepcopy(pixel_coors_ori)
        cam_coors = copy.deepcopy(cam_coors_ori)

        #image aug
        # aug_flag = True
        # if self.mode == 'train':
        #     if self.cnf.aug_flip and aug_flag:
        #         cam_coors, pixel_coors, img, aug_flag = aug_flip(cnf=self.cnf, img=img,
        #                                                         cam_coors=cam_coors, pixel_coors=pixel_coors,
        #                                                         aug_flag=aug_flag,flip_width=img_shape[1])
        #     if self.cnf.aug_rotate and aug_flag:  #after rotate, need crop to meet net's size
        #         pixel_coors, img, aug_flag = aug_rotate(cnf=self.cnf, img=img,
        #                                                 coors=pixel_coors, aug_flag=aug_flag)
        #         pixel_coors, img = aug_crop(cnf=self.cnf,img=img,coors=pixel_coors)
        #     if self.cnf.aug_trans and aug_flag:
        #         pixef_coors, img, aug_flag = aug_trans(cnf=self.cnf,img=img,coors=pixel_coors,aug_flag=aug_flag)

        # img aug
        # img, pixel_coors = Scale(self.cnf).make_scale(img, pixel_coors)
        img, pixel_coors = Rotate().make_rotate(img, pixel_coors)
        img, pixel_coors = CropPad(self.cnf).make_crop(img, pixel_coors)  # meet net's input_size

        cam_coors, pixel_coors = prepare_keypoints(ori_shape = crop_shape,
                                                    out_shape = out_shape,
                                                    cam_coors = cam_coors,
                                                    pixel_coors = pixel_coors,
                                                    dataset=self.cnf.data_format) #(X,Y,Z) (x,y,Z)

        # if need aug,centers are the fomat after aug
        centers = prepare_centers( coors = pixel_coors)  #(x,y)
        # self.show_img(pixel_coors, img)

        ssl = SingleStageLabel(self.cnf.outh,
                               self.cnf.outw,
                               centers,
                               pixel_coors,
                               cam_coors,
                               self.cnf)
    
        TwoD_label = ssl.create_2D_label()
        # self.show_center_map(TwoD_label['offset_map'][0],img,img_path)

        ThreeD_label = ssl.create_3D_label()  # --> root_relative_depth_map 第一个channel是root joint的depth map
        # self.show_center_map(ThreeD_label['depth_map'][0],img,img_path)

        # if self.cnf.do_normlize:
        #     img = self.transform_img(img=img)
        # else:
        #     img = img.transpose((2,0,1)).astype(np.float32)
        #     img = torch.from_numpy(img).float()

        img = img.astype(np.float32)
        img = (img - 128) / 256
        img = img.transpose((2,0,1))
        img = torch.from_numpy(img).float()

        if self.mode == 'train':
            return {
                   'img':img,
                   'center_map':torch.Tensor(TwoD_label['center_map']).unsqueeze(0),
                   'center_mask':torch.Tensor(TwoD_label['center_mask']).unsqueeze(0),
                   'offset_map':torch.Tensor(TwoD_label['offset_map']),
                   'offset_mask':torch.Tensor(TwoD_label['offset_mask']),
                   'depth_map':torch.Tensor(ThreeD_label['depth_map']),
                   'depth_mask':torch.Tensor(ThreeD_label['depth_mask'])
            }

        if self.mode == 'val':
            return {
                   'img':img,
                   'ccoors_ori':torch.Tensor(cam_coors_ori),
                   'cam':torch.Tensor(cam)
            }
            # (3,19)，　cam_coors：重投影回来的坐标

    """
    
    """
    def transform_img(self,img):
        img = transforms.ToTensor()(img)  #(3,1080,1920)
        img = transforms.Normalize(mean=self.cnf.norm_mean, std=self.cnf.norm_std)(img) 
        return img
    

    def show_center_map(self,center_map,img_copy,img_path):

        center_map = center_map * 255
        plt.subplot(111)
        plt.imshow(center_map)
        plt.axis('off')
        plt.show()
        
        # center_map = center_map * 255
        # for i in range(15):
        #     plt.subplot(5,3,i+1)
        #     plt.imshow(center_map[i])
        #     plt.axis('off')
        #     # plt.savefig(fname="/home/xuchengjun/Desktop/zx/SPM_Depth/results/center_map/" + img_name + '_offset.jpg')
        # plt.show()
        print(img_path)
        cv2.imshow("current_img", img_copy)
        # # cv2.imwrite("/home/xuchengjun/Desktop/zx/SPM_Depth/results/center_map/" + img_name + '_ori.jpg', img_copy)
        cv2.waitKey(0)
    def show_img(self,coors,img):
        
        for coor in coors:
            for i in range(15):
                cv2.circle(img, center=(int(coor[0][i]),int(coor[1][i])), radius=4, color=(255,255,0))
        
        cv2.imshow('result', img)
        key = cv2.waitKey()
        if key == 27:
            pass

if __name__ == '__main__':
    cnf = config.set_param()
    cmu = CMU_Dataset(cnf,mode='train')

    for i in range(len(cmu)):
        dict_ = cmu[i]
        print('working ..')


        
        



        
