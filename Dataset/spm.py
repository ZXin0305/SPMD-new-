"""
SPM
"""
import enum
from IPython.terminal.embed import embed
import numpy as np
import math
from torch.nn.functional import hinge_embedding_loss
from torch.onnx import select_model_mode_for_export
from lib.utils import create_center_map , create_heatmap
import torch

# joint_num = len(useful_points)

class SingleStageLabel():
    def __init__(self,height,width,centers,pcoors,ccoors,cnf) -> None:
        """
        pcoors --> pixel_coors（scaled）the third row is the depth(in world coordinate...)
        ccoors --> cam_coors
        """
        self.cnf = cnf
        if self.cnf.data_format == 'cmu':
            self.level = [[0,1],
                          [3,4,5],
                          [6,7,8],
                          [9,10,11],
                          [12,13,14]]
            self.pcoors = np.array(pcoors)
            self.ccoors = np.array(ccoors)
             
        self.centers = centers
        self.height = height
        self.width = width
        self.Z = math.sqrt(self.height ** 2 + self.width ** 2)
        self.th = cnf.th
        self.joint_num = self.cnf.joint_num

        #center_map
        self.center_map = np.zeros(shape=(self.height,self.width), dtype=np.float32)
        self.center_mask = np.ones(shape=self.center_map.shape,dtype=np.float32)
        #offset_map
        self.offset_map = np.zeros(shape=((self.joint_num) * 2 , self.height,self.width),
                                           dtype=np.float32)         #offset_map:offset放在center中心点的一定范围中 18*2 channels
        self.offset_mask = np.zeros(shape=self.offset_map.shape, dtype=np.float32)
        self.kps_count = np.zeros(shape=self.offset_map.shape,dtype=np.float32)
        
        #relative root map & root depth map
        self.depth_map = np.zeros(shape=(self.joint_num,self.height,self.width),
                                  dtype=np.float32)  #相对父节点深度图
        self.depth_mask = np.zeros(shape=self.depth_map.shape,dtype=np.float32)
        

    def create_2D_label(self):
        """
        centers:(x,y)
        pcoors:(x,y)
        """
            
        for i, center in enumerate(self.centers):  #对一张图中的所有人进行遍历
            #if center out of area , pass
            if center[0] < 0 or center[0] > self.width or \
                    center[1] < 0 or center[1] > self.height:
                continue
            self.center_map,self.center_mask = create_center_map(self.center_map,\
                                                                center,self.center_mask,\
                                                                sigma = self.cnf.sigma[0],th=self.th)
            self.body_joint_displacement(center,self.pcoors[i],self.cnf.sigma[1])
        
        self.kps_count[self.kps_count == 0] = 1
        self.offset_map = np.divide(self.offset_map , self.kps_count)

        # self.center_mask[self.center_mask == 0] = 0.01
        # self.offset_mask[self.offset_mask == 0] = 0.01

        return  { 
                    'center_map':self.center_map,
                    'center_mask':self.center_mask,
                    'offset_map':self.offset_map,
                    'offset_mask':self.offset_mask
                }

    def body_joint_displacement(self,center,coor,sigma):
        """
        在中心点一个范围内，都进行偏移的计算
        """
        for single_path in self.level:   #遍历每一条通道 
            start_joint = [center[0], center[1]] # 每条分支的起始点应该都是center joint
            for i, index in enumerate(single_path):

                end_joint = coor[:2,index]

                # make new end_joint based offset
                offset_x, offset_y = end_joint[0]-start_joint[0], end_joint[1]-start_joint[1]
                next_x = center[0] + offset_x
                next_y = center[1] + offset_y

                self.create_dense_displacement_map(index,center,[next_x, next_y],sigma)   
                start_joint[0], start_joint[1] = end_joint[0], end_joint[1]

    def create_dense_displacement_map(self,index,start_joint,end_joint,sigma):
        """
        start joint always is center joint 
        """
        start_x = int(start_joint[0])
        start_y = int(start_joint[1])

        #这里是得到父节点附近的范围
        x0 = int(max(0, start_x - sigma + 0.5))    
        x1 = int(min(self.width, start_x + sigma + 0.5))
        y0 = int(max(0, start_y - sigma + 0.5))
        y1 = int(min(self.height, start_y + sigma + 0.5))

        x_offset = 0
        y_offset = 0
        
        for x in range(x0,x1):
            for y in range(y0,y1):
                x_offset = (end_joint[0] - x) / self.Z
                y_offset = (end_joint[1] - y) / self.Z

                self.offset_map[2*index,y,x] += y_offset  # first-ch is y_offset
                self.offset_map[2*index+1,y,x] += x_offset # second-ch is x_offset
                self.offset_mask[2*index:2*index+2,y,x] = 1
                #center周围点(x,y)不和关节点重合，则人数增加一个
                if end_joint[1] != y or end_joint[0] != x:
                    self.kps_count[2*index:2*index+2,y,x] += 1

    def create_3D_label(self,sigma=3):
        """
        1):the root depth value --> absoute depth
        2):other joints depth value --> relative root joint's depth
        """   
        root_sigma = self.cnf.sigma[2]
        rel_sigma = self.cnf.sigma[3]

        count = np.zeros(shape=self.depth_map.shape,dtype='float')

        for idx, pcoor in enumerate(self.pcoors): # 遍历
            all_joint_depth = pcoor[2,:]
            center = pcoor[:,self.cnf.root_id]  # with depth

            #if center not in img's area
            if int(center[0]) < 0 or int(center[0]) > self.width or \
                    int(center[1]) < 0 or int(center[1]) > self.height:
                continue

            center_x0 = int(max(0,center[0] -  + 0.5))
            center_x1 = int(min(self.width,center[0] + root_sigma +0.5))
            center_y0 = int(max(0,center[1] - root_sigma + 0.5))
            center_y1 = int(min(self.height, center[1] + root_sigma + 0.5))
            center_area = [center_x0,center_x1,center_y0,center_y1]

            # 2-th channel is the center joint 
            self.depth_map[0], self.depth_mask[0] = self.setDepthMap(center,center[2],self.depth_map[0],self.depth_mask[0],center_area,count[0]) 

            #create relative depth between start joint & end joint
            for single_path in self.level:
                start_joint = center
                for i, index in enumerate(single_path):
                    end_joint = pcoor[:, index]  # with depth

                    if index == 0 or index == 1:
                        index += 1

                    if start_joint[0] < 0 or start_joint[0] > self.width or \
                            start_joint[1] < 0 or start_joint[1] > self.height:
                        start_joint = end_joint
                        continue
                    
                    if end_joint[0] < 0 or end_joint[0] > self.width or \
                            end_joint[1] < 0 or end_joint[1] > self.height:
                        start_joint = end_joint
                        continue

                    self.depth_map[index], self.depth_mask[index] = self.putVecDepthMap(centerA=start_joint, centerB=end_joint, \
                                                                                        rel_demap=self.depth_map[index], depth_mask=self.depth_mask[index], \
                                                                                        count=count[index],thre=2)
                    start_joint = end_joint     
                        
        count[count == 0] += 1
        # self.depth_mask[self.depth_mask == 0] == 0.01
        self.depth_map = np.divide(self.depth_map, count)  

        return {
                'depth_map':self.depth_map,
                'depth_mask':self.depth_mask,
               }

    
    def setDepthMap(self,coor,depth,depth_map,depth_mask,area,count):
        """
        common function to create root depth map
        """
        for x in range(area[0],area[1]):
            for y in range(area[2],area[3]):
                if depth_map[y,x] != 0:
                    count[y,x] += 1
                depth_map[y,x] += depth / self.Z
                depth_mask[y,x] = 1

        return depth_map, depth_mask

    def putVecDepthMap(self,centerA,centerB,rel_demap,depth_mask,count,thre=1.):
        """
        here:
        centerA: center joint
        centerB: other body joint
        """
        centerA = centerA.astype(float)  #has scaled
        centerB = centerB.astype(float)
        z_A = centerA[2]
        z_B = centerB[2]
        centerA = centerA[:2]
        centerB = centerB[:2]

        limb_vec = centerB - centerA
        limb_z = z_B - z_A

        norm = np.linalg.norm(limb_vec)
        if norm < 1.0:        # limb is too short, ignore it
            # print('return')
            return rel_demap, depth_mask
        
        limb_vec_unit = limb_vec / norm

        # To make sure not beyond the border of this two points
        x0 = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
        x1 = min(int(round(max(centerA[0], centerB[0]) + thre)), self.width)
        y0 = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
        y1 = min(int(round(max(centerA[1], centerB[1]) + thre)), self.height)

        #create the x/y area
        range_x = list(range(int(x0), int(x1), 1))
        range_y = list(range(int(y0), int(y1), 1))
        xx, yy = np.meshgrid(range_x, range_y)   # to be a grid
        
        ba_x = xx - centerA[0]
        ba_y = yy - centerA[1]
        limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
        mask = limb_width <= thre  #

        #temp representation
        rdep_vec_map = np.copy(rel_demap) * 0.0
        rdep_vec_map[yy, xx] = np.repeat(mask[:,:], 1, axis=0)
        rdep_vec_map[yy, xx] *= limb_z

        mask = np.logical_or.reduce(
            (np.abs(rdep_vec_map[:, :]) != 0, np.abs(rdep_vec_map[:, :]) != 0))  #求并集，(height,width)

        rel_demap += rdep_vec_map
        count[mask == True] += 1
        depth_mask[mask == True] = 1

        return rel_demap, depth_mask

