import random
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from Config import config
from IPython import embed
from lib.utils import imread
import math


"""
in rotation:because has resized the img shape when read it
            so, after rotate img,the size has changed
            but, final size must match net's size
            using factor_x / factor_y to refactor the coors(has rotated) to ori image size's reprentation
            refactored the coors has offsets in x-direction and y-direction
            how to calculate the offsets?
            first -- factor the rotated center to ori image size's reprentation
            second -- compare the factord center with the ori center, calculate the difference
"""
def aug_crop(cnf,img,coors):
    """
    img: has rotated , the size is changed
    coors: has rotated
    """
    crop_x = int(cnf.orih)
    crop_y = int(cnf.oriw)
    scale_x = cnf.crop_x / float(img.shape[1])
    scale_y = cnf.crop_y / float(img.shape[0])
    
    center = np.array((img.shape[1] // 2 ,img.shape[0] // 2))  #after rotate
    center = center.astype(int)

    center[0] = int(center[0] * scale_x + 0.5)
    center[1] = int(center[1] * scale_y + 0.5)  # to ori img size's represent
    
    img = cv2.resize(img, (0,0), fx=scale_x, fy=scale_y)  #旋转之后，再resize到网络需要的形状  (h,w,c)-->(1080,1920)
    # print('int aug_crop')

    #turn rotated coors to ori img size's representation
    for i in range(len(coors)):
        coors[i][0,:] *= scale_x
        coors[i][1,:] *= scale_y

    offset_left = (crop_x // 2 - center[0])  #ori_center / rotate_center
    offset_up = (crop_y // 2 - center[1])
    offset = np.array([offset_left, offset_up], np.int)
    
    # change the rotate coors to ori size representation
    for i in range(len(coors)):
        coors[i][0,:] += offset[0]  # x
        coors[i][1,:] += offset[1]  # y
    # -------------------------------------------------------------
    return coors, img


def aug_flip(cnf,img,cam_coors,pixel_coors,aug_flag,flip_width):
    dice = random.random()
    #随机翻转
    doflip = dice <= cnf.aug_prob

    if doflip:
        aug_flag = False
        flip_order = cnf.flip_order
        img = img.copy()
        width = flip_width                        
        cv2.flip(src=img, flipCode=1, dst=img)    #水平翻转

        # for cam_coor in cam_coors:
        #     cam_coor[:,:] = cam_coor[:,flip_order]

        for pixel_coor in pixel_coors:
            pixel_coor[0,:] = width - 1 - pixel_coor[0,:]
            pixel_coor[:,:] = pixel_coor[:,flip_order]

    return cam_coors, pixel_coors, img, aug_flag

def aug_rotate(cnf,img,coors,aug_flag):
    """
    coors --> pixel coors 
    """
    dice = random.random()
    if dice <= cnf.aug_prob:
        aug_flag = False
        degree = (dice - 0.5) * 2 * cnf.rotate_max
        img_rot, R = rotate_bound(img=img,angle=np.copy(degree), bordervalue=(128,128,128))
        
        for i in range(len(coors)):
            coors[i][:2,:] = rotate_skel2d(coors[i][:2,:],R,cnf.joint_num)
     
        return coors, img_rot, aug_flag

    return coors, img, aug_flag

def rotate_bound(img, angle, bordervalue):
    """The correct way to rotation an image
       http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    """
    # grab the dimensions of the image and then determine the
    # center    
    (h, w) = img.shape[0], img.shape[1] #这个时候还是cv的形式(h,w,c)
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # M --> the rotation matrix
    # 这个opencv中为了让图像能够在任意位置进行旋转变换
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    # 因为上面旋转之后，输出的图像还是同样尺寸的，所以进行放射变换后，图像的边框要重新设置
    # nW --> new width
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bordervalue), M

def rotate_skel2d(coor,R,joint_num):
    aug_pcoor = np.concatenate((coor, np.ones((1,joint_num))), axis=0)
    rot_pcoor = (R @ aug_pcoor)
    return rot_pcoor[:2,:]

def add_mask(cnf,img):
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2

    pass  

def aug_trans(cnf,img,coors,aug_flag):
    """
    1-left 2-right 3-up 4-down
    """
    dice = random.random()

    if dice <= cnf.aug_prob:
        aug_flag = False
        direct_list = [1,2,3,4]
        direction = random.sample(direct_list,1)
        img_trans = np.ones((img.shape),dtype=np.uint8) * 128
        w = img.shape[1]
        h = img.shape[0]
        trans = cnf.trans_pixel

        if direction[0] == 1:
            img_trans[:,trans:,:] = img[:,:(w-trans),:]
            for i in range(len(coors)):
                coors[i][0,:] += trans
        elif direction[0] == 2:
            img_trans[:,:(w-trans),:] = img[:,trans:,:]
            for i in range(len(coors)):
                coors[i][0,:] -= trans
        elif direction[0] == 3:
            img_trans[trans:,:,:] = img[:(h-trans),:,:]
            for i in range(len(coors)):
                coors[i][1,:] += trans
        elif direction[0] == 4:        
            img_trans[:(h-trans),:,:] = img[trans:,:,:]
            for i in range(len(coors)):
                coors[i][1,:] -= trans

        return coors, img_trans, aug_flag
    return coors, img, aug_flag

"""
--------------------------------------------------------------------------------------------------------------
"""

class Scale:
    def __init__(self, cnf, prob=1, min_scale=0.5, max_scale=1.1, target_dist=0.6):
        self._prob = prob
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._target_dist = target_dist
        self._cnf = cnf

    def make_scale(self, img, coors):
        prob = random.random()
        scale_multiplier = 1

        # random to do the scale operation
        if prob <= self._prob:
            # select_add_factor = random.sample(add_list, 1)
            prob = random.random()
            scale_multiplier = (self._max_scale - self._min_scale) * prob + self._min_scale  # 0.6 * prob + 0.5
        scale_provide = min(self._cnf.resize_x / self._cnf.oriw,
                            self._cnf.resize_y / self._cnf.orih)
        scale_abs = self._target_dist / scale_provide
        scale = scale_abs * scale_multiplier
        # print('scale factor:{}'.format(scale))
        img = cv2.resize(img, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # joint --> scale
        # 遍历
        for coor in coors:
            coor[0] *= scale
            coor[1] *= scale

        return img, coors

class Rotate():
    def __init__(self, pad_value=(128,128,128),max_rotate_degree=10):
        self._pad_value = pad_value
        self._max_rotate_degree = max_rotate_degree
        self._rotate_prob = 0.5

    def make_rotate(self, img, pcoors):
        prob = random.random()
        if prob <= self._rotate_prob:
            degree = (prob - 0.5) * 2 * self._max_rotate_degree

            img, M = self.rotate_bound(img,degree, pad_value=self._pad_value)

            for coor in pcoors:
                point = [coor[0], coor[1]]  # x , y
                point = self.rotate_point(point, M)
                coor[0], coor[1] = point[0], point[1]

        return img, pcoors


    def rotate_bound(self,img,degree,pad_value):
        """The correct way to rotation an image
           http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        """
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = img.shape[0], img.shape[1]  # 这个时候还是cv的形式(h,w,c)
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        # M --> the rotation matrix
        # 这个opencv中为了让图像能够在任意位置进行旋转变换
        M = cv2.getRotationMatrix2D((cX, cY), -degree, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        # 因为上面旋转之后，输出的图像还是同样尺寸的，所以进行放射变换后，图像的边框要重新设置
        # nW --> new width
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=pad_value), M

    def rotate_point(self,coor,M):
        return [M[0, 0] * coor[0] + M[0, 1] * coor[1] + M[0, 2],
                M[1, 0] * coor[0] + M[1, 1] * coor[1] + M[1, 2]]

class CropPad():
    def __init__(self,cnf, min_scale=0.5, max_scale=1.1, center_perterb_max=40):
        """
        :param cnf:
        :param min_scale:
        :param max_scale:
        :param center_perterb_max:
        crop中就包含了scale的操作
        """
        self._crop_x = cnf.resize_x
        self._crop_y = cnf.resize_y
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._center_perterb_max = center_perterb_max
        self._aug_flag = True

    def make_crop(self, img, coors):
        """
        img: has rotated , the size is changed
        coors: has rotated
        """
        h, w, _ = img.shape
        dice_x = random.random()
        dice_y = random.random()
        scale_random = random.random()
        scale_multiplier = ((self._max_scale - self._min_scale) * scale_random + self._min_scale)

        crop_x = int(self._crop_x)
        crop_y = int(self._crop_y)

        scale = min(self._crop_y / h,
                    self._crop_x / w)
        if self._aug_flag:
            scale *= scale_multiplier
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        for coor in coors:
            coor[0] *= scale
            coor[1] *= scale

        x_offset = int((dice_x - 0.5) * 2 * self._center_perterb_max)
        y_offset = int((dice_y - 0.5) * 2 * self._center_perterb_max)
        center = np.array([img.shape[1]//2 + x_offset, img.shape[0]//2 + y_offset])    #x,y
        center = center.astype(int)

        # pad up and down
        pad_v = np.ones((crop_y, img.shape[1], 3), dtype=np.uint8) * 128
        img = np.concatenate((pad_v, img, pad_v), axis=0)

        # pad right and left
        pad_h = np.ones((img.shape[0], crop_x, 3), dtype=np.uint8) * 128
        img = np.concatenate((pad_h, img, pad_h), axis=1)

        img = img[int(center[1] + crop_y / 2):int(center[1] + crop_y / 2 + crop_y),
                  int(center[0] + crop_x / 2):int(center[0] + crop_x / 2 + crop_x), :]

        offset_left = crop_x / 2 - center[0]
        offset_up = crop_y / 2 - center[1]
        offset = np.array([offset_left, offset_up], np.int)
        for coor in coors:
            coor[0] += offset[0]
            coor[1] += offset[1]

        return img, coors


if __name__ == '__main__':
    filename = '/media/xuchengjun/datasets/panoptic-toolbox/170407_haggling_a1/hdImgs/00_05/00_05_00003563.jpg'
    img = imread(filename)
    # cv2.flip(src=img,flipCode=1,dst=img)

    # cv2.imshow('flip', img)
    # cv2.waitKey(0)

    cnf = config.set_param()
    coor = np.ones((2,19))
    R = np.ones((2,3))
    rotate_skel2d(coor, R=R, joint_num=cnf.joint_num)
    # aug_rotate(cnf, img, coors=None)
