import random 


class GetDataset():
    def __init__(self,sub_dirs,useful_train_dirs,useful_img_dirs_train,\
                      useful_val_dirs,useful_img_dirs_val):
        self.sub_dirs = sub_dirs
        self.useful_train_dirs = useful_train_dirs
        self.useful_img_dirs_train = useful_img_dirs_train
        self.useful_val_dirs = useful_val_dirs
        self.useful_img_dirs_val = useful_img_dirs_val

    def get_train_data(self):
        """
        random select total 160k images in four different image dirs
        every image dir have 10 different views to use
        """
        data_list = []
        for sub_dir in self.sub_dirs:
            if sub_dir.basename() in self.useful_train_dirs:  #最终用到的训练集有四种
                # print(f'{sub_dir.basename()}')
                img_dir_path = sub_dir / 'hdImgs'
                annotation_dir_path = sub_dir / 'hdPose3d_stage1_coco19'

                img_dirs = img_dir_path.dirs()
                annotation_files = annotation_dir_path.files()  #annotations 这里没有文件夹，是所有的
                # sample_annotation_files = random.sample(annotation_files, 6000)

                for img_dir in img_dirs:
                    if img_dir.basename() in self.useful_img_dirs_train:
                        # imgs = img_dir.files()  #len(imgs) == 16716 所有的数据集
                        cali_file_path = sub_dir / ('calibration_' + sub_dir.basename() + '.json') #标定文件的路径
                        for idx in range(len(annotation_files)):
                            basename = annotation_files[idx].basename()
                            if basename.endswith('.json'):   #读取的时候有错误。。
                                anno_num = basename.split('.')[0].split('_')[1]  #只要这个标签的文件数值就好
                                img_path = img_dir / (img_dir.basename() +  '_' + anno_num + '.jpg')
                                data_list.append((img_path,annotation_files[idx],cali_file_path,img_dir.basename())) #img_dir.basename()　--> 主要是为了得到对应的相机参数
        return data_list
        
    def get_val_data(self):
        data_list = []
        for sub_dir in self.sub_dirs:
            if sub_dir.basename() in self.useful_val_dirs:
                # print(f'{sub_dir.basename()}')
                img_dir_path = sub_dir / 'hdImgs'
                annotation_dir_path = sub_dir / 'hdPose3d_stage1_coco19'

                img_dirs = img_dir_path.dirs()
                annotation_files = annotation_dir_path.files()  #annotations 这里没有文件夹，是所有的

                for img_dir in img_dirs:
                    if img_dir.basename() in self.useful_img_dirs_val:
                        cali_file_path = sub_dir / ('calibration_' + sub_dir.basename() + '.json') #标定文件的路径
                        for idx in range(len(annotation_files)):
                            basename = annotation_files[idx].basename()
                            if basename.endswith('.json'):   #读取的时候有错误。。
                                anno_num = basename.split('.')[0].split('_')[1]  #只要这个标签的文件数值就好
                                img_path = img_dir / (img_dir.basename() +  '_' + anno_num + '.jpg')

                                data_list.append((img_path,annotation_files[idx],cali_file_path,img_dir.basename())) #img_dir.basename()　--> 主要是为了得到对应的相机参数
        return data_list


if __name__ == '__main__':

    import sys 
    sys.path.append('/home/zx/code2020/SPMD')
    from path import Path
    from IPython import embed
    from utils import read_json
    from Dataset.CMU.project import reproject
    import json
    import numpy as np
    import os
    import time
    dataset_path = Path('/home/zx/panoptic-toolbox')
    sub_dirs = dataset_path.dirs()
    useful_train_dirs = ['170407_haggling_a1'] # '170221_haggling_b1',
    useful_val_dirs = ['160422_ultimatum1','160906_pizza1'] 
    useful_img_dirs_train = ['00_00'] #,'00_01','00_02','00_03','00_04','00_05','00_06','00_07','00_08','00_09',
    useful_img_dirs_val = ['00_16','00_30']

    get_data = GetDataset(sub_dirs,useful_train_dirs,useful_img_dirs_train,useful_val_dirs,useful_img_dirs_val)
    train_data_list = get_data.get_train_data()
    # train_data_list = get_data.get_val_data()

    #create annotation json
    count = 1
    s_time = time.time()
    for idx in range(len(train_data_list)):
        img_path , anno_path = train_data_list[idx][0] , train_data_list[idx][1]
        cali_path , cam_id = train_data_list[idx][2] , train_data_list[idx][3]
        
        anno_file = read_json(anno_path)  
        cali_file = read_json(cali_path)
        cam_id = str(cam_id)
        lnum , rnum = int(cam_id.split('_')[0]) , int(cam_id.split('_')[1])

        cam_coors , pixel_coors , skel_with_conf , cam, resolution = reproject(anno_file,cali_file,(lnum,rnum))

        tmp = str(img_path).split('/')

        img_anno_name = tmp[-4] + '--' + tmp[-2] + "--" + tmp[-1].split('.')[0].split('_')[-1] 

        output_json_root = dataset_path / f'{tmp[-4]}' / 'json_file'          #/media/xuchengjun/datasets/CMU/170407_haggling_a1/json_file

        json_sub_dirs = output_json_root / f'{tmp[-2]}'

        if not json_sub_dirs.exists():
            os.makedirs(json_sub_dirs)

        output_json_path = json_sub_dirs / f'{img_anno_name}.json'
        output_json = dict()
        output_json['img_path'] = img_path

        if len(cam_coors) > 0:
            output_json['cam_coors'] = np.array(cam_coors)[:,:,:15].tolist()
            output_json['pixel_coors'] = np.array(pixel_coors)[:,:,:15].tolist()
            output_json['skel_with_conf'] = np.array(skel_with_conf)[:,:,:15].tolist()
        else:
            output_json['cam_coors'] = cam_coors
            output_json['pixel_coors'] = pixel_coors
            output_json['skel_with_conf'] = skel_with_conf   

        output_json['cam'] = cam.tolist()   #fx,fy,cx,cy
        output_json['img_width'] = resolution[0]
        output_json['img_height'] = resolution[1]   

        with open(output_json_path, 'w') as f:
            json.dump(output_json, f)
        
        print(f'working .. {count} / {len(train_data_list)}')
        count += 1
    e_time = time.time()
    print(f'done .. {(e_time - s_time) / 3600}')