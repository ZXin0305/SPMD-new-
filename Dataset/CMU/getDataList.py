import random
from IPython import embed
import random

class GetDataset():
    def __init__(self,sub_dirs,useful_train_dirs,useful_img_dirs_train,\
                      useful_val_dirs,useful_img_dirs_val):
        self.sub_dirs = sub_dirs
        self.useful_train_dirs = useful_train_dirs
        self.useful_img_dirs_train = useful_img_dirs_train
        self.useful_val_dirs = useful_val_dirs
        self.useful_img_dirs_val = useful_img_dirs_val

    def get_train_data(self, max_iter_train, use_all_train_files=False):
        data_list = []
        for sub_dir in self.sub_dirs:
            if sub_dir.basename() in self.useful_train_dirs:
                json_file_path = sub_dir / 'json_file'
                json_file_dirs = json_file_path.dirs()

                for json_file_dir in json_file_dirs:
                    if json_file_dir.basename() in self.useful_img_dirs_train:
                        json_files_list = json_file_dir.files()  # current img_dir's all img file

                        for idx in range(len(json_files_list)):
                            file_path = json_files_list[idx]
                            data_list.append(file_path)
                            
        sample_data_list = []
        if use_all_train_files:
            pass
        else:
            sample_data_list = random.sample(data_list, max_iter_train)
        return sample_data_list


    def get_val_data(self, max_iter_val, use_all_val_files=False):
        data_list = []
        for sub_dir in self.sub_dirs:
            if sub_dir.basename() in self.useful_val_dirs:
                json_file_path = sub_dir / 'json_file'
                json_file_dirs = json_file_path.dirs()

                for json_file_dir in json_file_dirs:
                    if json_file_dir.basename() in self.useful_img_dirs_val:
                        json_files_list = json_file_dir.files()

                        for idx in range(len(json_files_list)):
                            file_path = json_files_list[idx]
                            data_list.append(file_path)
        sample_data_list = []
        if use_all_val_files:
            pass
        else:
            sample_data_list = random.sample(data_list, max_iter_val)
        return sample_data_list
