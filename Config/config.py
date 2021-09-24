
import argparse
import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 

def set_param():
    parser= argparse.ArgumentParser()
    #dataset & image 
    parser.add_argument('--data_path',type=str,default='/media/xuchengjun/datasets/CMU',help='the path to load the dataset')
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--resize_x',type=int,default=456)   # 456
    parser.add_argument('--resize_y',type=int,default=256)   # 256
    parser.add_argument('--outw',type=int,default=228)       # 改
    parser.add_argument('--outh',type=int,default=128)       # 改
    parser.add_argument('--oriw',type=int,default=1920)
    parser.add_argument('--orih',type=int,default=1080)
    parser.add_argument('--sigma',type=list,default=[5.0,5.0,4.0,4.0])   #control heatmap spread speed --> [center map, offset map, root depth map, depth map]
    parser.add_argument('--rel_thre',type=int,default=1)

    parser.add_argument('--data_format',type=str,default='cmu') #choose different dataset's joint format
    parser.add_argument('--stage_num',type=int,default=3)
    parser.add_argument('--root_id',type=int,default=2)
    parser.add_argument('--th',type=float,default=4.6052)  #only create heatmap uses

    """
    train 3D model config
    """
    #feature channels
    #if delete ears or eyes, the num is about 15
    parser.add_argument('--joint_num',type=int,default=15)
    parser.add_argument('--joint_ch',type=int,default=15)
    parser.add_argument('--offset_ch',type=int,default=15*2)
    parser.add_argument('--depth_ch',type=int,default=15)  # root deepth & root-relative channel
    parser.add_argument('--upsample_ch',type=int,default=64)

    #train
    parser.add_argument('--epoch',type=int,default=20)
    parser.add_argument('--log_dir',type=str,default='./log',help='the dir to store the summary')
    parser.add_argument('--checkpoint_period',type=int,default=2000)
    parser.add_argument('--ck_path',type=str,default='./checkpoint')
    parser.add_argument('--weights_only',type=bool,default=False)
    parser.add_argument('--gpu_num',type=int,default=2)
    parser.add_argument('--gpu_ids',type=list,default=[0,1])  #using dp , not ddp, DDP's setting in engine.py
    parser.add_argument('--max_iter_train',default=150000)
    parser.add_argument('--max_iter_val', default=0)

    #optimizer
    parser.add_argument('--base_lr',type=float,default=2e-4) # learning rate
    parser.add_argument('--weight_decay',type=float,default=8e-6)

    #dataloader
    parser.add_argument('--batch_size',type=int,default=8)  #if using ddp mode, the batch_size will be divide into per_gpu
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--model_path',default='/home/xuchengjun/ZXin/SPMD/pth')

    #image augmentation
    parser.add_argument('--do_normlize',type=bool, default=True)
    parser.add_argument('--norm_mean',type=list,default=[0.406, 0.456, 0.485])
    parser.add_argument('--norm_std',type=list,default=[0.225, 0.224, 0.229])
    parser.add_argument('--aug_flip',type=bool,default=False)
    parser.add_argument('--aug_prob',type=float,default=0.5)  #翻转的概率
    parser.add_argument('--flip_order',type=list,default=[0,1,2,9,10,11,12,13,14,3,4,5,6,7,8],help='cmu dataset flip order if do flip')
    parser.add_argument('--aug_rotate',type=bool,default=False) #旋转
    parser.add_argument('--rotate_max',type=int,default=10)
    parser.add_argument('--aug_trans',type=bool,default=False)
    parser.add_argument('--trans_pixel',type=int,default=20)

    # ---------------------------------------------------------
    # test
    parser.add_argument('--visthre',type=float,default=0.75)
    parser.add_argument('--max_people',type=int,default=20)

    arg = parser.parse_args()
    return arg

if __name__ == "__main__":
    from path import Path
    cnf = set_param()
    model_path = Path(cnf.model_path)
    epoch = 1
    print(model_path / f'epoch_{epoch}.pth')
