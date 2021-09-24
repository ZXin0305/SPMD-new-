import sys
sys.path.append('/home/xuchengjun/ZXin/SPMD')
from tensorboardX import SummaryWriter
import torch
from base_net import Single_hourglass_block , input_net
from model.net import Global_Net
from IPython import embed
from Config.config import set_param
import numpy as np

if __name__ == '__main__':
    cnf = set_param()
    net = Global_Net(cnf)
    input_value = torch.rand((1,3,540,960),dtype=torch.float32)

    res_c,res_o,res_r,kps_weight = net(input_value)
    with SummaryWriter(comment='net') as w:
        w.add_graph(net,res_c[0][0])