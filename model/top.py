import sys
sys.path.append('/home/xuchengjun/ZXin/SPMD')
import torch
import torch.nn as nn
from model.conv import conv_bn_relu
from model.PRM import PRM
from IPython import embed

class HeadTop(nn.Module):
    def __init__(self, cnf, in_ch=3, out_ch=64):
        super(HeadTop, self).__init__()
        self.ori_shape = (cnf.resize_y // 2, cnf.resize_x // 2)
        self.top_conv = conv_bn_relu(in_ch, out_ch, kernel_size=7, stride=2, padding=3, has_bn=True, has_relu=True)  # 这里为什么都用的是（7,2,3）
        self.top_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # first prm layer followed by a max-pooling layer
        self.pry_1 = PRM(out_ch, out_ch, self.ori_shape, cnf, 'no_preact')
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # second prm layer
        # self.pry_2 = PRM(out_ch, out_ch, self.ori_shape, cnf, 'preacat')

    def forward(self, x):
        out = self.top_conv(x)   # the first layer to process feature
        out = self.top_maxpool(out)
        out = self.pry_1(out)
        out = self.maxpool_1(out)
        # out = self.pry_2(out)
        return out


if __name__ == '__main__':
    from Config.config import set_param
    from IPython import embed
    cnf = set_param()

    top = HeadTop(cnf, in_ch=64, out_ch=64)
    input = torch.ones(size=(1,64,512,832), dtype=torch.float32)
    out = top(input)
    embed()