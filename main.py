#!/usr/bin/python3
# encoding: utf-8
import Config.config as conf
from trainer import Trainer
import os
import random
import torch
import numpy as np
from IPython import embed

#设置随机数种子
def set_seed(seed=2020):
    # seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def main():
    cnf = conf.set_param()
    trainer = Trainer(cnf)
    trainer.run()

if __name__ == '__main__':
    set_seed()
    main()