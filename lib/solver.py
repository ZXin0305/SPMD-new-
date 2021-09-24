"""
when training, set optimizor & lr scheduler
"""
import torch.optim as optim
from torch.optim import lr_scheduler

# self.optimizer = optim.Adam(params=self.net.parameters(),lr=self.lr) #,betas=(0.9,0.999),eps=1e-08,weight_decay=8e-6
# self.optimizer = optim.Adagrad(params=self.net.parameters(),lr=self.lr)
#学习率调整器

def make_optimizer(cnf, net, num_gpu):
    optimizer = optim.Adam(net.parameters(),
                           lr=cnf.base_lr * num_gpu,
                           betas=(0.9,0.999), eps=1e-08,
                           weight_decay=cnf.weight_decay)
    return optimizer


def make_lr_scheduler(optimizer, is_multiStep=True, is_step=True):
    scheduler = None
    if is_multiStep:
        drop_after_epoch = [5, 10, 15]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.1)
    elif is_step:
        scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)

    return scheduler
