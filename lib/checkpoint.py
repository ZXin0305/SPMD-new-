import torch
from collections import OrderedDict
from IPython import embed

def load_ck(log_dir, epoch, net, optimizer, scheduler):
    """
    loading training checkpoint
    """
    ck_path = log_dir / 'training.ck'
    if ck_path.exists():
        ck = torch.load(ck_path,map_location=torch.device('cpu'))
        print(f'[loading checkpoint --> {ck_path}')
        epoch = ck['epoch'] #last time's epoch

        # new_state_dict = OrderedDict()
        # for k,v in ck['model'].items(): #k:键名，v:对应的权值参数
        #     name = k[7:]
        #     new_state_dict[name] = v 
        # net.load_state_dict(new_state_dict)

        net.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
    return epoch, net, optimizer, scheduler

def save_ck(log_dir,epoch,net,optimizer,scheduler):
    """
    save training checkpoint
    """
    ck = {
        'epoch':epoch,
        'model':net.state_dict(),
        'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict()
    }

    torch.save(ck,log_dir / 'training.ck')

def load_state(net, checkpoint):
    source_state = checkpoint['state_dict'] # no module
    target_state = net.state_dict()  #module
    new_target_state = OrderedDict()

    #1.if no cuda using, remove the "module."
    # for k,v in target_state.items(): #k:键名，v:对应的权值参数
    #     name = k[7:]
    #     new_target_dict[name] = v

    # 2.
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))
    
    net.load_state_dict(new_target_state)

def load_state_with_no_ck(net, source_state):
    new_source_dict = OrderedDict()
    for k,v in source_state.items(): #k:键名，v:对应的权值参数
        name = k[7:]
        new_source_dict[name] = v
    net.load_state_dict(new_source_dict)  
