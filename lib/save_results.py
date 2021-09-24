from matplotlib import pyplot as plt
from path import Path
loss_path = Path('/home/xuchengjun/ZXin/SPMD')

#save results

def save_results(loss_name,loss_list, epoch):
    
    plt.figure(figsize=(10,5))
    plt.title(loss_name)
    plt.plot(loss_list)
    plt.savefig(fname=loss_path / f'{epoch}_' + loss_name + '.jpg')
