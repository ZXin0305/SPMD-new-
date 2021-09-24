1.重新规范了代码，使用DP模式进行训练
2.改变了loss, 先让mask乘上predict,再和gt作loss
3.加进了parymid learning block & residual block， 但是这样的话网络就太大了， 把网络层的通道数减少
即：[2048,1024,512,256] --> [1024,512,256,128]
upsample channel 变成64, net input size 变成了(456,256),并且加进了scale\rotate\crop等图像增强的方式，
使用150000张图片进行训练，一个epoch的时间是3.15h
但是这种方法的效果并不是特别的好，一些姿态并不能很好的拟合出来，可能是因为训练的图像中并没有这种姿态
这个也就是openpose这些直接使用heatmap的优点

现在这个对于center map的输出效果还可以，基本上可以较为精确的找到center joint
