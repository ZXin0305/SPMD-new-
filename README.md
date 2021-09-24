1.重新规范了代码，使用DP模式进行训练
2.改变了loss, 先让mask乘上predict,再和gt作loss
3.加进了parymid learning block & residual block， 但是这样的话网络就太大了， 把网络层的通道数减少
即：[2048,1024,512,256] --> [1024,512,256,128]
upsample channel 变成128
但是由于loss的计算方式吧，训练的时候特别慢， 并且center map的效果特别的差
