# -*- coding: UTF-8 -*-
import sys
import caffe
from caffe import layers as L
from caffe import params as P


def ResNet(lmdb,batch_size,mean_file,model):
    n=caffe.NetSpec()
    #数据层
    if model==False:
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=lmdb,
                              include=dict(phase=0),transform_param=dict(scale=1./255,mirror=True,
                              crop_size=227,mean_file=mean_file),ntop=2)
    if model==True:
        n.data,n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                                 include=dict(phase=1), transform_param=dict(scale=1. / 255,
                                mirror=True,crop_size=227,mean_file=mean_file), ntop=2)

    #卷积层conv1
    n.conv1 = L.Convolution(n.data,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=7,stride=2,num_output=64,pad=3,weight_filler=dict(type="gaussian",std=0.01),
                          bias_filler=dict(type='constant',value=0),name = "conv1/7x7_s2")
    #ReLu层
    n.relu1 = L.ReLU(n.conv1, in_place=True,name = "conv1/relu_7x7")

    #Pooling层
    n.pool1 = L.Pooling(n.conv1,kernel_size=2,stride=2,pool=P.Pooling.MAX,name = "pool1/3x3_s2")

    n.conv2 = L.Convolution(n.pool1,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=64,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "conv2/3x3_reduce")
    n.relu2 = L.ReLU(n.conv2,in_place=True,name = "conv2/relu_3x3_reduce")
    n.conv2_3x3 = L.Convolution(n.conv2,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=3,num_output=192,pad=1,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "conv2/3x3")
    n.relu2 = L.ReLU(n.conv2_3x3,in_place=True,name = "conv2/relu_3x3")
    n.pool2 = L.Pooling(n.conv2_3x3,kernel_size=2,stride=2,pool=P.Pooling.MAX,name = "pool2/3x3_s2")





def create_net(lmdb, batch_size,mean_file,model):

    n=caffe.NetSpec()
    #数据层
    if model==False:
        n.data,n.label=L.Data(batch_size=batch_size,backend=P.Data.LMDB,source=lmdb,
                              include=dict(phase=0),transform_param=dict(scale=1./255,mirror=True,
                              crop_size=227,mean_file=mean_file),ntop=2)
    if model==True:
        n.data,n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                                 include=dict(phase=1), transform_param=dict(scale=1. / 255,
                                mirror=True,crop_size=227,mean_file=mean_file), ntop=2)

    #卷积层conv1
    n.conv1=L.Convolution(n.data,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=11,stride=4,num_output=96,weight_filler=dict(type="gaussian",std=0.01),
                          bias_filler=dict(type='constant',value=0))
    #ReLu层
    n.relu1 = L.ReLU(n.conv1, in_place=True)

    #LRN层
    n.norm1=L.LRN(n.conv1,local_size=5,alpha=0.0001,beta=0.75)

    #Pooling层
    n.pool1=L.Pooling(n.norm1,kernel_size=3,stride=2,pool=P.Pooling.MAX)

    #卷积层conv2
    n.conv2 = L.Convolution(n.pool1,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                            kernel_size=5,num_output=256,pad=2,group=2,weight_filler=dict(type="gaussian",std=0.01),
                            bias_filler=dict(type='constant',value=0.1))

    # ReLu2层
    n.relu2 = L.ReLU(n.conv2, in_place=True)

    # LRN2层
    n.norm2 = L.LRN(n.conv2, local_size=5, alpha=0.0001, beta=0.75)

    # Pooling2层
    n.pool2 = L.Pooling(n.norm2, kernel_size=3, stride=2, pool=P.Pooling.MAX)


    # 卷积层conv3
    n.conv3 = L.Convolution(n.pool2, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                            kernel_size=3, num_output=384, pad=1, weight_filler=dict(type="gaussian", std=0.01),
                            bias_filler=dict(type='constant', value=0))
    # ReLu3层
    n.relu3 = L.ReLU(n.conv3, in_place=True)


    # 卷积层conv4
    n.conv4 = L.Convolution(n.conv3, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                            kernel_size=3, num_output=384, pad=1, group=2,
                            weight_filler=dict(type="gaussian", std=0.01),
                            bias_filler=dict(type='constant', value=0.1))
    # ReLu4层
    n.relu4 = L.ReLU(n.conv4, in_place=True)


    # 卷积层conv5
    n.conv5 = L.Convolution(n.conv4, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                            kernel_size=3, num_output=256, pad=1, group=2,
                            weight_filler=dict(type="gaussian", std=0.01),
                            bias_filler=dict(type='constant', value=0.1))
    # ReLu5层
    n.relu5 = L.ReLU(n.conv5, in_place=True)

    # Pooling5层
    n.pool5 = L.Pooling(n.conv5, kernel_size=3, stride=2, pool=P.Pooling.MAX)

    #全连接层fc6
    n.fc6 = L.InnerProduct(n.pool5, param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                           num_output=4096, weight_filler=dict(type="gaussian",std=0.005),
                           bias_filler=dict(type='constant',value=0.1))

    n.relu6 = L.ReLU(n.fc6, in_place=True)

    #Dropout6层
    n.drop6=L.Dropout(n.fc6,dropout_ratio=0.5,in_place=True)    #丢弃数据的概率

    # 全连接层fc7
    n.fc7 = L.InnerProduct(n.fc6, param=[dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                           num_output=4096, weight_filler=dict(type="gaussian", std=0.005),
                           bias_filler=dict(type='constant', value=0.1))

    # ReLu7层
    n.relu7 = L.ReLU(n.fc7, in_place=True)

    # Dropout7层
    n.drop7 = L.Dropout(n.fc7,dropout_ratio=0.5,in_place=True)  # 丢弃数据的概率

    # 全连接层fc8

    n.fc8 = L.InnerProduct(n.fc7, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                           num_output=1000, weight_filler=dict(type="gaussian", std=0.01),
                           bias_filler=dict(type='constant', value=0))

    if model:
        n.acc = L.Accuracy(n.fc8, n.label)
    else:
        n.loss = L.SoftmaxWithLoss(n.fc8, n.label)

    return n.to_proto()


def write_net(args):
    caffe_root = "./"    
    train_lmdb = caffe_root + args.train_lmdb                          # train.lmdb文件的位置
    test_lmdb = caffe_root + args.test_lmdb                            # test.lmdb文件的位置
    mean_file = caffe_root + args.mean_file                            # 均值文件的位置
    train_proto = caffe_root + args.train_proto                        # 保存train_prototxt文件的位置
    test_proto = caffe_root + args.test_proto                          # 保存test_prototxt文件的位置
    #写入prototxt文件
    with open(train_proto, 'w') as f:
        f.write(str(create_net(train_lmdb, mean_file, batch_size=64,False)))

    #写入prototxt文件
    with open(test_proto, 'w') as f:
        f.write(str(create_net(test_lmdb, mean_file, batch_size=32, True)))


if __name__ == '__main__':
    write_net()

