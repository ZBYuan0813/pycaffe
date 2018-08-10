# -*- coding: UTF-8 -*-
import sys
import caffe
from caffe import layers as L
from caffe import params as P
'''
This file uses pycaffe to generate train.prototxt and test.prototxt
if you have train.prototxt you can ignore this file.
'''

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
    
    # inception_3a

    n.inception_3a_1x1 = L.Convolution(n.pool2,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=64,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_3a/1x1")
    n.inception_3a_relu_1x1 = L.ReLU(n.inception_3a_1x1,in_place=True,name = "inception_3a/relu_1x1")



    n.inception_3a_3x3_reduce = L.Convolution(n.pool2,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=96,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_3a/3x3_reduce")
    n.inception_3a_relu_3x3_reduce = L.ReLU(n.inception_3a_3x3_reduce,in_place=True,name = "inception_3a/relu_3x3_reduce")
    n.inception_3a_3x3 = L.Convolution(n.inception_3a_3x3_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=3,num_output=128,pad=1,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_3a/3x3")
    n.inception_3a_relu_3x3 = L.ReLU(n.inception_3a_3x3,in_place=True,name = "inception_3a/relu_3x3")



    n.inception_3a_5x5_reduce = L.Convolution(n.pool2,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=16,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_3a/5x5_reduce")
    n.inception_3a_relu_5x5_reduce = L.ReLU(n.inception_3a_5x5_reduce,in_place=True,name = "inception_3a/relu_5x5_reduce")
    n.inception_3a_5x5 = L.Convolution(n.inception_3a_5x5_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=5,num_output=32,pad=2,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_3a/5x5")
    n.inception_3a_relu_5x5 = L.ReLU(n.inception_3a_5x5,in_place=True,name = "inception_3a/relu_5x5")



    n.inception_3a_pool = L.Pooling(n.pool2,kernel_size=3,stride=1,pad=1,pool=P.Pooling.MAX,name = "inception_3a/pool")
    n.inception_3a_pool_proj = L.Convolution(n.inception_3a_pool,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                  kernel_size=1,num_output=32,weight_filler=dict(type="xavier"),
                  bias_filler=dict(type='constant',value=0.2),name = "inception_3a/pool_proj")
    n.inception_3a_relu_pool_proj = L.ReLU(n.inception_3a_pool_proj,in_place=True,name = "inception_3a/relu_pool_proj")
    

    n.inception_3a_output = L.Concat(n.inception_3a_1x1,n.inception_3a_3x3,n.inception_3a_5x5,n.inception_3a_pool_proj,name="inception_3a/output")
    ####
    # inception_3b
    n.inception_3b_1x1 = L.Convolution(n.inception_3a_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=128,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_3b/1x1")
    n.inception_3b_relu_1x1 = L.ReLU(n.inception_3b_1x1,in_place=True,name = "inception_3b/relu_1x1")



    n.inception_3b_3x3_reduce = L.Convolution(n.inception_3a_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=128,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_3b/3x3_reduce")
    n.inception_3b_relu_3x3_reduce = L.ReLU(n.inception_3b_3x3_reduce,in_place=True,name = "inception_3b/relu_3x3_reduce")
    n.inception_3b_3x3 = L.Convolution(n.inception_3b_3x3_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=3,num_output=192,pad=1,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_3b/3x3")
    n.inception_3b_relu_3x3 = L.ReLU(n.inception_3b_3x3,in_place=True,name = "inception_3b/relu_3x3")



    n.inception_3b_5x5_reduce = L.Convolution(n.inception_3a_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=32,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_3b/5x5_reduce")
    n.inception_3b_relu_5x5_reduce = L.ReLU(n.inception_3b_5x5_reduce,in_place=True,name = "inception_3b/relu_5x5_reduce")
    n.inception_3b_5x5 = L.Convolution(n.inception_3b_5x5_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=5,num_output=96,pad=2,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_3b/5x5")
    n.inception_3b_relu_5x5 = L.ReLU(n.inception_3b_5x5,in_place=True,name = "inception_3b/relu_5x5")



    n.inception_3b_pool = L.Pooling(n.inception_3a_output,kernel_size=3,stride=1,pad=1,pool=P.Pooling.MAX,name = "inception_3b/pool")
    n.inception_3b_pool_proj = L.Convolution(n.inception_3b_pool,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                  kernel_size=1,num_output=64,weight_filler=dict(type="xavier"),
                  bias_filler=dict(type='constant',value=0.2),name = "inception_3b/pool_proj")
    n.inception_3b_relu_pool_proj = L.ReLU(n.inception_3b_pool_proj,in_place=True,name = "inception_3b/relu_pool_proj")
    

    n.inception_3b_output = L.Concat(n.inception_3b_1x1,n.inception_3b_3x3,n.inception_3b_5x5,n.inception_3b_pool_proj,name="inception_3b/output")

    n.pool3 = L.Pooling(n.inception_3b_output,kernel_size=3,stride=2,pool=P.Pooling.MAX,name = "pool3/3x3_s2")
    ######################################################################################################
    # inception_4a
    n.inception_4a_1x1 = L.Convolution(n.pool3,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=192,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4a/1x1")
    n.inception_4a_relu_1x1 = L.ReLU(n.inception_4a_1x1,in_place=True,name = "inception_4a/relu_1x1")



    n.inception_4a_3x3_reduce = L.Convolution(n.pool3,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=96,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4a/3x3_reduce")
    n.inception_4a_relu_3x3_reduce = L.ReLU(n.inception_4a_3x3_reduce,in_place=True,name = "inception_4a/relu_3x3_reduce")
    n.inception_4a_3x3 = L.Convolution(n.inception_4a_3x3_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=3,num_output=208,pad=1,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_4a/3x3")
    n.inception_4a_relu_3x3 = L.ReLU(n.inception_4a_3x3,in_place=True,name = "inception_4a/relu_3x3")



    n.inception_4a_5x5_reduce = L.Convolution(n.pool3,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=16,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4a/5x5_reduce")
    n.inception_4a_relu_5x5_reduce = L.ReLU(n.inception_4a_5x5_reduce,in_place=True,name = "inception_4a/relu_5x5_reduce")
    n.inception_4a_5x5 = L.Convolution(n.inception_4a_5x5_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=5,num_output=48,pad=2,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_4a/5x5")
    n.inception_4a_relu_5x5 = L.ReLU(n.inception_4a_5x5,in_place=True,name = "inception_4a/relu_5x5")



    n.inception_4a_pool = L.Pooling(n.pool3,kernel_size=3,stride=1,pad=1,pool=P.Pooling.MAX,name = "inception_4a/pool")
    n.inception_4a_pool_proj = L.Convolution(n.inception_4a_pool,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                  kernel_size=1,num_output=64,weight_filler=dict(type="xavier"),
                  bias_filler=dict(type='constant',value=0.2),name = "inception_4a/pool_proj")
    n.inception_4a_relu_pool_proj = L.ReLU(n.inception_4a_pool_proj,in_place=True,name = "inception_4a/relu_pool_proj")
    

    n.inception_4a_output = L.Concat(n.inception_4a_1x1,n.inception_4a_3x3,n.inception_4a_5x5,n.inception_4a_pool_proj,name="inception_4a/output")   

    #######################################################################################################
    # inception_4b
    n.inception_4b_1x1 = L.Convolution(n.inception_4a_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=160,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4b/1x1")
    n.inception_4b_relu_1x1 = L.ReLU(n.inception_4b_1x1,in_place=True,name = "inception_4b/relu_1x1")



    n.inception_4b_3x3_reduce = L.Convolution(n.inception_4a_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=112,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4b/3x3_reduce")
    n.inception_4b_relu_3x3_reduce = L.ReLU(n.inception_4a_3x3_reduce,in_place=True,name = "inception_4b/relu_3x3_reduce")
    n.inception_4b_3x3 = L.Convolution(n.inception_4b_3x3_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=3,num_output=224,pad=1,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_4b/3x3")
    n.inception_4b_relu_3x3 = L.ReLU(n.inception_4b_3x3,in_place=True,name = "inception_4b/relu_3x3")



    n.inception_4b_5x5_reduce = L.Convolution(n.inception_4a_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=24,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4b/5x5_reduce")
    n.inception_4b_relu_5x5_reduce = L.ReLU(n.inception_4b_5x5_reduce,in_place=True,name = "inception_4b/relu_5x5_reduce")
    n.inception_4b_5x5 = L.Convolution(n.inception_4b_5x5_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=5,num_output=64,pad=2,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_4b/5x5")
    n.inception_4b_relu_5x5 = L.ReLU(n.inception_4b_5x5,in_place=True,name = "inception_4b/relu_5x5")



    n.inception_4b_pool = L.Pooling(n.inception_4a_output,kernel_size=3,stride=1,pad=1,pool=P.Pooling.MAX,name = "inception_4b/pool")
    n.inception_4b_pool_proj = L.Convolution(n.inception_4b_pool,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                  kernel_size=1,num_output=64,weight_filler=dict(type="xavier"),
                  bias_filler=dict(type='constant',value=0.2),name = "inception_4b/pool_proj")
    n.inception_4b_relu_pool_proj = L.ReLU(n.inception_4b_pool_proj,in_place=True,name = "inception_4b/relu_pool_proj")
    

    n.inception_4b_output = L.Concat(n.inception_4b_1x1,n.inception_4b_3x3,n.inception_4b_5x5,n.inception_4b_pool_proj,name="inception_4b/output")   
    ####################################################
    # inception_4c
    n.inception_4c_1x1 = L.Convolution(n.inception_4b_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=128,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4c/1x1")
    n.inception_4c_relu_1x1 = L.ReLU(n.inception_4c_1x1,in_place=True,name = "inception_4c/relu_1x1")



    n.inception_4c_3x3_reduce = L.Convolution(n.inception_4b_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=128,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4c/3x3_reduce")
    n.inception_4c_relu_3x3_reduce = L.ReLU(n.inception_4c_3x3_reduce,in_place=True,name = "inception_4c/relu_3x3_reduce")
    n.inception_4c_3x3 = L.Convolution(n.inception_4c_3x3_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=3,num_output=256,pad=1,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_4c/3x3")
    n.inception_4c_relu_3x3 = L.ReLU(n.inception_4c_3x3,in_place=True,name = "inception_4c/relu_3x3")



    n.inception_4c_5x5_reduce = L.Convolution(n.inception_4b_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=24,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4c/5x5_reduce")
    n.inception_4c_relu_5x5_reduce = L.ReLU(n.inception_4c_5x5_reduce,in_place=True,name = "inception_4c/relu_5x5_reduce")
    n.inception_4c_5x5 = L.Convolution(n.inception_4c_5x5_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=5,num_output=64,pad=2,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_4c/5x5")
    n.inception_4c_relu_5x5 = L.ReLU(n.inception_4c_5x5,in_place=True,name = "inception_4c/relu_5x5")



    n.inception_4c_pool = L.Pooling(n.inception_4b_output,kernel_size=3,stride=1,pad=1,pool=P.Pooling.MAX,name = "inception_4c/pool")
    n.inception_4c_pool_proj = L.Convolution(n.inception_4c_pool,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                  kernel_size=1,num_output=64,weight_filler=dict(type="xavier"),
                  bias_filler=dict(type='constant',value=0.2),name = "inception_4c/pool_proj")
    n.inception_4c_relu_pool_proj = L.ReLU(n.inception_4c_pool_proj,in_place=True,name = "inception_4c/relu_pool_proj")
    

    n.inception_4c_output = L.Concat(n.inception_4c_1x1,n.inception_4c_3x3,n.inception_4c_5x5,n.inception_4c_pool_proj,name="inception_4c/output")       
    #######################################
    # inception_4d
    n.inception_4d_1x1 = L.Convolution(n.inception_4c_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=112,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4d/1x1")
    n.inception_4d_relu_1x1 = L.ReLU(n.inception_4d_1x1,in_place=True,name = "inception_4d/relu_1x1")



    n.inception_4d_3x3_reduce = L.Convolution(n.inception_4c_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=144,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4d/3x3_reduce")
    n.inception_4d_relu_3x3_reduce = L.ReLU(n.inception_4d_3x3_reduce,in_place=True,name = "inception_4d/relu_3x3_reduce")
    n.inception_4d_3x3 = L.Convolution(n.inception_4d_3x3_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=3,num_output=288,pad=1,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_4d/3x3")
    n.inception_4d_relu_3x3 = L.ReLU(n.inception_4d_3x3,in_place=True,name = "inception_4d/relu_3x3")



    n.inception_4d_5x5_reduce = L.Convolution(n.inception_4c_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=32,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4d/5x5_reduce")
    n.inception_4d_relu_5x5_reduce = L.ReLU(n.inception_4d_5x5_reduce,in_place=True,name = "inception_4d/relu_5x5_reduce")
    n.inception_4d_5x5 = L.Convolution(n.inception_4d_5x5_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=5,num_output=64,pad=2,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_4d/5x5")
    n.inception_4d_relu_5x5 = L.ReLU(n.inception_4d_5x5,in_place=True,name = "inception_4d/relu_5x5")



    n.inception_4d_pool = L.Pooling(n.inception_4c_output,kernel_size=3,stride=1,pad=1,pool=P.Pooling.MAX,name = "inception_4d/pool")
    n.inception_4d_pool_proj = L.Convolution(n.inception_4d_pool,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                  kernel_size=1,num_output=64,weight_filler=dict(type="xavier"),
                  bias_filler=dict(type='constant',value=0.2),name = "inception_4d/pool_proj")
    n.inception_4d_relu_pool_proj = L.ReLU(n.inception_4d_pool_proj,in_place=True,name = "inception_4d/relu_pool_proj")
    

    n.inception_4d_output = L.Concat(n.inception_4d_1x1,n.inception_4d_3x3,n.inception_4d_5x5,n.inception_4d_pool_proj,name="inception_4d/output")   
    ####################################################
    # inception_4e
    n.inception_4e_1x1 = L.Convolution(n.inception_4d_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=256,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4e/1x1")
    n.inception_4e_relu_1x1 = L.ReLU(n.inception_4e_1x1,in_place=True,name = "inception_4e/relu_1x1")



    n.inception_4e_3x3_reduce = L.Convolution(n.inception_4d_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=160,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4e/3x3_reduce")
    n.inception_4e_relu_3x3_reduce = L.ReLU(n.inception_4e_3x3_reduce,in_place=True,name = "inception_4e/relu_3x3_reduce")
    n.inception_4e_3x3 = L.Convolution(n.inception_4e_3x3_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=3,num_output=320,pad=1,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_4e/3x3")
    n.inception_4e_relu_3x3 = L.ReLU(n.inception_4e_3x3,in_place=True,name = "inception_4e/relu_3x3")



    n.inception_4e_5x5_reduce = L.Convolution(n.inception_4d_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=32,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_4e/5x5_reduce")
    n.inception_4e_relu_5x5_reduce = L.ReLU(n.inception_4e_5x5_reduce,in_place=True,name = "inception_4e/relu_5x5_reduce")
    n.inception_4e_5x5 = L.Convolution(n.inception_4d_5x5_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=5,num_output=128,pad=2,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_4e/5x5")
    n.inception_4e_relu_5x5 = L.ReLU(n.inception_4e_5x5,in_place=True,name = "inception_4e/relu_5x5")



    n.inception_4e_pool = L.Pooling(n.inception_4d_output,kernel_size=3,stride=1,pad=1,pool=P.Pooling.MAX,name = "inception_4e/pool")
    n.inception_4e_pool_proj = L.Convolution(n.inception_4e_pool,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                  kernel_size=1,num_output=128,weight_filler=dict(type="xavier"),
                  bias_filler=dict(type='constant',value=0.2),name = "inception_4e/pool_proj")
    n.inception_4e_relu_pool_proj = L.ReLU(n.inception_4e_pool_proj,in_place=True,name = "inception_4e/relu_pool_proj")
    

    n.inception_4e_output = L.Concat(n.inception_4e_1x1,n.inception_4e_3x3,n.inception_4e_5x5,n.inception_4e_pool_proj,name="inception_4d/output")   
    ####################
    n.pool4 = L.Pooling(n.inception_4e_output,kernel_size=3,stride=2,pool=P.Pooling.MAX,name = "pool4/3x3_s2")

    #inception_5a
    n.inception_5a_1x1 = L.Convolution(n.pool4,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=256,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_5a/1x1")
    n.inception_5a_relu_1x1 = L.ReLU(n.inception_5a_1x1,in_place=True,name = "inception_5a/relu_1x1")



    n.inception_5a_3x3_reduce = L.Convolution(n.pool4,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=160,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_5a/3x3_reduce")
    n.inception_5a_relu_3x3_reduce = L.ReLU(n.inception_5a_3x3_reduce,in_place=True,name = "inception_5a/relu_3x3_reduce")
    n.inception_5a_3x3 = L.Convolution(n.inception_5a_3x3_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=3,num_output=320,pad=1,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_5a/3x3")
    n.inception_5a_relu_3x3 = L.ReLU(n.inception_5a_3x3,in_place=True,name = "inception_5a/relu_3x3")



    n.inception_5a_5x5_reduce = L.Convolution(n.pool4,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=32,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_5a/5x5_reduce")
    n.inception_5a_relu_5x5_reduce = L.ReLU(n.inception_5a_5x5_reduce,in_place=True,name = "inception_5a/relu_5x5_reduce")
    n.inception_5a_5x5 = L.Convolution(n.inception_5a_5x5_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=5,num_output=128,pad=2,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_5a/5x5")
    n.inception_5a_relu_5x5 = L.ReLU(n.inception_5a_5x5,in_place=True,name = "inception_5a/relu_5x5")



    n.inception_5a_pool = L.Pooling(n.pool4,kernel_size=3,stride=1,pad=1,pool=P.Pooling.MAX,name = "inception_5a/pool")
    n.inception_5a_pool_proj = L.Convolution(n.inception_5a_pool,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                  kernel_size=1,num_output=128,weight_filler=dict(type="xavier"),
                  bias_filler=dict(type='constant',value=0.2),name = "inception_5a/pool_proj")
    n.inception_5a_relu_pool_proj = L.ReLU(n.inception_5a_pool_proj,in_place=True,name = "inception_5a/relu_pool_proj")
    

    n.inception_5a_output = L.Concat(n.inception_5a_1x1,n.inception_5a_3x3,n.inception_5a_5x5,n.inception_5a_pool_proj,name="inception_5a/output")     
    ##############
    # inception_5b
    n.inception_5b_1x1 = L.Convolution(n.inception_5a_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=384,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_5b/1x1")
    n.inception_5b_relu_1x1 = L.ReLU(n.inception_5b_1x1,in_place=True,name = "inception_5b/relu_1x1")



    n.inception_5b_3x3_reduce = L.Convolution(n.inception_5a_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=192,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_5b/3x3_reduce")
    n.inception_5b_relu_3x3_reduce = L.ReLU(n.inception_5b_3x3_reduce,in_place=True,name = "inception_5b/relu_3x3_reduce")
    n.inception_5b_3x3 = L.Convolution(n.inception_5b_3x3_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=3,num_output=384,pad=1,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_5b/3x3")
    n.inception_5b_relu_3x3 = L.ReLU(n.inception_5b_3x3,in_place=True,name = "inception_5b/relu_3x3")



    n.inception_5b_5x5_reduce = L.Convolution(n.inception_5a_output,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                          kernel_size=1,num_output=48,weight_filler=dict(type="xavier"),
                          bias_filler=dict(type='constant',value=0.2),name = "inception_5b/5x5_reduce")
    n.inception_5b_relu_5x5_reduce = L.ReLU(n.inception_5b_5x5_reduce,in_place=True,name = "inception_5b/relu_5x5_reduce")
    n.inception_5b_5x5 = L.Convolution(n.inception_5b_5x5_reduce,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                      kernel_size=5,num_output=128,pad=2,weight_filler=dict(type="xavier"),
                      bias_filler=dict(type='constant',value=0.2),name = "inception_5b/5x5")
    n.inception_5b_relu_5x5 = L.ReLU(n.inception_5b_5x5,in_place=True,name = "inception_5b/relu_5x5")



    n.inception_5b_pool = L.Pooling(n.inception_5a_output,kernel_size=3,stride=1,pad=1,pool=P.Pooling.MAX,name = "inception_5b/pool")
    n.inception_5b_pool_proj = L.Convolution(n.inception_5b_pool,param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                  kernel_size=1,num_output=128,weight_filler=dict(type="xavier"),
                  bias_filler=dict(type='constant',value=0.2),name = "inception_5b/pool_proj")
    n.inception_5b_relu_pool_proj = L.ReLU(n.inception_5b_pool_proj,in_place=True,name = "inception_5b/relu_pool_proj")
    

    n.inception_5b_output = L.Concat(n.inception_5b_1x1,n.inception_5b_3x3,n.inception_5b_5x5,n.inception_5b_pool_proj,name="inception_5a/output")         
    ########
    n.fc = L.InnerProduct(n.inception_5b_output, param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                           num_output=4096, weight_filler=dict(type="gaussian",std=0.005),
                           bias_filler=dict(type='constant',value=0.1))

    n.relu = L.ReLU(n.fc, in_place=True)

    #Dropout6层
    n.drop=L.Dropout(n.fc,dropout_ratio=0.5,in_place=True)    #丢弃数据的概率

    # 全连接层fc7
    n.fc1 = L.InnerProduct(n.fc, param=[dict(lr_mult=1,decay_mult=1), dict(lr_mult=2,decay_mult=0)],
                           num_output=1024, weight_filler=dict(type="gaussian", std=0.005),
                           bias_filler=dict(type='constant', value=0.1))    
    if model:
      n.acc = L.Accuracy(n.fc8, n.label)
    else:
      if (args.loss_type=='SoftmaxWithLoss'):
        n.loss = L.SoftmaxWithLoss(n.fc1, n.label)
      else if(args.loss_type=='triplet_loss'):
        n.loss = L.RankHardLoss(n.fc1,n.label,neg_num=4,pair_size = 2,hard_ratio=0.5,rand_ratio=0.5,margin=1)


    return n.to_proto



def write_net():
    caffe_root = "solver/"    
    train_lmdb = caffe_root + "train_lmdb"                          # train.lmdb文件的位置
    test_lmdb = caffe_root + "test_lmdb"                            # test.lmdb文件的位置
    mean_file = caffe_root + "mean_file"                            # 均值文件的位置
    train_proto = caffe_root + "train_proto"                        # 保存train_prototxt文件的位置
    test_proto = caffe_root + "test_proto"                          # 保存test_prototxt文件的位置
    #写入prototxt文件
    with open(train_proto, 'w') as f:
        f.write(str(create_net(train_lmdb,64,mean_file,False)))

    #写入prototxt文件
    with open(test_proto, 'w') as f:
        f.write(str(create_net(test_lmdb, 32,mean_file, True)))
    return train_proto,test_proto


if __name__ == '__main__':
    write_net()

