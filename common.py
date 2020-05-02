#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-04-29
# File name   : common.py 
# Description : Defined convolution block, residual block, upsampling block
#
#=====================================================

import tensorflow as tf

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    
        If training = False, in training mode, it relies on the moving average 
    and variance of each batch for batch normalization, while in inference mode, 
    it relies on the average and variance of the entire training set for normalization.

        If training = True / None, in training mode, the mean and variance of the current
    batch are used for batch normalization, while in inference mode, the mean and variance
    of the entire training set are used for BatchNormalization
        
        Therefore,there are two separate concepts."training" controls the data normalization
    process,depending on which type of mean and variance, and "trainable" controls whether to
    update beta and gamma during normalization.

        In "keras.BatchNormalization",the default value of training is None. Obviously,we need
    the first option,training = False
    """
    def call(self,inputs,training = False):
        if not training:
            training = tf.constant(False)

        training = tf.logical_and(training,self.trainable)
        return super().call(inputs,training)

def conv(input_layer,filters_shape,padding = "same",activation = True,bn = True):
    """
    Argument:
        input_layer:a 4-d tensor,[num_samples,rows,cols,channels]
        filters_shape:a list containing size of filters and num of filters,[size,num]
        padding:style of padding,"same" or "valid"
        activation:select activate Z or not
        bn:select BatchNormalization or not
    Return:
        the output of convolutional layer
    """
    if padding == "same":
        stride = 1
    elif padding == "valid":
        stride = 2
        input_layer = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(input_layer)

    conv = tf.keras.layers.Conv2D(filters=filters_shape[1],kernel_size = filters_shape[0],strides = stride,
            padding = padding,use_bias = not bn,kernel_regularizer = tf.keras.regularizers.l2(0.0005),
            kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
            bias_initializer = tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = tf.keras.layers.BatchNormalization()(conv)
    if activation:
        conv = tf,keras.activations.relu(conv,alpha = 0.1)
    return conv

def res_block(input_layer,num_filter):
    """
    Argument:
        input_layer:a 4-d tensor,[samples,rows,cols,channels]
        num_filters:a tuple or list containing two integers,the num of filters of two convolutional layers
    Return:
        the out of res_block    
    """
    filters1,filters2 = num_filter
    short_cut = input_layer

    conv_out = conv(input_layer = input_layer,filters_shape = (1,filters1))
    conv_out = conv(input_layer = conv_out,filters_shape = (3,filters2))

    return tf.keras.layers.Add()([conv_out,short_cut])

def upsample(input_layer,method = "deconv"):
    """
    Double the ouput's h and w
    """
    assert method in ["resize","deconv"]
    
    if method == "resize":
        out_put = tf.compat.v1.image.resize(input_layer,(input_layer[1]*2,input_layer[2]*2),method = "nearest")
    elif method == "deconv":
        out_put = tf.keras.layers.Conv2DTranspose(filters = input_layer[-1],kernel_size = 2,strides = 2,
                padding = "same",kernel_initializer = tf.random_normal_initializer())
    return out_put




























