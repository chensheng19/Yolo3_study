#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-04-29
# File name   : backbone.py 
# Description : Implement the darknet53 
#
#=====================================================

import common as common
import tensorflow as tf

def backbone(input_layer):
    """
    Argument:
        input_layer:a 4-d tensor,[samples,rows,cols,channels]
    Return:
        three output
    """
    out_put = common.conv(input_layer,filter_shape = (3,32))

    out_put = common.conv(out_put,filter_shape = (3,64),padding = "valid")
    for i in range(1):
        out_put = common.res_block(out_put,num_filter = (32,64))
    
    out_put = common.conv(out_put,filter_shape = (3,128),padding = "valid")
    for i in range(2):
        out_put = common.res_block(out_put,num_filter = (64,128))

    out_put = common.conv(out_put,filter_shape = (3,256),padding = "valid")
    for i in range(8):
        out_put = common.res_block(out_put,num_filter = (128,256))
    branch_1 = out_put

    out_put = common.conv(out_put,filter_shape = (3,512),padding = "valid")
    for i in range(8):
        out_put = common.res_block(out_put,num_filter = (256,512))
    branch_2 = out_put

    out_put = common.conv(out_put,filter_shape = (3,1024),padding = "valid")
    for i in range(4):
        out_put = common.res_block(out_put,num_filter = (512,1024))
    
    return branch_1,branch_2,out_put
