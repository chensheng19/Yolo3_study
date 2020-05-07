#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-05-02
# File name   : yolo3.py
# Description : Define yolo3 model,loss function,etc.
#
#=====================================================

import numpy as np
import tensorflow as tf
import utils
import common
import backbone
import getConfig

cfg = {}
cfg = getConfig.getConfig()

NUM_CLASS = len(utils.read_class_names(cfg['yolo_classes_names']))
ANCHORS = utils.get_anchors(cfg['yolo_anchors'])
STRIDES = np.array(cfg['yolo_strides'])
IOU_LOSS_THRESHOLD = cfg['yolo_iou_loss_threshold']

def yolo3(input_layer):
    branch_1,branch_2,conv = backbone.backbone(input_layer)

    conv = common.conv(conv,(1,512))
    conv = common.conv(conv,(3,1024))
    conv = common.conv(conv,(1,512))
    conv = common.conv(conv,(3,1024))
    conv = common.conv(conv,(1,512))

    conv_l_branch = common.conv(conv,(3,1024))
    l_output = common.conv(conv_l_branch,(1,3 * (NUM_CLASS + 5)),False,False)

    conv = common.conv(conv,(1,256))
    conv = common.upsample(conv,"resize")

    conv = tf.concat([conv,branch_2],axis = -1)
    conv = common.conv(conv,(1,256))
    conv = common.conv(conv,(3,512))
    conv = common.conv(conv,(1,256))
    conv = common.conv(conv,(3,512))
    conv = common.conv(conv,(1,256))

    conv_m_branch = common.conv(conv,(3,512))
    m_output = common.conv(conv_m_branch,(1,3 * (NUM_CLASS + 5)),False,False)

    conv = common.conv(conv,(1,128))
    conv = tf.concat([conv,branch_1],axis = -1)
    conv = common.conv(conv,(1,128))
    conv = common.conv(conv,(3,256))
    conv = common.conv(conv,(1,128))
    conv = common.conv(conv,(3,256))
    conv = common.conv(conv,(1,128))
    
    conv_s_branch = common.conv(conv,(3,256))
    s_output = common.conv(conv_s_branch,(1,3 * (NUM_CLASS + 5)),False,False)
    
    return [s_output,m_output,l_output]

def decode(conv_output,i=0):
    """
    Argument:
        conv_output : One of three types of convolution output,dim = [batch_size,output_size,output_size,anchor_per_scale*(5+num_class)]
        i           : the i th type convolution output
    Return:
        tensor of shape [batch_size,output_size,output_size,anchor_per_scale,5+num_class]
    """
    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(conv_output,(batch_size,output_size,output_size,3,5+NUM_CLASS))

    conv_raw_dxdy = conv_output[...,0:2]
    conv_raw_dwdh = conv_output[...,2:4]
    conv_raw_conf = conv_output[...,4:5]
    conv_raw_prob = conv_output[...,5:]

    y = tf.tile(tf.range(output_size,dtype=tf.int32)[:,tf.newaxis],[1,output_size])
    x = tf.tile(tf.range(output_size,dtype=tf.int32)[tf.newaxis,:],[output_size,1])

    xy_grid = tf.concat([x[...,tf.newaxis],y[...,tf.newaxis]],axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis,:,:,tf.newaxis,:],[batch_size,1,1,3,1])
    xy_grid = tf.cast(xy_grid,tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]
    pred_xywh = tf.concat([pred_xy,pred_wh],axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh,pred_conf,pred_prob],axis=-1)

def bbox_iou(boxes1,boxes2):
    """
    boxes : dim = [num_box,4], 4 : [x,y,w,h]
    """
    boxes1_area = boxes1[...,2] * boxes1[...,3]
    boxes2_area = boxes2[...,2] * boxes2[...,3]

    boxes1_xy = tf.concat([boxes1[...,:2] - boxes1[...,2:] * 0.5,
                           boxes1[...,:2] + boxes1[...,2:] * 0.5],axis=-1)
    boxes2_xy = tf.concat([boxes2[...,:2] - boxes2[...,2:] * 0.5,
                           boxes2[...,:2] + boxes2[...,2:] * 0.5],axis=-1)

    left_up = tf.maximum(boxes1_xy[...,:2],boxes2_xy[...,:2])
    right_down = tf.minimum(boxes1_xy[...,2:],boxes2_xy[...,2:])

    inter_section = tf.maximum(right_down - left_up,0.0)
    inter_area = inter_section[...,0] * inter_section[...,1]
    uniou_area = boxes1_area + boxes2_area - inter_area
    ious = 1.0 * inter_area / uniou_area

    return ious

def bbox_giou(boxes1,boxes2):
    """
     boxes : dim = [num_box,4], 4 : [x,y,w,h]
    """
    boxes1 = tf.concat([boxes1[...,:2] - boxes1[...,2:] * 0.5,
                        boxes1[...,:2] + boxes1[...,2:] * 0.5],axis=-1)
    boxes2 = tf.concat([boxes2[...,:2] - boxes2[...,2:] * 0.5,
                        boxes2[...,:2] + boxes2[...,2:] * 0.5],axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[...,:2],boxes1[...,2:]),
                        tf.maximum(boxes1[...,:2],boxes1[...,2:])],axis = -1)
    boxes2 = tf.concat([tf.minimum(boxes2[...,:2],boxes2[...,2:]),
                        tf.maximum(boxes2[...,:2],boxes2[...,2:])],axis = -1)

    boxes1_area = (boxes1[...,2] - boxes1[...,0]) * (boxes1[...,3] - boxes1[...,1])
    boxes2_area = (boxes2[...,2] - boxes2[...,0]) * (boxes2[...,3] - boxes2[...,1])

    left_up = tf.maximum(boxes1[...,:2],boxes2[...,:2])
    right_down = tf.minimum(boxes1[...,2:],boxes2[...,2:])

    inter_section = tf.maximum(right_down - left_up,0.)
    inter_area = inter_section[...,0] * inter_section[...,1]
    uniou_area = boxes1_area + boxes2_area - inter_area
    ious = 1.0 * inter_area / uniou_area

    enclose_left_up = tf.minimum(boxes1[...,:2],boxes2[...,:2])
    enclose_right_down = tf.maximum(boxes1[...,2:],boxes2[...,2:])
    enclose_section = tf.maximum(enclose_right_down - enclose_left_up,0.0)
    enclose_area = enclose_section[...,0] * enclose_section[...,1]
    gious = ious - 1.0 * (enclose_area - uniou_area) / enclose_area

    return gious

def loss_layer(conv,pred,label,bboxes,i = 0):
    """
    Argument:
        conv: 
    """
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = output_size * STRIDES[i]
    conv = tf.reshape(conv,(batch_size,output_size,output_size,3,5 + NUM_CLASS))

    conv_raw_conf = conv[...,4:5]
    conv_raw_prob = conv[...,5:]
    pred_xywh = pred[...,:4]
    pred_conf = pred[...,4:5]

    label_xywh = label[...,:4]
    respond_bbox = label[...,4:5]
    label_prob = label[...,5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh,label_xywh),axis = -1)
    input_size = tf.cast(input_size,tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[...,2] * label_xywh[...,3] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:,:,:,:,np.newaxis,:],bboxes[:,np.newaxis,np.newaxis,np.newaxis,:,:])
    max_iou = tf.expand_dims(tf.reduce_max(iou,axis = -1),axis = -1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESHOLD,tf.float32)
    conf_focal = tf.pow(tf.abs(respond_bbox - pred_conf),2)
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels = respond_bbox,logits = conv_raw_conf) + 
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels = respond_bbox,logits = conv_raw_conf))

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels = label_prob,logits = conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss,axis = [1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss,axis = [1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss,axis = [1,2,3,4]))

    return giou_loss,conf_loss,prob_loss

