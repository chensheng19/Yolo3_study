#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-04-29
# File name   : utils.py 
# Description : Implement load_weight,read classes name,read prior anchors,image preprocess,draw image
#               compute iou,NMS,post process box
#
#=====================================================

import cv2
import random
import colorsys
import numpy as np
import getConfig

cfg = {}
cfg = getConfig.getConfig()

def read_class_names(class_file_path):
    names = {}
    with open(class_file_path,"r") as f:
        for indx ,name in enumerate(f):
            names[indx] = name.strip('\n')
    return names

def get_anchors(anchors_path):
    with open(anchors_path,"r") as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(","),dtype = np.float32)
    return anchors.reshape(3,3,2)

def image_preprocess(image,target_size,get_boxes = None):
    """
    Argument:
        image       : dim = [h,w,3]
        target_size : the input size of model
        get_boxes   : whether get boxes from the image,dim = (num_boxes,[x_min,y_min,x_max,y_max])

    Return:
        target_image: the size of image equals the size of input of model  
        get_boxes   : convert box coordinate according the same ratio
    """
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)

    s_h,s_w,_ = image.shape
    t_h,t_w = target_size

    ratio = min(t_h / s_h,t_w / s_w)
    new_h,new_w = int(ratio*s_h),int(ratio*s_w)
    new_image = cv2.resize(image,(new_w,new_h))

    mask = np.full((t_h,t_w,3),128.0)
    dh,dw = (t_h - new_h) // 2,(t_w - new_w) // 2
    mask[dh:dh+new_h,dw:dw+new_w,:] = new_image
    image_padd = mask / 255.0

    if get_boxes is None:
        return image_padd
    else:
        get_boxes[:,[0,2]] = get_boxes[:,[0,2]] * ratio + dw
        get_boxes[:,[1,3]] = get_boxes[:,[1,3]] * ratio + dh
        return image_padd,get_boxes

def draw_bbox(image,bboxes,classes = read_class_names(cfg['yolo_classes_names']),show_label = True):
    """
    Argument:
        image      : original image,dim = (h,w,3)
        bboxes     : all true/predict boxes of the image,dim = (num_box,[x_min,y_min,x_max,y_max,prob,class_id])
        classes    : class name list
        show_label : whether show the label of box
    Return:
        image with box
    """
    num_classes = len(classes)
    h,w,_ = image.shape

    hsv_tuples = [(1.0 * x / num_classes,1.0,1.0) for x in range(num_classes)]
    colors = list(map(lambda x:colorsys.hsv_to_rgb(*x),hsv_tuples))
    colors = list(map(lambda x:(int(x[0] * 255),int(x[1] * 255),int(x[2] * 255)),colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i,bbox in enumerate(bboxes):
        coordinates = np.array(bbox[:4],dtype = np.int32)
        font_sacle = 0.5
        score = bbox[4]
        class_id = int(bbox[5])
        bbox_color = colors[class_id]
        bbox_thick = int(0.6 * (h + w) / 600)
        coor1,coor2 = (coordinates[0],coordinates[1]),(coordinates[2],coordinates[3])
        cv2.rectangle(image,coor1,coor2,bbox_color,bbox_thick)

        if show_label:
            bbox_mes = '%s: %.2f'%(classes[class_id],score)
            t_size = cv2.getTextSize(bbox_mes,0,font_sacle,thickness = bbox_thick // 2)[0]
            cv2.rectangle(image,c1,(c1[0] + t_size[0],c1[1] - t_size[1] -3),bbox_color,-1)
            cv2.putText(image,bbox_mes,(c1[0],c1[1]-2),cv2.FONT_HERSHEY_SIMPLEX,font_scale,(0,0,0),
                        bbox_thick // 2,lineType = cv2.LINE_AA)

        return image

def bboxes_iou(boxes1,boxes2):
    """
    Argument:
        bboxes:dim = (num_box,4), 4 : [x_min,y_min,x_max,y_max]
    Retiurn:
        a np.array,dim = (num_box,1)
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[...,2] - boxes1[...,0]) * (boxes1[...,3] - boxes1[...,1])
    boxes2_area = (boxes2[...,2] - boxes2[...,0]) * (boxes2[...,3] - boxes2[...,1])

    left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
    right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])

    inter_section = np.maximum(right_down - left_up,0.)
    inter_area = inter_section[...,0] * inter_section[...,1]

    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area , np.finfo(np.float32).eps)

    return ious

def nms(bboxes,iou_threshold,sigma = 0.3,method = 'nms'):
    """
    Argument:
        bboxes        : dim = (num_box,6),6 : [x_min,y_min,x_max,y_max,prob,class]
        iou_threshold : filter the boxes whose iou over this value
        sigma         : argument for 'softnms'
    Return:
        a list containing boxes,every box for one object
    Description:    
        Non-maximum suppression
    """
    classes_in_image = list(set(bboxes[:,5]))
    best_bboxes = []

    for cls in classes_in_image:
        cls_mask = (bboxes[:,5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_idx = np.argmax(cls_bboxes[:,4])
            best_bbox = cls_bboxes[max_idx]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(cls_bboxes[:max_idx],cls_bboxes[max_idx+1:])
            iou = bboxes_iou(best_bbox[np.newaxis,:4],cls_bboxes[:,:4])
            weight = np.ones((len(iou),),dtype = np.float32)

            assert method in ['nms','softnms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'softnms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:,4] = cls_bboxes[:,4] * weight
            score_mask = cls_bboxes[:,4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def load_weights(model,weights_file):

    weight_file = open(weights_file,'rb')
    major,minor,revision,seen,_ = np.fromfile(weight_file,dtype=np.int32,count=5)

    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        input_channels = conv_layer.input_shape[-1]

        if i not in [58,66,74]:
            #darknet:[beta,gamma,mean,variance]
            bn_weights = np.fromfile(weight_file,dtype=np.float32,count=4*filters)
            #tensorflow:[gamma,beta,mean,variance]
            bn_weights = bn_weights.reshape((4,filters))[[1,0,2,3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(weight_file,dtype=np.float32,count=filters)

        #darknet:[out_ch,in_ch,h,w]
        conv_shape = (filters,input_channels,k_size,k_size)
        conv_weight = np.fromfile(weight_file,dtype=np.float32,count=np.product(conv_shape))
        #tensorflow:[h,w,in_ch,out_ch]
        conv_weight = conv_weight.reshape(conv_shape).transpose([2,3,1,0])

        if i not in [58,66,74]:
            conv_layer.set_weights([conv_weight])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weight,conv_bias])

    assert len(weight_file.read()) == 0
    weight_file.close()

def postprocess_boxes(pred_bbox,org_img_shape,input_size,score_threshold):
    """
    Argument:
        pred_bbox       : dim = (num_box,5+num_class)
        org_img_shape   : the h and w of original image,(h,w)
        input_size      : the h and w of input layer,(h,w)
        score_threshold :
    Return:
        
    """
    valid_scale = [0,np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:,0:4]
    pred_conf = pred_bbox[:,4]
    pred_prob = pred_bbox[:,5:]

    pred_coor = np.concatenate([pred_xywh[:,:2] - pred_xywh[:,2:4] * 0.5,
                                pred_xywh[:,:2] + pred_xywh[:,2:4] * 0.5],axis = -1)

    org_h,org_w = org_img_shape
    resize_ratio = min(input_size / org_h,input_size / org_w)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:,0::2] = 1.0 * (pred_coor[:,0::2] - dw) / resize_ratio
    pred_coor[:,1::2] = 1.0 * (pred_coor[:,1::2] - dh) / resize_ratio

    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)













































