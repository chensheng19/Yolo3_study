#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-05-01
# File name   : dataset.py
# Description : Create a source dataset from your input data,which is a iterator
#
#=====================================================

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import utils
import getConfig

cfg = {}
cfg = getConfig.getConfig()

class Dataset(object):

    def __init__(self,dataset_type):
        self.annot_path = cfg['train_annot_path'] if  dataset_type == 'train' else cfg['test_annot_path']
        self.input_size = cfg['train_input_size'] if dataset_type == 'train' else cfg['test_input_size']
        self.batch_size = cfg['train_batch_size'] if dataset_type == 'train' else cfg['test_batch_size']
        self.data_aug = cfg['train_data_augement'] if dataset_type == 'train' else cfg['test_data_augement']

        self.train_input_sizes = cfg['train_input_size']
        self.strides = np.array(cfg['yolo_strides'])
        self.classes = utils.read_class_names(cfg['yolo_classes_names'])
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg['yolo_anchors']))
        self.anchor_per_scale = cfg['yolo_anchor_per_scale']
        self.max_bbox_per_scale = 150
        self.train_output_size = self.train_input_sizes // self.strides

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotation(self,dataset_type):
        with open(self.annot_path,'r') as _:
            txt = _.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def random_horizotal_flip(self,image,bboxes):
        if random.random() < 0.5:
            _,w_ = image.shape
            image = image[:,::-1,:]
            bboxes[:,[0,2]] = w - bboxes[:,[2,0]]
        return image,bboxes

    def random_crop(self,image,bboxes):
        if random.random() < 0.5:
            h,w,_ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:,0:2],axis=0),np.max(bboxes[:,2:4],axis=0)],axis=-1)

            max_l_random = max_bbox[0]
            max_u_random = max_bbox[1]
            max_r_random = w - max_bbox[2]
            max_d_random = h - max_bbox[3]

            crop_x_min = max(0,int(max_bbox[0] - random.uniform(0,max_l_random)))
            crop_y_min = max(0,int(max_bbox[1] - random.uniform(0,max_u_random)))
            crop_x_max = max(w,int(max_bbox[2] + random.uniform(0,max_r_random)))
            crop_y_max = max(h,int(max_bbox[3] + random.uniform(0,max_d_random)))

            image = image[crop_y_min:crop_y_max,crop_x_min:crop_x_max]

            bboxes[:,[0,2]] = bboxes[:,[0,2]] - crop_x_min
            bboxes[:,[1,3]] = bboxes[:,[1,3]] - crop_y_min

        return image,bboxes

    def random_translate(self,image,bboxes):
        if random.random() < 0.5:
            h,w,_ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:,0:2],axis=0),np.max(bboxes[:,2:4],axis=0)],axis=-1)
            
            max_l_random = max_bbox[0]
            max_u_random = max_bbox[1]
            max_r_random = w - max_bbox[2]
            max_d_random = h - max_bbox[3]

            tx = random.uniform(-(max_l_random - 1),(max_r_random - 1))
            ty = random.uniform(-(max_u_random - 1),(max_d_random - 1))

            M = np.array([[1,0,tx],[0,1,ty]])
            image = cv2.warpAffine(image,M,(w,h))

            bboxes[:,[0,2]] = bboxes[:,[0,2]] + tx
            bboxes[:,[1,3]] = bboxes[:,[1,3]] + ty

        return image,bboxes

    def parse_annotation(self,annotation):
        line = annotation.split()
        image_path = line[0]
        
        image = np.array(cv2.imread(image_path))
        bboxes = np.array(list(map(lambda x:int(float(x)),box.split(','))) for box in line[1:])
        image,bboxes = utils.image_preprocess(np.copy(image),
                [self.train_input_sizes,self.train_input_sizes],np.copy(bboxes))
        return image,bboxes

    def bbox_iou(self,boxes1,boxes2):
        """
        Argument:
            boxes:[num_box,4],4 : x,y,h,w (x,y refers to the center coordinates)
        """
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[...,2] * boxes1[...,3]
        boxes2_area = boxes2[...,2] * boxes2[...,3]

        #[x,y,h,w] -> [x_min,y_min,x_max,y_max]
        boxes1 = np.concatenate([boxes1[...,:2] - boxes1[...,2:] * 0.5],
                                [boxes1[...,:2] + boxes1[...,2:] * 0.5],axis = -1)
        boxes2 = np.concatenate([boxes2[...,:2] - boxes2[...,2:] * 0.5],
                                [boxes2[...,:2] + boxes2[...,2:] * 0.5],axis = -1)

        left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
        right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])

        inter_section = np.maximum(right_down - left_up,0.0)
        inter_area = inter_section[...,0] * inter_section[...,1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = 1.0 * inter_area / union_area

        return ious

    def preprocess_true_boxes(self,bboxes):
        """
        Argument:
            bboxes: the true boxes in image,dim=[num_box,5], 5 : x_min,y_min,x_max,y_max,class_id
        Return:
            label_*: the label containing [x_cneter,y_center,w,h,conf,class_onehot] of box in source image,only the label 
                     is bound to prior whose conf= 1,others is 0.dim=(out_size,out_size,3,classes+5),  
            *bboxes: the box params containing [x_cneter,y_center,w,h],dim=(num_box,3).Each column represents 
                     a type of convolution output,if the box belongs of the column,this is a value of [idx_box,column],else is 0.
            sbboxes: s refers to the grid of output is smaller.
        """
        label = [np.zeros((self.train_output_size[i],self.train_output_size[i],self.anchor_per_scale,
                            5+self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale,4)) for i in range(3)]
        bbox_count = np.zeros((3,))

        for box in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_id = bbox[4]

            onehot = np.zeros(self.num_classes,dtype = np.float32)
            onehot[bbox_class_id] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[:2] + bbox_coor[2:]) * 0.5],
                                        [np.abs(bbox_coor[2:] - bbox_coor[:2])],axis = -1)
            bbox_xywh_scale = 1.0 * bbox_xywh[np.newaxis,:] / self.strides[:,np.newaxis]

            iou = []
            exist_positive = False
            # i refers to i th output scale
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale,4))
                anchors_xywh[:,0:2] = np.floor(bbox_xywh_scale[i,0:2]).astype(np.int32) + 0.5
                anchors_xywh[:,2:4] = self.anchors[i]

                iou_sacle = self.bbox_iou(bbox_xywh_scale[i][np.newaxis,:],anchors_xywh)
                iou.append(iou_sacle)
                iou_mask = iou_sacle > 0.3

                if np.any(iou_mask):
                    xind,yind = np.floor(bbox_xywh_scale[i,0:2]).astype(np.int32)
                    
                    label[i][xind,yind,iou_mask,:] = 0
                    label[i][xind,yind,iou_mask,0:4] = bbox_xywh
                    label[i][xind,yind,iou_mask.4:5] = 1.0
                    label[i][xind,yind,iou_mask.5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind,:4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_idx = np.argmax(np.array(iou).reshape(-1),axis = -1)
                best_detect = int(best_anchor_idx / self.anchor_per_scale)
                best_anchor = int(best_anchor_idx % self.anchor_per_scale)
                xind,yind = np.floor(bbox_xywh_scale[best_detect,0:2]).astype(np.int32)

                label[best_detect][xind,yind,best_anchor,:] = 0
                label[best_detect][xind,yind,best_anchor,:4] = bbox_xywh
                label[best_detect][xind,yind,best_anchor,4:5] = 1
                label[best_detect][xind,yind,best_anchor,5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind,:4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox,load_mbbox,label_lbbox = label
        sbboxes,mbboxes,lbboxes = bboxes_xywh

        return label_sbbox,load_mbbox,label_lbbox,sbboxes,mbboxes,lbboxes

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)

            batch_images = np.zeros((self.batch_size,self.train_input_size,train_input_size,3),dtype=np.float32)
            batch_label_sbbox = np.zeros((self.batch_size,self.train_output_size[0],self.train_output_size[0],self.anchor_per_scale,
                                          self.num_classes+5),dtype = np.float32)
            batch_label_mbbox = np.zeros((self.batch_size,self.train_output_size[1],self.train_output_size[1],self.anchor_per_scale,
                                          self.num_classes+5),dtype = np.float32)
            batch_label_lbbox = np.zeros((self.batch_size,self.train_output_size[2],self.train_output_size[2],self.anchor_per_scale,
                                          self.num_classes+5),dtype = np.float32)

            batch_sbboxes = np.zeros((self.batch_size,self.max_bbox_per_scale,4),dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size,self.max_bbox_per_scale,4),dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size,self.max_bbox_per_scale,4),dtype=np.float32)
            
            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples

                    annotation = self.annotations[index]
                    image,bboxes = self.parse_annotation(annotation)
                    label_sbbox,load_mbbox,label_lbbox,sbboxes,mbboxes,lbboxes = self.preprocess_true_boxes(bboxes)

                    batch_images[num,...] = image
                    batch_label_sbbox[num,...] = label_sbbox
                    batch_label_mbbox[num,...] = label_mbbox
                    batch_label_lbbox[num,...] = label_lbbox

                    batch_sbboxes[num,...] = sbboxes
                    batch_mbboxes[num,...] = mbboxes
                    batch_lbboxes[num,...] = lbboxes

                    num += 1
                self.batch_count += 1
                return batch_images,batch_label_sbbox,batch_label_mbbox,batch_label_lbbox,batch_sbboxes,batch_mbboxes,batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def __len__(self):
        return self.num_batchs

