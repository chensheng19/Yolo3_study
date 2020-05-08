#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-05-07
# File name   : generate_data_txt.py 
# Description : generate a txt file containing path of image,and corresponding to the label box information
#               of the object in the image
#
#=====================================================

import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'),('2007', 'val')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(year, image_id, list_file):
    in_file = open('data/dataset/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


for year, image_set in sets:
    image_ids = open('data/dataset/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('data/dataset/%s_%s.txt'%(image_set,year), 'w')
    for image_id in image_ids:
        list_file.write('data/dataset/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
