#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-05-07
# File name   : image_detect.py 
# Description : execute object detection on one image
#
#=====================================================

import cv2
import numpy as np
import tensorflow as tf
from yolo3 import yolo3,decode
import utils
from PIL import Image

input_size = 416
image_path = "./predict_image/hourse.jpg"

input_layer = tf.keras.layers.Input([input_size,input_size,3])
feature_maps = yolo3(input_layer)

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = utils.image_preprocess(np.copy(original_image),(input_size,input_size))
image_data = image_data[np.newaxis,...].astype(np.float32)

bbox_tensors = []
for i,fm in enumerate(feature_maps):
    bbox_tensor = decode(fm,i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer,bbox_tensors)
utils.load_weights(model,"./model_dir/yolov3")
model.summary()

pred_bbox = model.predict(image_data)
pred_bbox = [tf.reshape(x,(-1,tf.shape(x)[-1])) for x in pred_bbox]
pred_bbox = tf.concat(pred_bbox,axis=0)
bboxes = utils.postprocess_boxes(pred_bbox,original_image_size,input_size,0.3)
bboxes = utils.nms(bboxes,0.45,'nms')

image = utils.draw_bbox(original_image,bboxes)
image = Image.fromarray(image)
image.show()














