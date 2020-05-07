#! usr/bin/env python
# coding:utf-8
#=====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
#
# Author      : Chen_Sheng19
# Editor      : VIM
# Create time : 2020-05-07
# File name   : train.py
# Description : for training model with VOC2007 dataset
#
#=====================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import utils
import dataset
from yolo3 import yolo3,decode,loss_layer
import getConfig

cfg = {}
cfg = getConfig.getConfig()

trainset = dataset.Dataset('train')
logdir = "./model_dir/log"
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1,trainable = False,dtype=tf.int64)
warmup_steps = cfg['train_warmup_epochs']
total_steps = cfg['train_epochs'] * steps_per_epoch

input_tensor = tf.keras.layers.Input([416,416,3])
conv_tensors = yolo3(input_tensor)

output_tensor = []
for i,conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor,i)
    output_tensor.append(conv_tensor)
    output_tensor.append(pred_tensor)

model = tf.keras.Model(input_tensor,output_tensor)
opt = tf.keras.optimizers.Adam()

if os.path.exists(logdir):
    shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

def train_step(image_data,target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data,training = True)
        giou_loss = conf_loss = prob_loss = 0

        for i in range(3):
            conv,pred = pred_result[i*2],pred_result[i*2+1]
            loss_items = loss_layer(pred,conv,target[i],target[i+3],i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss,model.trainable_variables)
        opt.apply_gradients(zip(gradients,model.trainable_variables))

        tf.print("=> STEP %4d  Learning-rate: %.6f giou-loss: %4.2f conf-loss: %4.2f prob-loss: %4.2f total-loss: %4.2f"%(global_steps,
            opt.lr.numpy(),giou_loss,conf_loss,prob_loss,total_loss))

        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg['train_learning_rate_init']
        else:
            lr = cfg['train_learning_rate_end'] + 0.5 *(cfg['train_learning_rate_init'] - cfg['train_learning_rate_end']) * (
                    1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))

        with writer.as_default():
            tf.summary.scalar("lr",opt.lr,step = global_steps)
            tf.summary.scalar("loss/total_loss",total_loss,step = global_steps)
            tf.summary.scalar("loss/giou_loss",giou_loss,step = global_steps)
            tf.summary.scalar("loss/conf_loss",conf_loss,step = global_steps)
            tf.summary.scalar("loss/prob_loss",prob_loss,step = global_steps)
        writer.flush()


for epoch in range(cfg['train_epochs']):
    for image_data,*target in trainset:
        train_step(image_data,target)
    model.save_weights("./yolo3")






























