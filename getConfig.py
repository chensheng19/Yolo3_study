#! /usr/bin/env python
# coding:utf-8
#====================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
# Editor      : VIM
# File name   : getConfig.py
# Author      : Chen_Sheng19
# Date        : 2020-04-29
# Description : This file is used to read the configuration file "config.ini"
#
#====================================================

import configparser

def getConfig(config_file = "./config.ini"):
    parser = configparser.ConfigParser()
    parser.read(config_file)

    _conf_yolo = [(key,value) for key,value in parser.items("yolo")]
    _conf_train = [(key,value) for key,value in parser.items("train")]
    _conf_test = [(key,value) for key,value in parser.items("test")]
    
    return dict(_conf_yolo + _conf_train + _conf_test)
