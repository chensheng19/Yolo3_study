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

    _conf_str = [(key,str(value)) for key,value in parser.items("strings")]
    _conf_int = [(key,int(value)) for key,value in parser.items("ints")]
    _conf_float = [(key,float(value)) for key,value in parser.items("floats")]
    mid_list= [(key,value) for key,value in parser.items("lists")]
    _conf_list = []
    for i in range(len(mid_list)):
        key = mid_list[i][0]
        value = mid_list[i][1]
        value = value.strip('[[]]')
        value = [int(x) for x in value.split(',')]
        mid = (key,value)
        _conf_list.append(mid)
    
    return dict(_conf_str + _conf_int + _conf_float + _conf_list)
    
