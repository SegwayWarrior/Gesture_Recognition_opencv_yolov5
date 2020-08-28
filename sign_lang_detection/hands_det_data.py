#!/usr/bin/env python
# coding: utf-8

# Simple script to move and resize images for yolo training

import argparse
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image, ImageEnhance
from IPython.display import display
from shutil import copy
import cv2
import csv
import glob
import random

parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='+', type=str, default='', help='where you saved the data from create_dataset.py')
# parser.add_argument('--data-dest',  type=str, default='', help='new save location for modified dataset')
parser.add_argument('--zero_class', nargs='+', type=str, default='', help='2 entries (data_root data_dest)')
opt = parser.parse_args()
print(opt)

# move all files from class folders to single image folder
image_root = opt.data[0] + '/images'
image_dest1 = opt.data[1] + '/images_orig'
if not os.path.exists(image_dest1):
    os.makedirs(image_dest1)
image_dest2 = opt.data[1] + '/images'
if not os.path.exists(image_dest2):
    os.makedirs(image_dest2)

label_root = opt.data[0] + '/labels'
label_dest1 = opt.data[1] + '/labels_orig'
if not os.path.exists(label_dest1):
    os.makedirs(label_dest1)
label_dest2 = opt.data[1] + '/labels'
if not os.path.exists(label_dest2):
    os.makedirs(label_dest2)

for folder in os.listdir(image_root):
    for image in os.listdir(os.path.join(image_root,folder)):
        if not os.path.exists(os.path.join(image_dest1,image)):
            copy(os.path.join(image_root,folder,image), os.path.join(image_dest1,image))

for folder in os.listdir(label_root):
    for label in os.listdir(os.path.join(label_root,folder)):
        if not os.path.exists(os.path.join(label_dest1,label)):
            copy(os.path.join(label_root,folder,label), os.path.join(label_dest1,label))


for image in os.listdir(image_dest1):
    img = cv2.imread(os.path.join(image_dest1,image))
    img = cv2.resize(img, (416,416))
    cv2.imwrite(os.path.join(image_dest1,image),img)


# random shuffle

image_root = image_dest1
image_dest_train = image_dest2 + '/train'
image_dest_test = image_dest2 + '/test'

label_root = label_dest1
label_dest_train = label_dest2 +'/train'
label_dest_test = label_dest2 + '/test'

image_list = []
label_list = []

for image in os.listdir(image_root):
    image_list.append(image)
for label in os.listdir(image_root):
    label_list.append(label)

random.shuffle(image_list)  # randomly shuffles the ordering of filenames
split = int(0.8 * len(image_list))
train_filenames = image_list[:split]
test_filenames = image_list[split:]

for file in train_filenames:
    label_train = file[:-4] + '.txt'
    if not os.path.exists(os.path.join(image_dest_train,file)):
        copy(os.path.join(image_root,file), os.path.join(image_dest_train,file))
    if not os.path.exists(os.path.join(label_dest_train,label_train)):
        copy(os.path.join(label_root,label_train), os.path.join(label_dest_train,label_train))

for file in test_filenames:
    label_test = file[:-4] + '.txt'
    if not os.path.exists(os.path.join(image_dest_test,file)):
        copy(os.path.join(image_root,file), os.path.join(image_dest_test,file))
    if not os.path.exists(os.path.join(label_dest_test,label_test)):
        copy(os.path.join(label_root,label_test), os.path.join(label_dest_test,label_test))


# Edit ds for general hands classification (change all to 0 class)
if opt.zero_class != '':
    train_root = opt.zero_class[0] + '/train'
    test_root = opt.zero_class[0] + '/test'
    train_dest = opt.zero_class[1] + '/train'
    if not os.path.exists(train_dest):
        os.makedirs(train_dest)
    test_dest = opt.zero_class[1] + '/test'
    if not os.path.exists(test_dest):
        os.makedirs(test_dest)

    for cnt,file in enumerate(os.listdir(train_root)):
    #     if cnt == 1:
        txt_file = open(os.path.join(train_root,file))
        txt_info = txt_file.read()
        row_split = txt_info.split('\n')
        for row in row_split:
            col_split = row.split(' ')
            if col_split != ['']:
                class_ = int(col_split[0])
                xcenter = float(col_split[1])
                ycenter = float(col_split[2])
                width = float(col_split[3])
                height = float(col_split[4])
                new_class = 0
                info_list = [new_class, xcenter, ycenter, width, height]
                info_string = ''
                for cnt,info in enumerate(info_list):
                    if cnt == 0:
                        info_string += str(info)
                    else:
                        info_string += ' ' + str(info)
                if not os.path.exists(os.path.join(train_dest,file)):
                    newf = open(os.path.join(train_dest,file),'a+')
                    newf.write(info_string)
                    newf.write('\n')
                    newf.close()
        txt_file.close()

    for cnt,file in enumerate(os.listdir(test_root)):
    #     if cnt == 1:
        txt_file = open(os.path.join(test_root,file))
        txt_info = txt_file.read()
        row_split = txt_info.split('\n')
        for row in row_split:
            col_split = row.split(' ')
            if col_split != ['']:
                class_ = int(col_split[0])
                xcenter = float(col_split[1])
                ycenter = float(col_split[2])
                width = float(col_split[3])
                height = float(col_split[4])
                new_class = 0
                info_list = [new_class, xcenter, ycenter, width, height]
                info_string = ''
                for cnt,info in enumerate(info_list):
                    if cnt == 0:
                        info_string += str(info)
                    else:
                        info_string += ' ' + str(info)
                if not os.path.exists(os.path.join(test_dest,file)):
                    newf = open(os.path.join(test_dest,file),'a+')
                    newf.write(info_string)
                    newf.write('\n')
                    newf.close()
        txt_file.close()
