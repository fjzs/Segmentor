# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:31:18 2021
copy file
@author: Adi Nugraha
"""
from shutil import copyfile, copy2
import numpy as np
import pandas as pd
import os

scr_path = 'C:/Users/adi/Downloads/VOC2012/SegmentationClass/'
tgt_path = 'C:/Users/adi/Documents/FourthBrain/CapstoneProject/GitRep/datasets/pascal/Segmentation_output/validation2/'
orgimg_path = 'C:/Users/adi/Documents/FourthBrain/CapstoneProject/GitRep/datasets/pascal/Segmentation_output/validation/'
csv_in   = 'pascal_segmented_classes_per_image.csv'
tst_path = 'C:/Users/adi/Documents/FourthBrain/CapstoneProject/GitRep/datasets/pascal/'

labels = pd.read_csv(tst_path + csv_in, index_col=1).drop('Unnamed: 0', axis = 1)
label_array = labels.to_numpy()
col_head = labels.columns
image_list = os.listdir(orgimg_path)


#copyfile(src, dst)

# 2nd option

for img in image_list:
    copy2(scr_path + img , tgt_path)  # dst can be a folder; use copy2() to preserve timestamp
