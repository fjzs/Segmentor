# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:04:40 2021

@author: Adi Nugraha
Utilities to process images
"""

import os
import numpy as np
import cv2
from PIL import Image

def create_image_array(image_list, image_path):
    image_array = []
    for image_name in image_list:
            image = np.array(Image.open(os.path.join(image_path, image_name)))
            image = np.true_divide(image,255)
            image = cv2.resize(image,(256,256))
            image_array.append(image)

    return np.asarray(image_array)