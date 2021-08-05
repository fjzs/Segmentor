import numpy as np
import pandas as pd
from stat_perform import *
from img_utils import *

tst_path = './datasets/pascal'
labels = pd.read_csv(tst_path + 'pascal_segmented_classes_per_image.csv', index_col='image_filename')
labels= labels.iloc[: , 1:]
label_array = labels.to_numpy()
prediction = label_array [1]
target = label_array [1]
image_list = None
image_path = tst_path + '/Segmentation_input/validation' 

# Get the test image
image_for_prediction = create_image_array(image_list, image_path)

# Calculate iou?
iou_m('modelDeepLabV3_Mila.tflite', labels, image_for_prediction)

# Calculate uou per class?
iou_per_class('modelDeepLabV3_Mila.tflite', labels, image_for_prediction) 