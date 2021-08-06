import numpy as np
import pandas as pd
from img_utils import *
#from stat_perform import *
from utilities import *

'''
This program will calculate the IoU per emage per class
Inputs:
    - .tflite: a  segmentation model
    - .jpg: a picture from pascal
    - pascal_segmented_classes_per_image.csv file

Output:
    - CSV file contains iou_score (float), 
        iou_per_class_array (float array size 1x20 an entry per class), 
        time_milisecs (time in miliseconds)

'''

tst_path = './datasets/pascal/'
mdl_path = './static/model/'
labels = pd.read_csv(tst_path + 'pascal_segmented_classes_per_image.csv', index_col=1).drop('Unnamed: 0', axis = 1)
label_array = labels.to_numpy()
col_head = labels.columns
image_path = tst_path + '/Segmentation_input/validation/' 
image_list = os.listdir(image_path)

# # Get the test image
# image_for_prediction = create_image_array(image_list, image_path)

# # Calculate iou?
# iou_m('modelDeepLabV3_Mila.tflite', labels, image_for_prediction)

# Calculate # uou per class
iou_out = []

for img in image_list:     
        iou_score, ipclass, time_milisecs = iou_per_class(mdl_path + 'modelDeepLabV3_Mila.tflite', image_path + img, labels)
        iou_out.append(np.hstack((iou_score, np.squeeze(ipclass), time_milisecs))) 
iou_out = np.array(iou_out)
header = np.hstack(('mIOU', col_head.tolist(), 'Speed (ms)'))
rst =pd.DataFrame(iou_out, columns = header, index = image_list[0:40])
print(rst.head())
rst.to_csv('IOU_perclass.csv')
