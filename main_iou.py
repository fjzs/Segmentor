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
mdl      = 'modelDeepLabV3_Mila.tflite'
csv_in   = 'pascal_segmented_classes_per_image.csv'
csv_out  = 'IOU_perclass.csv'

labels = pd.read_csv(tst_path + csv_in, index_col=1).drop('Unnamed: 0', axis = 1)
label_array = labels.to_numpy()
col_head = labels.columns
val_path = tst_path + 'Segmentation_input/validation2/' 
val_path = 'C:/Users/adi/Downloads/VOC2012/SegmentationClass/'
image_list = os.listdir(val_path)

# # Get the test image
# image_for_prediction = create_image_array(image_list, val_path)


# Calculate # uou per class
iou_out = []
i=1
nimg = len(image_list)
# Loop to compute the IOU
for img in image_list:
        print('Processing image:' + img + ', immage no ' + str(i) + ' of ' + str(nimg))
        anno_mat =  np.array(Image.open(val_path + img))
        iou_score, ipclass, time_milisecs = iou_per_class(mdl_path + mdl, val_path + img, labels)
        iou_out.append(np.hstack((iou_score, np.squeeze(ipclass), time_milisecs)))
        i=i+1
iou_out = np.array(iou_out)

# Create header for CSV
header = np.hstack(('mIOU', col_head.tolist(), 'Speed (ms)'))
rst =pd.DataFrame(iou_out, columns = header, index = image_list[:iou_out.shape[0]])
print(rst.head())

# Do mean for evaluating the model performace
iouave = iou_out.mean(axis=0)
maiou = iouave[0]
mspd = iouave[-1]
print('MAIOU: ' + str(maiou) + ', mean speed: ' + str(mspd)) 
# MAIOU: 0.8326892109500805, mean speed: 3.5134649413388503

# Create the csv file
rst.to_csv(csv_out)
