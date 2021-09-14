import numpy as np
import pandas as pd
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
#mdl_name = 'modelDeepLabV3_Mila'
#mdl_name = 'lite-model_deeplabv3-mobilenetv2_dm05_1_default_2'
#mdl_name = 'lite-model_deeplabv3-xception65_1_default_2'
#mdl_name = 'lite-model_mobilenetv2-coco_dr_1'
#mdl_name = 'lite-model_mobilenetv2-coco_dr_1'

tst_path = './datasets/pascal/'
#mdl_path = './models/deep_lab_v3_plus/'
mdl_path = './models/slow/'
csv_in   = 'pascal_segmented_classes_per_image.csv'

mdls = os.listdir(mdl_path)
mdl = 'lite-model_deeplabv3-mobilenetv2_dm05-int8_1_default_2.tflite'
#mdl = 'lite-model_deeplabv3-mobilenetv2-int8_1_default_1.tflite'

labels = pd.read_csv(tst_path + csv_in, index_col=1).drop('Unnamed: 0', axis = 1)
label_array = labels.to_numpy()
col_head = labels.columns
val_path = tst_path + 'Segmentation_input/validation/'
seg_path = tst_path + 'Segmentation_output/validation/' 
#val_path = 'C:/Users/adi/Downloads/VOC2012/SegmentationClass/'
image_list = os.listdir(val_path)
# Calculate # uou per class
nimg = len(image_list)


iou_out = []
i=1
# Loop to compute the IOU
for img in image_list:
      print('Processing image:' + img + ', image no ' + str(i) + ' of ' + str(nimg))
      #iou_score, ipclass, time_milisecs = iou_per_class(mdl_path + mdl_name + '.tflite', val_path + img, labels)
      #iou_out.append(np.hstack((iou_score, np.squeeze(ipclass), time_milisecs)))
      label = image2segmap(seg_path + os.path.splitext(img)[0] + '.png')
      #time_milisecs, iou_score = meanIou(mdl_path + os.path.splitext(mdl)[0] + '.tflite', val_path + img, seg_path + os.path.splitext(img)[0] + '.png')
      _, iou_score, ioupclass ,time_milisecs = iou_per_pixelclass1(mdl_path + os.path.splitext(mdl)[0] + '.tflite', val_path + img, seg_path + os.path.splitext(img)[0] + '.png')
      iou_out.append(np.hstack((iou_score, time_milisecs, ioupclass)))
      i=i+1
      if i == 50:
          break
iou_out = np.array(iou_out)

# Create header for CSV
_, label_names = get_pascal_labels()
label_names = label_names[:-1]
header = np.hstack(('mIOU', 'Speed (ms)', label_names))
#header = np.hstack(('mIOU', 'Speed (ms)'))
rst =pd.DataFrame(iou_out, columns = header, index = image_list[:iou_out.shape[0]])
print(rst.head())

# Do mean for evaluating the model performace
iouave = iou_out.mean(axis=0)
maiou = iouave[0]
#mspd = iouave[-1]
mspd = iouave[1]
prtout = 'MAIOU: ' + str(maiou) + ', mean speed: ' + str(mspd)
print(prtout) 
print(prtout, file=open(mdl + '_maiou.txt', "a"))
# Create the csv file
rst.to_csv(mdl + '_miou.csv')
