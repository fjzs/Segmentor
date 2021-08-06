import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time


#first definition, compact
def iou_(target, prediction):
  """takes in two arrays as inputs
  """
  intersection = np.logical_and(target, prediction)
  union = np.logical_or(target, prediction)
  iou_score = np.sum(intersection) / np.sum(union)
  iou_per_class_array = np.nan_to_num(intersection/union)
  index = np.where(intersection) #check for the classes that appear in both
  i=0
  for el in iou_per_class_array[index]:
    print("for class: ", index[0][i], " the iou is : ", el)
    i+=1

  return iou_score, iou_per_class_array
  
  
# Find IOU Given image and model
# Load the model and image.

def iou_m(model, labels, image_for_prediction): #can change the model
  image_name = image_for_prediction

  # read_img = mpimg.imread(image_for_prediction)
  # image = Image.fromarray(read_img)

  interpreter = tf.lite.Interpreter(model_path=model)#model_path = model_dict[tflite_model_type])
      # Interpreter interface for TensorFlow Lite Models.

    # Gets model input and output details.
  input_index = interpreter.get_input_details()[0]["index"]
  input_details = interpreter.get_input_details()
  output_index = interpreter.get_output_details()[0]["index"]
  interpreter.allocate_tensors()
  input_img = mpimg.imread(image_for_prediction)
  image = Image.fromarray(input_img)

    # Get image size - converting from BHWC to WH
  input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]

  old_size = image.size  # old_size is in (width, height) format
  desired_ratio = input_size[0] / input_size[1]
  old_ratio = old_size[0] / old_size[1]

  if old_ratio < desired_ratio: # '<': cropping, '>': padding
      new_size = (old_size[0], int(old_size[0] / desired_ratio))
  else:
      new_size = (int(old_size[1] * desired_ratio), old_size[1])

    # Cropping the original image to the desired aspect ratio
  delta_w = new_size[0] - old_size[0]
  delta_h = new_size[1] - old_size[1]
  padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
  cropped_image = ImageOps.expand(image, padding)

    # Resize the cropped image to the desired model size
  resized_image = cropped_image.convert('RGB').resize(input_size, Image.BILINEAR)

    # Convert to a NumPy array, add a batch dimension, and normalize the image.
  image_for_prediction = np.asarray(resized_image).astype(np.float32)
  image_for_prediction = np.expand_dims(image_for_prediction, 0)
  image_for_prediction = image_for_prediction / 127.5 - 1
    
    # Invoke the interpreter to run inference.

  interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
  interpreter.invoke()

    #get values of input sizes **********
  input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]

    # Sets the value of the input tensor
  interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
    # Invoke the interpreter.
  interpreter.invoke()

  predictions_array = interpreter.get_tensor(output_index)
  raw_prediction = predictions_array
    # Post-processing: convert raw output to segmentation output
    ## Method 1: argmax before resize - this is used in some frozen graph
    # seg_map = np.squeeze(np.argmax(raw_prediction, axis=3)).astype(np.int8)
    # seg_map = np.asarray(Image.fromarray(seg_map).resize(image.size, resample=Image.NEAREST))
    ## Method 2: resize then argmax - this is used in some other frozen graph and produce smoother output
  width, height = cropped_image.size
  seg_map = tf.argmax(tf.image.resize(raw_prediction, (height, width)), axis=3)
  seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)

  #real iou part 
  prediction = np.unique(seg_map)
  #For the target, find the label array corresponding to the input image
  labels = pd.read_csv('pascal_segmented_classes_per_image.csv', index_col='image_filename')
  labels= labels.iloc[: , 1:]
  specific_pic_classes = labels.filter(like= image_name	,  axis =0)#here test
  specific_pic_class_array = specific_pic_classes.to_numpy()

  target = np.argwhere(specific_pic_class_array >0) #
  intersection = np.logical_and(target, prediction)
  union = np.logical_or(target, prediction)
  iou_score = np.sum(intersection) / np.sum(union)
  # iou_per_class_array = np.nan_to_num(intersection/union)
  return iou_score
  
  # Find IOU Given image and model
# Load the model and image

def iou_per_class(model, image_for_prediction, labels): #can change the model as needed
  '''
  call function like: utilities.iou_per_class('modelDeepLabV3_Mila.tflite', '2008_000491.jpg')
    What does this function do?
    - This function will calculate the IoU
    The input image will be transformed to a segmentation map where the classes per pixel value will be found.
    After having the classes per image that the model detected, it will be compared to the labeled classes for that image:
    these are found in the pascal_segmented_classes_per_image.csv' folder. The IoU will be compared by detecting the subsets of classes in the intersection per union.
    Arguments:
    - param1 (.tflite): a  segmentation model
    - param2 (.jpg): a picture from pascal

    Returns:
    - iou_score (float), iou_per_class_array (float array size 1x20 an entry per class), time_milisecs (time in miliseconds)
    
  '''

  image_name = image_for_prediction
  interpreter = tf.lite.Interpreter(model_path=model)
  # Interpreter interface for TensorFlow Lite Models.


  # Gets model input and output details.
  input_index = interpreter.get_input_details()[0]["index"]
  input_details = interpreter.get_input_details()
  output_index = interpreter.get_output_details()[0]["index"]
  interpreter.allocate_tensors()
  input_img = mpimg.imread(image_for_prediction)
  image = Image.fromarray(input_img)


  # Get image size - converting from BHWC to WH
  input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]

  old_size = image.size  # old_size is in (width, height) format
  desired_ratio = input_size[0] / input_size[1]
  old_ratio = old_size[0] / old_size[1]

  if old_ratio < desired_ratio: # '<': cropping, '>': padding
      new_size = (old_size[0], int(old_size[0] / desired_ratio))
  else:
      new_size = (int(old_size[1] * desired_ratio), old_size[1])
 
  # Cropping the original image to the desired aspect ratio
  delta_w = new_size[0] - old_size[0]
  delta_h = new_size[1] - old_size[1]
  padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
  cropped_image = ImageOps.expand(image, padding)

  # Resize the cropped image to the desired model size
  resized_image = cropped_image.convert('RGB').resize(input_size, Image.BILINEAR)

  # Convert to a NumPy array, add a batch dimension, and normalize the image.
  image_for_prediction = np.asarray(resized_image).astype(np.float32)
  image_for_prediction = np.expand_dims(image_for_prediction, 0)
  image_for_prediction = image_for_prediction / 127.5 - 1
    
  # Invoke the interpreter to run inference.
  
  interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
  interpreter.invoke()

  #get values of input sizes **********
  input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]

  # Sets the value of the input tensor
  interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
   # Invoke the interpreter.
  interpreter.invoke()
  start = time.time()
  predictions_array = interpreter.get_tensor(output_index)
  end = time.time()
  raw_prediction = predictions_array
  ##  resize then argmax - this is used in some other frozen graph and produce smoother output
  width, height = cropped_image.size
  seg_map = tf.argmax(tf.image.resize(raw_prediction, (height, width)), axis=3)
  seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)

  #real iou part 
  prediction = np.unique(seg_map)
  #For the target, find the label array corresponding to the input image
  #change pascal_segmented_classes_per_image.csv to local path
  #labels = pd.read_csv('https://raw.githubusercontent.com/fjzs/Segmentor/main/datasets/pascal/pascal_segmented_classes_per_image.csv', index_col=1).drop('Unnamed: 0', axis = 1)
  specific_pic_classes = labels.filter(like= os.path.basename(image_name),  axis =0)#here test
  specific_pic_class_array = specific_pic_classes.to_numpy()
  

  vector_form = np.zeros((20,1))
  vector_form [prediction] = 1
  vector_form = vector_form.T
  

  target = specific_pic_class_array.astype('float64')
  intersection = np.logical_and(target , vector_form)
  union = np.logical_or(target, vector_form)
  iou_score = np.sum(intersection) / np.sum(union)
  iou_per_class_array = np.nan_to_num(intersection/union)

  time_milisecs= np.round((end-start) * 1000,4)
  

  return iou_score, iou_per_class_array, time_milisecs