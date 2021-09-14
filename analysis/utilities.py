import numpy as np
import time
import tensorflow as tf
from PIL import Image
import matplotlib.image as mpimg
from PIL import ImageOps
import os
import imageio
import cv2
import cProfile, pstats

#This class is to open graph type files .pb
class DeepLabModel(object):
  """Class to load deeplab model and run inference."""
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = (500, 375)
  FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'

  def __init__(self,model):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    self.model = model
    
    with tf.io.gfile.GFile(model, "rb") as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
    self.sess = tf.compat.v1.Session(graph=self.graph)

  def run(self, image):
        """Runs inference on a single image.

        Args:
        image: A PIL.Image object, raw input image.

        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        time_milisecs: Time it takes to run inference on an image
        """
       
        image = mpimg.imread(image)
        image = Image.fromarray(image)
        
        start = time.time()
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)]})
        end = time.time()
        seg_map = batch_seg_map[0] # expected batch size = 1
        time_milisecs= round((end-start) * 1000,4)
        return image, seg_map, time_milisecs
  def run_norm(self, image):
        """Runs inference on a single image.

        Args:
        image: A PIL.Image object, raw input image.

        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        time_milisecs: Time it takes to run inference on an image
        """
       
        image = mpimg.imread(image)
        image = Image.fromarray(image)
        
        start = time.time()
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)/np.max(np.asarray(image))]})
        end = time.time()
        seg_map = batch_seg_map[0] # expected batch size = 1
        time_milisecs= round((end-start) * 1000,4)
        return image, seg_map, time_milisecs
      
def meanIougraph_2(model, image_for_prediction, image_target):
  '''
    call function like: meanIou3graph('deeplabv3_mnv2_pascal_train_aug/saved_model.pb','2007_000063.jpg', "2007_000063.png")
    What does this function do?
    - This function will calculate the IoU per class
    The image_for_prediction will be transformed to a segmentation map where the classes per pixel value will be found.This will be flattened into a 1D array
    The image_target array will be obtained by converting the pixels of the image_target to classes
     Arguments:
    - param1 (.pb): a  segmentation model
    - param2 (.jpg): a picture from pascal
    - param3 (.png): a picture from pascal
    
    Returns:
    - time_milisecs (time in miliseconds), kmiou (float)
  '''
  MODEL = DeepLabModel(model)
  image, seg_map, time = MODEL.run(image_for_prediction)

  predicted = np.array(seg_map).ravel() #375 x 500 , original pascal voc 212 size
  num_classes=21
  
  target = np.array(Image.open(image_target)).ravel() # 375 x 500


  valid_mask = (target <= num_classes)
  target = target[valid_mask]
  predicted =  predicted[valid_mask]


  conf_matrix = tf.cast(tf.math.confusion_matrix(target, predicted, num_classes=num_classes), 'float32')
  


  # Compute the IoU and mean IoU from the confusion matrix
  true_positive = np.diag(conf_matrix)
  false_positive = np.sum(conf_matrix, 0) - true_positive
  false_negative = np.sum(conf_matrix, 1) - true_positive

  denominator = true_positive + false_positive + false_negative

  # print(f' the classes that appear in the calculations are: {denominator.nonzero()}')

  num_valid_entries = np.count_nonzero(denominator)
  out = np.zeros( len( denominator))  #preinit
  iou = np.divide(true_positive, denominator, out=out, where=denominator!=0)
  meaniou = np.sum(iou).astype(np.float32)/num_valid_entries #there will always be at least one entry (background)

  iou[denominator ==0 ]=np.nan

  # keras
#   k = tf.keras.metrics.MeanIoU(num_classes=21)
#   k.update_state(target, predicted) 
#   kmiou = k.result().numpy()
#   k.reset_state()

  return round(meaniou, 8), iou, time

def meanIougraph_norm(model, image_for_prediction, image_target):
  '''
    call function like: meanIou3graph('deeplabv3_mnv2_pascal_train_aug/saved_model.pb','2007_000063.jpg', "2007_000063.png")
    What does this function do?
    - This function will calculate the IoU per class, mIou, and the same but with normalized input images
    The image_for_prediction will be transformed to a segmentation map where the classes per pixel value will be found.This will be flattened into a 1D array
    The image_target array will be obtained by converting the pixels of the image_target to classes
     Arguments:
    - param1 (.pb): a  segmentation model
    - param2 (.jpg): a picture from pascal
    - param3 (.png): a picture from pascal
    
    Returns:
    - time_milisecs (time in miliseconds), kmiou (float)
  '''
  MODEL = DeepLabModel(model)
  image, seg_map, time = MODEL.run(image_for_prediction)
  image_n, seg_map_n, time_n = MODEL.run_norm(image_for_prediction)


  predicted = np.array(seg_map).ravel() #375 x 500 , original pascal voc 212 size
  predicted_n = np.array(seg_map_n).ravel()
  num_classes=21
  
  target = np.array(Image.open(image_target)).ravel() # 375 x 500
  target_n = (np.array(Image.open(image_target))).ravel() # 375 x 500

  valid_mask = (target <= num_classes)
  target = target[valid_mask]
  target_n = target_n[valid_mask]
  predicted =  predicted[valid_mask]
  predicted_n =  predicted_n[valid_mask]

  conf_matrix = tf.cast(tf.math.confusion_matrix(target, predicted, num_classes=num_classes), 'float32')
  conf_matrix_n = tf.cast(tf.math.confusion_matrix(target_n, predicted_n, num_classes=num_classes), 'float32')


  # Compute the IoU and mean IoU from the confusion matrix
  true_positive = np.diag(conf_matrix)
  true_positive_n = np.diag(conf_matrix_n)

  false_positive = np.sum(conf_matrix, 0) - true_positive
  false_positive_n = np.sum(conf_matrix_n, 0) - true_positive_n

  false_negative = np.sum(conf_matrix, 1) - true_positive
  false_negative_n = np.sum(conf_matrix_n, 1) - true_positive_n

  denominator = true_positive + false_positive + false_negative
  denominator_n = true_positive_n + false_positive_n + false_negative_n

  num_valid_entries = np.count_nonzero(denominator)
  num_valid_entries_n = np.count_nonzero(denominator_n)

  out = np.zeros( len( denominator))  #preinit
  out_n = np.zeros( len( denominator_n))  #preinit

  iou = np.divide(true_positive, denominator, out=out, where=denominator!=0)
  iou_n = np.divide(true_positive_n, denominator_n, out=out_n, where=denominator_n!=0)

  meaniou = np.sum(iou).astype(np.float32)/num_valid_entries #there will always be at least one entry (background)
  meaniou_n = np.sum(iou_n).astype(np.float32)/num_valid_entries_n #there will always be at least one entry (background)
  
  iou[denominator ==0 ]=np.nan
  iou_n[denominator_n ==0 ]=np.nan

  # keras
  k = tf.keras.metrics.MeanIoU(num_classes=21)
  k.update_state(target, predicted) 
  kmiou = k.result().numpy()
  #k.reset_state()

  return round(meaniou, 8),round(meaniou_n, 8), iou,iou_n, time    

def meanIougraph(model, image_for_prediction, image_target): #this function is to be used for .pb models
  '''
    call function like: meanIou3graph('deeplabv3_mnv2_pascal_train_aug/saved_model.pb','2007_000063.jpg', "2007_000063.png")
    What does this function do?
    - This function will calculate the IoU
    The image_for_prediction will be transformed to a segmentation map where the classes per pixel value will be found.This will be flattened into a 1D array
    The image_target array will be obtained by converting the pixels of the image_target to classes
     Arguments:
    - param1 (.pb): a  segmentation model
    - param2 (.jpg): a picture from pascal
    - param3 (.png): a picture from pascal
    
    Returns:
    - meaniou (float computed mean iou), kmiou (float , keras miou), iou (array of floats mean iou per class),time_milisecs (time in miliseconds),
  '''
  MODEL = DeepLabModel(model)
  image, seg_map, time = MODEL.run(image_for_prediction)
  #predicted is 375x500 before raveling
  predicted = np.array(seg_map).ravel()
  #target is 375x500 before raveling
  target = np.array(Image.open(image_target)).ravel()

  #Filter the valid classes
  
  num_classes=21
  
  valid_mask = (target <= num_classes)
  target = target[valid_mask]
  predicted =  predicted[valid_mask]
  
  #Obtain confusion matrix
  conf = tf.cast(tf.math.confusion_matrix(target, predicted, num_classes=num_classes), 'float32')
  
  # Compute the IoU and mean IoU from the confusion matrix
  true_positive = np.diag(conf)
  false_positive = np.sum(conf, 0) - true_positive
  false_negative = np.sum(conf, 1) - true_positive

  denominator = (true_positive + false_positive + false_negative)
  num_valid_entries = np.count_nonzero(denominator)
  
  out = np.zeros( len( denominator))  #preinit
  iou = np.divide(true_positive, denominator, out=out, where=denominator!=0)

  meaniou = np.sum(iou).astype(np.float32)/num_valid_entries

  # keras
  k = tf.keras.metrics.MeanIoU(num_classes=21)
  k.update_state(target.flatten(), np.array(seg_map).flatten())
  kmiou = k.result().numpy()
  #k.reset_state()


  return  round(meaniou, 8), kmiou, iou, time



def iou_per_pixelclass1(model, image_for_prediction, image_target): #this function is to be used for tflite
  '''
    call function like: utilities.iou_per_class('modelDeepLabV3_Mila.tflite', '2008_000491.jpg', '2008_000491.png')
    What does this function do?
    - This function will calculate the IoU
    The image_for_prediction will be transformed to a segmentation map where the classes per pixel value will be found.This will be flattened into a 1D array
    The image_target array will be obtained by converting the pixels of the image_target to classes
      Arguments:
    - param1 (.tflite): a  segmentation model
    - param2 (.jpg): a picture from pascal
    - param3 (.png): a picture from pascal
    
    Returns:

     - meaniou (float computed mean iou), kmiou (float , keras miou), iou (float array size 1x20 an miou entry per class),time_milisecs (time in miliseconds),
  '''
  
  # profiler = cProfile.Profile()
  # profiler.enable()
  image_name = image_for_prediction
  interpreter = tf.lite.Interpreter(model_path=model)
  image_name = image_for_prediction
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
  resized_image = image.convert('RGB').resize(input_size, Image.BILINEAR)


  # Convert to a NumPy array, add a batch dimension, and normalize the image.
  image_for_prediction = np.asarray(resized_image).astype(np.float32)
  image_for_prediction = np.expand_dims(image_for_prediction, 0)
  image_for_prediction = image_for_prediction / 127.5 - 1
    
  
  # Calculate latency
  start = time.time()
  # Sets the value of the input tensor
  interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
  # Invoke the interpreter.
  interpreter.invoke()
  predictions_array = interpreter.get_tensor(output_index)
  end = time.time()
  # profiler.disable()
  # stats = pstats.Stats(profiler).sort_stats('tottime')
  # stats.print_stats()   


  #obtain the predicted array
  seg_map = np.argmax(tf.image.resize(predictions_array, image.size[::-1] ), axis=3)#(height, width) revert back to original image
  predicted = np.array(seg_map).ravel()
  
  #obtain the true array, we will call target
  target = np.array(Image.open(image_target)).ravel() #transform image to labels, removes colormap
 
  # exclude the deliniation pixels of class 255
  num_classes=21
  valid_mask = (target <= num_classes) 
  target = target[valid_mask]
  predicted =  predicted[valid_mask]
  
  #create the confucion matrix
  conf = tf.cast(tf.math.confusion_matrix(target, predicted, num_classes=num_classes), 'float32')
  
  # Compute the IoU and mean IoU from the confusion matrix
  true_positive = np.diag(conf)
  false_positive = np.sum(conf, 0) - true_positive
  false_negative = np.sum(conf, 1) - true_positive

  denominator = (true_positive + false_positive + false_negative)
  num_valid_entries = np.count_nonzero(denominator)
  out = np.zeros( len( denominator))  #preinit
  iou = np.divide(true_positive, denominator, out=out, where=denominator!=0)

  meaniou = np.sum(iou, 0).astype(np.float32)/num_valid_entries


  # keras
  k = tf.keras.metrics.MeanIoU(num_classes=num_classes)
  k.update_state(target, predicted)
  kmiou = k.result().numpy()
  #k.reset_state()
  time_milisecs= round((end-start) * 1000,4)

  return round(meaniou, 8),kmiou, iou ,time_milisecs


def iou_per_class(model, image_for_prediction, labels): #can change the model as needed
  '''
  call function like: utilities.iou_per_class('modelDeepLabV3_Mila.tflite', '2008_000491.jpg', labels)
    What does this function do?
    - This function will calculate the IoU
    The input image will be transformed to a segmentation map where the classes per pixel value will be found.
    After having the classes per image that the model detected, it will be compared to the labeled classes for that image:
    these are found in the pascal_segmented_classes_per_image.csv' folder. The IoU will be compared by detecting the subsets of classes in the intersection per union.
    Arguments:
    - param1 (.tflite): a  segmentation model
    - param2 (.jpg): a picture from pascal
    - labels (pd.DataFrame) : A dataFrame containing the labels to be compared , first column must include the images and the name should be image_filename	
    the rest of the columns should be labeled 0 to 19 and correspond to the class number of the data set
    see:  pd.read_csv('https://raw.githubusercontent.com/fjzs/Segmentor/main/datasets/pascal/pascal_segmented_classes_per_image.csv', index_col=1).drop('Unnamed: 0', axis = 1)

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


  # Resize the cropped image to the desired model size
  resized_image = image.convert('RGB').resize(input_size, Image.BILINEAR)

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
  specific_pic_classes = labels.filter(like= os.path.basename(image_name),  axis =0)#here test
  specific_pic_class_array = specific_pic_classes.to_numpy()
  

  vector_form = np.zeros((20,1))
  prediction=prediction[prediction != 0]-1
  vector_form [prediction] = 1
  vector_form = vector_form.T
  

  target = specific_pic_class_array.astype('float64')
  intersection = np.logical_and(target, vector_form)
  union = np.logical_or(target, vector_form)
  iou_score = np.sum(intersection) / np.sum(union)
  iou_per_class_array = np.nan_to_num(intersection/union)

  time_milisecs= round((end-start) * 1000,4)
  

  return iou_score, iou_per_class_array, time_milisecs

def image2segmap(img):
    """Encode segmentation label images as pascal classes
    Args:
       - img: png label image from Pascal VOC database
    Returns:
       - (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = imageio.imread(img,pilmode='RGB')
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    
    # Mask border as 255 (from 21)
    label_mask[label_mask == 21] = 255
    
    return label_mask


def segmap2image(label_mask, filename, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        -label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        -filename (str): name the file to save
        -plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    n_classes = 21
    label_colours = get_pascal_labels()
    
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        cv2.imshow(rgb)
        cv2.waitKey(0)
        cv2.imwrite(filename, rgb)
    else:
        return rgb

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
       - np.ndarray with dimensions (21, 3) this include bacground (0) and 
         border line separating the class (255)
    """
    label = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128],[224, 224, 192]])
    
    label_names = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv','border'
    ])
    return label, label_names
