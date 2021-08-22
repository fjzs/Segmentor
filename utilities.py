import numpy as np
import time
import tensorflow as tf
from PIL import Image
import matplotlib.image as mpimg
from PIL import ImageOps
import os
import imageio
import cv2


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
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
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)]})#/127.5 -1]})
        end = time.time()
        seg_map = batch_seg_map[0]
        time_milisecs= round((end-start) * 1000,4)
        return image, seg_map, time_milisecs

def meanIougraph(model, image_for_prediction, image_target):
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
    - time_milisecs (time in miliseconds), kmiou (float)
  '''
  MODEL = DeepLabModel(model)
  image, seg_map, time = MODEL.run(image_for_prediction)

  # seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
  target = np.array(Image.open(image_target))

  target = target.ravel()
  # target = np.clip(target, 0, 21)
  target[target == 255] = 0
  
  predicted = np.array(seg_map).ravel()
  num_classes=21
  # Trick for bincounting 2 arrays together
  x = predicted + num_classes * target
  bincount_2d = np.bincount(x.astype(np.int32), minlength=num_classes**2)
  # assert bincount_2d.size == num_classes**2
  conf = bincount_2d.reshape((num_classes, num_classes))
  x = predicted 
  pred_count_2d = np.bincount(x.astype(np.int32), minlength=21)
  y = target
  targ_count_2d = np.bincount(y.astype(np.int32), minlength=21)
  temp = target * 21 + predicted
  cm = np.bincount(temp, weights = None, minlength = 441)
  cm = cm.reshape((21, 21))
  # Compute the IoU and mean IoU from the confusion matrix
  true_positive = np.diag(conf)
  false_positive = np.sum(conf, 0) - true_positive
  false_negative = np.sum(conf, 1) - true_positive

  denominator = (true_positive + false_positive + false_negative)
  num_valid_entries = np.count_nonzero(denominator)
  iou = true_positive/denominator
  iou = np.nan_to_num(iou)
  meaniou = np.sum(iou, 0).astype(np.float32)/num_valid_entries

  # keras
  k = tf.keras.metrics.MeanIoU(num_classes=21)
  k.update_state(target.flatten(), np.array(seg_map).flatten())
  kmiou = k.result().numpy()
  k.reset_state()


  return  meaniou, kmiou, iou, time

def iou_per_pixelclass1(model, image_for_prediction, image_target):
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
    - iou_score (float), iou_per_class_array (float array size 1x20 an entry per class),kmiou (keras miou float), time_milisecs (time in miliseconds)
    
  '''
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
    
  # Invoke the interpreter to run inference.

  interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
  interpreter.invoke()

  #get values of input sizes **********
  input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]

  # Sets the value of the input tensor
  interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
    # Invoke the interpreter.
  interpreter.invoke()
  # 
  start = time.time()
  predictions_array = interpreter.get_tensor(output_index)
  end = time.time()

  raw_prediction = predictions_array
  ##  resize then argmax - this is used in some other frozen graph and produce smoother output
  seg_map = tf.argmax(tf.image.resize(raw_prediction, image.size[::-1] ), axis=3)#(height, width) revert back to original image
  seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
  target = np.array(Image.open(image_target)) #transform image to labels, removes colormap
 
  #https://stackoverflow.com/questions/67445086/meaniou-calculation-approaches-for-semantic-segmentation-which-one-is-correct
  target = target.ravel()
  # target = np.clip(target, 0, 21)
  target[target == 255] = 0
  
  predicted = np.array(seg_map).ravel()
  num_classes=21
  # Trick for bincounting 2 arrays together
  x = predicted + num_classes * target
  bincount_2d = np.bincount(x.astype(np.int32), minlength=num_classes**2)
  # assert bincount_2d.size == num_classes**2
  conf = bincount_2d.reshape((num_classes, num_classes))
  x = predicted 
  pred_count_2d = np.bincount(x.astype(np.int32), minlength=21)
  y = target
  targ_count_2d = np.bincount(y.astype(np.int32), minlength=21)
  temp = target * 21 + predicted
  cm = np.bincount(temp, weights = None, minlength = 441)
  cm = cm.reshape((21, 21))
  # Compute the IoU and mean IoU from the confusion matrix
  true_positive = np.diag(conf)
  false_positive = np.sum(conf, 0) - true_positive
  false_negative = np.sum(conf, 1) - true_positive

  denominator = (true_positive + false_positive + false_negative)
  num_valid_entries = np.count_nonzero(denominator)
  iou = true_positive/denominator
  # iou[np.isnan(iou)] = 1
  #iou= np.mean(iou)
  iou = np.nan_to_num(iou)
  meaniou = np.sum(iou, 0).astype(np.float32)/num_valid_entries
  # meaniou = np.nanmean(iou).astype(np.float32)

  # keras
  k = tf.keras.metrics.MeanIoU(num_classes=21)
  k.update_state(target.flatten(), np.array(seg_map).flatten())
  kmiou = k.result().numpy()
  k.reset_state()
  time_milisecs= round((end-start) * 1000,4)

  return meaniou, iou ,kmiou,time_milisecs

def meanIou(model, image_for_prediction, image_target):
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
    - time_milisecs (time in miliseconds), kmiou (float)
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

  # Resize the image to the desired model size
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
  # 
  start = time.time()
  predictions_array = interpreter.get_tensor(output_index)
  end = time.time()

  raw_prediction = predictions_array
  ##  resize then argmax - this is used in some other frozen graph and produce smoother output
  seg_map = tf.argmax(tf.image.resize(raw_prediction, image.size[::-1] ), axis=3)#(height, width) revert back to original image
  seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
  target = np.array(Image.open(image_target))
  target[target == 255] = 0

  k = tf.keras.metrics.MeanIoU(num_classes=21)
  k.update_state(target.flatten(), np.array(seg_map).flatten())
  kmiou = k.result().numpy()
  k.reset_states()
  time_milisecs= round((end-start) * 1000,4)

  return  time_milisecs, kmiou



def iou_per_pixelclass(model, image_for_prediction, image_target):
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
    - iou_score (float), iou_per_class_array (float array size 1x20 an entry per class), time_milisecs (time in miliseconds)
  '''
  
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


  # Resize the cropped image to the desired model size
  resized_image = image.convert('RGB').resize(input_size, Image.BILINEAR)

  # Convert to a NumPy array, add a batch dimension, and normalize the image.
  image_for_prediction = np.asarray(resized_image).astype(np.float32)
  image_for_prediction = np.expand_dims(image_for_prediction, 0)
  image_for_prediction = image_for_prediction / 127.5 - 1
    
  # Invoke the interpreter to run inference.

  #interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
  #interpreter.invoke()

  #get values of input sizes **********
  #input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]

  # Sets the value of the input tensor
  interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
  # Invoke the interpreter.
  interpreter.invoke()
  
  # calculate the time difference
  start = time.time()
  predictions_array = interpreter.get_tensor(output_index)
  end = time.time()

  raw_prediction = predictions_array
  ##  resize then argmax - this is used in some other frozen graph and produce smoother output
  seg_map = tf.argmax(tf.image.resize(raw_prediction, image.size[::-1] ), axis=3)#(height, width) revert back to original image
  seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
  target = np.array(Image.open(image_target))
  target[target == 255] = 0

  #https://stackoverflow.com/questions/67445086/meaniou-calculation-approaches-for-semantic-segmentation-which-one-is-correct
  target = target.flatten()
  predicted = np.array(seg_map).flatten()
  num_classes=21
  # Trick for bincounting 2 arrays together
  x = predicted + num_classes * target
  bincount_2d = np.bincount(x.astype(np.int32), minlength=num_classes**2)
  assert bincount_2d.size == num_classes**2
  conf = bincount_2d.reshape((num_classes, num_classes))

  # Compute the IoU and mean IoU from the confusion matrix
  true_positive = np.diag(conf)
  false_positive = np.sum(conf, 0) - true_positive
  false_negative = np.sum(conf, 1) - true_positive

  iou = true_positive / (true_positive + false_positive + false_negative)
  #   iou[np.isnan(iou)] = 1
  
  meaniou = np.nanmean(iou).astype(np.float32)  # nanmean is used to neglect 0/0 case which arise due to absence of any class
  time_milisecs= round((end-start) * 1000,4)


  k = tf.keras.metrics.MeanIoU(num_classes=21)
  k.update_state(target.flatten(), np.array(seg_map).flatten())
  kmiou = k.result().numpy()
  k.reset_state()


  return meaniou, iou, time_milisecs, kmiou

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
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128],[224, 224, 192]])
