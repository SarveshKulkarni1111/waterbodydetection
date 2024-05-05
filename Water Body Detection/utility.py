from dataclasses import dataclass, field
import os
import requests
import cv2
import tensorflow as tf
import numpy as np

@dataclass(frozen=True)
class DatasetConfig:
    IMAGE_SIZE:        tuple = (256, 256)
    BATCH_SIZE:          int = 16
    NUM_CLASSES:         int = 2
    BRIGHTNESS_FACTOR: float = 0.2
    CONTRAST_FACTOR:   float = 0.2
    
    
def preprocess(image):
    image = tf.convert_to_tensor(image)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=(256,256), method = "bicubic")
    image = tf.cast(tf.clip_by_value(image, 0., 255.), tf.float32)
    return image

def create_overlayed_image(image , predictions):
    rgb_pred_mask = num_to_rgb(predictions)
    mask_to_overlay = rgb_pred_mask
    
    overlayed_iamge = image_overlay(image , mask_to_overlay)
    
    return overlayed_iamge

def image_overlay(image, segmented_image):

    alpha = 1.0 # Transparency for the original image.
    beta  = 0.7 # Transparency for the segmentation map.
    gamma = 0.0 # Scalar added to each sum.

    image = image.astype(np.uint8)

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

id2color = {
    0: (0,  0,    0),    # Background
    1: (102, 204, 255),  # Waterbody
 }

# Function to convert a single channel mask representation to an RGB mask.
def num_to_rgb(num_arr, color_map=id2color):
    
    # single_layer = np.squeeze(num_arr)
    output = np.zeros(num_arr.shape[:2]+(3,))
    
    for k in color_map.keys():
        output[num_arr==k] = color_map[k]
        
    return output.astype(np.uint8)

