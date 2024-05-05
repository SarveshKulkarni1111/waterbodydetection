import os , io
import base64
import requests
import glob
import random
import numpy as np
import cv2
import tensorflow as tf
import keras_cv
import matplotlib.pyplot as plt
from flask_cors import CORS, cross_origin
import zipfile

from dataclasses import dataclass, field
from flask import Flask , request, send_file , jsonify, send_from_directory
from zipfile import ZipFile
from PIL import Image

from model import create_model
from utility import preprocess , create_overlayed_image , num_to_rgb

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model_name = "deeplabv3_plus_resnet50_v2.h5"
weights_folder = "weights"
path_to_weights = os.path.join(os.getcwd() , weights_folder , model_name)

model = create_model()
model.summary()

model.load_weights(path_to_weights)

@cross_origin
@app.route("/get_overlayed_image" , methods = ['POST'])
def return_output():
    
    image_file= request.files['image']
    img = Image.open(image_file)
    preprocessed_image = preprocess(img)
    
    predictions = (model.predict(np.expand_dims((preprocessed_image), axis=0))).astype('float32')
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=-1)
    
    np_image = preprocessed_image.numpy()
    overlayed_image = create_overlayed_image(np_image , predictions)
    print(overlayed_image.shape)
    output = Image.fromarray(overlayed_image)
    rawBytes = io.BytesIO()
    output.save(rawBytes, "JPEG")
    rawBytes.seek(0) 
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'overlayed_image':str(img_base64)})

@cross_origin() 
@app.route("/get_prediction_mask" , methods = ['POST'])
def return_pred_mask():
    image_file= request.files['image']
    img = Image.open(image_file)
    preprocessed_image = preprocess(img)
    
    predictions = (model.predict(np.expand_dims((preprocessed_image), axis=0))).astype('float32')
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=-1)
    
    intermediate = num_to_rgb(predictions)
    print(intermediate.shape)
    output = Image.fromarray(intermediate)
    
    rawBytes = io.BytesIO()
    output.save(rawBytes, "JPEG")
    rawBytes.seek(0) 
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'pred_mask':str(img_base64)})

# @app.route("/get_prediction11" , methods = ['GET'])
# def get_data():
   
#     return "HELLO WORLD"

    

if __name__ == "__main__":
    app.run(debug=True)

