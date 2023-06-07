from tqdm import tqdm
import cv2      
from keras.applications.vgg16 import VGG16
from keras.applications import MobileNet
import numpy as np
import os
from dotenv import load_dotenv
import json
from sklearn.utils import shuffle
from utils import load_h5, save_in_h5, get_args 
from minio_client.minio_client import MinioClient

load_dotenv()

def preprocessing(minio_client, bucket_name, object, image_size):
    image_data = minio_client.client.get_object(bucket_name, object.object_name).data
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    return image

def load_inference_data(path_to_args, bucket_name):
    args = get_args(path_to_args)
    minio_client = MinioClient(os.getenv("MINIO_ACCESS_KEY"), os.getenv("MINIO_SECRET_KEY"),os.getenv("MINIO_URL"))
    objects = minio_client.get_data(bucket_name)
    images = []
    for obj in objects:
        image = preprocessing(minio_client, bucket_name, obj, tuple(args['image_size']))
        images.append(image)
    images = np.array(images, dtype = 'float32')
    images = images / 255.0
    save_in_h5(args["interim_path_data"] + bucket_name + "_images.h5", images)

def load_training_data(path_to_args, bucket_name):
    args = get_args(path_to_args)
    minio_client = MinioClient(os.getenv("MINIO_ACCESS_KEY"), os.getenv("MINIO_SECRET_KEY"),os.getenv("MINIO_URL"))
    folders = args["class_names"]
    images = []
    labels = []
    for folder in folders:
        objects = minio_client.get_data(bucket_name, folder)    
        for obj in objects:
            image = preprocessing(minio_client, bucket_name, obj, tuple(args['image_size']))
            images.append(image)
            if 'dirty' in obj.object_name:
                labels.append(0)
            elif 'clean' in obj.object_name:
                labels.append(1)
    images = np.array(images, dtype = 'float32')
    images = images / 255.0
    labels = np.array(labels, dtype = 'int32') 
    images, labels = shuffle(images, labels, random_state=25)
    save_in_h5(args["interim_path_data"] + bucket_name + "_images.h5", images)
    save_in_h5(args["interim_path_data"] + bucket_name + "_labels.h5", labels)

def get_features(args, images):
    model = MobileNet(input_shape=tuple(args["mobile_net_input_shape"]), weights='imagenet', include_top=False)
    features = model.predict(images)
    return features

def process_data(path_to_args, interim_filename, processed_filename):
    args = get_args(path_to_args)
    images = load_h5(args["interim_path_data"] + interim_filename)

    features = get_features(args, images)
    save_in_h5(args["processed_path_data"] + processed_filename, features)