from tqdm import tqdm
import cv2      
from keras.applications.vgg16 import VGG16
from keras.applications import MobileNet
import numpy as np
import os
import json
from sklearn.utils import shuffle
from image_classification.utils import get_args  

def preprocessing(img_path, image_size):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    return image

def load_data(path_to_folder, class_names, image_size):
    images = []
    labels = []
    dataset = path_to_folder
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
    print("Loading {}".format(dataset))
    for folder in os.listdir(dataset):
        label = class_names_label[folder]
        for file in tqdm(os.listdir(os.path.join(dataset, folder))):
            img_path = os.path.join(os.path.join(dataset, folder), file)
            image = preprocessing(img_path, tuple(image_size))
            images.append(image)
            labels.append(label)
    images = np.array(images, dtype = 'float32')
    labels = np.array(labels, dtype = 'int32')  
    images = images / 255.0
    return (images, labels)

def get_data(path_to_args):
    args = get_args(path_to_args)

    (train_images, train_labels)  = load_data(args["path_to_data"] + 'Train1', args["class_names"], args["image_size"])
    (test_images, test_labels)  = load_data(args["path_to_data"] + 'Test1', args["class_names"], args["image_size"])
    train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
    
    np.save(args["interim_path_data"] + "train_images.npy", train_images)
    np.save(args["interim_path_data"] + "test_images.npy", test_images)
    np.save(args["interim_path_data"] + "train_labels.npy", train_labels)
    np.save(args["interim_path_data"] + "test_labels.npy", test_labels)

def process_data(path_to_args):
    args = get_args(path_to_args)
    train_images = np.load(args["interim_path_data"] + "train_images.npy")
    test_images = np.load(args["interim_path_data"] + "test_images.npy")

    model = MobileNet(input_shape=tuple(args["mobile_net_input_shape"]), weights='imagenet', include_top=False)
    train_features = model.predict(train_images)
    test_features = model.predict(test_images)

    np.save(args["processed_path_data"] + 'train_features.npy', train_features)
    np.save(args["processed_path_data"] + 'test_features.npy', test_features)