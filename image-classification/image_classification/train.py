from tensorflow.keras.preprocessing.image import ImageDataGenerator   
import tensorflow as tf    
import numpy as np
import wandb
from wandb.keras import WandbCallback
from image_classification.utils import get_args

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_model(args):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = tuple(args["input_shape"])),
        tf.keras.layers.Dense(args["layers"], activation=args["activation"]),
        tf.keras.layers.Dense(2, activation=args["activation1"])])
    model.compile(optimizer = args["optimizer"], loss = args["loss"], metrics=['accuracy'])
    return model

def get_data(args):
    train_images = np.load(args["processed_path_data"] + "train_features.npy")
    test_images = np.load(args["processed_path_data"] + "test_features.npy")

    train_labels = np.load(args["interim_path_data"] + "train_labels.npy")
    test_labels = np.load(args["interim_path_data"] +"test_labels.npy")
    return train_images, test_images, train_labels, test_labels

def train(config_path):
    args = get_args(config_path)
    if args["use_wandb"]:
        wandb.init(project=args["wandb_project"], config=args)
    datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    
    train_features, test_features, train_labels, test_labels = get_data(args)
    model = get_model(args)
    if args["use_wandb"]:
        model.fit(datagen.flow(train_features, train_labels, batch_size=args["batch_size"], shuffle=True), epochs=args["epochs"], validation_data=(test_features, test_labels), callbacks=[WandbCallback()])
    else:
        model.fit(datagen.flow(train_features, train_labels, batch_size=args["batch_size"], shuffle=True), epochs=args["epochs"], validation_data=(test_features, test_labels))
    model.save(args["model_path"] + args["model_name_save"])