import pandas as pd
from alibi_detect.cd import MMDDrift
from alibi_detect.saving import save_detector, load_detector
import pickle
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator   
import tensorflow as tf    
import logging
from utils import load_h5, get_args, setup_logger, compute_metrics 
from pathlib import Path
import numpy as np

SWEEP_PROJECT = 'solar_panels'

logger = logging.getLogger(__name__)

def get_model(args):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape = tuple(args["input_shape"])),
        tf.keras.layers.Dense(args["layers"], activation=args["activation"]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(args["number_of_classes"], activation=args["activation1"])
    ])

    model.compile(optimizer=args["optimizer"], loss=args["loss"], metrics=['accuracy'])
    return model

def get_data(args):
    train_images = load_h5(args["processed_path_data"] + "train_features.h5")
    test_images = load_h5(args["processed_path_data"] + "test_features.h5")

    train_labels = load_h5(args["interim_path_data"] + "train_labels.h5")
    test_labels = load_h5(args["interim_path_data"] +"test_labels.h5")
    return train_images, test_images, train_labels, test_labels

def train(config_path):
    args = get_args(config_path)
    if args["use_wandb"]:
        wandb.init(project=args["wandb_project"], config=args)
    setup_logger(logger)
    logger.info(f"Training/evaluation parameters {args}")
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
        model.fit(datagen.flow(train_features, train_labels, batch_size=args["batch_size"], shuffle=True), epochs=args["epochs"], 
                                                    validation_data=(test_features, test_labels), callbacks=[WandbCallback()])
    else:
        model.fit(datagen.flow(train_features, train_labels, batch_size=args["batch_size"], shuffle=True), 
                                                    epochs=args["epochs"], validation_data=(test_features, test_labels))
    model.save(args["model_path"])
    
    mmd_drift_detector = MMDDrift(train_features, backend="tensorflow", p_val=.01)
    save_detector(mmd_drift_detector, args["detector_path"])

    train_metrics = compute_metrics(model.predict(train_features), train_labels)
    val_metrics = compute_metrics(model.predict(test_features), test_labels)

    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Validation metrics: {val_metrics}")

def hyperparameter_train(config):
    datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    train_features, test_features, train_labels, test_labels = get_data(config)
    model = get_model(config)
    model.fit(datagen.flow(train_features, train_labels, batch_size=config["batch_size"], shuffle=True), 
                                epochs=config["epochs"], validation_data=(test_features, test_labels))
    accuracy = compute_metrics(model.predict(train_features), train_labels)['accuracy']
    return accuracy

def start_training():
    wandb.init(project='solar panels classification')
    accuracy = hyperparameter_train(wandb.config)
    wandb.log({'accuracy': accuracy})

def hyperparameter_search(path_to_params):
    sweep_params = get_args(path_to_params)

    sweep_configuration = {
        'method': 'random',
        'metric': 
        {
            'goal': 'maximize', 
            'name': 'accuracy'
        },
        'parameters': sweep_params
    }

    sweep_id = wandb.sweep(
            sweep=sweep_configuration, 
            project='solar panels classification'
        )

    wandb.agent(sweep_id, function=start_training, count=10)