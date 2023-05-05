from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator      
import numpy as np  
import cv2      
import numpy as np
from image_classification.train import get_model, train
from image_classification.predict import predict


def test_overfit_batch(data_testing, args):
    train_features, train_labels, test_features, test_labels = data_testing
    model = get_model(args)
    datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    history = model.fit(datagen.flow(train_features, train_labels, batch_size=args["batch_size"], shuffle=True), epochs=args["epochs"], validation_data=(test_features, test_labels))
    
    assert history.history['loss'][-1] < 0.01

def test_train_to_completion(args):
    train('./data/config.json')
    result_path = Path(args["path_to_save_model"])
    assert (result_path).exists()

def test_minimum_functionality(args):
    predict(args["processed_path_data"] + 'test_features.npy', args["path_to_save_model"], args["results_path"])
    results = np.load( args["results_path"])
    assert results[0] == 1