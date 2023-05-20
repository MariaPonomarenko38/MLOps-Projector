from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator      
import numpy as np        
from image_classification.train import get_model, train
from image_classification.predict import predict
from image_classification.utils import load_h5

def test_overfit_batch(processed_data, args):
    train_features, train_labels, test_features, test_labels = processed_data
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
    
    assert history.history['loss'][-1] < 0.1

def test_train_to_completion(args_path, args):
    train(args_path)
    result_path = Path(args["path_to_save_model"])
    assert (result_path).exists()

def test_minimum_functionality(args):
    predict(args["processed_path_data"] + 'test_features.h5', args["path_to_retrieve_model"], args["results_path"])
    predicted_label = np.load(args["results_path"])[0]
    correct_label = load_h5(args["interim_path_data"] + 'test_labels.h5')[0]
    assert predicted_label == correct_label
