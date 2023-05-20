# Solar Panel Cleaning Model Card
## Model Details
### Model name

Solar Panel Cleaning Model
### Model version

v1.0
### Model type

Image Classification
### Model description

The Solar Panel Cleaning Model is a machine learning model that can analyze images of solar panels and classify them as either clean or dirty. The model was trained on a dataset of images of solar panels, both clean and dirty. The model can be used by an automated solar panel cleaning system to prioritize the cleaning schedule and notify the maintenance team of the panels that need cleaning.
### Model performance

The model was trained and evaluated using a dataset of 2551 images of solar panels, split into 2035 training images and 516 validation images. 

accuracy        0.92383
loss            0.26643
val_accuracy    0.74612
val_loss        0.74442

## Model Use
### Intended use

The Solar Panel Cleaning Model is intended to be used by an automated solar panel cleaning system to prioritize the cleaning schedule and notify the maintenance team of the panels that need cleaning. The model can also be used by solar panel manufacturers to detect dirty panels in the production process.
## User Guide
### Inputs

The model takes as input an image of a solar panel in JPEG or PNG format.
### Outputs

The model outputs a prediction of whether the solar panel is clean or dirty.

### Example usage
```
from PIL import Image
import requests
from io import BytesIO
from keras.applications import MobileNet
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model.h5')

url = 'https://www.electrochem.org/wp-content/uploads/2018/09/iStock-Dirty-Solar-Panel-min-1.jpg'
response = requests.get(url)
img = Image.open(BytesIO(response.content))

img = img.resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
img_array = tf.expand_dims(img_array, axis=0)
mobile_net = MobileNet(input_shape=(224,224,3), weights='imagenet', include_top=False)
features = mobile_net.predict(np.array(img_array))

pred = model.predict(features)

if pred[0][0] > 0.5:
    print('The solar panel is dirty.')
else:
    print('The solar panel is clean.')
```
## Model Training

The Solar Panel Cleaning Model uses a MobileNetV2 architecture, with the last layer replaced with a dense layer with two output nodes, one for clean panels and one for dirty panels. The model was trained using sparse categorical cross entropy loss and the Adam optimizer.

## Hyperparameters

The following hyperparameters were used during training:

    Batch size: 64
    Number of epochs: 15