import cv2
import numpy as np


def preprocessing(img_path):
    images = []
    image = cv2.imread(img_path)
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = cv2.resize(image, (224, 224))
    images.append(processed_image)
    images = np.array(images, dtype = 'float32')
    images = images / 255.0
    return images