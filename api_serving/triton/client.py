import numpy as np
import tritonclient.http as httpclient
from utils import preprocessing

SAVE_INTERMEDIATE_IMAGES = False

def get_features(client, img_path):
    preprocessed_image = preprocessing(img_path)

    mobilenet_input = httpclient.InferInput(
        "input_2", preprocessed_image.shape, datatype="FP32"
    )
    mobilenet_input.set_data_from_numpy(preprocessed_image, binary_data=True)
    mobilenet_response = client.infer(
        model_name="mobilenet", inputs=[mobilenet_input]
    )

    features = mobilenet_response.as_numpy("conv_pw_13_relu")
    return features

def get_predictions(client, features):
    imageclass_input = httpclient.InferInput(
        "flatten_input", features.shape, datatype="FP32"
    )
    imageclass_input.set_data_from_numpy(features, binary_data=True)
    imageclass_response = client.infer(
        model_name="imageclassification", inputs=[imageclass_input]
    )
    predictions = imageclass_response.as_numpy("dense_1")
    return predictions

if __name__ == "__main__":
    triton_client = httpclient.InferenceServerClient(url='localhost:8000', verbose=False)
    
    features = get_features(triton_client, "./Imgdirty_872_1.jpg")
    predictions = get_predictions(triton_client, features)
  
    print('Prediction:', np.argmax(predictions))
