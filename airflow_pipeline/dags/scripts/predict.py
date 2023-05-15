from train import get_model
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

def predict(path_to_save, model_name):
    X_test = pd.read_csv(path_to_save + 'data_test.csv')[:20]

    model = load_model(path_to_save + model_name)

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_test['text'])
    sequences = tokenizer.texts_to_sequences(X_test['text'])
    X = pad_sequences(sequences, maxlen=100)

    predictions = model.predict(X)
    with open(path_to_save + 'results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(predictions)