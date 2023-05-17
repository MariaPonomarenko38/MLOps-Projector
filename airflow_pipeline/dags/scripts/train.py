import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from utils import get_args

def get_model(args):
    model = Sequential()
    model.add(Embedding(input_dim=args["input_dim"], output_dim=args["output_dim"], input_length=args["input_length"]))
    model.add(Flatten())
    model.add(Dense(args["output_dim"], activation='relu'))
    model.add(Dropout(args["dropout_rate"]))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model

def train(path_to_save, path_to_config):
    args = get_args(path_to_config)
    X_train = pd.read_csv(path_to_save + 'data_train.csv')

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train['text'])
    sequences = tokenizer.texts_to_sequences(X_train['text'])
    padded_sequences = pad_sequences(sequences, maxlen=100)

    model = get_model(args)

    model.fit(padded_sequences, X_train['target'], epochs=args["epochs"], validation_split=0.2)
    model.save(path_to_save + args["model_name"])