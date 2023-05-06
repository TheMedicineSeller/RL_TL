import os
import numpy as np
import sys
import random

from tensorflow import keras
from keras import layers, backend
from keras.optimizers import Adam
from keras.losses import mean_squared_error
# from keras.utils import plot_model
from keras.models import load_model


class TLModel:
    def __init__(self, n_layers, width, batch_size, learning_rate, input_dim, output_dim, max_history_size):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._history = []
        self._MAX_history = max_history_size
        self._model = self._build_model(n_layers, width)

    def add_sample(self, sample):
        self._history.append(sample)
        if len(self._history) > self._MAX_history:
            self._history.pop(0)

    def get_samples(self, n):
       if n > len(self._history):
           return self._history
       else:
           return random.sample(self._history, n)


    def _build_model(self, n_layers, widths):
        # DNN
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(widths[0], activation='relu')(inputs)
        for i in range(1, n_layers):
            x = layers.Dense(widths[i], activation='relu')(x)
            
        outputs = layers.Dense(self._output_dim, activation='sigmoid')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='TrafficLightRL')
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        return model
    
    def load_trained_model(self, path):
        self._model = load_model(path)
    

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=0)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states, verbose=0)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)
        backend.clear_session()


    def save_model(self, name, path):
        self._model.save(os.path.join(path, name), save_format='h5')


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size
    

# def load_trained_model(path, model_name):
#     """
#     Load the model stored in the folder specified by the model number, if it exists
#     """
#     model_path = os.path.join(path, model_name)
#     if os.path.isfile(model_path):
#         loaded_model = load_model(model_path)
#         return loaded_model
#     else:
#         sys.exit("Model number not found")
