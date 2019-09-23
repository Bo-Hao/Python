from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from NoisyDense import NoisyDense
import tensorflow as tf 
import copy 
import numpy as np
from build_model import Build_Model
from Memory import Memory
from tensorflow.compat.v1.losses import huber_loss
from tensorflow.compat.v1.train import get_or_create_global_step
from tensorflow.keras.utils import Progbar


class Build_Model():
    def __init__(self, state_size, neurons, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.neurons = neurons
        self.optimizer = tf.optimizers.Adam(lr = 0.01)
        self.Noisy = True
        tf.keras.backend.set_floatx('float64')
    def build(self):
        #input layer
        state_input = tf.keras.layers.Input(shape = (self.state_size, ))
        
        X = tf.keras.layers.Dense(self.neurons, activation = 'tanh')(state_input)
        X = tf.keras.layers.Dense(self.action_size, activation = 'linear')(X)

        
        #最後compile
        self.model = tf.keras.models.Model(inputs = state_input, outputs = X)
        return self.model

    def _loss(self, model, x, y):
        x = np.array(x)
        y_ = model(x)
        loss = huber_loss(y, y_)
        return loss
	
    def _grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self._loss(model, inputs, targets)
            
            return tape.gradient(loss_value, self.model.trainable_variables)

    def train(self, model, s, q):
        grads = self._grad(model, s, q)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables),
            get_or_create_global_step())



if __name__ == "__main__":
    M = Build_Model(2, 10, 2)
    model = M.build()
    x = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 3],[2, 0], [2, 3], [3, 0], [3, 1], [3, 2], [3, 3]]
    y = [[1, 1], [0, 1], [0, 1], [1, 0], [1, 0], [-1, -1], [1, 0], [1, 0], [0, -1], [0, 1], [0, 1], [-1, 0], [-1, -1]]
    for i in range(200):
        M.train(model, x, y)
    print(model.predict([x]))
    