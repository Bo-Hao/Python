from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import os, sys 
os.chdir("/Users/pengbohao/Python_code/for_import")
sys.path.append(".")
from NoisyDense import NoisyDense
import decorator
os.chdir(os.path.dirname(__file__))


class Build_Model():
    def __init__(self, state_size, neurons, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.neurons = neurons

        self.Noisy = True
        tf.keras.backend.set_floatx('float64')
    def build(self):
        #input layer
        state_input = tf.keras.layers.Input(shape = (self.state_size, ))
        
        X = tf.keras.layers.Dense(self.neurons, activation = 'tanh')(state_input)
        
        X = tf.keras.layers.Dense(4, activation = 'linear')(X)

        
        #最後compile
        self.model = tf.keras.models.Model(inputs = state_input, outputs = X)
        return self.model
if __name__ == "__main__":
    M = Build_Model