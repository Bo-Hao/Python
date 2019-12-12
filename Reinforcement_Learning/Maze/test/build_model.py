from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys 
os.chdir("/Users/pengbohao/Python_code/for_import")
sys.path.append(".")
import NoisyDense
import decorator
os.chdir(os.path.dirname(__file__))

import tensorflow as tf
from NoisyDense import NoisyDense


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
 
        X = NoisyDense(self.neurons, activation = 'tanh', Noisy = False, bias = False)(state_input)
        X = tf.keras.layers.Dropout(0.4)(X)
        X = NoisyDense(4, activation = 'linear', Noisy = False, bias = False)(X)
       

        
        #最後compile
        self.model = tf.keras.models.Model(inputs = state_input, outputs = X)
        return self.model

if __name__ == "__main__":
    M = Build_Model(4, 6, 4)
    model = M.build()
    inputs = [[1, 1, 1, 1]]
    a = model([inputs])
    print(a)