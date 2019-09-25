import tensorflow as tf 
import numpy as np 
import copy 
import decorator
from NoisyDense import NoisyDense

class Generator():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.neurons = neurons
        self.output_shape = input_shape
        

    def build(self):
        data_input = tf.keras.layers.Input(shape = (self.input_shape, ))
        X = NoisyDense(self.neurons, activation = 'tanh')(data_input)
        X = NoisyDense(self.input_shape, activation ='linear')(X)
        self.model = tf.keras.models.Model(inputs = data_input, outputs = X)
        return self.model




class Discriminator():
    def __init__(self):
        self.input_shape = input_shape
        self.neurons = neurons


if __name__ == "__main__":
    G = Generator(4, 4)
    G.build()
    inputs = [[[1, 1, 1, 1], [2, 2, 2, 2]]]
    ans = G.model.predict(inputs)
    print(ans)