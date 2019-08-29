from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from NoisyDense import NoisyDense
import copy

class Build_Model():
    def __init__(self, state_size, neurons, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = 0.1

        #input layer
        state_input = tf.keras.layers.Input(shape = (self.state_size, ))
        D1 = tf.keras.layers.Dense(neurons, activation = 'tanh')(state_input)

        #連結層&Noisy Net
        d1 = NoisyDense(neurons, neurons, activation = 'tanh', Noisy = False)(D1)
        d2 = NoisyDense(neurons, neurons,activation = 'tanh', Noisy = False)(d1)
        #dueling
        d3_a = NoisyDense(neurons, int(neurons/2), activation = 'linear', Noisy = True)(d2)
        d3_v = NoisyDense(neurons, int(neurons/2), activation = 'linear', Noisy = True)(d2)

        a = ＮoisyDense(int(neurons/2), self.action_size, activation = 'linear', Noisy = False)(d3_a)
        value = NoisyDense(int(neurons/2), 1, activation='linear', Noisy = False)(d3_v)
        a_mean = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1, keepdims=True))(a)
        advantage = tf.keras.layers.Subtract()([a, a_mean])
        q = tf.keras.layers.Add()([value, advantage])

        #最後compile
        self.model = tf.keras.models.Model(inputs = state_input, outputs = q)
        self.model.compile(loss = 'mse', optimizer = tf.optimizers.Adam(lr = self.lr))