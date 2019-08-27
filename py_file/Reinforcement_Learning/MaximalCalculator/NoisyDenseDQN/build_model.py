from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from NoisyDense import NoisyDense
import copy

class Build_Model():
    def __init__(self, state_size, neurons, action_size):
        self.state_size = state_size
        self.neurons = neurons
        self.action_size = action_size
        self.lr = 0.1

        state_input = tf.keras.layers.Input(shape=(self.state_size, ))
        D1 = tf.keras.layers.Dense(self.neurons, activation='tanh')(state_input)

        #連結層
        d1 = tf.keras.layers.Dense(self.neurons,activation='tanh')(D1)
        d2 = tf.keras.layers.Dense(self.neurons,activation='tanh')(d1)
        #dueling
        d3_a = tf.keras.layers.Dense(self.neurons/2, activation='sigmoid')(d2)
        d3_v = tf.keras.layers.Dense(self.neurons/2, activation='sigmoid')(d2)
        a = tf.keras.layers.Dense(self.action_size,activation='sigmoid')(d3_a)
        value = tf.keras.layers.Dense(1,activation='sigmoid')(d3_v)
        a_mean = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1, keepdims=True))(a)
        advantage = tf.keras.layers.Subtract()([a, a_mean])
        q = tf.keras.layers.Add()([value, advantage])
        # noisy
        self.noise = NoisyDense(self.action_size)
        final = self.noise(q)

        #最後compile
        self.model = tf.keras.models.Model(inputs=state_input, outputs=final)
        self.model.compile(loss = 'mse', optimizer = tf.optimizers.Adam(lr = self.lr))