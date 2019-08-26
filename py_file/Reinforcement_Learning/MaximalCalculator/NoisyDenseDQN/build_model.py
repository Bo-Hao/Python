from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from NoisyDense import NoisyDense
import copy

class Build_Model():
    def __init__(self, state_size, neurons, action_size):
        self.state_size = state_size
        self.neurons = neurons
        self.action_size = action_size
        self.bias_noisy = True
        self.weight_noisy = True
        self.lr = 0.1

        state_input = tf.keras.layers.Input(shape=(self.state_size, ))
        D1 = tf.keras.layers.Dense(self.neurons, activation='tanh')(state_input)

        #連結層
        d1 = tf.keras.layers.Dense(self.neurons,activation='tanh')(D1)
        d2 = tf.keras.layers.Dense(self.neurons,activation='tanh')(d1)
        
        #dueling
        d3_a = tf.keras.layers.Dense(self.neurons/2, activation='tanh')(d2)
        d3_v = tf.keras.layers.Dense(self.neurons/2, activation='tanh')(d2)
        a = tf.keras.layers.Dense(self.action_size,activation='tanh')(d3_a)
        value = tf.keras.layers.Dense(1,activation='tanh')(d3_v)
        a_mean = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=1, keepdims=True))(a)
        advantage = tf.keras.layers.Subtract()([a, a_mean])
        q = tf.keras.layers.Add()([value, advantage])
        # noisy

        #noise = tf.keras.layers.GaussianNoise(0.5)(q)
        noise = NoisyDense(self.action_size, bias= self.bias_noisy, training= self.weight_noisy)(q)

        #self.noise = copy.copy(noise)
        #最後compile
        self.model = tf.keras.models.Model(inputs=state_input, outputs=noise)
        self.model.compile(loss = 'mse', optimizer = tf.optimizers.Adam(lr = self.lr))