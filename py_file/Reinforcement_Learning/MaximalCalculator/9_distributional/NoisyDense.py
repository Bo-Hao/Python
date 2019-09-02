from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import copy
import numpy as np

# Noisy Network 
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, in_shape, units, activation = None, Noisy = True, bias = True):
        super(NoisyDense, self).__init__()
        self.units = units
        self.noisy = Noisy
        self.activation_function = tf.keras.layers.Activation(activation = activation)

        # mu亂數，sigma常數0.1 
        mu_init = tf.random_uniform_initializer(minval=-1, maxval=1)
        sigma_init = tf.constant_initializer(value=0.1)
        # need Bias or not
        mu_bias_init = mu_init if bias else tf.zeros_initializer()
        sigma_bias_init = sigma_init if bias else tf.zeros_initializer()

        # mu + sigma * epsilon for weight
        self.mu_w = tf.Variable(initial_value=mu_init(shape=(in_shape,units),
        dtype='float32'),trainable=True)
        self.sigma_w = tf.Variable(initial_value=sigma_init(shape=(in_shape,units),
        dtype='float32'),trainable=True)
        # mu + sigma * epsilon for bias
        self.mu_bias = tf.Variable(initial_value=mu_bias_init(shape=(units,),
        dtype='float32'),trainable=True)
        self.sigma_bias = tf.Variable(initial_value=sigma_bias_init(shape=(units,),
        dtype='float32'),trainable=True)
        
    def call(self, inputs):
        #是訓練階段給高斯雜訊，不是就把epsilon設0
        if self.noisy:
            p = tf.random.normal([inputs.shape[1], self.units])
            q = tf.random.normal([1, self.units])
            f_p = tf.multiply(tf.sign(p), tf.pow(tf.abs(p), 0.5))
            f_q = tf.multiply(tf.sign(q), tf.pow(tf.abs(q), 0.5))
            epsilon_w = f_p*f_q
            epsilon_b = tf.squeeze(f_q)
        else:
            epsilon_w = 0
            epsilon_b = 0 

        weights = tf.add(self.mu_w, tf.multiply(self.sigma_w, epsilon_w))
        bias = tf.add(self.mu_bias, tf.multiply(self.sigma_bias, epsilon_b))
        result = self.activation_function(tf.matmul(inputs, weights) + bias)
        return result

