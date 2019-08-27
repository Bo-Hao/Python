from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import copy
import numpy as np

# Noisy Network 
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super(NoisyDense, self).__init__()
        self.units = units
        self.bias = True
        self.training = True
        # mu亂數，sigma常數0.1 (我要當rainbow)
        mu_init = tf.random_uniform_initializer(minval=-1, maxval=1)
        sigma_init = tf.constant_initializer(value=0.1)
        # 看要不要bias
        if self.bias:  
            mu_bias_init = mu_init
            sigma_bias_init = sigma_init
        else:
            mu_bias_init = tf.zeros_initializer()
            sigma_bias_init = tf.zeros_initializer()
        # mu + sigma * epsilon for weight
        self.mu_w = tf.Variable(initial_value=mu_init(shape=(units,units),
        dtype='float32'),trainable=True)
        self.sigma_w = tf.Variable(initial_value=sigma_init(shape=(units,units),
        dtype='float32'),trainable=True)
        # mu + sigma * epsilon for bias
        self.mu_bias = tf.Variable(initial_value=mu_bias_init(shape=(units,),
        dtype='float32'),trainable=True)
        self.sigma_bias = tf.Variable(initial_value=sigma_bias_init(shape=(units,),
        dtype='float32'),trainable=True)
        
    def call(self, inputs):
        #是訓練階段給高斯雜訊，不是就把epsilon設0
        if self.training:
            p = tf.random.normal([self.units, self.units])
            q = tf.random.normal([1, self.units])
            f_p = tf.multiply(tf.sign(p), tf.pow(tf.abs(p), 0.5))
            f_q = tf.multiply(tf.sign(q), tf.pow(tf.abs(q), 0.5))
            epsilon_w = f_p*f_q
            epsilon_b = tf.squeeze(f_q)

            weights = tf.add(self.mu_w, tf.multiply(self.sigma_w, epsilon_w))
            bias = tf.add(self.mu_bias, tf.multiply(self.sigma_bias, epsilon_b))
            return tf.matmul(inputs, weights) + bias

        elif self.training:
            weights = self.mu_w
            bias = self.mu_bias
            return tf.matmul(inputs, weights) + bias

       
