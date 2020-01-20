from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np


# Noisy Network
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation = None, Noisy = True, bias = True, **kwargs):  # 要加上**kwargs，主層在存取時才不會報錯
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.noisy = Noisy
        self.bias = bias
        self.activation_function = tf.keras.layers.Activation(activation = activation)
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        # mu亂數，sigma常數0.1 
        mu_init = tf.keras.initializers.glorot_normal()
        sigma_init = tf.constant_initializer(value=0.1)
        # need Bias or not
        mu_bias_init = mu_init if self.bias else tf.zeros_initializer()
        sigma_bias_init = sigma_init if self.bias else tf.zeros_initializer()

        # mu + sigma * epsilon for weight
        self.mu_w = tf.Variable(initial_value=mu_init(shape=(self.input_dim, self.units),
        dtype='float64'), trainable=True)
        self.sigma_w = tf.Variable(initial_value=sigma_init(shape=(self.input_dim, self.units),
        dtype='float64'), trainable=True)
        # mu + sigma * epsilon for bias
        self.mu_bias = tf.Variable(initial_value=mu_bias_init(shape=(self.units,),
        dtype='float64'), trainable=True)
        self.sigma_bias = tf.Variable(initial_value=sigma_bias_init(shape=(self.units,),
        dtype='float64'), trainable=True)

        super(NoisyDense, self).build(input_shape)
        
    def call(self, inputs):
        # Factor 式的 noisy
        #是訓練階段給高斯雜訊，不是就把epsilon設0
        if self.noisy:
            p = tf.random.normal([inputs.shape[-1], self.units], dtype='float64')
            q = tf.random.normal([1, self.units], dtype='float64')
            f_p = tf.multiply(tf.sign(p), tf.pow(tf.abs(p), 0.5))
            f_q = tf.multiply(tf.sign(q), tf.pow(tf.abs(q), 0.5))
            epsilon_w = f_p*f_q
            epsilon_b = tf.squeeze(f_q)
        else:
            epsilon_w = 0
            epsilon_b = 0 
        weights = tf.add(self.mu_w, tf.multiply(self.sigma_w, epsilon_w))
        bias = tf.add(self.mu_bias, tf.multiply(self.sigma_bias, epsilon_b))
        return self.activation_function(tf.matmul(inputs, weights) + bias)



class Mydense(tf.keras.layers.Layer):
    def __init__(self, units, activation = None, bias = True, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Mydense, self).__init__(**kwargs)
        self.units = units
        self.bias = bias
        self.activation_function = tf.keras.layers.Activation(activation = activation)
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        # mu亂數
        mu_init = tf.keras.initializers.glorot_uniform()
        # need Bias or not
        mu_bias_init = mu_init if self.bias else tf.zeros_initializer()

        # mu for weight
        self.mu_w = tf.Variable(initial_value=mu_init(shape=(self.input_dim, self.units),
        dtype='float32'), trainable=True)
        # mu for bias
        self.mu_bias = tf.Variable(initial_value=mu_bias_init(shape=(self.units,),
        dtype='float32'), trainable=self.bias)
        super(Mydense, self).build(input_shape)
        
        
    def call(self, inputs):
        weights = self.mu_w
        bias = self.mu_bias
        result = self.activation_function(tf.matmul(inputs, weights) + bias)
        return result