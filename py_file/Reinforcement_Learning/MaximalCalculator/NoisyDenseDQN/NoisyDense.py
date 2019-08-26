from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import copy

# Noisy Network 
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, bias=True, training=True):
        super(NoisyDense, self).__init__()
        #是訓練階段給高斯雜訊，不是就把epsilon設0
        if training:
            epsilon = tf.random_normal_initializer(mean=0, stddev=1)
        else:
            epsilon = tf.constant_initializer(value=0)
        
        # mu亂數，sigma常數0.1 (我要當rainbow)
        mu_init = tf.random_uniform_initializer(minval=-1, maxval=1)
        sigma_init = tf.constant_initializer(value=0.5)
        
        # 看要不要bias
        if bias:  
            mu_bias_init = mu_init
            sigma_bias_init = sigma_init
        else:
            mu_bias_init = tf.zeros_initializer()
            sigma_bias_init = tf.zeros_initializer()
        
        # mu + sigma * epsilon for weight
        self.epsilon_w = tf.Variable(initial_value=epsilon(shape=(units,units),
        dtype='float32'),trainable=False)
        self.mu_w = tf.Variable(initial_value=mu_init(shape=(units,units),
        dtype='float32'),trainable=True)
        self.sigma_w = tf.Variable(initial_value=sigma_init(shape=(units,units),
        dtype='float32'),trainable=True)
        # mu + sigma * epsilon for bias
        self.epsilon_bias = tf.Variable(initial_value=epsilon(shape=(units,),
        dtype='float32'),trainable=False)
        self.mu_bias = tf.Variable(initial_value=mu_bias_init(shape=(units,),
        dtype='float32'),trainable=True)
        self.sigma_bias = tf.Variable(initial_value=sigma_bias_init(shape=(units,),
        dtype='float32'),trainable=True)
        
    def call(self, inputs):
        weights = tf.add(self.mu_w, tf.multiply(self.sigma_w, self.epsilon_w))
        self.bias = tf.add(self.mu_bias, tf.multiply(self.sigma_bias, self.epsilon_bias))
        return tf.matmul(inputs, weights) + self.bias
