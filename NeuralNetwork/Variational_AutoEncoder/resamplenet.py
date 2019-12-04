from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np



class ResampleNet(tf.keras.layers.Layer):
    def __init__(self, **kwargs):  # 要加上**kwargs，主層在存取時才不會報錯
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ResampleNet, self).__init__(**kwargs)
  
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        super(ResampleNet, self).build(input_shape)
        
    def call(self, inputs):
        random_weights = tf.random.normal([int(inputs.shape[-1]), 1])
        return tf.matmul(inputs, random_weights)