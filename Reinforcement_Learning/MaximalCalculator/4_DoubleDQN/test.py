from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


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
        sigma_init = tf.constant_initializer(value=0.1)
        
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
        bias = tf.add(self.mu_bias, tf.multiply(self.sigma_bias, self.epsilon_bias))
        return tf.matmul(inputs, weights) + bias


#####################################################################################################################


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Subtract, Lambda, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from agent.Noisynet import NoisyDense

#Tensorflow 2.0 Beta

class Dueling_model():
    def build_model(self, state_size, neurons, action_size):
        #前面的LSTM層
        state_input = Input(shape=state_size)
        lstm1 = LSTM(neurons, activation='sigmoid',return_sequences=False)(state_input)

        #連結層
        d1 = Dense(neurons,activation='relu')(lstm1)
        d2 = Dense(neurons,activation='relu')(d1)
        
        #dueling
        d3_a = Dense(neurons/2, activation='relu')(d2)
        d3_v = Dense(neurons/2, activation='relu')(d2)
        a = Dense(action_size,activation='linear')(d3_a)
        value = Dense(1,activation='linear')(d3_v)
        a_mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = Subtract()([a, a_mean])
        q = Add()([value, advantage])

        # noisy
        noise = NoisyDense(action_size, bias=True, training=True)
        final = noise(q)

        #最後compile
        model = Model(inputs=state_input, outputs=final)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        
        return model
