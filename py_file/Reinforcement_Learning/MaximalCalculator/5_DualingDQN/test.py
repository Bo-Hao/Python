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
