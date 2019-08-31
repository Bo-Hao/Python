from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from NoisyDense import NoisyDense
import copy

class Build_Model():
    def __init__(self, state_size, neurons, action_size, atoms = 51, lr = 0.1):
        self.state_size = state_size
        self.action_size = action_size

        #input layer
        state_input = tf.keras.layers.Input(shape = (self.state_size, ))
        D1 = tf.keras.layers.Dense(neurons, activation = 'tanh')(state_input)

        #連結層&Noisy Net
        d1 = NoisyDense(neurons, neurons, activation = 'tanh', Noisy = False)(D1)
        d2 = NoisyDense(neurons, neurons,activation = 'tanh', Noisy = False)(d1)

        #dueling
        d3_a = NoisyDense(neurons, int(neurons/2), activation = 'linear', Noisy = True)(d2)
        d3_v = NoisyDense(neurons, int(neurons/2), activation = 'linear', Noisy = True)(d2)
        value = NoisyDense(int(neurons/2), 1, activation='linear', Noisy = False)(d3_v)

        a_list = []
        for act_ in range(action_size):
            act = ＮoisyDense(int(neurons/2), atoms, activation = 'linear', Noisy = False)(d3_a)
            a_list.append(act)

        a_mean = tf.reduce_mean(a_list, 0)
        a_mean = tf.reduce_mean(a_mean, 1)

        subtract_list = []
        for sub_ in range(action_size):
            sub = tf.keras.layers.Subtract()([a_list[sub_], a_mean])
            subtract_list.append(sub)
        
        add_list = []
        for add_ in range(action_size):
            add1 = tf.nn.softmax(tf.keras.layers.Add()([value, subtract_list[add_]]))
            add_list.append(add1)

        output_list = add_list



        #最後compile
        self.model = tf.keras.models.Model(inputs = state_input, outputs = output_list)
        self.model.compile(loss = 'mse', optimizer = tf.optimizers.Adam(lr = lr))



if __name__ == "__main__":
    M = Build_Model(1, 10 ,4)
    model = M.model
    ans = model.predict([0 for i in range(2)])
    import numpy as np 
    print(np.array(ans).shape)

    