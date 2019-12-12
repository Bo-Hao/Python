
import tensorflow as tf
import os, sys 
os.chdir("/Users/pengbohao/Python_code/for_import")
sys.path.append(".")
from NoisyDense import NoisyDense
import decorator
os.chdir(os.path.dirname(__file__))
import copy
import tensorflow.keras.backend as K

class Build_Model():
    def __init__(self, state_size, neurons, action_size, atoms = 51, lr = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.neurons = neurons
        self.atoms = atoms
        self.lr = lr
        self.Noisy = True


    def build(self):
        #input layer
        state_input = tf.keras.layers.Input(shape = (self.state_size, ))
        D1 = tf.keras.layers.Dense(self.neurons, activation = 'tanh')(state_input)
        

        d1 = NoisyDense(self.neurons, self.neurons, activation = 'tanh', Noisy = self.Noisy, bias = False)(D1)
        #d2 = NoisyDense(self.neurons, self.neurons, activation = 'tanh', Noisy = self.Noisy, bias = False)(d1)

        #dueling
        d3_a = NoisyDense(self.neurons, int(self.neurons/2), activation = 'tanh', Noisy = False, bias = False)(d1)
        d3_v = NoisyDense(self.neurons, int(self.neurons/2), activation = 'tanh', Noisy = False, bias = False)(d1)
        value = NoisyDense(int(self.neurons/2), 1, activation='tanh', Noisy = False, bias = False)(d3_v)

        advantage = NoisyDense(int(self.neurons/2), self.action_size, activation = 'tanh', Noisy = self.Noisy, bias = False)(d3_a)
        a_mean = tf.reduce_mean(advantage, 1)
        sub = tf.keras.layers.Subtract()([advantage, a_mean])
        add = tf.keras.layers.Add()([sub, value])

        N_list = []
        for i in range(self.action_size):
            N_list.append(
                NoisyDense(self.action_size, 1, activation = 'tanh', Noisy = False, bias = False)(add)
                )
        
        out_list = []
        for i in range(self.action_size):
            out_list.append(
                NoisyDense(1, self.atoms, activation = 'tanh', Noisy = self.Noisy, bias = True)(N_list[i])
                )

        output_list = out_list

        self.model = tf.keras.models.Model(inputs = state_input, outputs = output_list)
        return self.model

    def remove_noisy(self):
        weight = self.model.get_weights()
        self.Noisy = False
        self.build()
        self.model.set_weights(weight)



if __name__ == "__main__":
    import numpy as np
    N = Build_Model(4, 10, 4)
    N.build()
    model = N.model

    import time
    t = time.time()
    inputs = [[[1, 0, 0, 0], [0, 1, 0, 0]]]
    print(np.array(inputs).shape)
    ans = model.predict(inputs)
    print(np.array(ans).shape)
    t = time.time()-t
    print("time cost: ", t)
