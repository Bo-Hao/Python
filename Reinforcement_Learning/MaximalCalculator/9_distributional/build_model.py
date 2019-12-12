
import tensorflow as tf
import os, sys 
os.chdir("/Users/pengbohao/Python_code/for_import")
sys.path.append(".")
from NoisyDense import NoisyDense
import decorator
os.chdir(os.path.dirname(__file__))
import copy

class Build_Model():
    def __init__(self, state_size, neurons, action_size, atoms = 51, lr = 0.1):
        self.state_size = state_size
        self.action_size = action_size

        #input layer
        state_input = tf.keras.layers.Input(shape = (self.state_size, ))
        D1 = tf.keras.layers.Dense(neurons, activation = 'tanh')(state_input)
        

        #連結層&Noisy Net
        d1 = NoisyDense(neurons, neurons, activation = 'tanh', Noisy = True, bias = False)(D1)
        #d2 = NoisyDense(neurons, neurons,activation = 'tanh', Noisy = False, bias = False)(d1)

        #dueling
        d3_a = NoisyDense(neurons, int(neurons/2), activation = 'tanh', Noisy = False, bias = False)(d1)
        d3_v = NoisyDense(neurons, int(neurons/2), activation = 'linear', Noisy = False, bias = False)(d1)
        value = NoisyDense(int(neurons/2), 1, activation='linear', Noisy = False, bias = False)(d3_v)

        advantage = NoisyDense(int(neurons/2), self.state_size, activation = 'linear', Noisy = True, bias = False)(d3_a)
        a_mean = tf.reduce_mean(advantage, 0)
        sub = tf.keras.layers.Subtract()([advantage, a_mean])
        add = tf.keras.layers.Add()([sub, value])

        out = []
        for i in range(self.action_size):
            out.append(
                tf.nn.softmax(NoisyDense(1, atoms, activation = 'linear', Noisy = False, bias = True)(add))
            )
        output_list = out

        '''a_list = []
        for act_ in range(action_size):
            act = ＮoisyDense(int(neurons/2), atoms, activation = 'linear', Noisy = False)(d3_a)
            a_list.append(act)

        a_mean = tf.reduce_mean(a_list, 0)
        a_mean = tf.reduce_mean(a_mean, 1)

        subtract_list = []
        for sub_ in range(action_size):
            sub = tf.keras.layers.Subtract()([a_list[sub_], a_mean])
            subtract_list.append(sub)
        # softmax function as activation function
        add_list = []
        for add_ in range(action_size):
            add1 = tf.nn.softmax(tf.keras.layers.Add()([value, subtract_list[add_]]))
            add_list.append(add1)

        output_list = add_list'''

        #最後compile
        self.model = tf.keras.models.Model(inputs = state_input, outputs = output_list)
        self.model.compile(
            loss = 'binary_crossentropy', 
            optimizer = tf.optimizers.Adam(lr = lr)

            )



if __name__ == "__main__":
    
    N = Build_Model(1, 10, 4)
    model = N.model
    import time
    t = time.time()
    print(model.predict([1]))
    t = time.time()-t
    print("time cost: ", t)