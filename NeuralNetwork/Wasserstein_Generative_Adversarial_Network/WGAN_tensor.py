import os, sys 
os.chdir("/Users/pengbohao/Python_code/for_import")
sys.path.append(".")
import NoisyDense
import decorator
os.chdir(os.path.dirname(__file__))

import tensorflow as tf 
import numpy as np 
import decorator
from tensorflow.compat.v1.train import get_or_create_global_step


class WGAN():
    def __init__(self):
        self.lr = 0.01
        self.G_input_shape = 3
        self.D_input_shape = 5
        self.neuron = 7
        self.D_output_shape = 1
        self.C = float(1)

        self.optimizer = tf.optimizers.RMSprop(learning_rate = self.lr)
        
    
    def build_model(self):

        G_input = tf.keras.Input(shape = (self.G_input_shape, ))
        G = tf.keras.layers.Dense(self.neuron, activation = "relu", autocast=False)(G_input)
        G = tf.keras.layers.Dense(self.neuron, activation = "relu", autocast=False)(G)

        G_output = tf.keras.layers.Dense(self.D_input_shape, activation = "linear", autocast=False)(G)

        D = tf.keras.layers.Dense(self.neuron, activation = 'relu', autocast=False)(G_output)
        D = tf.keras.layers.Dense(self.neuron, activation = 'relu', autocast=False)(D)
        D_output = tf.keras.layers.Dense(self.D_output_shape, activation = 'linear', autocast=False)(D)

        self.training_model = tf.keras.models.Model(G_input, D_output)
        self.generator = tf.keras.models.Model(G_input, G_output)

        D_input_extra = tf.keras.Input(shape = (self.D_input_shape, ))
        d = self.training_model.layers[-3](D_input_extra)
        d = self.training_model.layers[-2](d)
        d_out = self.training_model.layers[-1](d)

        self.discriminator = tf.keras.models.Model(D_input_extra, d_out)


    def _loss(self, model, x, y):
        # JS divergence
        x = np.array(x)
        y_ = model(x)
        mse_loss = tf.reduce_sum(tf.square(tf.subtract(y, y_)))
        

        penalty = self._loss_clip_weight(model)
        return mse_loss+penalty

    def _loss_clip_weight(self, model):
        penalty = 0
        for layer in model.layers:
            s = layer.input_shape
            if len(s) == 2:
                ss = int(layer.input_shape[-1])
                ones = np.array([[1. for i in range(ss)]])
                weight = layer(ones)
                weight_loss = tf.square(tf.maximum(tf.abs(weight) - self.C, 0.0))/ss
                penalty = tf.add(penalty, tf.reduce_sum(weight_loss))
            print(penalty)

        return penalty


    def _grad(self, model, x, y):
        with tf.GradientTape() as tape:
            loss_value = self._loss(model, x, y)
            #self.epoch_loss_avg(loss_value)
            return tape.gradient(loss_value, model.trainable_variables)


    def train_disciminator(self, x, y):
        self.discriminator.trainable = True
        grads = self._grad(self.discriminator, x, y)
        self.optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables),
            get_or_create_global_step())

    def train_generator(self, x, y):
        self.discriminator.trainable = False
        grads = self._grad(self.training_model, x, y)
        self.optimizer.apply_gradients(
            zip(grads, self.training_model.trainable_variables),
            get_or_create_global_step())

    


if __name__ == "__main__":
    import numpy as np 
    r = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
    x = [np.random.normal(size = 3) for i in range(3)]

    r = np.array(r)
    x = np.array(x)

    G = WGAN()
    G.build_model()
    
    
    for i in range(100):
        G.train_generator(x, [[0], [0], [0]])
    print("training finished")
    
