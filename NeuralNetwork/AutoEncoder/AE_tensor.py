import tensorflow as tf
from tensorflow.compat.v1.train import get_or_create_global_step
from tensorflow.compat.v1.losses import huber_loss
import numpy as np 
import matplotlib.pyplot as plt 

class AE():
    def __init__(self):
        self.lr = 0.01
        self.neuron = 4
        self.input_shape = (3, )
        self.latent_number = 3
        self.output_shape = 3
        self.optimizer = tf.optimizers.Adam(lr = self.lr)

    def build_model(self):
        encoder_input = tf.keras.Input(shape = self.input_shape)
        encoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(encoder_input)
        encoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(encoder)
        
        latent = tf.keras.layers.Dense(self.latent_number, activation = "relu")(encoder)

        decoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(latent)
        decoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(decoder)
        d_out = tf.keras.layers.Dense(self.output_shape, activation = 'linear')(decoder)

        
        self.training_model = tf.keras.models.Model(encoder_input, d_out)

        self.encoder = tf.keras.models.Model(encoder_input, latent)

        decoder_input = tf.keras.Input(shape = (self.output_shape, )) # Extra input layer
        d = self.training_model.layers[-3](decoder_input)
        d = self.training_model.layers[-2](d)
        d = self.training_model.layers[-1](d)

        self.decoder = tf.keras.models.Model(decoder_input, d)
    
    
    def _loss(self, x, y):
        shape1, shape2 = len(x), len(x[0])
        x = np.array(x).reshape((shape1, shape2))
        y_ = self.training_model(x)

        loss = tf.compat.v1.losses.mean_squared_error(y_, y)
        return loss
        
    def _grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self._loss(inputs, targets)
            #self.epoch_loss_avg(loss_value)
            return tape.gradient(loss_value, self.training_model.trainable_variables)

    def train(self, x, y):
        grads = self._grad(x, y)
        self.optimizer.apply_gradients(
            zip(grads, self.training_model.trainable_variables),
            get_or_create_global_step())



if __name__ == "__main__":
    A = AE()
    A.build_model()

    x = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    y = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    x = np.array(x)
    print(A.training_model(x))
    
    for i in range(100):
        A.train(x, y)
    print(A.training_model(x))
    print('train finished')
    
    
    
    
