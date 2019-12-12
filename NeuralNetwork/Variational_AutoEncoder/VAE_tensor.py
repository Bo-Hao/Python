import os, sys 
os.chdir("/Users/pengbohao/Python_code/for_import")
sys.path.append(".")
import NoisyDense
import decorator
os.chdir(os.path.dirname(__file__))

import tensorflow as tf
from tensorflow.compat.v1.train import get_or_create_global_step
from tensorflow.compat.v1.losses import huber_loss
import numpy as np 
from resamplenet import ResampleNet



class VAE():
    def __init__(self):
        self.lr = 0.01
        self.neuron = 4
        self.inoutput_dim = 3
        self.distribution_number = 3
        self.optimizer = tf.optimizers.Adam(lr = self.lr)



    def build_model(self):
        # encode net
        encoder_input = tf.keras.Input(shape = (self.inoutput_dim, ))
        encoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(encoder_input)
        encoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(encoder)

        # distribution        
        distribution_mu = tf.keras.layers.Dense(self.distribution_number, activation = "linear")(encoder)
        distribution_logvar = tf.keras.layers.Dense(self.distribution_number, activation = "relu")(encoder)


        std = tf.multiply(0.5, distribution_logvar)
        std = tf.keras.activations.exponential(std)
        
        # random normal sample 
        epstd = ResampleNet()(std)

        # mu + std * epsilon
        z = tf.keras.layers.Add()([distribution_mu, epstd])
        
        # decode net
        decoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(z)
        decoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(decoder)
        d_out = tf.keras.layers.Dense(self.inoutput_dim, activation = 'linear')(decoder)

        
        self.training_model = tf.keras.models.Model(encoder_input, d_out)

        self.encoder = tf.keras.models.Model(encoder_input, [distribution_mu, distribution_logvar])
        self.zencoder = tf.keras.models.Model(encoder_input, z)

        decoder_input = tf.keras.Input(shape = (self.distribution_number, )) # Extra input layer take z as input
        d = self.training_model.layers[-3](decoder_input)
        d = self.training_model.layers[-2](d)
        d = self.training_model.layers[-1](d)

        self.decoder = tf.keras.models.Model(decoder_input, d)


    def _loss(self, x):
        shape1, shape2 = len(x), len(x[0])
        x = np.array(x).reshape((shape1, shape2))
        x_ = self.training_model(x)
        mu, logvar = self.encoder(x)

        # Mean square error as loss function 
        mse_loss = tf.compat.v1.losses.mean_squared_error(x_, x)

        # KL divergence as loss function 
        kl_loss = -0.5*tf.reduce_sum(logvar-tf.square(mu)-tf.exp(logvar)+1)

        return mse_loss, kl_loss/len(x)
        
    def _grad(self, x):
        with tf.GradientTape() as tape:
            loss_value = self._loss(x)
            return tape.gradient(loss_value, self.training_model.trainable_variables)

    def train(self, x):
        grads = self._grad(x)
        self.optimizer.apply_gradients(
            zip(grads, self.training_model.trainable_variables),
            get_or_create_global_step())



if __name__ == "__main__":
    A = VAE()
    A.build_model()

    x = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    #x = [[1, 2, 3]]
    
    x = np.array(x)

    print(A.decoder(A.zencoder(x)))
    print(A.training_model(x))
    for i in range(100):
        A.train(x)
    print("training fininshed")
    print(A.training_model(x))
    
    
    