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
        encoder_input = tf.keras.Input(shape = (self.inoutput_dim, ))
        encoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(encoder_input)
        encoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(encoder)
        
        distribution_mu = tf.keras.layers.Dense(self.distribution_number, activation = "linear")(encoder)
        distribution_var = tf.keras.layers.Dense(self.distribution_number, activation = "relu")(encoder)

        #multiplier = tf.constant([0.5])
        #std = tf.keras.layers.Multiply()([multiplier, distribution_var])
        std = tf.multiply(0.5, distribution_var)
        std = tf.keras.activations.exponential(std)
        epstd = ResampleNet()(std)
        z = tf.keras.layers.Add()([distribution_mu, epstd])

        decoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(z)
        decoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(decoder)
        d_out = tf.keras.layers.Dense(self.inoutput_dim, activation = 'linear')(decoder)


        self.training_model = tf.keras.models.Model(encoder_input, d_out)

        self.encoder = tf.keras.models.Model(encoder_input, [distribution_mu, distribution_var])
        self.zencoder = tf.keras.models.Model(encoder_input, z)

        decoder_input = tf.keras.Input(shape = (self.distribution_number, )) # Extra input layer
        d = self.training_model.layers[-3](decoder_input)
        d = self.training_model.layers[-2](d)
        d = self.training_model.layers[-1](d)

        self.decoder = tf.keras.models.Model(decoder_input, d)


    def _loss(self, x):
        shape1, shape2 = len(x), len(x[0])
        x = np.array(x).reshape((shape1, shape2))
        x_ = self.training_model(x)
        mu, logvar = self.encoder(x)

        mse_loss = tf.compat.v1.losses.mean_squared_error(x_, x)


        mu2 = tf.multiply(tf.pow(mu, 2), -1)
        var = tf.multiply(tf.exp(logvar), -1)
        kl = tf.add(tf.add(mu2, var), tf.add(logvar, 1))
        kl_loss = tf.multiply(tf.reduce_sum(kl), -0.5)
        
        #kl_loss = -0.5*tf.reduce_sum(logvar-tf.square(mu)-tf.exp(logvar)+1)

        
        return mse_loss, kl_loss/len(x)
        
    def _grad(self, x):
        with tf.GradientTape() as tape:
            loss_value = self._loss(x)
            #self.epoch_loss_avg(loss_value)
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
    
    
    