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

        eps = tf.random.normal([self.distribution_number, ])

        multiplier = tf.constant([0.5])
        std = tf.keras.layers.Multiply()([multiplier, distribution_var])
        std = tf.keras.activations.exponential(std)
        
        epstd = ResampleNet()(std)
        #epstd = tf.keras.layers.Multiply()([eps, std])
        #epstd = tf.multiply(std, eps)
        z = tf.keras.layers.Add()([distribution_mu, epstd])

        decoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(z)
        decoder = tf.keras.layers.Dense(self.neuron, activation = "linear")(decoder)
        d_out = tf.keras.layers.Dense(self.inoutput_dim, activation = 'linear')(decoder)

        self.m1 = tf.keras.models.Model(encoder_input, distribution_mu)
        self.m2 = tf.keras.models.Model(encoder_input, distribution_var)
        self.m3 = tf.keras.models.Model(encoder_input, std)
        self.m4 = tf.keras.models.Model(encoder_input, epstd)
        self.m5 = tf.keras.models.Model(encoder_input, z)

        self.training_model = tf.keras.models.Model(encoder_input, d_out)

        '''self.encoder = tf.keras.models.Model(encoder_input, latent)

        decoder_input = tf.keras.Input(shape = (self.inoutput_dim, )) # Extra input layer
        d = self.training_model.layers[-3](decoder_input)
        d = self.training_model.layers[-2](d)
        d = self.training_model.layers[-1](d)

        self.decoder = tf.keras.models.Model(decoder_input, d)'''
    
    
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
    A = VAE()
    A.build_model()

    #x = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    x = [[1, 2, 3]]
    y = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    x = np.array(x)

    print(A.m1(x))
    print(A.m2(x))
    print('---------------')
    print(A.m3(x))
    print(A.m4(x))
    print(A.m4(x))
    print(A.m4(x))
    #print(A.m5(x))
    print('---------------')
    print(A.training_model(x))

    '''eps = tf.random.normal([3, ])
    p = tf.multiply(eps, eps)
    q = tf.add(p, eps)
    print(eps)
    print(p)
    print(q)'''
    
    
    
    