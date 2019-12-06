import tensorflow as tf 
import numpy as np 
import decorator

from tensorflow.compat.v1.train import get_or_create_global_step



class GAN():
    def __init__(self):
        self.lr = 0.01
        self.G_input_shape = 3
        self.D_input_shape = 5
        self.neuron = 7
        self.D_output_shape = 1

        self.optimizer = tf.optimizers.Adam(lr = self.lr)
        
    
    def build_model(self):

        G_input = tf.keras.Input(shape = (self.G_input_shape, ))
        G = tf.keras.layers.Dense(self.neuron, activation = "relu")(G_input)
        G = tf.keras.layers.Dense(self.neuron, activation = "relu")(G)

        G_output = tf.keras.layers.Dense(self.D_input_shape, activation = "linear")(G)

        D = tf.keras.layers.Dense(self.neuron, activation = 'relu')(G_output)
        D = tf.keras.layers.Dense(self.neuron, activation = 'relu')(D)
        D_output = tf.keras.layers.Dense(self.D_output_shape, activation = 'sigmoid')(D)

        self.training_model = tf.keras.models.Model(G_input, D_output)
        self.generator = tf.keras.models.Model(G_input, G_output)

        D_input_extra = tf.keras.Input(shape = (self.D_input_shape, ))
        d = self.training_model.layers[-3](D_input_extra)
        d = self.training_model.layers[-2](d)
        d_out = self.training_model.layers[-1](d)

        self.discriminator = tf.keras.models.Model(D_input_extra, d_out)


    def _loss(self, model, x, y):
        x = np.array(x)
        y_ = model(x)
        mse_loss = tf.reduce_sum(tf.square(tf.subtract(y, y_)))
        return mse_loss

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
            zip(grads, self.discriminator.trainable_variables),
            get_or_create_global_step())

    


if __name__ == "__main__":
    import numpy as np 
    r = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
    x = [np.random.normal(size = 3) for i in range(3)]

    r = np.array(r)
    x = np.array(x)

    G = GAN()
    G.build_model()
    
    
    for i in range(10):
        G.train_generator(x, [[0], [0], [0]])
    print("training finished")
    
