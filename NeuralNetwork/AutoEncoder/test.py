import tensorflow as tf
from tensorflow.compat.v1.losses import huber_loss
from tensorflow.compat.v1.train import get_or_create_global_step
import numpy as np
from decorator import timecost

class test():
    def __init__(self):
        self.lr = 0.01
        self.neuron = 3
        self.shape = (1, )
        self.output = 1
        self.optimizer = tf.optimizers.Adam(lr = self.lr)
        
    
    def build_model(self):
        inputs = tf.keras.Input(shape = self.shape, dtype=tf.dtypes.float64)
        X = tf.keras.layers.Dense(self.neuron, activation = 'relu', dtype=tf.dtypes.float64)(inputs)
        X = tf.keras.layers.Dense(self.neuron, activation = 'relu', dtype=tf.dtypes.float64)(X)
        X = tf.keras.layers.Dense(self.neuron, activation = 'relu', dtype=tf.dtypes.float64)(X)
        X = tf.keras.layers.Dense(self.output, activation = "linear", dtype=tf.dtypes.float64)(X)

        self.model = tf.keras.models.Model(inputs, X)

        return self.model
    
    def _loss(self, model, x, y):
        x = np.array(x).reshape((400, 1))
        y_ = model(x)

        loss = huber_loss(y_[:, 0], y)
        return loss
        
    def _grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self._loss(model, inputs, targets)
            #self.epoch_loss_avg(loss_value)
            return tape.gradient(loss_value, self.model.trainable_variables)

    def train(self, model, x, y):
        grads = self._grad(model, x, y)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables),
            get_or_create_global_step())



if __name__ == "__main__":
    print()
