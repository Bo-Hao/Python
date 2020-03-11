import tensorflow as tf
from tensorflow.compat.v1.losses import huber_loss
from tensorflow.compat.v1.train import get_or_create_global_step


class test():
    def __init__(self):
        self.lr = 0.1
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
    import numpy as np
    import matplotlib.pyplot as plt 
    from sklearn.model_selection import train_test_split
    import pickle


    with open("regression_2degree_fakedata.pickle", 'rb') as f:
        data = pickle.load(f)
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size = 0.2, random_state = 0)
    
    t = test()
    M = t.build_model()

    
    for i in range(1000):
        t.train(M, X_train, y_train)
    l = 0
    for i in range(len(X_test)):
        invector = np.array([[X_test[i]]])
        l += huber_loss(M(invector)[0][0], y_test[i])
    print(l/len(X_test))






plt.scatter(X_train, y_train, c = 'green')
for i in range(len(X_test)):
    x = np.array([[X_test[i]]])
    plt.scatter(x, M(x), c = 'red')

plt.show()
