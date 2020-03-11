import os, sys 
os.chdir("/Users/pengbohao/Python_code/for_import")
sys.path.append(".")
from NoisyDense import Mydense
os.chdir(os.path.dirname(__file__))

import tensorflow as tf
import numpy as np 
import torch



class Residual_learning():
    def __init__(self):
        self.lr = 0.1
        self.neuron = 10
        self.shape = (1, )
        self.output = 1
        self.optimizer = tf.optimizers.Adam(self.lr)
        
        
        
    
    def build_model(self):
        inputs = tf.keras.Input(shape = self.shape)
        #X = Mydense(self.neuron, activation = 'linear', )(inputs)
        #X = Mydense(self.neuron, activation = 'relu', )(X)
        
        
        X = Mydense(self.output, activation = "linear", bias = True)(inputs)

        self.model = tf.keras.models.Model(inputs, X)
        
        return self.model
        


    def _loss(self, y_, y):
        '''x = np.array(list(map(self.clip, x)))
        y_ = self.model(x)'''
        y = np.array(list(y))
        loss = tf.reduce_mean(tf.square(y - y_))
        
        return loss
    
    def train(self, x, y):
        x = np.array(x).reshape(len(x), 1)
        
        with tf.GradientTape() as g:  
            y_ = self.model(x)
            y_ = tf.squeeze(y_, axis=1)
            loss = self._loss(y_, y)
            
        gradients = g.gradient(loss,  self.model.trainable_variables)
        print(gradients)
        self.optimizer.apply_gradients(zip(gradients,  self.model.trainable_variables))
        


if __name__ == "__main__":
    import pickle
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt 


    with open("/Users/pengbohao/Python_code/NeuralNetwork/regression_2degree_fakedata.pickle", 'rb') as f:
        data = pickle.load(f)


    x = torch.unsqueeze(torch.linspace(-1, 1, 200), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.mul(2) +3+ 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

    x = np.array(list(x))
    y = np.array(list(y))
    


    #X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size = 0.2, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    
    t = Residual_learning()
    t.build_model()

    for _epoch in range(100):
        t.train( X_train, y_train)

    print("finished")

    plt.scatter(X_train, y_train, c = 'green')
    #for i in range(len(X_test)):
    x = np.array(X_test).reshape(len(X_test), 1)
    plt.scatter(x, t.model(x), c = 'red')

    
    plt.show()
