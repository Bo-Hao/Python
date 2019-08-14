import tensorflow as tf
import numpy as np 

class Autoencoder:
    def __init__(self, data):
        self.data = np.array(data)/100
        self.N = 2
        self.epochs = 5
        self.shape = self.data.shape[1:]
        
        
    def fit(self):
        
        inputs = tf.keras.Input(shape = (4, ))
        #encoded1 = tf.keras.layers.Dense(4, activation='relu')(inputs)
        encoded1 = tf.keras.layers.Dense(2, activation='tanh')(inputs)
        
        decoded1 = tf.keras.layers.Dense(2, activation='tanh')(encoded1)
        decoded2 = tf.keras.layers.Dense(4, activation='tanh')(decoded1)
        
        autoencoder = tf.keras.models.Model(inputs, decoded2)
        self.encoder = tf.keras.models.Model(inputs, encoded1)
        
        
        autoencoder.compile(optimizer='adam',loss='MSE', metrics = ['accuracy'])
        autoencoder.fit(self.data, self.data, epochs = self.epochs)
        

    def predict(self, X_test):
        self.pred = self.encoder(X_test)
        return self.encoder(X_test)
    
    def draw(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.pred[:,0], self.pred[:,1], c = 'red')
        plt.show()
                
        

if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    
    A = Autoencoder(X)
    A.fit()
    A.predict(X)
    A.draw()
    