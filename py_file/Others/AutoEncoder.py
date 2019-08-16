import tensorflow as tf
import numpy as np 

class Autoencoder:
    def __init__(self, data):
        self.data = np.array(data)/10
        self.N = 2
        self.epochs = 15
        self.shape = self.data.shape[1:]
        
        
    def fit(self):
        print(self.shape)
        inputs = tf.keras.Input(shape = self.shape)
        encoded1 = tf.keras.layers.Dense(self.shape[0], activation='sigmoid')(inputs)
        encoded2 = tf.keras.layers.Dense(self.N, activation='sigmoid')(encoded1)
        
        decoded1 = tf.keras.layers.Dense(self.N, activation='sigmoid')(encoded2)
        decoded2 = tf.keras.layers.Dense(self.shape[0], activation='sigmoid')(decoded1)
        
        autoencoder = tf.keras.models.Model(inputs, decoded2)
        self.encoder = tf.keras.models.Model(inputs, encoded2)
        
        
        autoencoder.compile(optimizer='adam',loss='MSE', metrics = ['accuracy'])
        autoencoder.fit(self.data, self.data, epochs = self.epochs)
        self.pred = self.encoder(self.data)

    def predict(self, X_test):
        self.pred = self.encoder(X_test)
        return self.encoder(X_test)
    
    def draw(self, target = ''):
        import matplotlib.pyplot as plt
        if target == '':
            plt.scatter(self.pred[:,0], self.pred[:,1], c = 'red')
            plt.title('Autoencoder')
            plt.show()
        else:
            plt.scatter(self.pred[:,0], self.pred[:,1], c = target)
            plt.title('Autoencoder')
            plt.show()
        

if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    
    A = Autoencoder(X)
    A.fit()
    A.predict(X)
    A.draw(target = Y)
    