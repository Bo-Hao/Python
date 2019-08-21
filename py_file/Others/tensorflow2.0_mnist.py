import time
import tensorflow as tf
import numpy as np

class Simple_classfyNN:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.epochs = 5
        
        
        inputshape = np.array(self.X).shape[1:]
        groups = len(set(self.Y))
        print(inputshape)
        self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape = inputshape), 
                tf.keras.layers.Dense(128, activation = 'relu'), 
                tf.keras.layers.Dropout(0.2), 
                tf.keras.layers.Dense(groups, activation = 'softmax')
                ])

    def fit(self):


        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model.fit(self.X, self.Y, epochs=self.epochs)


    def predict(self, X):
        #self.model.evaluate(X, Y)
        prediction = self.model(X)
        prediction = [np.argmax(i) for i in prediction]
        return prediction

def Simple_classfyNN_main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    S = Simple_classfyNN(x_train, y_train)
    S.epochs = 5
    S.fit()
    pre = S.predict(x_test)
    
    losss = (np.array(pre) - np.array(y_test))
    print(1 - sum([1 for i in losss if i != 0])/len(x_test))


Simple_classfyNN_main()
