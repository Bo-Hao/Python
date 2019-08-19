# Cross Vailation
# K folder cross vailation

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import tensorflow.keras as tf

epochs = 10


iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn import preprocessing 
X = preprocessing.scale(X)


result = []
for i in range(30):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle=True)
    

    model = tf.models.Sequential([ 
        tf.layers.Dense(4, activation = 'tanh'), 
        tf.layers.Dense(100, activation = 'tanh'),
        tf.layers.Dense(100, activation = 'tanh'),
        tf.layers.Dense(100, activation = 'tanh'),
        tf.layers.Dense(100, activation = 'tanh'),
        tf.layers.Dense(100, activation = 'tanh'),
        tf.layers.Dense(100, activation = 'tanh'),
        tf.layers.Dense(3, activation = 'softmax')
    ])

    model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epochs)
    res = model.evaluate(x_test, y_test)
    result.append(res[1])

print(result)
print(sum(result)/len(result))



from SVM import SVM
result = []
R = result.append 
for i in range(30):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle=True)
    S = SVM(x_train, y_train)
    S.fit()
    pre = S.predict(x_test)
    array = np.array(pre) - np.array(y_test)
    res = sum([1 for i in array if i == 0])/len(y_test)
    R(res)

print(result)
print(sum(result)/len(result))‘’‘