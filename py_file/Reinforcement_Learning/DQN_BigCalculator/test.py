import tensorflow.keras as tf 
from sklearn import datasets
import numpy as np

data = datasets.load_diabetes()
v = np.array(data.data)
t = data.target
t = np.array([[i, (i/2)**2] for i in t])




model = tf.models.Sequential([
            tf.layers.Dense(4, input_shape = (10, ), activation = 'linear'), 
            tf.layers.Dense(10, activation = 'linear'), 
            tf.layers.Dropout(0.2), 
            tf.layers.Dense(2, activation = 'linear')
        ])

model.compile(optimizer='adam', loss='MSE')
print(np.array(v[0]).reshape(1, 10).shape)
model.fit(np.array(v[0]).reshape(1, 10), np.array(t[0]).reshape(1, 2), epochs = 1)
