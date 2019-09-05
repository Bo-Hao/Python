import tensorflow as tf 

activation_function = {
            'relu': tf.nn.relu, 
            'tanh': tf.nn.tanh, 
            'softmax': tf.nn.softmax,
            'linear': lambda x: x   
            }
name = ['relu',  
            'linear', 'tanh', 
            'softmax']
print(tf.keras.activations.tanh(8))
for i in range(len(name)):
    A = tf.keras.layers.Activation(activation = name[i])
    print(A(i))
