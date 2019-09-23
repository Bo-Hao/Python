from keras.layers import SimpleRNN, Activation, Dense
import numpy as np 
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

time_steps = 28		# same as the height of the image
input_size = 28		# same as the width of the image	
batch_size = 50
batch_index = 0
cell_size = 50
output_size = 10
lr = 1e-3

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 28, 28) / 255		# normalize
X_test = X_test.reshape(-1, 28, 28) / 255		# normalize
y_train = np_utils.to_categorical(y_train, num_classes = 10)
y_test = np_utils.to_categorical(y_test, num_classes = 10)

# build RNN model 
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape = (None, time_steps, input_size),       
    output_dim = cell_size,
    unroll = True,
))

# output layer
model.add(Dense(output_size))
model.add(Activation("softmax"))

# optimizer
adam = Adam(lr)
model.compile(optimizer = adam, 
	loss = "categorical_crossentropy",
	metrics = ["accuracy"])

# training
for step in range(4001):
	X_batch = X_train[batch_index: batch_size + batch_index, :, :]
	Y_batch = y_train[batch_index: batch_size + batch_index, :]
	cost = model.train_on_batch(X_batch, Y_batch)

	batch_index += batch_size
	batch_index = 0 if batch_index >= X_train.shape[0] else batch_index

	if step % 500 == 0:
		cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
		print('test cost: ', cost, 'test accuracy: ', accuracy)
















