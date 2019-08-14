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


'''
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model







mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# add a channel dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Create an instance of the model
model = MyModel()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
  
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
      
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()      

EPOCHS = 3

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

# Reset the metrics for the next epoch
train_loss.reset_states()
train_accuracy.reset_states()
test_loss.reset_states()
test_accuracy.reset_states()

    

'''


