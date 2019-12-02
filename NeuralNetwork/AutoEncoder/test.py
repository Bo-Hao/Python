

def learn(self):
    self.loss_record = []
    batch, index, is_weight = self.memory.sample(self.batch_size)
    # initial the training data
    X_train = np.zeros((self.batch_size, self.state_size))
    Y_train = [np.zeros(len(self.actions)) for i in range(self.batch_size)]
    for i in range(self.batch_size):
        s, a, r, s_ = batch[i][0], batch[i][1], batch[i][2], batch[i][3], 
        q_table = self.model.predict([[s]])[0]
        q_predict = q_table[a]
        if s_ != 'terminal':
            q_next_table = self.target_model.predict([[s_]])[0]
            next_action = np.argmax(self.model.predict([[s]])[0])
            q_target = r + self.gamma * q_next_table[next_action]  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        loss = abs(q_target) 
        q_table[a] += (q_target - q_predict) 
        # store memory
        self.loss_record.append(loss)
        # setup training data
        X_train[i] = s
        for i_ in range(len(self.actions)):
            Y_train[i][i_] = q_table[i_]

    #training
    for epoch in range(self.epochs):
        self.train(self.model, X_train, Y_train)

    # memory update 
    for i in range(self.batch_size):
        self.memory.update(index[i], self.loss_record[i])




import tensorflow as tf
from tensorflow.compat.v1.losses import huber_loss
from tensorflow.compat.v1.train import get_or_create_global_step
class test():
    def __init__(self):
        self.lr = 0.01
        self.neuron = 2
        self.shape = (1, )
        self.output = 1
        self.optimizer = tf.optimizers.Adam(lr = self.lr)
        
    
    def build_model(self):
        inputs = tf.keras.Input(shape = self.shape, dtype=tf.dtypes.float64)
        X = tf.keras.layers.Dense(self.neuron, activation = "relu", dtype=tf.dtypes.float64)(inputs)
        X = tf.keras.layers.Dense(self.neuron, activation = "relu", dtype=tf.dtypes.float64)(X)
        X = tf.keras.layers.Dense(self.neuron, activation = "relu", dtype=tf.dtypes.float64)(X)
        X = tf.keras.layers.Dense(self.output, activation = "linear", dtype=tf.dtypes.float64)(X)

        self.model = tf.keras.models.Model(inputs, X)

        return self.model

    def _loss(self, model, x, y):
        y_ = []
        for i in range(len(x)):
            _ = np.array([[x[i]]])
            y_ += [model(_)[0][0]]
        
        '''x = np.array([x])
        y_ = model(x)
        '''        
        loss = huber_loss(y, y_)

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
    
    
    for i in range(10):
        t.train(M, X_train, y_train)
    loss = 0
    for i in range(len(X_test)):
        invector = np.array([[X_test[i]]])
        loss += huber_loss(M(invector)[0][0], y_test[i])
    print(loss/len(X_test))

    for i in range(10):
        t.train(M, X_train, y_train)

    loss = 0
    for i in range(len(X_test)):
        invector = np.array([[X_test[i]]])
        loss += huber_loss(M(invector)[0][0], y_test[i])
    print(loss/len(X_test))

    for i in range(50):
        t.train(M, X_train, y_train)
    loss = 0
    for i in range(len(X_test)):
        invector = np.array([[X_test[i]]])
        loss += huber_loss(M(invector)[0][0], y_test[i])
    print(loss/len(X_test))


for i in range(len(X_test)):
    x = np.array([[X_test[i]]])
    plt.scatter(x, M(x))

plt.show()
