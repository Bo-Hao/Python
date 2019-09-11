import tensorflow as tf
import numpy as np
import pandas as pd
from build_model import Build_Model
from tensorflow.compat.v1.losses import huber_loss
from tensorflow.compat.v1.train import get_or_create_global_step
from Memory import Memory
import copy 

class QLearningTable:
    def __init__(self, actions, learning_rate=0.001, reward_decay=0.9, e_greedy=0.4):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.batch_size = 25
        self.state_size = 4
        
        # neural network
        M = Build_Model(4, 4, 4)
        self.model = M.build()
        self.target_model = copy.copy(self.model)
        self.optimizer = tf.optimizers.Adam(lr = self.lr)
        self.epochs = 1

        # memory
        self.capacity = 200
        self.memory = Memory(self.capacity)
        self.store_times = 0
        

    def choose_action(self, s):
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            s = np.array(s)
            state_action = self.model.predict([[s]])[0]
            print(state_action)
            # some actions may have the same value, randomly choose on in these actions
            action = np.argmax(state_action)
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def store_memory(self, s, a, r, s_):
        if r in [1, -1]:
            self.memory.add(100, [s, a, r, s_])
            self.memory.add(100, [s, a, r, s_])
        else:
            self.memory.add(1, [s, a, r, s_])
        self.store_times += 1

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

    def _loss(self, model, x, y):
        x = np.array(x)

        y_ = model([[x]])
        
        loss = huber_loss(y, y_)
        return loss
	
    def _grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self._loss(model, inputs, targets)
            #self.epoch_loss_avg(loss_value)
            return tape.gradient(loss_value, self.model.trainable_variables)

    def train(self, model, s, q):
        grads = self._grad(model, s, q)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables),
            get_or_create_global_step())
        