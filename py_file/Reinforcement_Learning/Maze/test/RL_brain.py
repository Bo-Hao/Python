import tensorflow as tf
import numpy as np
import pandas as pd
from build_model import Build_Model
from tensorflow.compat.v1.losses import huber_loss
from tensorflow.compat.v1.train import get_or_create_global_step

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.batch_size = 1
        self.state_size = 4
        

        M = Build_Model(4, 10, 4)
        self.model = M.build()
        self.optimizer = tf.optimizers.Adam(lr = self.lr)
        self.epochs = 10
        

    def choose_action(self, observation):
        
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            observation = np.array(observation)
            state_action = self.model.predict([[observation]])[0]
            # some actions may have the same value, randomly choose on in these actions
            action = np.argmax(state_action)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        print(action)
        return action


    def learn(self, s, a, r, s_):
        q_table = self.model.predict([[s]])[0]
        q_predict = q_table[a]
        
        if s_ != 'terminal':
            q_next_table = self.model.predict([[s_]])[0]
            q_target = r + self.gamma * max(q_next_table)  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        print(q_target - q_predict)
        q_table[a] += (q_target - q_predict)  # update
        
        
        X_train = np.zeros((self.batch_size, self.state_size))
        Y_train = [np.zeros(len(self.actions)) for i in range(self.batch_size)]
        X_train[0] = s
        for i_ in range(len(self.actions)):
            Y_train[0][i_] = q_table[i_]



        for epoch in range(self.epochs):
            self.train(self.model, X_train, Y_train)


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
        