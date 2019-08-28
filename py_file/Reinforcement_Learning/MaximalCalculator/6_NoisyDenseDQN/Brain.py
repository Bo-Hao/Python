from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
import copy 
import numpy as np
from build_model import Build_Model

class NoisyQ:
    def __init__(self, actions, gamma=0.8, e_greedy = 0.9):
        self.actions = actions  # a list
        self.gamma = gamma
        self.epsilon = e_greedy
        self.record = []
        self.lr = 0.5
        self.count = 0

        self.m = Build_Model(1, 10, len(actions))
        self.model = self.m.model
        self.dump_model = copy.copy(self.model)

    def choose_action(self, s):
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.model.predict([s])[0]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice([i for i in range(len(state_action)) if state_action[i] == max(state_action)])
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        batch = 50
        record_size = 100

        q_list = self.model.predict([s])[0]
        q_predict = q_list[a]
        qvalue = self.dump_model.predict([s_])[0][np.argmax(q_list)]



        loss = r + (self.gamma * qvalue - q_predict)
        
        #q_list[a] += self.lr * loss  # update
        q_list[a] += loss 
        self.record.append([s, a, r, s_, q_list, loss])
        self.count += 1



        if len(self.record) > record_size:
            self.record = self.record[-record_size:]

        if self.count % record_size == 0:
            ind = np.random.choice(range(record_size), size = batch)
            Train = [self.record[i] for i in ind]

            X_train = np.array(Train)[:, 0]
            Y_train = np.array([i for i in np.array(Train)[:, 4]])
            self.model.fit(X_train, Y_train, epochs = 5)
            self.record = []

            self.m.noise.epsilon = 0
            print(self.model.predict([s])[0])
\

