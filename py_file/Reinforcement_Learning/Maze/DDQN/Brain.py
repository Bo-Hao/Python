from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
import copy 
import numpy as np
from build_model import Build_Model
from Memory import Memory

class NoisyQ:
    def __init__(self, actions, gamma=0.1, e_greedy = 0.9):
        self.actions = actions  # a list
        self.gamma = gamma
        self.epsilon = e_greedy
        self.lr = 0.1
        self.count = 0
        self.epochs = 5
        # initial model include the hard-working one and the dump one.
        self.m = Build_Model(1, 10, len(actions))
        self.model = self.m.model
        self.dump_model = copy.copy(self.model)
        # initial memory with sum tree
        self.capacity = 200
        self.memory = Memory(self.capacity)

    def choose_action(self, s):
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.model.predict([s])[0]
            # Doulbe method 
            action = np.random.choice([i for i in range(len(state_action)) if state_action[i] == max(state_action)])
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        batch_size = 100
        record_size = 300
        s, a, r, s_, q_list, loss = self.q_value_cal(s, a, r, s_)
        self.memory.add(loss, [s, a, r, s_, q_list])
        self.count += 1

        # train when do record_size times actions.
        if self.count % record_size == 0:
            batch, idxs, is_weight = self.memory.sample(batch_size)
            Train = copy.copy(batch)
            X_train = np.array(Train)[:, 0]
            Y_train = np.array([i for i in np.array(Train)[:, 4]])
            print(X_train.shape)
            print(Y_train.shape)
            self.model.fit(X_train, Y_train, epochs = self.epochs)

            # update prioritized experience
            for i in range(batch_size):
                _s, _a, _r, _s_ = batch[i][0], batch[i][1], batch[i][2], batch[i][3]
                loss = self.q_value_cal(_s, _a, _r, _s_)[5]
                self.memory.update(idxs[i], loss)

    
    def q_value_cal(self, s, a, r, s_):
        # calculate q value
        q_list = self.model.predict([s])[0]
        q_predict = q_list[a]
        qvalue = self.dump_model.predict([s_])[0][np.argmax(q_list)]
        #q_target = r + self.gamma * qvalue
        loss = qvalue - q_predict
        q_list[a] += r + self.gamma * qvalue - q_predict 
        return s, a, r, s_, q_list, loss
            


