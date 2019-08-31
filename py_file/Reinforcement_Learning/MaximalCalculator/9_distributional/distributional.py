from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
import copy 
import numpy as np
from build_model import Build_Model
from Memory import Memory
import math 

class DistributionalRL:
    def __init__(self, actions, gamma=0.1, e_greedy = 0.9):
        state_size = 1
        neurons = 24

        self.actions = actions
        self.gamma = gamma
        self.epsilon = e_greedy
        self.lr = 0.1
        self.count = 0
        self.epochs = 5

        self.v_max = 10
        self.v_min = -10
        self.atoms = 5
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]
    
        self.m = Build_Model(state_size, neurons, len(actions), atoms = self.atoms)
        self.model = self.m.model
        self.dump_model = copy.copy(self.model)

        self.capacity = 200
        self.memory = Memory(self.capacity)

    def choose_action(self, s):
        '''q_list = []
        for i in self.model.predict([s]):
            q_list.append(sum([self.z[j] * i[0][j] for j in range(self.atoms)]))'''

        if np.random.uniform() < self.epsilon:
            # choose the best action
            state_action = self.get_q_list(s, self.model)
            action = np.random.choice([i for i in range(len(state_action)) if state_action[i] == max(state_action)])
        else:
            # choose action randomly
            action = np.random.choice(self.actions)
        return action


    def learn(self, s, a, r, s_):
        batch_size = 50
        record_size = 200
        s, a, r, s_, loss, q_distribution = self.get_q_value(s, a, r, s_)
        self.memory.add(loss, [s, a, r, s_, q_distribution])
        self.count += 1

        # train when do record_size times actions.
        if self.count % record_size == 0:
            batch, idxs, is_weight = self.memory.sample(batch_size)
            Train = copy.copy(batch)

            X_train = np.array(Train)[:, 0]
            Y_train = np.array([i for i in np.array(Train)[:, 4]]).reshape(len(self.actions), batch_size ,self.atoms)
 
            self.model.fit(
                X_train,
                [np.array(Y_train[i]) for i in range(len(self.actions))],
                epochs = self.epochs,
                #verbose=0
                )
            
            # update prioritized experience
            for i in range(batch_size):
                _s, _a, _r, _s_ = batch[i][0], batch[i][1], batch[i][2], batch[i][3]
                loss = self.get_q_value(_s, _a, _r, _s_)[4]
                self.memory.update(idxs[i], loss)

    
    def get_q_value(self, s, a, r, s_):
        # calculate q value
        q_distribution = self.model.predict([s]) # shape: (action_size, atoms)
        q_distribution_next = self.model.predict([s_])
        q_list = self.get_q_list(s, self.model)
        m_prop = [np.zeros(self.atoms)]
        for j in range(self.atoms - 1):
            Tz = min(self.v_max, max(self.v_min, r + self.gamma * self.z[j]))
            bj = (Tz - self.v_min) / self.delta_z
            l, u =  int(math.floor(bj)), int(math.ceil(bj))

            m_prop[0][l] += q_distribution_next[a][0][j] * (u - bj)
            m_prop[0][u] += q_distribution_next[a][0][j] * (bj - l)
        loss = -sum([m_prop[0][i] * np.log(q_distribution[a][0][i]) + 10**(-10) for i in range(self.atoms)])
        q_distribution[a][:] = m_prop[:]
        return s, a, r, s_, loss, q_distribution

    def get_q_list(self, s, model):
        q_list = []
        for i in model.predict([s]):
            q_list.append(sum([self.z[j] * i[0][j] for j in range(self.atoms)]))
        return q_list