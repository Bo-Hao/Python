from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
from tensorflow.compat.v1.train import get_or_create_global_step
from tensorflow.compat.v1.losses import huber_loss
import copy 
import numpy as np
from build_model import Build_Model
from Memory import Memory
import math 
from decorator import *


class Agent:
    def __init__(self, actions, gamma=0.1, e_greedy = 0.9):
        self.state_size = 4
        neurons = 24

        self.actions = actions
        self.gamma = gamma
        self.epsilon = e_greedy
        self.lr = 0.1
        self.count = 0
        self.epochs = 50

        self.v_max = 10
        self.v_min = -10
        self.atoms = 51
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]
    
        self.m = Build_Model(self.state_size, neurons, len(actions), atoms = self.atoms)
        self.m.build()
        self.model = self.m.model
        self.dump_model = copy.copy(self.model)
        self.optimizer = tf.optimizers.Adam(lr = self.lr)
        self.batch_size = 100

        self.capacity = 300
        self.memory = Memory(self.capacity)

        self.record_size = self.capacity

        
    @timecost
    def choose_action(self, s):
        if np.random.uniform() < self.epsilon:
            # choose the best action
            state_action = []
            for i in self.model.predict([[s]]):
                state_action.append(np.sum([self.z[j] * i[0][j] for j in range(self.atoms)]))
            action = np.random.choice([i for i in range(len(state_action)) if state_action[i] == max(state_action)])
        else:
            # choose action randomly
            action = np.random.choice(self.actions)
        return action

    #@timecost
    def learn(self, s, a, r, s_, done):
        loss, q_distribution = self.get_q_value(s, a, r, s_, done)
        self.memory.add(loss, [s, a, r, s_, q_distribution])
        self.count += 1

        # train when do record_size times actions.
        if self.count % self.record_size == 0:
            batch, idxs, is_weights = self.memory.sample(self.batch_size)
            X_train = np.zeros((self.batch_size, self.state_size))
            Y_train =[np.zeros((self.batch_size, self.atoms)) for i in range(len(self.actions))]

            for i in range(self.batch_size): 
                X_train[i] = batch[i][0]
                for i_ in range(len(self.actions)):
                    Y_train[i_][i][:] = batch[i][4][i_][:]

            print('-----training-----')
            for i in range(self.epochs):
                self.train(X_train, Y_train)


            # update prioritized experience
            for i in range(self.batch_size):
                _s, _a, _r, _s_, is_weight = batch[i][0], batch[i][1], batch[i][2], batch[i][3], is_weights[i]
                loss = self.get_q_value(_s, _a, _r, _s_, done)[0]
                self.memory.update(idxs[i], is_weight*loss)
            

    #@timecost
    def get_q_value(self, s, a, r, s_, done):
        p = self.model.predict([[s]])
        old_q = np.sum(np.multiply(np.vstack(p), np.array(self.z)), axis=1) 
		# 一樣有 double dqn
        p_next = self.model.predict([[s_]])
        q = np.sum(np.multiply(np.vstack(p_next), np.array(self.z)), axis=1) 

        p_d_next = self.dump_model.predict([[s_]])
        next_action_idxs = np.argmax(q)
		# init m 值
        m_prob = [np.zeros((1, self.atoms))]
		# action 後更新 m 值
        if done: # Distribution collapses to a single point
            Tz = min(self.v_max, max(self.v_min, r))
            bj = (Tz - self.v_min) / self.delta_z 
            m_l, m_u = math.floor(bj), math.ceil(bj)
            m_prob[0][0][int(m_l)] += (m_u - bj)
            m_prob[0][0][int(m_u)] += (bj - m_l)
        else:
            for j in range(self.atoms):
                Tz = min(self.v_max, max(self.v_min, r + self.gamma * self.z[j]))
                bj = (Tz - self.v_min) / self.delta_z
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[0][0][int(m_l)] += p_d_next[next_action_idxs][0][j] * (m_u - bj)
                m_prob[0][0][int(m_u)] += p_d_next[next_action_idxs][0][j] * (bj - m_l)
		# 更新後放回p，回去訓練
        p[a][0][:] = m_prob[0][0][:]
		# 計算q估計
        new_q = np.sum(np.multiply(np.vstack(p), np.array(self.z)), axis=1) 
		#計算 error 給PER
        error = abs(old_q[a] - new_q[a])
        return error, p


    def _loss(self, model, x, y):
        y_ = self.model(x)
        #loss = sum(sum(tf.nn.softmax_cross_entropy_with_logits(y, y_)))
        loss = tf.nn.softmax_cross_entropy_with_logits(y, y_)

        return loss
	
    def _grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self._loss(self.model, inputs, targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def train(self, s, q):
        loss_value, grads = self._grad(self.model, s, q)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables),
            get_or_create_global_step()
                )
