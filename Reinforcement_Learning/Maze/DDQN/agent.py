from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
import copy 
import numpy as np
from build_model import Build_Model
from Memory import Memory
from tensorflow.compat.v1.losses import huber_loss
from tensorflow.compat.v1.train import get_or_create_global_step
from tensorflow.keras.utils import Progbar

class Agent:
    def __init__(self, actions, gamma=0.7, e_greedy = 0.7):
        self.actions = actions  # a list
        self.gamma = gamma
        self.epsilon = e_greedy
        self.lr = 0.01
        self.count = 0
        self.epochs = 50
        self.bar = Progbar(self.epochs)
        self.epoch_loss_avg = tf.keras.metrics.Mean()

        self.batch_size = 100
        self.state_size = 2
        self.record_size = 200

        # initial model include the hard-working one and the dump one.
        M = Build_Model(self.state_size, 16, len(actions))
        self.model = M.build()
        self.dump_model = copy.copy(self.model)
        self.optimizer = tf.optimizers.Adam(lr = self.lr)
        # initial memory with sum tree
        self.capacity = 200
        self.memory = Memory(self.capacity)

    def choose_action(self, s):
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.model.predict([[s]])[0]
            action = np.argmax(state_action)
            #action = np.random.choice([i for i in range(len(state_action)) if state_action[i] == max(state_action)])
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def store(self, s, a, r, s_, done):
        self.memory.add(1, [s, a, r, s_, done])
        self.count += 1

    def learn(self):
        loss_record = []
        batch, idxs, is_weight = self.memory.sample(self.batch_size)
        X_train = np.zeros((self.batch_size, self.state_size))
        Y_train = [np.zeros(len(self.actions)) for i in range(self.batch_size)]
        for i in range(self.batch_size):
            _s, _a, _r, _s_, done_ = batch[i][0], batch[i][1], batch[i][2], batch[i][3], batch[i][4]
            q_list, loss = self.get_loss(_s, _a, _r, _s_, done_)
            loss_record.append(loss)
            X_train[i] = _s
            for i_ in range(len(self.actions)):
                Y_train[i][i_] = q_list[i_]


        # Train!
        print('-----------Training-----------')
        for i in range(self.epochs):
            self.train(self.model, X_train, Y_train)
            self.bar.update(i, values=[('loss', self.epoch_loss_avg.result().numpy())])

        # update prioritized experience
        for i in range(self.batch_size):
            loss = loss_record[i] * is_weight[i]
            self.memory.update(idxs[i], loss)

    
    def get_loss(self, s, a, r, s_, done):
        # calculate q value
        q_list = self.model.predict([[s]])[0]
        q_predict = q_list[a]
        qvalue = self.dump_model.predict([[s_]])[0][np.argmax(q_list)]
        loss = r + self.gamma * qvalue - q_predict
        if done:
            q_target = r
            
        else:
            q_target = loss
        q_list[a] =r + self.gamma * qvalue
  
        return q_list, loss

    def _loss(self, model, x, y):
        x = np.array(x)
        y_ = model(x)
        loss = huber_loss(y, y_)
        return loss
	
    def _grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self._loss(model, inputs, targets)
            self.epoch_loss_avg(loss_value)
            return tape.gradient(loss_value, self.model.trainable_variables)

    def train(self, model, s, q):
        grads = self._grad(model, s, q)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables),
            get_or_create_global_step())
        

