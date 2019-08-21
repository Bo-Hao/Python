import pandas as pd 
import numpy as np 

class QLearningTable:
    def __init__(self, actions, learning_rate=0.1, gamma=0.9, e_greedy=0.7):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greedy
        self.q_table = [[0]*len(self.actions)]

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table[observation]

            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice([i for i in range(len(state_action)) if state_action[i] == max(state_action)])
            print('action = ', action)
        else:
            # choose random action
            action = np.random.choice(self.actions)
 
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table[s][a]
        #print(max(self.q_table[s_]) ,self.q_table[s][a])
        q_target = r + self.gamma * (max(self.q_table[s_]))

        self.q_table[s][a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state > len(self.q_table)-1:
            # append new state to q table
            self.q_table.append([0]*len(self.actions))
