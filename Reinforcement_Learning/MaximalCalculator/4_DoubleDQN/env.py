import numpy as np 

class GrowUp:
    def __init__(self, num = 2, fin_step = 5):
        self.num = num
        self.fin_step = fin_step
        self.action_space = ['+', '-', '*', '/']
        self.n_actions = len(self.action_space)
        self.ini_step = 0
        self.score = 0
        
    def step(self, action):
        tmp = self.score
        if action == 0:
            self.score += self.num
        elif action == 1:
            self.score -= self.num
        elif action == 2:
            self.score = self.num * self.score
        elif action == 3:
            self.score = self.score / self.num
        '''elif action == 4:
            self.score = self.score**self.num'''
        
        # reward function
        if self.ini_step == self.fin_step - 1:
            done = True
            reward = self.score - tmp

        else:
            reward = self.score - tmp

            done = False
        self.ini_step += 1
        s_ = self.ini_step

        return s_, reward, done        

    