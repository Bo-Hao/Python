import numpy as np 
import pickle


class GrowUp:
    def __init__(self, num = 2, fin_step = 2):
        self.num = num
        self.fin_step = fin_step
        self.action_space = ['+', '-', '*', '/']
        self.n_actions = len(self.action_space)
        self.ini_step = 0
        self.score = 1
        self.record = []
        self.bonus = 1
        
    def step(self, action):
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
            reward = self.score
        else:
            done = False
            reward = 0
        self.ini_step += 1
        s_ = self.ini_step


        return s_, reward, done        


if __name__ == "__main__":
    import numpy.random as np
    G = GrowUp()
    done = False
    actions = ['+', '-', '*', '/', '**']
    while done == False:
        action = actions[np.randint(4, size = 1)[0]]
        s_, reward, done = G.step(action)
    
    print(s_, reward, done)