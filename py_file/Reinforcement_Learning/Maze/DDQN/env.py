import copy 
class Env:
    def __init__(self, number = 2, final_step = 5):
        self.number = number
        self.final_step = final_step
        self.action_space = ['+', '-', '*', '/']
        self.n_actions = len(self.action_space)
        self.initial_step = 0
        self.score = 0
        
    def step(self, action):
        tmp = copy.copy(self.score)
        if action == 0:
            self.score += self.number
        elif action == 1:
            self.score -= self.number
        elif action == 2:
            self.score = self.number * self.score
        elif action == 3:
            self.score = self.score / self.number
        '''elif action == 4:
            self.score = self.score**self.num'''
        
        # reward function
        if self.initial_step == self.final_step - 1:
            done = True
            reward = self.score - tmp

        else:
            if tmp >= self.score:
                reward = -0.1 + self.score - tmp
            else:
                reward = 0.1 + self.score - tmp
            done = False

        self.initial_step += 1
        s_ = self.initial_step

        return s_, reward, done        

    