import copy 

class Env:
    def __init__(self, number = 2, final_step = 4):
        self.number = number
        self.final_step = final_step
        self.action_space = ['+', '-', '*', '/']
        self.n_actions = len(self.action_space)
        self.initial_step = 0
        self.score = 0
        self.grow = 0
        self.tmp = 0

    def step(self, action):
        self.grow = copy.copy(self.tmp)
        self.tmp = copy.copy(self.score)

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
            r =  -0.1 + (self.score - self.tmp) - (self.tmp - self.grow)

        else:
            done = False
            r = -0.1 + (self.score - self.tmp) - (self.tmp - self.grow)

        s = copy.copy(self.initial_step)
        a = action
        
        self.initial_step += 1
        s_ = self.initial_step

        return s, a, r, s_, done        

    