import numpy as np 



class Env:
    def __init__(self):
        # ['up', 'down', 'left', 'right']
        self.action_space = [0, 1, 2, 3]
        self.n_actions = len(self.action_space)
        self.state = [0, 0]
        self.step_num = 0

        self.badpoint = [[1, 2], [2, 1]] 
        self.goal = [[2, 2]]
        
        self.map = np.array([[0 for i in range(4)] for j in range(4)])
        for i in self.badpoint:
            self.map[i[0]][i[1]] = -1
        for i in self.goal:
            self.map[i[0]][i[1]] = 1

    def return_s(self):
        return self.state
        
    def step(self, a):
        self.step_num += 1
        s = self.state
        r = 0

        if a == 0: # up
            self.state[1] -= 1
        elif a == 1: # down
            self.state[1] += 1
        elif a == 2: # left
            self.state[0] -= 1
        elif a == 3: # right
            self.state[0] += 1

        
        if self.state in self.badpoint:
            done = True
            r = -1
        elif self.state in self.goal:
            done = True
            r = 1
        elif self.state[0] > 3 or self.state[0] < 0:
            done = True
            r = -1
        elif self.state[1] > 3 or self.state[1] < 0:
            done = True
            r = -1
        else:
            done = False
            r = 0
        

        s_ = self.state

        return s, a, r, s_, done




    